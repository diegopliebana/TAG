"""Batched neural learner decisions and value-only OSLAPython evaluation."""
import math
import json
from dataclasses import replace
import numpy as np
from tag_rust.engine.api import EncodingIdentity
from tag_rust.search.evaluation import OSLAPython, EvaluatorIdentity
from tag_rust.learning.inputs import pack_rows, _dimension


def observer_values(contract, values, observers):
    """Select each row's declared observer value without changing perspective."""
    players = contract.value.players
    if players is None:
        return values.copy()
    if any(int(observer) not in players for observer in observers):
        raise ValueError("model value outputs omit a requested observer")
    columns = [players.index(int(observer)) for observer in observers]
    return values[np.arange(len(observers)), columns].copy()


def pack_episode_rows(contract, observations, infos):
    """Construct an inference batch from Gym observation dictionaries and player
    metadata. The batch contains independent copies of the selected features."""
    if len(observations) != len(infos) or not observations:
        raise ValueError("provide matching nonempty observation/info rows")
    for info in infos:
        identity = info["identity"]
        if not isinstance(identity, EncodingIdentity):
            identity = EncodingIdentity(**identity)
        if identity != contract.identity:
            raise ValueError("episode encoding differs from model")
    masks = []
    for observation in observations:
        mask = observation["action_mask"]
        if (not isinstance(mask, np.ndarray) or mask.dtype not in (np.dtype("int8"), np.dtype("bool"))
                or not np.isin(mask, (0, 1)).all()):
            raise ValueError("Gym masks must contain Boolean or int8 zero/one entries")
        masks.append(mask.astype(bool))
    return pack_rows(contract, {t.name:np.stack([o[t.name] for o in observations]) for t in contract.tensors},
                     np.array([_dimension(i["observer"], "observer", zero=True) for i in infos], dtype=np.int64),
                     np.array([-1 if i["next_player"] is None else _dimension(i["next_player"], "next player", zero=True) for i in infos], dtype=np.int64),
                     legal_mask=np.stack(masks))


class NeuralPolicy:
    """Choose actions for active episodes with one batched policy evaluation.

    Each episode samples actions using its own random-number generator."""
    def __init__(self, snapshot, rngs):
        if snapshot.contract.actions.mapping == "candidates":
            raise ValueError("current learner episodes require fixed action IDs")
        self._snapshot, self.rngs = snapshot, rngs
        self.last_statistics = {}

    @property
    def snapshot(self):
        return self._snapshot

    def choose_batch(self, members, observations, infos):
        """Return one action ID for each active episode. Retain the fixed policy
        snapshot's value predictions and action log-probabilities for training."""
        self.last_statistics = {}
        if len(members) != len(observations) or len(set(members)) != len(members):
            raise ValueError("members must uniquely identify the supplied rows")
        batch = pack_episode_rows(self.snapshot.contract, observations, infos)
        if not batch.legal_mask.any(axis=1).all():
            raise ValueError("policy sampling requires a decision on every active row")
        for member in members:
            if member not in self.rngs or not isinstance(self.rngs[member], np.random.Generator):
                raise ValueError("provide a NumPy Generator for every active member")
        output = self.snapshot.predict(batch)
        values = observer_values(self.snapshot.contract, output.values, np.array([i["observer"] for i in infos]))
        actions, statistics = [], {}
        for row, member in enumerate(members):
            p = output.probabilities[row].astype(np.float64)
            p /= p.sum()
            index = int(self.rngs[member].choice(len(p), p=p))
            actions.append(int(output.action_ids[row, index]))
            statistics[member] = dict(value=float(values[row]), log_probability=math.log(float(p[index])),
                                      model_identity=self.snapshot.identity)
        self.last_statistics = statistics
        return actions


class NeuralValueEvaluator:
    """Score successor requests for their original observers using one frozen model.

    Terminal rewards correspond to Win/Draw/Loss result tags, in that order.
    Their units/discount clock must agree with the model's declared value meaning.
    Rust retains request validity, candidate selection and RNG/commit ownership.
    """
    def __init__(self, snapshot, *, terminal_rewards=(1., 0., -1.),
                 units="win=1,draw=0,loss=-1", gamma=0.99, discount_clock="learner-step"):
        value = snapshot.contract.value
        if (value.units, value.gamma, value.discount_clock) != (units, gamma, discount_clock):
            raise ValueError("model value units/discount convention differs from evaluator")
        rewards = np.asarray(terminal_rewards, dtype=np.float64)
        if rewards.shape != (3,) or not np.isfinite(rewards).all():
            raise ValueError("declare finite win/draw/loss terminal rewards")
        if value.bounds is not None and ((rewards < value.bounds[0]).any() or (rewards > value.bounds[1]).any()):
            raise ValueError("terminal rewards exceed declared value bounds")
        self._snapshot = snapshot
        self._rewards = rewards.copy()
        configuration = json.dumps(dict(snapshot=snapshot.identity, digest_version=1,
                                       terminal_rewards=rewards.tolist(), units=units, gamma=gamma,
                                       discount_clock=discount_clock), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._identity = EvaluatorIdentity("neural-value", 1, configuration)

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def identity(self):
        return self._identity

    def controller(self, *, seed=None, **limits):
        """Create existing OSLAPython settings tied to this snapshot/evaluation meaning."""
        return OSLAPython(evaluator_id=self.identity.name, evaluator_version=self.identity.version,
                          evaluator_parameters=self.identity.parameters, seed=seed,
                          payloads=tuple(t.name for t in self.snapshot.contract.tensors), **limits)

    def evaluate(self, request):
        """Return owned float64 values in original candidate order; perform no commits."""
        contract = self.snapshot.contract
        if request.identity != contract.identity or request.evaluator != self.identity:
            raise ValueError("request model/evaluator/encoding identity mismatch")
        ids, results, next_players = request.action_ids, request.results, request.next_players
        if not isinstance(ids, np.ndarray) or ids.dtype != np.int64:
            raise TypeError("candidate IDs must be int64")
        rows, players = len(ids), contract.identity.player_count
        if ids.shape != (rows,) or rows == 0 or (ids < 0).any():
            raise ValueError("invalid candidate ID layout")
        if (not isinstance(results, np.ndarray) or results.dtype != np.int8
                or results.shape != (rows, players) or not np.isin(results, (0, 1, 2, 3)).all()):
            raise ValueError("invalid result codes/layout")
        observers = request.observers if hasattr(request, "observers") else np.full(rows, _dimension(request.actor, "actor", zero=True), dtype=np.int64)
        batch = pack_rows(contract, request.payloads, observers, next_players)
        # Check vector coverage even when every row is terminal.
        if contract.value.players is not None and any(int(o) not in contract.value.players for o in observers):
            raise ValueError("model omits a requested observer")
        codes = results[np.arange(rows), observers]
        if ((next_players == -1) & (codes == 0)).any():
            raise ValueError("ended rows require a declared observer result")
        values = np.empty(rows, dtype=np.float64)
        terminal = codes != 0
        values[terminal] = self._rewards[codes[terminal] - 1]
        ongoing = np.flatnonzero(~terminal)
        if len(ongoing):
            selected = replace(batch, payloads={k:v[ongoing].copy() for k,v in batch.payloads.items()},
                               context=batch.context[ongoing].copy(), legal_mask=batch.legal_mask[ongoing].copy(),
                               action_ids=batch.action_ids[ongoing].copy(),
                               candidate_features=None if batch.candidate_features is None else batch.candidate_features[ongoing].copy())
            predictions = self.snapshot.predict(selected)
            values[ongoing] = observer_values(contract, predictions.values, observers[ongoing])
        return values
