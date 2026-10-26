"""Position-level search targets and bounded, reproducible training storage."""
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
import numpy as np
from tag_rust.learning.inputs import pack_observations


def owned_observation(observation):
    """Copy training arrays without retaining an executable live-game action token."""
    return replace(observation,position=None,
        payloads=MappingProxyType({name:value.copy() for name,value in observation.payloads.items()}),
        progress=observation.progress.copy(),results=observation.results.copy(),
        action_mask=observation.action_mask.copy(),offered_action_ids=observation.offered_action_ids.copy())


def visit_target(observation,report):
    """Align completed visits with permanent IDs; zero visits cannot teach a policy."""
    actions = tuple(int(action) for action in observation.offered_action_ids)
    rows = report["root_actions"]
    if (len(rows) != len(actions) or any(type(row["action_id"]) is not int for row in rows)
        or {row["action_id"] for row in rows} != set(actions)):
        raise ValueError("search action identities differ from the collected decision")
    visits = np.zeros(len(observation.action_mask),dtype=np.int64)
    for row in rows:
        count = row["visits"]
        if type(count) is not int or not 0 <= count <= np.iinfo(np.int64).max:
            raise ValueError("visit count is outside the stored int64 range")
        visits[row["action_id"]] = count
    total = sum(int(value) for value in visits)
    if total == 0:
        raise ValueError("self-play requires at least one completed root visit")
    if report["action_id"] not in actions:
        raise ValueError("search selected an unavailable action")
    return visits,(visits.astype(np.float64)/total).astype(np.float32)


@dataclass(frozen=True)
class PositionExample:
    observations: tuple
    actor: int
    policy: np.ndarray
    visits: np.ndarray
    terminal_values: np.ndarray
    action: int
    game_id: int
    ply: int
    model_identity: str
    search_seed: int
    noise_seed: int
    search_report: dict

    @property
    def nbytes(self):
        return self.policy.nbytes+self.visits.nbytes+self.terminal_values.nbytes+sum(
            sum(a.nbytes for a in obs.payloads.values())+obs.progress.nbytes+obs.results.nbytes+
            obs.action_mask.nbytes+obs.offered_action_ids.nbytes for obs in self.observations)

    def copy(self):
        return replace(self,observations=tuple(owned_observation(obs) for obs in self.observations),
            policy=self.policy.copy(),visits=self.visits.copy(),terminal_values=self.terminal_values.copy(),
            search_report=json.loads(json.dumps(self.search_report,allow_nan=False)))


def validate_example(example,contract,*,max_bytes):
    players,actions = contract.identity.player_count,contract.actions.size
    for name in ("actor","action","game_id","ply","search_seed","noise_seed"):
        value = getattr(example,name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid example {name}")
    if example.actor >= players or example.search_seed >= 2**64 or example.noise_seed >= 2**64:
        raise ValueError("invalid example player/seed")
    if not isinstance(example.model_identity,str) or not example.model_identity:
        raise ValueError("example model identity is required")
    expected = set(range(players)) if contract.value.players is None else {example.actor}
    observers = [obs.observer for obs in example.observations]
    if len(observers) != len(expected) or set(observers) != expected:
        raise ValueError("example observations do not cover the required player perspectives")
    pack_observations(contract,example.observations,max_bytes=max_bytes)
    for observation in example.observations:
        ids = observation.offered_action_ids
        if (observation.progress.dtype != np.int64 or observation.progress.shape != (4,)
            or (observation.progress < 0).any() or observation.results.dtype != np.int8
            or observation.results.shape != (players,) or not np.isin(observation.results,(0,1,2,3)).all()
            or ids.dtype != np.int64 or ids.ndim != 1 or len(ids) != len(set(ids.tolist()))
            or set(ids.tolist()) != set(np.flatnonzero(observation.action_mask).tolist())):
            raise ValueError("invalid recorded observation metadata/action mapping")
    if any(obs.next_player != example.actor for obs in example.observations):
        raise ValueError("example observations disagree about the acting player")
    if contract.identity.game == "loveletter":
        if contract.value.players != tuple(range(players)) or "last_private_history" in {
            tensor.name for tensor in contract.tensors
        }:
            raise ValueError("Love Letter examples require full player values and supported search inputs")
        if any((obs.results != 0).any() or (
            "terminal_values" in obs.payloads and obs.payloads["terminal_values"].any()
        ) for obs in example.observations):
            raise ValueError("Love Letter decision inputs must precede the whole-game outcome")
    policy,visits,values = example.policy,example.visits,example.terminal_values
    if (not isinstance(policy,np.ndarray) or policy.dtype != np.float32 or policy.shape != (actions,)
        or not np.isfinite(policy).all() or (policy < 0).any() or not np.isclose(policy.sum(),1,atol=1e-6)
        or not isinstance(visits,np.ndarray) or visits.dtype != np.int64 or visits.shape != (actions,) or (visits < 0).any()
        or not isinstance(values,np.ndarray) or values.dtype != np.float32 or values.shape != (players,)
        or not np.isin(values,(-1.,0.,1.)).all()):
        raise ValueError("invalid policy, visits or complete terminal reward vector")
    if contract.identity.game == "loveletter" and (not (values == 1).any() or (values == 0).any()):
        raise ValueError("Love Letter whole-game targets require winners and win/loss values only")
    actor = next(obs for obs in example.observations if obs.observer == example.actor)
    total = sum(int(value) for value in visits)
    if (not 0 <= example.action < actions or not actor.action_mask[example.action]
        or (policy[~actor.action_mask] != 0).any() or (visits[~actor.action_mask] != 0).any()
        or total == 0 or not np.allclose(policy,visits.astype(np.float64)/total,atol=1e-6,rtol=1e-6)):
        raise ValueError("policy target does not match legal completed visits")
    if example.search_report["settings"]["seed"] != example.search_seed:
        raise ValueError("example search seed differs from its effective settings")
    json.dumps(example.search_report,allow_nan=False)
    if example.nbytes > max_bytes:
        raise MemoryError("one training position exceeds its numeric byte limit")


class ReplayBuffer:
    """FIFO capacity is measured in positions; scalar player rows stay together."""
    def __init__(self,settings,contract):
        self.settings,self.contract = settings,contract
        self._positions = deque()
        self.nbytes = 0
        self.dropped = 0

    def __len__(self):
        return len(self._positions)

    @property
    def positions(self):
        return tuple(position.copy() for position in self._positions)

    def add_group(self,positions):
        positions = tuple(positions)
        # Reject the whole incoming group before evicting any existing examples.
        for position in positions:
            validate_example(position,self.contract,max_bytes=self.settings.max_bytes)
        if self.settings.mode == "newest_group":
            self.dropped += len(self._positions)
            self._positions.clear()
            self.nbytes = 0
        for position in positions:
            owned = position.copy()
            self._positions.append(owned)
            self.nbytes += owned.nbytes
            while len(self) > self.settings.capacity or self.nbytes > self.settings.max_bytes:
                self.nbytes -= self._positions.popleft().nbytes
                self.dropped += 1

    def sample(self,count,rng):
        if not self._positions:
            raise ValueError("cannot sample an empty replay buffer")
        indices = rng.integers(len(self),size=count)
        positions = list(self._positions)
        return tuple(positions[int(index)].copy() for index in indices),tuple(int(i) for i in indices)

    def digest(self):
        digest = hashlib.sha256()
        for position in self._positions:
            digest.update(json.dumps([position.game_id,position.ply,position.actor,position.action,
                position.model_identity,position.search_seed,position.noise_seed],separators=(",",":")).encode())
            for array in (position.policy,position.visits,position.terminal_values):
                digest.update(array.tobytes())
            for obs in position.observations:
                digest.update(str(obs.observer).encode())
                for array in (*obs.payloads.values(),obs.progress,obs.results,obs.action_mask,obs.offered_action_ids):
                    digest.update(array.tobytes())
        return digest.hexdigest()
