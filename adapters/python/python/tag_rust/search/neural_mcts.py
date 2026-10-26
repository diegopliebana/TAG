"""Frozen PyTorch policy/value predictions for batched Rust MCTS requests."""
import json
import numpy as np
from tag_rust.learning.inputs import pack_observations
from tag_rust.learning.snapshot import FrozenModel
from .mcts import MctsResponse


class NeuralMctsEvaluator:
    """Bind a frozen model to declared supported observers and turn contexts.

    any_turn admits non-acting observers for value prediction. own_turn admits only
    observations where the observer is the next player. These are explicit model
    compatibility declarations, not evidence of prediction quality on those inputs.
    """
    def __init__(self, snapshot, *, supported_observers=None, turn_context="any_turn"):
        if not isinstance(snapshot, FrozenModel):
            raise TypeError("snapshot must be FrozenModel")
        self._snapshot = snapshot
        players = snapshot.contract.identity.player_count
        observers = tuple(range(players)) if supported_observers is None else tuple(supported_observers)
        if (not observers or len(set(observers)) != len(observers)
                or any(type(p) is not int or not 0 <= p < players for p in observers)
                or turn_context not in ("any_turn", "own_turn")):
            raise ValueError("declare supported observers and turn context")
        self._observers, self._turn_context = observers, turn_context
        self._identity = json.dumps({"type":"neural-mcts", "version":1,
            "model":snapshot.identity, "observers":observers,"turn_context":turn_context},
            sort_keys=True, separators=(",", ":"))

    @property
    def identity(self):
        return self._identity

    @property
    def payloads(self):
        return tuple(t.name for t in self._snapshot.contract.tensors)

    def validate(self, spec, kind, observers, completion):
        contract = self._snapshot.contract
        if spec.identity != contract.identity or spec.actions != contract.actions:
            raise ValueError("MCTS model game/encoding/action specification mismatch")
        available = {payload.name: payload.tensor for payload in spec.payloads}
        if any(available.get(tensor.name) != tensor for tensor in contract.tensors):
            raise ValueError("MCTS model observation tensor specification mismatch")
        if contract.value.units != "win=1,draw=0,loss=-1":
            raise ValueError("neural MCTS currently requires win=1,draw=0,loss=-1 value units")
        if contract.actions.mapping == "candidates":
            raise ValueError("candidate-feature neural MCTS needs a declared candidate encoder")
        if spec.identity.game in ("cantstop", "loveletter"):
            game_name = "Love Letter" if spec.identity.game == "loveletter" else "Can't Stop"
            if spec.identity.game == "loveletter" and any(t.name == "last_private_history" for t in contract.tensors):
                raise ValueError("Love Letter MCTS cannot supply previous-round private history for hypothetical opponents")
            if (contract.value.players != tuple(range(spec.identity.player_count))
                    or contract.value.gamma != 1):
                raise ValueError(f"{game_name} neural MCTS requires absolute-seat full player values with gamma=1")
            if set(self._observers) != set(range(spec.identity.player_count)):
                raise ValueError(f"{game_name} neural MCTS requires every acting observer")
            if kind == "values" and (observers != "deciding_player" or completion != "require_full_vector"):
                raise ValueError(f"{game_name} neural values require deciding_player and require_full_vector")
        if kind == "values":
            full = contract.value.players is not None and set(contract.value.players) == set(range(spec.identity.player_count))
            if contract.value.players is not None and not full:
                raise ValueError("MCTS vector model must provide every player value")
            if not full and observers != "all_players" and completion == "require_full_vector":
                raise ValueError("strict values require all-player scalar observations or a full-vector model")
            if observers == "all_players" and (set(self._observers) != set(range(spec.identity.player_count))
                                               or self._turn_context == "own_turn"):
                raise ValueError("all-player evaluation requires every observer and non-acting turn contexts")

    def evaluate(self, requests, *, max_bytes=64 * 1024 * 1024):
        requests = tuple(requests)
        if not requests or any(request.evaluator_identity != self.identity for request in requests):
            raise ValueError("request model binding identity mismatch")
        observations = tuple(obs for request in requests for obs in request.observations)
        for obs in observations:
            if obs.observer not in self._observers or (self._turn_context == "own_turn" and obs.next_player != obs.observer):
                raise ValueError("model does not support this observer/turn context")
        batch = pack_observations(self._snapshot.contract, observations, max_bytes=max_bytes)
        predictions = self._snapshot.predict(batch)
        responses, offset = [], 0
        players = self._snapshot.contract.identity.player_count
        order = self._snapshot.contract.value.players
        for request in requests:
            if request.kind == "policy":
                mapping = {int(action): float(probability) for action, probability in
                           zip(predictions.action_ids[offset], predictions.probabilities[offset]) if action >= 0}
                if any(action not in mapping for action in request.action_ids):
                    raise ValueError("neural policy omitted a requested action ID")
                response = MctsResponse.for_request(request, [mapping[action] for action in request.action_ids])
            elif order is not None:
                response = MctsResponse.for_request(request,
                    [predictions.values[offset, order.index(player)] for player in range(players)])
            elif len(request.observations) == players and {obs.observer for obs in request.observations} == set(range(players)):
                values = np.empty(players, dtype=np.float64)
                for row, obs in enumerate(request.observations):
                    values[obs.observer] = predictions.values[offset + row]
                response = MctsResponse.for_request(request, values)
            elif len(request.observations) == 1:
                response = MctsResponse.for_request(request, [predictions.values[offset]],
                                                     scalar_player=request.observations[0].observer)
            else:
                raise ValueError("incomplete or ambiguous scalar player observations")
            responses.append(response)
            offset += request.rows
        return tuple(responses)
