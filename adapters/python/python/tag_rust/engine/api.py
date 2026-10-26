"""Give the common Rust interface readable Python objects, without game-specific rules."""
from dataclasses import dataclass, asdict
from operator import index
from types import MappingProxyType
from collections.abc import Mapping, Sequence
import numpy as np
from .. import _engine


@dataclass(frozen=True)
class EncodingIdentity:
    """The game, settings and versions that determine what array entries mean."""
    game: str
    rules_version: int
    encoding_version: int
    # Legacy placement fields stay available for existing users and saved tables.
    # General games leave these zero and use configuration for their settings.
    width: int = 0
    height: int = 0
    target: int = 0
    player_count: int = 2
    configuration: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        settings = self.configuration
        if isinstance(settings, Mapping):
            settings = settings.items()
        settings = tuple(sorted(tuple(pair) for pair in settings))
        if (any(len(pair) != 2 or any(not isinstance(s, str) for s in pair) for pair in settings)
                or len({pair[0] for pair in settings}) != len(settings)):
            raise ValueError("configuration requires unique string keys and string values")
        object.__setattr__(self, "configuration", settings)


@dataclass(frozen=True)
class TensorSpec:
    """Describe an array's name, element type, dimensions and entry meanings."""
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    components: tuple[str, ...]
    bounds: tuple[int, int] | None = None


def _tensor(value):
    return TensorSpec(value["name"], np.dtype(value["dtype"]), tuple(value["shape"]),
                      tuple(value["axes"]), tuple(value["components"]),
                      None if value["bounds"] is None else tuple(value["bounds"]))


@dataclass(frozen=True)
class PayloadSpec:
    """One available representation. Image/graph producers remain future work."""
    kind: str
    name: str
    tensor: TensorSpec | None
    details: Mapping


def _payload(value):
    value = dict(value)
    kind = value.pop("kind")
    tensor = _tensor(value.pop("tensor")) if "tensor" in value else None
    name = tensor.name if tensor is not None else value.pop("name")
    for field in ("node_features", "edge_features", "global_features", "edge_index"):
        if field in value:
            value[field] = _tensor(value[field])
    return PayloadSpec(kind, name, tensor, MappingProxyType(value))


@dataclass(frozen=True)
class ActionSpec:
    """Count and mapping of fixed action IDs; an observation gives current legality."""
    size: int
    mapping: str


@dataclass(frozen=True)
class HeuristicSupport:
    """Fixed line groups for hand-written scoring, separate from model input arrays."""
    version: int
    window_length: int
    indices: np.ndarray


@dataclass(frozen=True)
class GameSpec:
    """Describe available model inputs, metadata, actions and separate evaluator support."""
    identity: EncodingIdentity
    payloads: tuple[PayloadSpec, ...]
    metadata: tuple[TensorSpec, ...]
    actions: ActionSpec
    heuristic: HeuristicSupport | None = None


@dataclass(frozen=True)
class Observation:
    """One player's view. NumPy arrays own data independently of the live game.

    progress is int64 [moves, completed turns, turns this round, completed rounds].
    results is int8 in absolute player order: ongoing=0, win=1, draw=2, loss=3.
    payloads contains only the requested arrays, in request order.
    """
    identity: EncodingIdentity
    position: _engine.Position
    observer: int
    next_player: int | None
    payloads: Mapping[str, np.ndarray]
    progress: np.ndarray
    results: np.ndarray
    action_mask: np.ndarray
    offered_action_ids: np.ndarray

    @property
    def done(self) -> bool:
        """Whether the whole game has ended."""
        return self.next_player is None


def _integer(value, label):
    """Accept integer-valued Python/NumPy indices and reject Boolean switches."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer")
    result = index(value)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


class Game:
    """Own a Rust game. Use create() to choose the game and parameters.

    Each accepted action/reset changes position. apply() requires the token from
    the observation being used, so old and cross-game decisions are rejected.
    """
    def __init__(self, native):
        self._native = native

    @property
    def position(self) -> _engine.Position:
        return self._native.position

    @property
    def next_player(self) -> int | None:
        return self._native.next_player

    @property
    def seed(self) -> int | None:
        """Resolved experiment seed; None for games without environment randomness.

        For hidden-information games this is privileged data that reproduces the
        deal. Pass observations, rather than the game handle/seed, to policies.
        """
        return self._native.seed

    def spec(self) -> GameSpec:
        """Discover input formats and action meanings using the Rust game definition."""
        value = self._native.spec()
        support = None if value["heuristic"] is None else HeuristicSupport(**value["heuristic"])
        if support is not None:
            support.indices.flags.writeable = False
        return GameSpec(
            EncodingIdentity(**value["identity"]),
            tuple(_payload(p) for p in value["payloads"]),
            tuple(_tensor(t) for t in value["metadata"]),
            ActionSpec(**value["actions"]), support,
        )

    def open_lines_evaluator(self):
        """Create a Python-callable Rust heuristic for this board configuration."""
        from tag_rust.search.scoring import RustOpenLinesEvaluator
        return RustOpenLinesEvaluator(self._native.open_lines_evaluator())

    def observe(self, observer: int, payloads: Sequence[str] | None = None) -> Observation:
        """Read a player's view; omission requests its advertised arrays.

        payloads=() requests only metadata and choices.
        """
        if isinstance(payloads, (str, bytes)):
            raise TypeError("payloads must be a sequence of names, such as ['cells']")
        names = self._native.payload_names() if payloads is None else list(payloads)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("each payload name must be a string")
        value = self._native.observe(_integer(observer, "observer"), names)
        value["identity"] = EncodingIdentity(**value["identity"])
        value["payloads"] = MappingProxyType(value["payloads"])
        return Observation(**value)

    def apply(self, action_id: int, *, position: _engine.Position, actor: int | None = None) -> _engine.Position:
        """Apply one action ID through Rust's rules and return the new position token."""
        actor = None if actor is None else _integer(actor, "actor")
        return self._native.apply(_integer(action_id, "action_id"), position, actor)

    def reset(self) -> _engine.Position:
        """Recreate the original setup and invalidate previous position tokens."""
        return self._native.reset()

    def clone(self) -> "Game":
        """Return an independent Rust state with its own identity and position token."""
        return Game(self._native.clone())

    def set_controller(self, player: int, controller=None):
        """Attach a fresh OSLAPython controller, or None for human input. Cancels pending work."""
        from tag_rust.search.evaluation import OSLAPython
        if controller is not None and not isinstance(controller, OSLAPython):
            raise TypeError("controller must be OSLAPython or None")
        checkpoint = None if controller is None else asdict(controller.checkpoint)
        return self._native.set_controller(_integer(player, "player"), checkpoint)

    def controller_checkpoint(self, player: int):
        """Read exact settings/RNG progress; None identifies a human seat."""
        from tag_rust.search.evaluation import ControllerCheckpoint
        value = self._native.controller_checkpoint(_integer(player, "player"))
        if value is None:
            return None
        value["payloads"] = tuple(value["payloads"])
        return ControllerCheckpoint(**value)

    def restore_controller(self, player: int, checkpoint):
        """Restore an independent controller from a complete checkpoint; cancels pending work."""
        from tag_rust.search.evaluation import ControllerCheckpoint
        if not isinstance(checkpoint, ControllerCheckpoint):
            raise TypeError("checkpoint must be a ControllerCheckpoint")
        return self._native.set_controller(_integer(player, "player"), asdict(checkpoint))

    def prepare_evaluation(self):
        """Prepare the first chunk for the current AI seat without changing live state/RNG."""
        from tag_rust.search.evaluation import _status
        return _status(self._native.prepare_evaluation())

    def finish_evaluation(self, request, values, *, action_ids=None, evaluator=None):
        """Validate float64 values and return the next chunk or a MoveProposal.

        Invalid responses retain the current request for retry or cancellation.
        Optional response IDs/identity make externally returned metadata explicit.
        """
        from tag_rust.search.evaluation import _response_arguments, _status
        return _status(self._native.finish_evaluation(*_response_arguments(request,values,action_ids,evaluator)))

    def advance(self, *, max_moves=1024):
        """Run Rust until finished, human input, external evaluation or a move limit.

        max_moves must be positive. Pending manual work must be resumed or cancelled.
        Earlier committed moves remain accepted if a later decision raises an error.
        """
        from tag_rust.search.evaluation import _execution_result
        return _execution_result(self._native.advance(_integer(max_moves,"max_moves")))

    def resume(self, request, values, *, max_moves=1024, action_ids=None, evaluator=None):
        """Supply a chunk's scores, commit its completed move and continue in Rust.

        Invalid responses preserve the pending chunk for retry or cancellation.
        The resumed move counts toward this call's max_moves.
        """
        from tag_rust.search.evaluation import _response_arguments, _execution_result
        arguments = _response_arguments(request,values,action_ids,evaluator)
        return _execution_result(self._native.resume(*arguments,_integer(max_moves,"max_moves")))

    def commit_evaluation(self, proposal):
        """Apply a ready action and install its next RNG checkpoint exactly once."""
        from tag_rust.search.evaluation import MoveProposal
        if not isinstance(proposal, MoveProposal):
            raise TypeError("proposal must be a MoveProposal")
        return self._native.commit_evaluation(proposal._handle)

    def cancel_evaluation(self, work):
        """Discard this request/proposal while retaining the live state and RNG."""
        from tag_rust.search.evaluation import EvaluationRequest, MoveProposal
        if not isinstance(work, (EvaluationRequest, MoveProposal)):
            raise TypeError("work must be an EvaluationRequest or MoveProposal")
        return self._native.cancel_evaluation(work._handle)

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        result = self.clone()
        memo[id(self)] = result
        return result

    def close(self) -> None:
        """Release the Rust state. Previously returned arrays remain usable."""
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def registered_games() -> tuple[str, ...]:
    """Discover game IDs from the Rust factories used by create()."""
    return tuple(_engine.registered_games())


def create(name: str, parameters: Mapping[str, object] | None = None, *, players: int = 2,
           max_cells: int = 4096, max_bytes: int = 64 * 1024 * 1024) -> Game:
    """Create a registered Rust game with checked settings and encoding limits.

    registered_games() lists the available game IDs. Unknown parameters fail
    explicitly. Settings, shapes and move meanings are defined in Rust.
    """
    parameters = {} if parameters is None else dict(parameters)
    if any(not isinstance(key, str) for key in parameters):
        raise TypeError("parameter names must be strings")
    def normalize(value, label):
        if value is None:
            if label == "seed":
                return None
            raise TypeError(f"{label} must be an integer, mapping or sequence")
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("parameter names must be strings")
            return {key: normalize(item, key) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [normalize(item, label) for item in value]
        result = _integer(value, label)
        if result > 2**64-1:
            raise OverflowError(f"{label} exceeds the unsigned 64-bit range")
        return result
    parameters = {key: normalize(value, key) for key, value in parameters.items()}
    arguments = (_integer(players, "players"), _integer(max_cells, "max_cells"), _integer(max_bytes, "max_bytes"))
    if all(isinstance(value, int) for value in parameters.values()):
        return Game(_engine.create(name, parameters, *arguments))
    import json
    return Game(_engine.create_json(name, json.dumps(parameters), *arguments))
