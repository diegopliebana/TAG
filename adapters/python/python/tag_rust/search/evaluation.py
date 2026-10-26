"""Describe external evaluator requests and complete OSLAPython controller checkpoints."""
from dataclasses import dataclass, replace, asdict
from types import MappingProxyType
import json
from ..randomness import resolve_seed
import numpy as np
from tag_rust.engine.api import EncodingIdentity, _integer
from .. import _engine


@dataclass(frozen=True)
class EvaluatorIdentity:
    """Caller-declared code/version/settings identity, checked on every response."""
    name: str
    version: int
    configuration: str

    @property
    def parameters(self):
        """Decode a fresh copy of the evaluator's recorded JSON settings."""
        return json.loads(self.configuration)


@dataclass(frozen=True)
class ControllerCheckpoint:
    """Complete persistent settings and RNG progress for exact continuation."""
    policy_version: int
    evaluator_name: str
    evaluator_version: int
    evaluator_configuration: str
    payloads: tuple[str, ...]
    max_cells: int
    max_bytes: int
    max_rows: int
    max_candidates: int
    rng_version: int
    seed: int
    stream: int
    word_position: int
    execution: str = "external"


@dataclass(frozen=True, init=False)
class OSLAPython:
    """Settings for a new independently seeded controller, attached with set_controller().

    evaluator_id/version/parameters identify the external function and its settings.
    Rust owns the attached controller and RNG. This object is a reusable setup value.
    """
    checkpoint: ControllerCheckpoint

    def __init__(self, *, evaluator_id: str, seed: int | None = None,
                 evaluator_version: int = 1, evaluator_parameters=None,
                 payloads=("cells",), max_rows=1024, max_candidates=4096,
                 max_cells=4096, max_bytes=64 * 1024 * 1024):
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise ValueError("evaluator_id must be a nonempty string")
        seed = resolve_seed() if seed is None else _integer(seed, "seed")
        if seed >= 2**64:
            raise OverflowError("seed exceeds uint64")
        if isinstance(payloads, (str, bytes)):
            raise TypeError("payloads must be a sequence of names")
        payloads = tuple(payloads)
        if any(not isinstance(n, str) for n in payloads):
            raise TypeError("each payload name must be a string")
        parameters = {} if evaluator_parameters is None else evaluator_parameters
        configuration = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
        settings = dict(
            evaluator_name=evaluator_id, evaluator_version=_integer(evaluator_version, "evaluator_version"),
            evaluator_configuration=configuration, payloads=payloads,
            max_cells=_integer(max_cells, "max_cells"), max_bytes=_integer(max_bytes, "max_bytes"),
            max_rows=_integer(max_rows, "max_rows"), max_candidates=_integer(max_candidates, "max_candidates"),
        )
        value = _engine.create_controller(settings, seed)
        value["payloads"] = tuple(value["payloads"])
        checkpoint = ControllerCheckpoint(**value)
        object.__setattr__(self, "checkpoint", checkpoint)

    @property
    def seed(self) -> int:
        return self.checkpoint.seed

    @classmethod
    def rust_heuristic(cls, *, seed=None, max_rows=1024, max_candidates=4096,
                       max_cells=4096, max_bytes=64 * 1024 * 1024):
        """Use the same batched policy with Rust's open-line scorer inside advance()."""
        result = cls(evaluator_id="rust-open-lines", seed=seed, payloads=("cells",),
                     max_rows=max_rows, max_candidates=max_candidates,
                     max_cells=max_cells, max_bytes=max_bytes)
        object.__setattr__(result, "checkpoint", replace(result.checkpoint, execution="rust_open_lines"))
        return result


@dataclass(frozen=True)
class EvaluationRequest:
    """A bounded candidate chunk. All rows use the original acting player's viewpoint."""
    _handle: _engine.EvaluationHandle
    position: _engine.Position
    actor: int
    identity: EncodingIdentity
    evaluator: EvaluatorIdentity
    offset: int
    total: int
    action_ids: np.ndarray
    payloads: object
    progress: np.ndarray
    results: np.ndarray
    next_players: np.ndarray

    @property
    def id(self) -> int:
        return self._handle.id


@dataclass(frozen=True)
class MoveProposal:
    """A selected action waiting for commitment; the live board and RNG are unchanged."""
    _handle: _engine.EvaluationHandle
    position: _engine.Position
    actor: int
    action_id: int


@dataclass(frozen=True)
class ExecutionResult:
    """A bounded Rust run stopped for finished/human/evaluation/move_limit.

    moves counts accepted moves in this call, including a completed resumed move.
    request is present only when reason is 'evaluation'. actor identifies the
    next player for the other unfinished stops. position reflects all accepted moves.
    """
    reason: str
    moves: int
    position: _engine.Position
    actor: int | None
    request: EvaluationRequest | None


def _execution_result(value):
    value = dict(value)
    if value["request"] is not None:
        value["request"] = _status(value["request"])
    return ExecutionResult(**value)


def _response_arguments(request, values, action_ids, evaluator, *, request_type=EvaluationRequest):
    """Validate reply types for manual, Rust-owned and grouped execution."""
    if not isinstance(request, request_type):
        raise TypeError(f"request must be an {request_type.__name__}")
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float64") or values.ndim != 1:
        raise TypeError("values must be a one-dimensional float64 NumPy array")
    action_ids = request.action_ids if action_ids is None else action_ids
    evaluator = request.evaluator if evaluator is None else evaluator
    if not isinstance(evaluator, EvaluatorIdentity):
        raise TypeError("evaluator must be an EvaluatorIdentity")
    return request._handle, values, action_ids, asdict(evaluator)


def _status(value):
    value = dict(value)
    kind = value.pop("kind")
    value["_handle"] = value.pop("handle")
    if kind == "proposal":
        return MoveProposal(**value)
    return EvaluationRequest(**_request_fields(value))


def _request_fields(value):
    """Convert fields shared by single-game and grouped candidate requests."""
    value = dict(value)
    if "handle" in value:
        value["_handle"] = value.pop("handle")
    value["identity"] = EncodingIdentity(**value["identity"])
    value["evaluator"] = EvaluatorIdentity(
        value.pop("evaluator_name"), value.pop("evaluator_version"),
        value.pop("evaluator_configuration"),
    )
    value["payloads"] = MappingProxyType(value["payloads"])
    return value
