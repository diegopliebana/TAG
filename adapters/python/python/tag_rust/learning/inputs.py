"""Validate and batch player-visible named arrays independently of any neural model."""
from dataclasses import dataclass
from collections.abc import Mapping
from math import prod, isfinite
from types import MappingProxyType
import numpy as np
from tag_rust.engine.api import EncodingIdentity, ActionSpec, GameSpec, TensorSpec

DTYPES = tuple(np.dtype(t) for t in ("float32", "float64", "int8", "int64", "uint8", "bool"))


def _dimension(value, label, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if value < (0 if zero else 1):
        raise ValueError(f"invalid {label}")
    return int(value)


def _tensor_spec(spec):
    """Snapshot a fixed-size numeric description; ragged structures need another batcher."""
    shape = tuple(_dimension(n, "tensor dimension") for n in spec.shape)
    dtype = np.dtype(spec.dtype)
    if not spec.name or dtype not in DTYPES or len(spec.axes) != len(shape):
        raise ValueError("expected a named fixed-size numeric tensor with one axis name per dimension")
    bounds = None if spec.bounds is None else tuple(spec.bounds)
    if bounds is not None and (len(bounds) != 2 or not all(isfinite(x) for x in bounds) or bounds[0] > bounds[1]):
        raise ValueError("invalid tensor bounds")
    return TensorSpec(spec.name, dtype, shape, tuple(spec.axes), tuple(spec.components), bounds)


@dataclass(frozen=True)
class ValueSpec:
    """Declare whose return each output estimates and how rewards are interpreted.

    players=None requests one value for each row's observer. An explicit tuple
    requests values in that absolute-player order, all using the row's permitted
    observation. bounds=None uses an unbounded linear output. units and discount
    clock identify reward meaning separately from game scores.
    """
    players: tuple[int, ...] | None = None
    bounds: tuple[float, float] | None = (-1.0, 1.0)
    units: str = "win=1,draw=0,loss=-1"
    gamma: float = 0.99
    discount_clock: str = "learner-step"

    def __post_init__(self):
        if not self.units or not self.discount_clock or not isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("declare value units, discount clock and gamma in [0,1]")
        if self.players is not None:
            players = tuple(_dimension(p, "value player", zero=True) for p in self.players)
            if not players or len(set(players)) != len(players):
                raise ValueError("value players must be nonempty and unique")
            object.__setattr__(self, "players", players)
        if self.bounds is not None:
            bounds = tuple(self.bounds)
            if len(bounds) != 2 or not all(isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("value bounds require finite lower < upper")
            object.__setattr__(self, "bounds", bounds)

    @property
    def size(self):
        return 1 if self.players is None else len(self.players)


@dataclass(frozen=True)
class ModelContract:
    """Specify the observations, actions and player values a model accepts or predicts.

    Record the game configuration, ordered observation features, action mapping
    and value convention so a model can check compatibility before inference.
    """
    identity: EncodingIdentity
    actions: ActionSpec
    tensors: tuple[TensorSpec, ...]
    value: ValueSpec = ValueSpec()
    candidate_tensor: TensorSpec | None = None
    feature_version: int = 2

    def __post_init__(self):
        players = _dimension(self.identity.player_count, "player count")
        if self.feature_version != 2:
            raise ValueError("unsupported feature format version")
        tensors = tuple(_tensor_spec(t) for t in self.tensors)
        if not tensors or len({t.name for t in tensors}) != len(tensors):
            raise ValueError("select at least one uniquely named input")
        object.__setattr__(self, "tensors", tensors)
        if self.value.players is not None and any(p >= players for p in self.value.players):
            raise ValueError("value player is outside the game's seats")
        size = _dimension(self.actions.size, "action count", zero=True)
        if self.actions.mapping == "candidates":
            if size != 0 or self.candidate_tensor is None:
                raise ValueError("candidate policy requires size=0 and a candidate tensor description")
            object.__setattr__(self, "candidate_tensor", _tensor_spec(self.candidate_tensor))
        elif self.actions.mapping not in ("cell_x_major", "column", "indexed") or size == 0 or self.candidate_tensor is not None:
            raise ValueError("fixed policy requires permanent IDs and a positive action count")

    @classmethod
    def from_spec(cls, spec: GameSpec, *, payloads, value=None, candidate_tensor=None):
        """Select ordered features/grids without inspecting game names or board dimensions.

        Image preprocessing and graph/ragged batching need explicit adapters.
        Callers select input names explicitly; no game representation is assumed.
        """
        if isinstance(payloads, str):
            raise TypeError("payloads must be a sequence of names")
        tensors = []
        for name in payloads:
            payload = next((p for p in spec.payloads if p.name == name), None)
            if payload is None or payload.kind not in ("features", "grid") or payload.tensor is None:
                raise ValueError(f"unsupported or absent fixed-size payload: {name}; images/graphs need an explicit adapter")
            tensors.append(payload.tensor)
        return cls(spec.identity, spec.actions, tuple(tensors), value or ValueSpec(), candidate_tensor)

    @property
    def context_size(self):
        """Number of context features: one one-hot vector for the observer, one
        for the next player, and a flag indicating whether the observer acts next."""
        return 2 * self.identity.player_count + 1

    @property
    def input_size(self):
        return sum(prod(t.shape) for t in self.tensors) + self.context_size


@dataclass(frozen=True)
class CandidateActions:
    """One row's legal action features and opaque IDs, bound by the caller to its decision.

    IDs identify supplied Rust actions within this decision and must be unique.
    They need not be permanent policy-output IDs. Apply through original position/
    request validation; IDs alone do not authorize applying an action later.
    """
    features: np.ndarray
    action_ids: np.ndarray


@dataclass(frozen=True)
class NeuralBatch:
    """An inference batch containing observation arrays, public player-context
    features, legal-action masks and optional action features. Arrays have
    independent storage; padding allows different candidate counts in one batch."""
    contract: ModelContract
    payloads: Mapping[str, np.ndarray]
    context: np.ndarray
    legal_mask: np.ndarray
    action_ids: np.ndarray
    candidate_features: np.ndarray | None = None

    @property
    def inputs(self):
        """Return a new float32 flat input for inspecting the fully connected model."""
        return flatten_inputs(self.contract, self.payloads, self.context)


def _array(value, spec, shape, *, check_bounds=True):
    if not isinstance(value, np.ndarray) or value.dtype != spec.dtype:
        raise TypeError(f"{spec.name} requires dtype {spec.dtype}")
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{spec.name} has invalid shape or non-finite entries")
    if check_bounds and spec.bounds is not None and ((value < spec.bounds[0]).any() or (value > spec.bounds[1]).any()):
        raise ValueError(f"{spec.name} entries exceed declared bounds")
    return np.array(value, order="C", copy=True)


def flatten_inputs(contract, payloads, context):
    """Explicitly convert selected arrays to float32; reject overflow/lost integer precision."""
    parts = []
    for spec in contract.tensors:
        raw = payloads[spec.name]
        with np.errstate(over="ignore", invalid="ignore"):
            converted = raw.astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError("input conversion exceeds float32 range")
        if raw.dtype.kind in "iu" and not np.array_equal(converted.astype(object), raw.astype(object)):
            raise ValueError("integer input loses float32 precision; provide a versioned scaled encoding")
        parts.append(converted.reshape(len(context), -1))
    return np.ascontiguousarray(np.concatenate((*parts, context), axis=1), dtype=np.float32)


def pack_observations(contract, observations, *, candidates=None, max_bytes=64 * 1024 * 1024):
    """Construct an inference batch from player-visible observation arrays and
    public player-context features. Copy inputs into independent storage.

    observations may come from any compatible game/seat. Optional candidates has
    one CandidateActions per observation, padded with a separate legal mask.
    A zero-length candidate list is a row with no policy operation.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("provide at least one observation")
    count, players = len(observations), contract.identity.player_count
    payloads = {t.name: [] for t in contract.tensors}
    dynamic = contract.actions.mapping == "candidates"
    if dynamic:
        candidates = tuple(candidates) if candidates is not None else ()
        if len(candidates) != count:
            raise ValueError("supply one candidate list per observation")
        lengths = [len(c.action_ids) for c in candidates]
        width = max(lengths)
    else:
        if candidates is not None:
            raise ValueError("fixed-ID policy does not accept candidate features")
        width = contract.actions.size
    budget = count * (4 * contract.context_size + sum(prod(t.shape) * t.dtype.itemsize for t in contract.tensors) + 9 * width)
    if dynamic:
        budget += count * width * prod(contract.candidate_tensor.shape) * contract.candidate_tensor.dtype.itemsize
    if budget > _dimension(max_bytes, "batch byte limit"):
        raise MemoryError("packed numeric arrays exceed max_bytes")
    context = np.zeros((count, contract.context_size), dtype=np.float32)
    candidate_features = (np.zeros((count, width, *contract.candidate_tensor.shape), dtype=contract.candidate_tensor.dtype)
                          if dynamic else None)
    mask = np.zeros((count, width), dtype=np.bool_)
    ids = np.full((count, width), -1, dtype=np.int64)
    for row, obs in enumerate(observations):
        if obs.identity != contract.identity:
            raise ValueError("observation encoding identity differs from model")
        observer = _dimension(obs.observer, "observer", zero=True)
        next_player = None if obs.next_player is None else _dimension(obs.next_player, "next player", zero=True)
        if observer >= players or (next_player is not None and next_player >= players):
            raise ValueError("player is outside the game's seats")
        context[row, observer] = 1
        if next_player is not None:
            context[row, players + next_player] = 1
        context[row, -1] = next_player == observer
        for spec in contract.tensors:
            payloads[spec.name].append(_array(obs.payloads.get(spec.name), spec, spec.shape))
        if dynamic:
            candidate, length = candidates[row], lengths[row]
            if not isinstance(candidate.action_ids, np.ndarray) or candidate.action_ids.dtype != np.int64:
                raise TypeError("candidate action IDs must be int64")
            if candidate.action_ids.shape != (length,) or (candidate.action_ids < 0).any() or len(set(candidate.action_ids.tolist())) != length:
                raise ValueError("candidate action IDs must be unique and nonnegative")
            candidate_features[row, :length] = _array(candidate.features, contract.candidate_tensor, (length, *contract.candidate_tensor.shape))
            mask[row, :length], ids[row, :length] = True, candidate.action_ids
        else:
            if not isinstance(obs.action_mask, np.ndarray) or obs.action_mask.dtype != np.bool_:
                raise TypeError("action mask must be Boolean")
            if obs.action_mask.shape != (width,):
                raise ValueError("action mask dimensions differ from model")
            mask[row], ids[row] = obs.action_mask, np.arange(width)
        if mask[row].any() and next_player != observer:
            raise ValueError("policy choices require the observer's decision")
    return NeuralBatch(contract, MappingProxyType({k: np.stack(v) for k, v in payloads.items()}),
                       context, mask, ids, candidate_features)


def validate_batch(batch):
    """Check manually constructed or modified buffers before model execution."""
    c, context = batch.contract, batch.context
    if not isinstance(context, np.ndarray) or context.dtype != np.float32:
        raise TypeError("context must be float32")
    if context.ndim != 2 or len(context) == 0 or context.shape[1] != c.context_size or not np.isin(context, (0, 1)).all():
        raise ValueError("invalid context shape or indicators")
    n, p = len(context), c.identity.player_count
    observer, actor = context[:, :p], context[:, p:2*p]
    if not (observer.sum(axis=1) == 1).all() or (actor.sum(axis=1) > 1).any() or not np.array_equal(context[:, -1], (observer * actor).sum(axis=1)):
        raise ValueError("inconsistent observer/actor context")
    if set(batch.payloads) != {t.name for t in c.tensors}:
        raise ValueError("named payloads differ from contract")
    for spec in c.tensors:
        _array(batch.payloads[spec.name], spec, (n, *spec.shape))
    mask, ids = batch.legal_mask, batch.action_ids
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("legal mask must be Boolean")
    if mask.ndim != 2 or mask.shape[0] != n or (mask.any(axis=1) & (context[:, -1] == 0)).any():
        raise ValueError("invalid legal mask")
    if not isinstance(ids, np.ndarray) or ids.dtype != np.int64:
        raise TypeError("action IDs must be int64")
    if ids.shape != mask.shape or (ids[mask] < 0).any():
        raise ValueError("invalid action ID dimensions/values")
    if c.actions.mapping == "candidates":
        _array(batch.candidate_features, c.candidate_tensor, (*mask.shape, *c.candidate_tensor.shape), check_bounds=False)
        _array(batch.candidate_features[mask], c.candidate_tensor, (int(mask.sum()), *c.candidate_tensor.shape))
        for row in range(n):
            if len(np.unique(ids[row, mask[row]])) != int(mask[row].sum()):
                raise ValueError("duplicate candidate action IDs")
    elif (mask.shape[1] != c.actions.size or batch.candidate_features is not None
          or not np.array_equal(ids, np.broadcast_to(np.arange(c.actions.size), ids.shape))):
        raise ValueError("fixed action IDs differ from contract")


def pack_rows(contract, payloads, observers, next_players, *, legal_mask=None, max_bytes=64*1024*1024):
    """Copy already-batched named arrays and public player metadata into a neural batch.

    observers/next_players are int64 [B], with -1 meaning no next player. An absent
    legal_mask requests values only. Candidate-policy models receive zero policy
    candidates in this mode; direct candidate play uses pack_observations instead.
    This adapter shares the ordinary batch validation and creates no per-row objects.
    """
    for values in (observers, next_players):
        if not isinstance(values, np.ndarray) or values.dtype != np.int64:
            raise TypeError("player metadata must be int64 arrays")
    if observers.ndim != 1 or len(observers) == 0 or next_players.shape != observers.shape:
        raise ValueError("player metadata must be nonempty [B] arrays")
    n, players = len(observers), contract.identity.player_count
    if ((observers < 0) | (observers >= players)).any() or ((next_players < -1) | (next_players >= players)).any():
        raise ValueError("player metadata is outside the declared seats")
    dynamic = contract.actions.mapping == "candidates"
    if dynamic and legal_mask is not None:
        raise ValueError("direct candidate choices require explicit candidate features")
    width = 0 if dynamic else contract.actions.size
    budget = n * (4*contract.context_size + 9*width + sum(prod(t.shape)*t.dtype.itemsize for t in contract.tensors))
    if budget > _dimension(max_bytes, "batch byte limit"):
        raise MemoryError("packed numeric arrays exceed max_bytes")
    context = np.zeros((n, contract.context_size), dtype=np.float32)
    context[np.arange(n), observers] = 1
    acting = next_players >= 0
    context[np.flatnonzero(acting), players + next_players[acting]] = 1
    context[:, -1] = observers == next_players
    payloads = MappingProxyType({t.name: _array(payloads.get(t.name), t, (n, *t.shape)) for t in contract.tensors})
    if legal_mask is None:
        mask = np.zeros((n, width), dtype=bool)
    else:
        if not isinstance(legal_mask, np.ndarray) or legal_mask.dtype != np.bool_:
            raise TypeError("legal mask must be Boolean")
        mask = legal_mask.copy()
    batch = NeuralBatch(contract, payloads, context, mask,
                        np.broadcast_to(np.arange(width, dtype=np.int64), (n, width)).copy(),
                        np.empty((n, 0, *contract.candidate_tensor.shape), dtype=contract.candidate_tensor.dtype) if dynamic else None)
    validate_batch(batch)
    return batch
