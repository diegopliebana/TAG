"""Evaluate candidates from several independent Rust games in one Python call."""
from dataclasses import dataclass
import numpy as np
from .. import _engine
from tag_rust.engine.api import Game, EncodingIdentity, _integer
from tag_rust.search.evaluation import EvaluatorIdentity, _request_fields, _response_arguments


@dataclass(frozen=True)
class EvaluationSlice:
    """Rows [start:stop] belonging to one game's current candidate chunk."""
    game_index: int
    start: int
    stop: int
    actor: int
    request_id: int
    offset: int
    total: int
    position: _engine.Position


@dataclass(frozen=True)
class GroupedEvaluation:
    """Compatible candidate arrays, with the game and player identified per row."""
    _handle: _engine.EvaluationGroupHandle
    identity: EncodingIdentity
    evaluator: EvaluatorIdentity
    action_ids: np.ndarray
    payloads: object
    progress: np.ndarray
    results: np.ndarray
    next_players: np.ndarray
    observers: np.ndarray
    game_indices: np.ndarray
    segments: tuple[EvaluationSlice, ...]

    @property
    def id(self):
        return self._handle.id


@dataclass(frozen=True)
class GameExecutionStop:
    """Where one game stopped; moves counts accepted moves during this call."""
    game_index: int
    reason: str
    moves: int
    position: _engine.Position
    actor: int | None
    error: str | None


@dataclass(frozen=True)
class BatchExecutionResult:
    """New evaluation groups and stops for the games visited by this call."""
    groups: tuple[GroupedEvaluation, ...]
    stops: tuple[GameExecutionStop, ...]


def _result(raw):
    groups = []
    for value in raw["groups"]:
        value = _request_fields(value)
        value["segments"] = tuple(EvaluationSlice(**s) for s in value["segments"])
        groups.append(GroupedEvaluation(**value))
    return BatchExecutionResult(tuple(groups), tuple(GameExecutionStop(**s) for s in raw["stops"]))


class GameBatch:
    """Keep original games and let Rust coordinate their bounded execution.

    Compatible requests share game/configuration, evaluator/settings and input
    layouts. Each original game's RNG and decision remain independent.
    Closing the coordinator cancels its requests and keeps its games open.
    """
    def __init__(self, games, *, max_group_rows=4096, max_bytes=64 * 1024 * 1024):
        self.games = tuple(games)
        if any(not isinstance(g, Game) for g in self.games):
            raise TypeError("games must contain Game instances")
        self._native = _engine.GameBatch([g._native for g in self.games],
            _integer(max_group_rows, "max_group_rows"), _integer(max_bytes, "max_bytes"))

    @property
    def pending_count(self):
        return self._native.pending_count

    def advance(self, *, max_moves_per_game=1024):
        """Run every game until it needs attention, then combine external requests."""
        return _result(self._native.advance(_integer(max_moves_per_game, "max_moves_per_game")))

    def resume(self, group, values, *, max_moves_per_game=1024, action_ids=None, evaluator=None):
        """Validate the whole response, then resume only this group's games."""
        response = _response_arguments(group, values, action_ids, evaluator,
                                       request_type=GroupedEvaluation)
        return _result(self._native.resume(*response,
            _integer(max_moves_per_game, "max_moves_per_game")))

    def cancel(self, group):
        """Discard still-matching requests in this group, preserving board and RNG."""
        self._native.cancel(group._handle)

    def cancel_all(self):
        self._native.cancel_all()

    def close(self):
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
