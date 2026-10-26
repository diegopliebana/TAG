"""Play Rust games and obtain independently stored NumPy observations.

Start with create(). Game formats and action meanings come from Rust LearningGame
implementations. The Python classes here give those descriptions readable names.
"""
from ._engine import ClosedGameError, EvaluationError, Position, StalePositionError
from .engine.api import (
    ActionSpec, EncodingIdentity, Game, GameSpec, HeuristicSupport,
    Observation, PayloadSpec, TensorSpec, create, registered_games,
)
from .search.evaluation import OSLAPython, ControllerCheckpoint, EvaluatorIdentity, EvaluationRequest, MoveProposal, ExecutionResult
from .search.scoring import RustOpenLinesEvaluator

__all__ = [
    "ActionSpec", "ClosedGameError", "EncodingIdentity", "Game", "GameSpec",
    "HeuristicSupport", "Observation", "PayloadSpec", "Position",
    "StalePositionError", "TensorSpec", "create", "registered_games",
    "OSLAPython", "ControllerCheckpoint", "EvaluatorIdentity", "EvaluationRequest", "MoveProposal", "EvaluationError",
    "RustOpenLinesEvaluator",
    "ExecutionResult",
]

from .search.grouping import GameBatch, GroupedEvaluation, EvaluationSlice, GameExecutionStop, BatchExecutionResult
__all__ += ["GameBatch", "GroupedEvaluation", "EvaluationSlice", "GameExecutionStop", "BatchExecutionResult"]

from .search.mcts import MctsSearch, MctsBatch, MctsRequest, MctsResponse, MctsGroup
__all__ += ["MctsSearch", "MctsBatch", "MctsRequest", "MctsResponse", "MctsGroup"]
