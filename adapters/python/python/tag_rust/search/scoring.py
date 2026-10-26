"""Python access to the native Rust heuristic."""
from dataclasses import asdict
import numpy as np
from tag_rust.search.evaluation import EvaluationRequest


class RustOpenLinesEvaluator:
    """Score received candidate arrays in Rust; the caller supplies them from Python.

    Create with game.open_lines_evaluator(). Each call copies numeric inputs into
    Rust-owned buffers, releases Python's lock while scoring, and returns float64
    values. It preserves the live board, pending request and agent RNG.
    """
    def __init__(self, native):
        self._native = native

    def evaluate(self, request: EvaluationRequest) -> np.ndarray:
        identity = asdict(request.identity)
        identity["configuration"] = dict(request.identity.configuration)
        return self._native.evaluate(
            identity, request.actor, request.payloads["cells"],
            request.results, request.action_ids,
        )


