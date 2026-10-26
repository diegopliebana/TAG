"""Complete-game workload and optional call timings used only by the benchmark."""
import sys
import time
from pathlib import Path
import tag_rust as tag

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "adapters/python/examples"))
from open_lines import evaluate_open_lines


def play(case, seed, rows, profile=False, *, scoring="python", record_moves=True):
    """Include creation, description, controllers, all decisions and closure."""
    moves = []
    totals = dict(prepare=0.0, evaluate=0.0, finish=0.0, commit=0.0, requests=0)
    with tag.create(case["game"], case["parameters"]) as game:
        support = game.spec().heuristic
        lines = support.indices.reshape(-1, support.window_length)
        rust_scorer = game.open_lines_evaluator() if scoring == "rust" else None
        for seat in range(2):
            game.set_controller(seat, tag.OSLAPython(
                evaluator_id="open-lines-benchmark", seed=seed + seat,
                payloads=("cells",), max_rows=rows,
            ))
        while game.next_player is not None:
            start = time.perf_counter() if profile else 0
            request = game.prepare_evaluation()
            if profile:
                totals["prepare"] += time.perf_counter() - start
            while isinstance(request, tag.EvaluationRequest):
                start = time.perf_counter() if profile else 0
                values = (rust_scorer.evaluate(request) if rust_scorer is not None
                          else evaluate_open_lines(request, lines))
                if profile:
                    totals["evaluate"] += time.perf_counter() - start
                    start = time.perf_counter()
                request = game.finish_evaluation(request, values)
                if profile:
                    totals["finish"] += time.perf_counter() - start
                    totals["requests"] += 1
            if record_moves:
                moves.append(request.action_id)
            start = time.perf_counter() if profile else 0
            game.commit_evaluation(request)
            if profile:
                totals["commit"] += time.perf_counter() - start
    return moves, totals


def play_owned(case, seed, rows, *, verify=False):
    """Python configures two agents; one advance call runs the whole match in Rust."""
    with tag.create(case["game"], case["parameters"]) as game:
        for seat in range(2):
            game.set_controller(seat,tag.OSLAPython.rust_heuristic(seed=seed+seat,max_rows=rows))
        result = game.advance(max_moves=4096)
        if result.reason != "finished":
            raise RuntimeError(f"Owned benchmark stopped for {result.reason}")
        if verify:
            final = game.observe(0,payloads=("cells",))
            checkpoints = [game.controller_checkpoint(seat) for seat in range(2)]
            return result.moves, final, checkpoints
        return result.moves


def verify_owned(case, seed, rows, expected_moves):
    """Check the recorded moves using bounded advances, then verify one-call parity.

    Validation observes every move outside timed regions. Timed owned games use
    one advance call and never request candidate arrays or intermediate states.
    """
    import numpy as np
    with tag.create(case["game"],case["parameters"]) as owned, tag.create(case["game"],case["parameters"]) as reference:
        for seat in range(2):
            owned.set_controller(seat,tag.OSLAPython.rust_heuristic(seed=seed+seat,max_rows=rows))
        for action_id in expected_moves:
            result = owned.advance(max_moves=1)
            assert result.moves == 1
            reference.apply(action_id,position=reference.position)
            actual, expected = owned.observe(0,payloads=("cells",)), reference.observe(0,payloads=("cells",))
            np.testing.assert_array_equal(actual.payloads["cells"],expected.payloads["cells"])
            np.testing.assert_array_equal(actual.results,expected.results)
            np.testing.assert_array_equal(actual.progress,expected.progress)
        assert result.reason == "finished"
        checkpoints = [owned.controller_checkpoint(seat) for seat in range(2)]
        moves, whole, whole_checkpoints = play_owned(case,seed,rows,verify=True)
        assert moves == len(expected_moves)
        np.testing.assert_array_equal(whole.payloads["cells"],actual.payloads["cells"])
        np.testing.assert_array_equal(whole.results,actual.results)
        assert checkpoints == whole_checkpoints


