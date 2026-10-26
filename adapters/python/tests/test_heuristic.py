"""Compare the example's Python heuristic with actual native evaluators and OSLA."""
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import tag_rust as tag

ADAPTER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("open_lines", ADAPTER / "examples/open_lines.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class HeuristicComparison(unittest.TestCase):
    def test_scores_and_seeded_games_match_native_osla(self):
        reference = subprocess.run(
            ["cargo", "run", "--offline", "--release", "--quiet", "--manifest-path",
             str(ADAPTER / "Cargo.toml"), "--example", "heuristic_reference"],
            check=True, capture_output=True, text=True,
        )
        decisions = 0
        for case in json.loads(reference.stdout):
            for rows in (1, 3, 1024):
                with self.subTest(game=case["game"], parameters=case["parameters"], rows=rows):
                    with tag.create(case["game"], case["parameters"]) as game:
                        rust_scorer = game.open_lines_evaluator()
                        support = game.spec().heuristic
                        lines = support.indices.reshape(-1, support.window_length)
                        for seat in range(2):
                            game.set_controller(seat, tag.OSLAPython(
                                evaluator_id="python-open-lines", seed=7 + seat,
                                payloads=("cells",), max_rows=rows,
                            ))
                        for expected in case["moves"]:
                            self.assertEqual(game.next_player, expected["actor"])
                            request = game.prepare_evaluation()
                            ids, scores = [], []
                            while isinstance(request, tag.EvaluationRequest):
                                values = example.evaluate_open_lines(request, lines)
                                position = game.position
                                checkpoint = game.controller_checkpoint(request.actor)
                                rust_values = rust_scorer.evaluate(request)
                                np.testing.assert_array_equal(rust_values, values)
                                self.assertEqual(rust_values.dtype, np.dtype("float64"))
                                self.assertEqual(game.position, position)
                                self.assertEqual(game.controller_checkpoint(request.actor), checkpoint)
                                self.assertEqual(values.dtype, np.dtype("float64"))
                                ids.extend(request.action_ids.tolist())
                                scores.extend(values.tolist())
                                request = game.finish_evaluation(request, rust_values)
                            self.assertEqual(ids, expected["ids"])
                            # Exact equality also checks the floating-point summation order.
                            self.assertEqual(scores, expected["values"])
                            self.assertEqual(request.action_id, expected["chosen"])
                            game.commit_evaluation(request)
                            decisions += 1
                        self.assertIsNone(game.next_player)
        print(f"Python/native heuristic: {decisions} decisions matched exactly across three batch sizes.")

    def test_python_calls_rust_validation_buffers_and_reuse(self):
        with tag.create("tictactoe") as game:
            scorer = game.open_lines_evaluator()
            game.set_controller(0, tag.OSLAPython(evaluator_id="rust-open-lines", seed=7,
                                                 payloads=("cells",), max_rows=3))
            request = game.prepare_evaluation()
            expected = scorer.evaluate(request)
            # A strided view keeps every second column of a larger array.
            storage = np.zeros((3,18), dtype=np.float32)
            storage[:,::2] = request.payloads["cells"]
            strided = replace(request, payloads={"cells": storage[:,::2]})
            np.testing.assert_array_equal(scorer.evaluate(strided), expected)
            expected[:] = 99  # Returned scores own their data.
            self.assertFalse(np.any(scorer.evaluate(request) == 99))
            bad = replace(request, identity=replace(request.identity, width=5))
            with self.assertRaises(ValueError):
                scorer.evaluate(bad)
            with self.assertRaises(TypeError):
                scorer.evaluate(replace(request, payloads={"cells":request.payloads["cells"].astype(np.float64)}))
            with self.assertRaises(ValueError):
                scorer.evaluate(replace(request, results=np.zeros((3,1),dtype=np.int8)))
            with self.assertRaises(ValueError):
                scorer.evaluate(replace(request, action_ids=np.full(3,-1,dtype=np.int64)))
            invalid = request.payloads["cells"].copy()
            invalid[0,0] = np.nan
            with self.assertRaises(ValueError):
                scorer.evaluate(replace(request,payloads={"cells":invalid}))
            with self.assertRaises(MemoryError):
                scorer.evaluate(replace(request, action_ids=np.zeros(4097,dtype=np.int64),
                                        payloads={"cells":np.zeros((4097,9),dtype=np.float32)},
                                        results=np.zeros((4097,2),dtype=np.int8)))
            # Failed scoring leaves the request usable; committing still goes
            # through the session's normal handle/position checks.
            game.finish_evaluation(request, scorer.evaluate(request))
        # The scorer owns the fixed description and can score saved arrays after closure.
        self.assertEqual(scorer.evaluate(request).shape,(3,))
        with self.assertRaises(tag.ClosedGameError):
            game.open_lines_evaluator()
