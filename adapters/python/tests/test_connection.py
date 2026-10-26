"""Compare every position in complete Rust/Python games and exercise adapter boundaries."""
import copy
import gc
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import tag_rust as tag

ADAPTER = Path(__file__).resolve().parents[1]


class NativeComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["cargo", "run", "--offline", "--quiet", "--manifest-path", str(ADAPTER / "Cargo.toml"),
             "--example", "native_reference"],
            cwd=ADAPTER, check=True, capture_output=True, text=True,
        )
        cls.records = json.loads(result.stdout)

    def test_every_intermediate_position_matches_native(self):
        positions = 0
        for record in self.records:
            case = record["case"]
            with self.subTest(case=case):
                game = tag.create(case["game"], case["parameters"])
                spec = game.spec()
                self.assertEqual(spec.actions.size, record["action_count"])
                self.assertEqual(spec.heuristic.window_length, record["window_length"])
                np.testing.assert_array_equal(spec.heuristic.indices, record["line_indices"])
                self.assertEqual(spec.heuristic.indices.dtype, np.dtype("int64"))
                self.assertEqual([p.name for p in spec.payloads], ["cells", "board"])
                self.assertEqual([p.kind for p in spec.payloads], ["features", "grid"])
                for step, expected_views in enumerate(record["positions"]):
                    for expected in expected_views:
                        view = game.observe(expected["observer"])
                        self.assertEqual(view.identity, spec.identity)
                        self.assertEqual(view.observer, expected["observer"])
                        self.assertEqual(view.position, game.position)
                        self.assertEqual(view.next_player, expected["next_player"])
                        self.assertEqual(view.done, expected["next_player"] is None)
                        for field in ("progress", "results", "action_mask", "offered_action_ids"):
                            np.testing.assert_array_equal(getattr(view, field), expected[field])
                        for name, tensor in expected["payloads"].items():
                            data = view.payloads[name]
                            description = next(p.tensor for p in spec.payloads if p.name == name)
                            self.assertEqual(data.shape, description.shape)
                            self.assertEqual(data.dtype, description.dtype)
                            self.assertEqual(data.shape, tuple(tensor["shape"]))
                            self.assertEqual(data.dtype, np.dtype("float32"))
                            self.assertTrue(data.flags.c_contiguous)
                            np.testing.assert_array_equal(data.ravel(), tensor["values"])
                        self.assertEqual(view.progress.dtype, np.dtype("int64"))
                        self.assertEqual(view.results.dtype, np.dtype("int8"))
                        self.assertEqual(view.action_mask.dtype, np.dtype("bool"))
                        self.assertEqual(view.offered_action_ids.dtype, np.dtype("int64"))
                        positions += 1
                    if step < len(case["moves"]):
                        game.apply(case["moves"][step], position=game.position, actor=game.next_player)
        print(f"Native/Python comparison: {len(self.records)} workloads, {positions} player observations.")


class OwnershipAndErrors(unittest.TestCase):
    def test_format_arrays_are_independent_of_cached_rust_description(self):
        for name in ("tictactoe", "connect4"):
            with self.subTest(game=name), tag.create(name) as game:
                original = game.spec()
                expected = original.heuristic.indices.copy()
                self.assertFalse(original.heuristic.indices.flags.writeable)
                game._native.spec()["heuristic"]["indices"][:] = -1
                # Editing a returned array must leave the retained Rust format intact.
                np.testing.assert_array_equal(game.spec().heuristic.indices, expected)
                game.reset()
                with game.clone() as clone:
                    clone_spec = clone.spec()
                    np.testing.assert_array_equal(clone_spec.heuristic.indices, expected)
                    clone._native.spec()["heuristic"]["indices"][:] = -2
                    np.testing.assert_array_equal(game.spec().heuristic.indices, expected)
                    clone.set_controller(0, tag.OSLAPython.rust_heuristic(seed=7))
                    self.assertEqual(clone.advance(max_moves=1).moves, 1)

    def test_snapshots_arrays_reset_and_close_are_independent(self):
        for name in ("tictactoe", "connect4"):
            with self.subTest(game=name):
                game = tag.create(name)
                retained = game.observe(0)
                pristine = game.observe(0)
                retained.payloads["cells"].fill(99)
                retained.payloads["board"].fill(99)
                retained.action_mask.fill(False)
                retained.progress.fill(99)
                np.testing.assert_array_equal(game.observe(0).payloads["cells"], pristine.payloads["cells"])
                game.apply(np.int64(0), position=game.position)
                snapshot = game.clone()
                self.assertNotEqual(snapshot.position, game.position)
                with self.assertRaises(tag.StalePositionError):
                    snapshot.apply(1, position=game.position)
                for duplicate in (snapshot, copy.copy(game), copy.deepcopy(game)):
                    duplicate.apply(1, position=duplicate.position)
                    self.assertEqual(game.observe(0).progress[0], 1)
                old = game.observe(0)
                old_cells = old.payloads["cells"].copy()
                token = game.position
                game.reset()
                self.assertEqual(game.observe(0).progress.tolist(), [0, 0, 0, 0])
                with self.assertRaises(tag.StalePositionError):
                    game.apply(0, position=token)
                game.close()
                game.close()
                with self.assertRaises(tag.ClosedGameError):
                    game.observe(0)
                with self.assertRaises(tag.ClosedGameError):
                    game.reset()
                del game
                gc.collect()
                np.testing.assert_array_equal(old.payloads["cells"], old_cells)
                old.payloads["cells"].fill(3)

    def test_rejected_actions_leave_position_and_data_unchanged(self):
        game = tag.create("tictactoe")
        token = game.position
        for kwargs in ({"action_id": 9}, {"action_id": 0, "actor": 1}, {"action_id": 0, "actor": 2}):
            with self.assertRaises(ValueError):
                game.apply(position=token, **kwargs)
            self.assertEqual(game.position, token)
            self.assertEqual(game.observe(0).progress[0], 0)
        game.apply(0, position=token)
        with self.assertRaises(tag.StalePositionError):
            game.apply(1, position=token)
        token = game.position
        with self.assertRaises(ValueError):
            game.apply(0, position=token)
        self.assertEqual(game.position, token)
        self.assertEqual(game.observe(1).progress[0], 1)

    def test_full_column_and_terminal_input(self):
        game = tag.create("connect4")
        for _ in range(6):
            game.apply(0, position=game.position)
        token = game.position
        with self.assertRaises(ValueError):
            game.apply(0, position=token)
        self.assertEqual(game.position, token)
        game = tag.create("tictactoe", {"grid_size": 1})
        game.apply(0, position=game.position)
        self.assertIsNone(game.next_player)
        self.assertEqual(game.observe(0).results.tolist(), [1, 3])
        with self.assertRaises(ValueError):
            game.apply(0, position=game.position)

    def test_payloads_limits_and_strict_parameters(self):
        game = tag.create("tictactoe")
        for payloads in (("image",), ("cells", "cells")):
            with self.assertRaises(ValueError):
                game.observe(0, payloads)
        self.assertEqual(list(game.observe(0, ("board", "cells")).payloads), ["board", "cells"])
        self.assertEqual(dict(game.observe(0, ()).payloads), {})
        with self.assertRaises(ValueError):
            game.observe(2)
        for arguments in (("missing", {}), ("tictactoe", {"grid_szie": 3}),
                          ("tictactoe", {"grid_size": 0}), ("connect4", {"winning_length": 0})):
            with self.assertRaises(ValueError):
                tag.create(*arguments)
        for kwargs in ({"players": 3}, {"players": -1}):
            with self.assertRaises(ValueError):
                tag.create("tictactoe", **kwargs)
        for value in (True, 3.2, "3", None):
            with self.assertRaises(TypeError):
                tag.create("tictactoe", {"grid_size": value})
        with self.assertRaises(OverflowError):
            tag.create("tictactoe", {"grid_size": 2**100})
        with self.assertRaises(MemoryError):
            tag.create("tictactoe", max_cells=8)
        with self.assertRaises(MemoryError):
            tag.create("tictactoe", max_bytes=1)
        # The description's line indices fit; the requested board values do not.
        limited = tag.create("tictactoe", max_bytes=200)
        with self.assertRaises(MemoryError):
            limited.observe(0)
        self.assertEqual(limited.observe(0, ()).progress[0], 0)
        with self.assertRaises(TypeError):
            game.observe(0, "cells")

    def test_large_board_and_context_manager(self):
        with tag.create("tictactoe", {"grid_size": 64}) as game:
            view = game.observe(0, ("board",))
            self.assertEqual(view.payloads["board"].shape, (3, 64, 64))
            self.assertEqual(view.offered_action_ids.size, 4096)
        with self.assertRaises(tag.ClosedGameError):
            _ = game.position
        self.assertEqual(view.payloads["board"][2].sum(), 4096)


if __name__ == "__main__":
    unittest.main()
