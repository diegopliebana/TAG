"""Can't Stop observations, chance continuation and Python capability boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import tag_rust as tag

NAMES = ("permanent", "temporary", "owners", "dice", "context")
ADAPTER = Path(__file__).resolve().parents[1]


def same_view(test, first, second):
    test.assertEqual(first.identity, second.identity)
    test.assertEqual(first.next_player, second.next_player)
    for name in ("progress", "results", "action_mask", "offered_action_ids"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    for name in first.payloads:
        np.testing.assert_array_equal(first.payloads[name], second.payloads[name])


class CantStopNativeComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
                                 str(ADAPTER / "Cargo.toml"), "--example", "cantstop_reference"],
                                cwd=ADAPTER, check=True, capture_output=True, text=True)
        cls.records = json.loads(result.stdout)

    def test_complete_native_games_match_every_python_observation_and_action(self):
        count = 0
        for case in self.records:
            with self.subTest(players=case["players"]), tag.create("cantstop", case["parameters"], players=case["players"]) as game:
                self.assertEqual(game.seed, case["parameters"]["seed"])
                self.assertEqual(game.spec().identity.player_count, case["players"])
                for step, expected_views in enumerate(case["positions"]):
                    for expected in expected_views:
                        view = game.observe(expected["observer"])
                        self.assertEqual(tuple(view.payloads), NAMES)
                        self.assertEqual(view.next_player, expected["next_player"])
                        for name in ("progress", "results", "action_mask", "offered_action_ids"):
                            np.testing.assert_array_equal(getattr(view, name), expected[name])
                        for name, tensor in expected["payloads"].items():
                            self.assertEqual(view.payloads[name].dtype, np.dtype("float32"))
                            self.assertEqual(view.payloads[name].shape, tuple(tensor["shape"]))
                            np.testing.assert_array_equal(view.payloads[name].ravel(), tensor["values"])
                        count += 1
                    if step < len(case["moves"]):
                        move = case["moves"][step]
                        game.apply(move["action_id"], actor=move["actor"], position=game.position)
                self.assertIsNone(game.next_player)
                self.assertEqual(game.observe(0).results.tolist().count(1), 1)
                self.assertFalse(game.observe(0).action_mask.any())
        self.assertGreater(count, 100)

    def test_midgame_copies_and_reset_preserve_exact_chance_continuation(self):
        for case in self.records:
            with tag.create("cantstop", case["parameters"], players=case["players"]) as game:
                initial = game.observe(0)
                for move in case["moves"][:5]:
                    game.apply(move["action_id"], actor=move["actor"], position=game.position)
                clones = [game.clone(), copy.copy(game), copy.deepcopy(game)]
                try:
                    for clone in clones:
                        self.assertEqual(clone.seed, game.seed)
                        self.assertNotEqual(clone.position, game.position)
                    for move in case["moves"][5:]:
                        for member in [game, *clones]:
                            member.apply(move["action_id"], actor=move["actor"], position=member.position)
                        for clone in clones:
                            same_view(self, game.observe(0), clone.observe(0))
                    old = game.position
                    game.reset()
                    self.assertNotEqual(game.position, old)
                    same_view(self, initial, game.observe(0))
                    for move in case["moves"]:
                        game.apply(move["action_id"], actor=move["actor"], position=game.position)
                    same_view(self, clones[0].observe(0), game.observe(0))
                finally:
                    for clone in clones:
                        clone.close()


class CantStopInterface(unittest.TestCase):
    def test_specs_seeds_and_legal_masks(self):
        for players in (2, 3, 4):
            with tag.create("cantstop", {"seed": np.uint64(2**64-1)}, players=players) as game:
                spec = game.spec()
                self.assertEqual(game.seed, 2**64-1)
                self.assertIsNone(spec.heuristic)
                self.assertEqual((spec.actions.size, spec.actions.mapping), (80, "indexed"))
                self.assertEqual(spec.payloads[0].tensor.shape, (players, 11, 2))
                for seat in range(players):
                    view = game.observe(seat)
                    self.assertEqual(view.results.dtype, np.dtype("int8"))
                    self.assertEqual(view.progress.dtype, np.dtype("int64"))
                    self.assertEqual(view.action_mask.dtype, np.dtype("bool"))
                    self.assertEqual(view.offered_action_ids.tolist(), [0] if seat == 0 else [])
                    self.assertEqual(view.payloads["context"][3+seat], 1)
                    self.assertNotIn("seed", dict(view.identity.configuration))
        with tag.create("cantstop") as automatic:
            self.assertIsInstance(automatic.seed, int)
            with tag.create("cantstop", {"seed": automatic.seed}) as explicit:
                self.assertEqual(automatic.spec().identity, explicit.spec().identity)
                for _ in range(10):
                    same_view(self, automatic.observe(automatic.next_player), explicit.observe(explicit.next_player))
                    for game in (automatic, explicit):
                        view = game.observe(game.next_player)
                        game.apply(int(view.offered_action_ids[0]), position=view.position)
        with tag.create("cantstop", {"seed": None}) as game:
            self.assertIsInstance(game.seed, int)
        with tag.create("tictactoe") as game:
            self.assertIsNone(game.seed)
            self.assertEqual(tuple(game.observe(0).payloads), ("cells", "board"))

    def test_invalid_actions_tokens_and_optional_operations_preserve_state(self):
        from tag_rust.search.evaluation import OSLAPython
        with tag.create("cantstop", {"seed": 42}, players=3) as game:
            before = game.observe(0)
            for action in (1, 2, 3, 79, 80):
                with self.assertRaises(ValueError):
                    game.apply(action, position=game.position)
                self.assertEqual(game.position, before.position)
                same_view(self, before, game.observe(0))
            with self.assertRaises(ValueError):
                game.apply(0, actor=1, position=game.position)
            for operation in (lambda: game.set_controller(0, OSLAPython(evaluator_id="test", seed=1)),
                              game.prepare_evaluation, game.open_lines_evaluator):
                with self.assertRaises(ValueError):
                    operation()
                self.assertEqual(game.position, before.position)
                same_view(self, before, game.observe(0))
            self.assertEqual(game.advance().reason, "human")
            with game.clone() as clone:
                with self.assertRaises(tag.StalePositionError):
                    clone.apply(0, position=before.position)
            game.apply(0, position=before.position)
            self.assertEqual(game.next_player, 0)
            with self.assertRaises(tag.StalePositionError):
                game.apply(0, position=before.position)

    def test_payload_ownership_limits_and_parameter_validation(self):
        with tag.create("cantstop", {"seed": 5}) as game:
            view = game.observe(0)
            view.payloads["permanent"].fill(123)
            self.assertFalse(game.observe(0).payloads["permanent"].any())
            self.assertEqual(tuple(game.observe(0, ("context", "dice")).payloads), ("context", "dice"))
            self.assertFalse(game.observe(0, ()).payloads)
            for names in (("cells",), ("dice", "dice")):
                with self.assertRaises(ValueError):
                    game.observe(0, names)
            retained = game.observe(0)
        self.assertEqual(retained.payloads["dice"].shape, (4, 7))
        with self.assertRaises(tag.ClosedGameError):
            _ = game.seed
        with tag.create("cantstop", {"seed": 1}, max_bytes=200) as game:
            self.assertEqual(game.observe(0, ()).offered_action_ids.tolist(), [0])
            with self.assertRaises(MemoryError):
                game.observe(0)
        for parameters in ({"rules": {"markers": 0}}, {"rules": {"column_heights": [1]*10}},
                           {"rules": {"columns_to_win": 0}}, {"rules": {"unknown": 1}}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                tag.create("cantstop", parameters)
        for parameters in ({"seed": True}, {"rules": {"markers": 1.5}}, {"rules": {"column_heights": [True]*11}}):
            with self.assertRaises(TypeError):
                tag.create("cantstop", parameters)
        for players in (0, 1, 5):
            with self.assertRaises(ValueError):
                tag.create("cantstop", players=players)
        with self.assertRaises(MemoryError):
            tag.create("cantstop", max_cells=21)
        with self.assertRaises(OverflowError):
            tag.create("cantstop", {"seed": 2**64})
        with tag.create("cantstop", {"rules": {"column_heights": [2**32-1]*11}, "seed": 1}, players=4) as game:
            self.assertEqual(game.spec().identity.player_count, 4)


if __name__ == "__main__":
    unittest.main()
