"""Love Letter game operations, permitted arrays, history and capability limits."""
import copy
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import tag_rust as tag

NAMES = ("hands", "players", "reserve", "discards", "context", "public_history",
         "private_history", "last_public_history", "last_private_history", "last_tokens", "terminal_values")


def same_view(test, left, right):
    test.assertEqual(left.identity, right.identity)
    test.assertEqual(left.observer, right.observer)
    test.assertEqual(left.next_player, right.next_player)
    for name in ("progress", "results", "action_mask", "offered_action_ids"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    test.assertEqual(tuple(left.payloads), tuple(right.payloads))
    for name in left.payloads:
        np.testing.assert_array_equal(left.payloads[name], right.payloads[name])


class LoveLetterComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = Path(__file__).resolve().parents[1]
        result = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
                                 str(adapter / "Cargo.toml"), "--example", "loveletter_reference"],
                                cwd=adapter, check=True, capture_output=True, text=True)
        cls.cases = json.loads(result.stdout)

    def test_every_native_view_matches_python_in_complete_games_and_shared_wins(self):
        count = 0
        shared = False
        for case in self.cases:
            with tag.create("loveletter", case["parameters"], players=case["players"]) as game:
                for step, views in enumerate(case["positions"]):
                    for expected in views:
                        view = game.observe(expected["observer"])
                        self.assertEqual(tuple(view.payloads), NAMES)
                        self.assertEqual(view.next_player, expected["next_player"])
                        for name in ("progress", "results", "action_mask", "offered_action_ids"):
                            np.testing.assert_array_equal(getattr(view, name), expected[name])
                        for name, tensor in expected["payloads"].items():
                            array = np.zeros(tensor["shape"], dtype=np.float32)
                            for index, value in tensor["nonzero"]:
                                array.flat[index] = value
                            self.assertEqual(view.payloads[name].dtype, np.dtype("float32"))
                            np.testing.assert_array_equal(view.payloads[name], array)
                        self.assertNotIn("seed", dict(view.identity.configuration))
                        count += 1
                    if step < len(case["moves"]):
                        move = case["moves"][step]
                        game.apply(move["action_id"], actor=move["actor"], position=game.position)
                final = game.observe(0)
                np.testing.assert_array_equal(final.payloads["terminal_values"],
                                              np.where(final.results == 1, 1, -1))
                shared |= np.count_nonzero(final.results == 1) > 1
                self.assertIsNone(game.next_player)
                self.assertFalse(final.action_mask.any())
                self.assertEqual(final.offered_action_ids.size, 0)
        self.assertGreater(count, 100)
        self.assertTrue(shared)

    def test_exact_copies_reset_stale_handles_and_owned_arrays(self):
        for case in self.cases[:3]:
            with tag.create("loveletter", case["parameters"], players=case["players"]) as game:
                initial = game.observe(0)
                split = min(3, len(case["moves"]))
                for move in case["moves"][:split]:
                    game.apply(move["action_id"], actor=move["actor"], position=game.position)
                clones = [game.clone(), copy.copy(game), copy.deepcopy(game)]
                try:
                    for clone in clones:
                        self.assertEqual(clone.seed, game.seed)
                        self.assertNotEqual(clone.position, game.position)
                        with self.assertRaises(tag.StalePositionError):
                            clone.apply(0, position=game.position)
                    for move in case["moves"][split:]:
                        for member in [game, *clones]:
                            member.apply(move["action_id"], actor=move["actor"], position=member.position)
                        for clone in clones:
                            for seat in range(case["players"]):
                                same_view(self, game.observe(seat), clone.observe(seat))
                    old = game.position
                    game.reset()
                    same_view(self, initial, game.observe(0))
                    with self.assertRaises(tag.StalePositionError):
                        game.apply(int(initial.offered_action_ids[0]), position=old)
                    changed = game.observe(0)
                    changed.payloads["hands"][:] = 999
                    changed.action_mask[:] = False
                    same_view(self, initial, game.observe(0))
                finally:
                    for clone in clones:
                        clone.close()


class LoveLetterInterface(unittest.TestCase):
    def test_specs_unknown_hands_seeds_limits_and_capability_errors(self):
        self.assertIn("loveletter", tag.registered_games())
        for n in (2, 3, 4):
            with tag.create("loveletter", {"seed": np.uint64(2**64-1)}, players=n) as game:
                self.assertEqual(game.seed, 2**64-1)
                spec = game.spec()
                self.assertEqual((spec.actions.size, spec.actions.mapping), (52, "indexed"))
                self.assertIsNone(spec.heuristic)
                for seat in range(n):
                    view = game.observe(seat)
                    self.assertEqual(view.action_mask.dtype, np.dtype("bool"))
                    self.assertEqual(view.results.dtype, np.dtype("int8"))
                    self.assertEqual(view.progress.dtype, np.dtype("int64"))
                    for other in range(n):
                        if other != seat:
                            self.assertEqual(view.payloads["hands"][other, 0], -1)
                            self.assertEqual(view.payloads["hands"][other, 1], -1 if other == 0 else 0)
                    if seat:
                        self.assertFalse(view.action_mask.any())
                before = game.observe(0)
                self.assertFalse(before.payloads["terminal_values"].any())
                for action in (52, 2**32, 20):  # Out-of-domain or target self with King.
                    with self.assertRaises(ValueError):
                        game.apply(action, position=game.position)
                    same_view(self, before, game.observe(0))
                with self.assertRaises(ValueError):
                    game.observe(n)
                for payloads in (("hands", "hands"), ("unknown",)):
                    with self.assertRaises(ValueError):
                        game.observe(0, payloads)
        with tag.create("loveletter", {"seed": None}) as automatic:
            self.assertIsInstance(automatic.seed, int)
            with tag.create("loveletter", {"seed": automatic.seed}) as repeated:
                same_view(self, automatic.observe(0), repeated.observe(0))
        with tag.create("loveletter", max_bytes=1024) as limited:
            self.assertEqual(limited.observe(0, ()).action_mask.size, 52)
            with self.assertRaises(MemoryError):
                limited.observe(0)
        for parameters in ({"unknown": 1}, {"rules": {"winning_tokens": [0, 1, 1]}},
                           {"rules": {"winning_tokens": [1, 1]}}, {"first_player": 4}):
            with self.assertRaises(ValueError):
                tag.create("loveletter", parameters)
        for n in (0, 1, 5):
            with self.assertRaises(ValueError):
                tag.create("loveletter", players=n)
        with self.assertRaises(MemoryError):
            tag.create("loveletter", max_cells=15)
        with self.assertRaises(TypeError):
            tag.create("loveletter", {"seed": True})
        with self.assertRaises(OverflowError):
            tag.create("loveletter", {"seed": 2**64})

    def test_distinct_actual_hidden_worlds_produce_identical_player_arrays(self):
        # Three-player initial rounds have no public reserve. Find two real deals
        # with identical ordered observer hands and different opponent hands.
        with tag.create("loveletter", {"seed": 0}, players=3) as left:
            expected = left.observe(0)
            found = False
            for seed in range(1, 257):
                with tag.create("loveletter", {"seed": seed}, players=3) as right:
                    other = right.observe(0)
                    if not np.array_equal(expected.payloads["hands"][0], other.payloads["hands"][0]):
                        continue
                    if all(np.array_equal(left.observe(seat).payloads["hands"][seat],
                                          right.observe(seat).payloads["hands"][seat])
                           for seat in (1, 2)):
                        continue
                    same_view(self, expected, other)
                    found = True
                    break
            self.assertTrue(found, "fixed seed range must include a paired hidden-world fixture")


if __name__ == "__main__":
    unittest.main()
