"""Check normalized game metrics with independently specified work counts."""
import unittest
from workflow_summary import game_metrics


class GameMetrics(unittest.TestCase):
    def test_uses_measured_work_not_warmup_or_a_fixed_length(self):
        row = dict(calls=32, elapsed_ns=96000, game_work=[32, 320, 80, 64, 320])
        result = game_metrics(row)
        self.assertEqual(result["actions_per_game"], 10)
        self.assertEqual(result["turns_per_game"], 2.5)
        self.assertEqual(result["ns_per_action"], 300)
        self.assertEqual(result["ns_per_turn"], 1200)

    def test_rejects_incomplete_game_denominators(self):
        with self.assertRaises(ValueError): game_metrics(dict(calls=32, elapsed_ns=1, game_work=[31, 80, 64, 0, 0]))
        with self.assertRaises(ValueError): game_metrics(dict(calls=32, elapsed_ns=1, game_work=[32, 80, 0, 0, 0]))

    def test_observation_calls_are_separate_from_turns(self):
        result = game_metrics(dict(calls=32, elapsed_ns=1000, game_work=[32, 80, 64, 0, 0]))
        self.assertEqual(result["observations_per_game"], 0)
        self.assertEqual(result["actions_per_game"], 2.5)
