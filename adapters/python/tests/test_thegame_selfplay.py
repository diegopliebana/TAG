"""Complete cooperative self-play, checkpoint replay and team evaluation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tag_rust as tag

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.selfplay import SelfPlaySettings, SelfPlaySession, save_checkpoint, load_checkpoint
    from tag_rust.learning.selfplay.collection import collect_games
    from tag_rust.learning.selfplay.evaluation import evaluate_model
    from tag_rust.learning.selfplay import lineup_evaluation
    from tag_rust.learning.selfplay.seeds import game_parameters, workflow_seed


def settings(players=2, *, communication="limited_choices", evaluation=False, **changes):
    document = {
        "schema_version": 1, "game": "thegame", "players": players,
        "parameters": {"communication": communication}, "seed": 42,
        "cycles": 2, "games_per_cycle": 1, "parallel_games": 1,
        "search": {"schema_version": 1, "structured_actions": True,
                   "tree_policy": "puct", "rollout_length": 0,
                   "budgets": {"iterations": 1},
                   "value_completion": "require_full_vector",
                   "state_evaluator": {"name": "network_value"},
                   "puct_preferences": {"source": "probabilities",
                                        "evaluator": {"name": "network_policy"}}},
        "model": {"payloads": ["hand", "piles", "players", "context"],
                  "hidden_sizes": [4], "value_output": "players"},
        "exploration": {"root_noise": False},
        "optimization": {"updates_per_cycle": 1, "batch_size": 4},
        "evaluation": {"enabled": evaluation, "games": players,
                       "opponent": {"name": "random", "agent": {"type": "random"}},
                       "max_actions": 1024},
        "limits": {"max_moves": 1024},
    }
    document.update(changes)
    return SelfPlaySettings.from_dict(document)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class TheGameSelfPlay(unittest.TestCase):
    def test_parallel_collection_preserves_fixed_work_trace(self):
        configured = settings(games_per_cycle=2)
        session = SelfPlaySession(configured)
        with torch.no_grad():
            for parameter in session.model.parameters():
                parameter.zero_()
        snapshot = FrozenModel(session.model)
        serial = collect_games(configured, snapshot)
        parallel = collect_games(replace(configured, parallel_games=2), snapshot)
        def trace(group):
            return [(game.game_id, game.game_seed, game.outcomes, game.score,
                     [(position.actor, position.action, position.search_seed, position.noise_seed)
                      for position in game.positions]) for game in group.games]
        self.assertEqual(trace(serial), trace(parallel))

    def test_configuration_and_seed_schedule(self):
        for players in range(1, 6):
            configured = settings(players)
            self.assertEqual([game_parameters(configured, index)["first_player"]
                              for index in range(players)], list(range(players)))
            self.assertEqual(len({game_parameters(configured, index)["seed"]
                                  for index in range(players)}), players)
        for change in ({"parameters": {"seed": 1}},
                       {"model": {"value_output": "observer"}},
                       {"evaluation": {"enabled": True, "ranking": "score"}}):
            with self.assertRaises(ValueError):
                settings(**change)
        fixed = replace(settings(), parameters={"communication": "silent", "first_player": 1})
        self.assertTrue(all(game_parameters(fixed, index)["first_player"] == 1 for index in range(3)))
        with tag.create("thegame", {"seed": 42, "communication": "silent"}, players=1) as game:
            self.assertIsNotNone(game.next_player)

    def test_train_resume_and_permitted_inputs(self):
        configured = settings(parameters={"communication": "limited_choices", "first_player": 1})
        session = SelfPlaySession(configured)
        collected = collect_games(configured, FrozenModel(session.model))
        game, = collected.games
        self.assertEqual(game.positions[0].actor, 0)  # First speaker, not starting card player.
        self.assertEqual(game.score, int(game.positions[-1].observations[0].payloads["context"][6])
                         + int(13 <= game.positions[-1].action <= 404))
        self.assertEqual(len(set(game.outcomes)), 1)
        for position in game.positions:
            view, = position.observations
            self.assertEqual(view.observer, position.actor)
            self.assertEqual(len(view.action_mask), 406)
            self.assertFalse(position.policy[~view.action_mask].any())
            np.testing.assert_array_equal(position.terminal_values,
                                          np.full(2, 1 if game.outcomes[0] == 1 else -1))
            self.assertEqual(position.search_seed,
                             workflow_seed(configured, "selfplay", game.game_id, position.ply, "search"))
        first = session.run_cycle()
        self.assertEqual(first["games"][0]["first_player"], 1)
        self.assertEqual(first["games"][0]["score"], game.score)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, session)
            resumed = load_checkpoint(path)
            expected = session.run_cycle()
            actual = resumed.run_cycle()
            for key in ("games", "model_identity", "optimization", "buffer_digest"):
                self.assertEqual(actual[key], expected[key])

    def test_rotated_team_evaluation_and_truncation(self):
        configured = settings(evaluation=True)
        session = SelfPlaySession(configured)
        result = evaluate_model(configured, FrozenModel(session.model))
        self.assertEqual(result["completed_games"], 2)
        self.assertEqual([row["candidate_seat"] for row in result["games"]], [0, 1])
        self.assertEqual([row["first_player"] for row in result["games"]], [0, 0])
        for row in result["games"]:
            self.assertEqual(len(set(row["scores"])), 1)
            self.assertEqual(row["score"], row["scores"][0])
            self.assertEqual(len(set(row["outcomes"])), 1)
        self.assertEqual(result["combined"]["average_score"],
                         sum(row["score"] for row in result["games"]) / 2)
        truncated = replace(configured, evaluation=replace(configured.evaluation, max_actions=1))
        limited = evaluate_model(truncated, FrozenModel(session.model))
        self.assertEqual(limited["completed_games"], 0)
        self.assertEqual(limited["truncated_games"], 2)
        self.assertIsNone(limited["combined"]["average_score"])

    def test_runtime_failure_is_not_a_team_loss(self):
        configured = settings(evaluation=True)
        snapshot = FrozenModel(SelfPlaySession(configured).model)
        actual_lineup = lineup_evaluation.native_lineup
        calls = 0
        def fail_after_validation(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return actual_lineup(*args)
            raise RuntimeError("injected teammate failure")
        with patch.object(lineup_evaluation, "native_lineup", side_effect=fail_after_validation):
            result = evaluate_model(configured, snapshot)
        self.assertEqual(result["failed_games"], 1)
        self.assertEqual(result["completed_games"], 0)
        self.assertEqual(result["unplayed_games"], 1)
        self.assertEqual(result["games"][0]["status"], "failed")
        self.assertIsNone(result["games"][0]["outcomes"])
        self.assertIsNone(result["combined"]["average_score"])


if __name__ == "__main__":
    unittest.main()
