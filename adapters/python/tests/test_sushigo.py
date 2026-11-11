"""Sushi Go! joint decisions and native MCTS through the compiled adapter."""
import unittest
import json
import importlib.util
import tempfile
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import tag_rust as tag
from tag_rust import _engine

HAS_GYM = importlib.util.find_spec("gymnasium") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SushiGoOperations(unittest.TestCase):
    def test_explicit_and_automatic_game_seeds(self):
        with tag.create("sushigo", players=2) as automatic:
            self.assertIsInstance(automatic.seed, int)
            self.assertGreaterEqual(automatic.seed, 0)
            with tag.create("sushigo", {"seed": automatic.seed}, players=2) as repeated:
                np.testing.assert_array_equal(
                    automatic.observe(0).payloads["hand"],
                    repeated.observe(0).payloads["hand"],
                )

    def test_joint_choices_do_not_change_rule_observations_until_reveal(self):
        self.assertIn("sushigo", tag.registered_games())
        with tag.create("sushigo", {"seed": 12}, players=3) as game:
            self.assertEqual(game.seed, 12)
            spec = game.spec()
            self.assertEqual((spec.actions.size, spec.actions.mapping), (156, "indexed"))
            before = [game.observe(seat) for seat in range(3)]
            first = int(before[0].offered_action_ids[0])
            game.apply(first, position=before[0].position, actor=0)
            self.assertEqual(game.position, before[0].position)
            self.assertEqual(game.next_player, 1)
            with self.assertRaises(ValueError):
                game.clone()
            with self.assertRaises(ValueError):
                game.apply(first, position=before[0].position, actor=0)
            for seat in range(3):
                after = game.observe(seat)
                self.assertEqual(after.next_player, before[seat].next_player)
                for name in before[seat].payloads:
                    np.testing.assert_array_equal(before[seat].payloads[name], after.payloads[name])
                if seat == 0:
                    self.assertEqual(after.offered_action_ids.size, 0)
            for seat in (1, 2):
                game.apply(int(before[seat].offered_action_ids[0]),
                           position=before[seat].position, actor=seat)
            self.assertNotEqual(game.position, before[0].position)
            with self.assertRaises(tag.StalePositionError):
                game.apply(first, position=before[0].position)
            self.assertEqual(game.next_player, 0)
            self.assertEqual(int(game.observe(0).progress[1]), 1)
            with game.clone() as clone:
                np.testing.assert_array_equal(game.observe(0).payloads["played"],
                                              clone.observe(0).payloads["played"])

    def test_complete_games_and_native_search_at_pending_boundary(self):
        for players in (2, 3, 4, 5):
            with tag.create("sushigo", {"seed": 31}, players=players) as game:
                if players == 3:
                    first = game.observe(0)
                    game.apply(int(first.offered_action_ids[0]), position=first.position)
                    with tag.MctsSearch(game, {"schema_version": 1, "seed": 9,
                                               "rollout_length": 0,
                                               "budgets": {"iterations": 8}}) as search:
                        with tag.MctsBatch([search]) as batch:
                            report, = batch.run()
                        self.assertEqual(report["counters"]["completed_iterations"], 8)
                        self.assertIn(report["action_id"], game.observe(1).offered_action_ids)
                while game.next_player is not None:
                    actor = game.next_player
                    view = game.observe(actor)
                    game.apply(int(view.offered_action_ids[0]), position=view.position, actor=actor)
                final = game.observe(0)
                self.assertEqual(int(final.progress[3]), 3)
                self.assertGreaterEqual(np.count_nonzero(final.results == 1), 1)
                self.assertEqual(final.offered_action_ids.size, 0)

    def test_native_opponents_preserve_joint_boundary_and_report_scores(self):
        with tag.create("sushigo", {"seed": 19}, players=3) as game:
            opponents = _engine.NativeOpponents(
                game._native, 1, json.dumps([
                    {"type": "osla", "seed": 21, "repetitions_per_action": 1}, None,
                    {"type": "mcts", "seed": 23, "config": {
                        "schema_version": 1, "rollout_length": 0,
                        "budgets": {"iterations": 4},
                    }},
                ]),
            )
            total = 0
            while True:
                result = json.loads(opponents.advance(game._native, 100))
                total += result["moves"]
                if result["reason"] == "finished":
                    break
                self.assertEqual(result["reason"], "candidate")
                self.assertEqual(game.next_player, 1)
                view = game.observe(1)
                opponents.apply_candidate(game._native, int(view.offered_action_ids[0]), view.position)
                total += 1
            state = json.loads(opponents.state(game._native))
            self.assertEqual(len(state["scores"]), 3)
            self.assertIn(1, state["outcomes"])
            self.assertEqual(total, 3 * 3 * (9 - 1))

    def test_initial_tokens_allow_out_of_order_submissions(self):
        with tag.create("sushigo", {"seed": 67}, players=3) as ordered:
            with ordered.clone() as reversed_game:
                views = [ordered.observe(seat) for seat in range(3)]
                other_views = [reversed_game.observe(seat) for seat in range(3)]
                for seat in (0, 1, 2):
                    ordered.apply(int(views[seat].offered_action_ids[0]),
                                  position=views[seat].position, actor=seat)
                for seat in (2, 0, 1):
                    reversed_game.apply(int(other_views[seat].offered_action_ids[0]),
                                        position=other_views[seat].position, actor=seat)
                for seat in range(3):
                    left, right = ordered.observe(seat), reversed_game.observe(seat)
                    for name in left.payloads:
                        np.testing.assert_array_equal(left.payloads[name], right.payloads[name])

    def test_native_lineup_detects_external_pending_submission(self):
        with tag.create("sushigo", {"seed": 67}, players=3) as game:
            opponents = _engine.NativeOpponents(
                game._native, 1, json.dumps([
                    {"type": "random", "seed": 21}, None,
                    {"type": "random", "seed": 23},
                ]),
            )
            view = game.observe(0)
            game.apply(int(view.offered_action_ids[0]), position=view.position, actor=0)
            self.assertEqual(game.position, view.position)
            with self.assertRaises(tag.StalePositionError):
                opponents.advance(game._native, 1)

    def test_external_value_requests_batch_deciding_player_views(self):
        class Values:
            identity = "sushigo-constant-values-v1"
            payloads = ("hand", "known_hands", "last_reveal")

            def __init__(self):
                self.calls = []

            def validate(self, spec, kind, observers, completion):
                assert spec.identity.game == "sushigo"
                assert kind == "values"
                assert observers == "deciding_player"
                assert completion == "require_full_vector"

            def evaluate(self, requests, *, max_bytes):
                self.calls.append(len(requests))
                replies = []
                for request in requests:
                    self_outer.assertEqual(len(request.observations), 1)
                    view = request.observations[0]
                    self_outer.assertEqual(set(view.payloads), set(self.payloads))
                    self_outer.assertEqual(view.payloads["known_hands"].shape, (3, 12))
                    replies.append(tag.MctsResponse.for_request(request, [0., 0., 0.]))
                return replies

        self_outer = self
        evaluator = Values()
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create("sushigo", {"seed": 41}, players=3))
                     for _ in range(2)]
            for game in games:
                first = game.observe(0)
                game.apply(int(first.offered_action_ids[0]), position=first.position, actor=0)
                self.assertEqual(game.next_player, 1)
            searches = [stack.enter_context(tag.MctsSearch(game, {
                "schema_version": 1, "seed": 9, "rollout_length": 0,
                "budgets": {"iterations": 8},
                "state_evaluator": {"name": "network_value"},
            }, bindings={"network_value": evaluator})) for game in games]
            with tag.MctsBatch(searches) as batch:
                reports = batch.run()
            self.assertFalse(batch.errors)
            self.assertTrue(any(count > 1 for count in evaluator.calls))
            self.assertEqual(reports[0]["action_id"], reports[1]["action_id"])


@unittest.skipUnless(HAS_GYM, "optional Gymnasium not installed")
class SushiGoEpisodes(unittest.TestCase):
    def test_single_episode_and_teacher_complete_all_rounds(self):
        from tag_rust.episodes.gym import GameEnv
        for opponent in ("random", "osla"):
            env = GameEnv("sushigo", players=3, learner=2,
                          opponent=opponent, teacher="osla",
                          include_intermediate=True)
            try:
                observation, info = env.reset(seed=17, options={"teacher_seed": 19})
                steps = 0
                while True:
                    action = int(np.flatnonzero(observation["action_mask"])[0])
                    observation, reward, done, truncated, info = env.step(action)
                    steps += 1
                    self.assertFalse(truncated)
                    if done:
                        self.assertEqual(steps, 3 * (9 - 1))
                        self.assertIn(reward, (-1., 1.))
                        break
            finally:
                env.close()


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class SushiGoSelfPlay(unittest.TestCase):
    def test_training_checkpoint_model_export_and_rotating_evaluation(self):
        from tag_rust.learning.selfplay import (
            SelfPlaySettings, SelfPlaySession, save_checkpoint, load_checkpoint,
        )
        from tag_rust.learning.artifacts import save_model, load_model
        settings = SelfPlaySettings.from_dict({
            "schema_version": 1, "game": "sushigo", "players": 3,
            "seed": 71, "cycles": 2, "games_per_cycle": 1, "parallel_games": 1,
            "search": {"schema_version": 1, "tree_policy": "ucb1", "rollout_length": 0,
                       "budgets": {"iterations": 4},
                       "state_evaluator": {"name": "network_value"}},
            "model": {"payloads": ["hand", "known_hands", "played", "paired_nigiri",
                                   "players", "context", "discard", "last_reveal",
                                   "round_scores"],
                      "hidden_sizes": [8], "value_output": "players"},
            "exploration": {"root_noise": False},
            "optimization": {"updates_per_cycle": 1, "batch_size": 4},
            "evaluation": {"enabled": True, "games": 3,
                           "opponent": {"name": "random", "agent": {"type": "random"}},
                           "ranking": "score"},
            "limits": {"max_moves": 100},
        })
        session = SelfPlaySession(settings)
        report = next(session.run(max_cycles=1))
        self.assertEqual(len(report["games"]), 1)
        self.assertEqual(report["evaluation"]["completed_games"], 3)
        self.assertEqual([row["games"] for row in report["evaluation"]["per_seat"]], [1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sushigo-checkpoint.pt"
            model = Path(directory) / "sushigo-model.json"
            save_checkpoint(checkpoint, session)
            resumed = load_checkpoint(checkpoint)
            self.assertEqual(resumed.provenance(), session.provenance())
            continuation = next(resumed.run(max_cycles=1))
            self.assertEqual(continuation["cycle"], 2)
            self.assertEqual(resumed.completed_cycles, 2)
            save_model(model, session.model, provenance=session.provenance())
            loaded = load_model(model)
            self.assertEqual(loaded.contract.identity.game, "sushigo")


if __name__ == "__main__":
    unittest.main()
