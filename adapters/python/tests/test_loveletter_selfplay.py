"""Complete private-observation self-play, shared wins and exact cycle resumes."""
from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tag_rust as tag
from test_loveletter import same_view
from test_selfplay import configuration

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.artifacts import save_model, load_model
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.selfplay import SelfPlaySettings, SelfPlaySession, save_checkpoint, load_checkpoint
    from tag_rust.learning.selfplay.collection import collect_games
    from tag_rust.learning.selfplay.seeds import seed_version, workflow_seed, game_parameters

INPUTS = ("hands", "players", "reserve", "discards", "context", "public_history",
          "private_history", "last_public_history", "last_tokens")


def letter_settings(players=3, **changes):
    value = configuration("loveletter", players=players, games_per_cycle=players,
        parameters={"rules": {"winning_tokens": [2, 2, 2]}},
        model={"payloads": INPUTS, "hidden_sizes": [4], "value_output": "players"},
        limits={"max_moves": 4096})
    value.update(changes)
    return SelfPlaySettings.from_dict(value)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class LoveLetterSelfPlay(unittest.TestCase):
    def test_real_collection_rotates_starters_and_replays_every_private_input(self):
        for n in (2, 3, 4):
            settings = letter_settings(n)
            session = SelfPlaySession(settings)
            with torch.no_grad():
                for p in session.model.parameters():
                    p.zero_()
            snapshot = FrozenModel(session.model)
            single = collect_games(replace(settings, parallel_games=1), snapshot)
            grouped = collect_games(replace(settings, parallel_games=n), snapshot)
            def trace(result):
                return [(g.game_id, g.game_seed, g.outcomes,
                         [(p.actor, p.action, p.visits.tolist(), p.search_seed, p.noise_seed)
                          for p in g.positions]) for g in result.games]
            self.assertEqual(trace(single), trace(grouped))
            self.assertEqual([g.positions[0].actor for g in grouped.games], list(range(n)))
            self.assertEqual(len({g.game_seed for g in grouped.games}), n)
            for completed in grouped.games:
                with tag.create("loveletter", game_parameters(settings, completed.game_id), players=n) as game:
                    for position in completed.positions:
                        self.assertEqual(len(position.observations), 1)
                        actual = game.observe(game.next_player, INPUTS)
                        saved = position.observations[0]
                        same_view(self, actual, saved)
                        self.assertTrue((saved.results == 0).all())
                        self.assertNotIn("last_private_history", saved.payloads)
                        self.assertNotIn("seed", dict(saved.identity.configuration))
                        self.assertEqual(saved.observer, position.actor)
                        self.assertEqual(position.search_seed, workflow_seed(settings, "selfplay", completed.game_id, position.ply, "search"))
                        self.assertEqual(position.noise_seed, workflow_seed(settings, "selfplay", completed.game_id, position.ply, "root-noise"))
                        self.assertFalse(position.policy[~saved.action_mask].any())
                        self.assertAlmostEqual(float(position.policy.sum()), 1.)
                        for other in range(n):
                            if other != saved.observer:
                                self.assertTrue(np.isin(saved.payloads["hands"][other], (-1, 0)).all())
                        game.apply(position.action, position=game.position)
                    self.assertIsNone(game.next_player)
                    self.assertEqual(tuple(game.observe(0, ()).results), completed.outcomes)
                expected = np.array([1 if code == 1 else -1 for code in completed.outcomes], dtype=np.float32)
                for position in completed.positions:
                    np.testing.assert_array_equal(position.terminal_values, expected)
            fixed = replace(settings, parameters={"rules": {"winning_tokens": [2, 2, 2]}, "first_player": n-1})
            self.assertTrue(all(game_parameters(fixed, i)["first_player"] == n-1 for i in range(n*2)))
            fixed_game = collect_games(replace(fixed, games_per_cycle=1), snapshot, first_game_id=n+1).games[0]
            self.assertEqual(fixed_game.positions[0].actor, n-1)
            self.assertEqual(seed_version(settings), 2)

    def test_train_export_and_completed_cycle_resume_reproduce_both_buffer_modes(self):
        for n in (2, 3, 4):
            for mode in ("replay_buffer", "newest_group"):
                with self.subTest(players=n, mode=mode), tempfile.TemporaryDirectory() as folder:
                    settings = letter_settings(n, data={"mode": mode, "capacity": 24})
                    session = SelfPlaySession(settings)
                    initial = session.initial_identity
                    first = session.run_cycle()
                    self.assertNotEqual(first["model_identity"], initial)
                    self.assertEqual(session.model.contract.value.players, tuple(range(n)))
                    self.assertEqual(first["evaluation"], {"enabled": False})
                    self.assertEqual([g["first_player"] for g in first["games"]], list(range(n)))
                    self.assertEqual([row["games"] for row in first["per_seat"]], [n]*n)
                    self.assertEqual(first["combined"]["games"], n*n)
                    path = Path(folder)/"state.pt"
                    save_checkpoint(path, session)
                    self.assertEqual(torch.load(path, weights_only=True)["version"], 2)
                    self.assertEqual(torch.load(path, weights_only=True)["information_access"], "privileged-research")
                    with patch("tag_rust.learning.selfplay.training._engine.automatic_search_seed",
                               side_effect=AssertionError("resume reseeded")):
                        resumed = load_checkpoint(path)
                    next_report = session.run_cycle()
                    continued = resumed.run_cycle()
                    for key in ("before_identity", "model_identity", "games", "optimization", "buffer_digest", "per_seat", "combined"):
                        self.assertEqual(next_report[key], continued[key])
                    self.assertEqual(session.sample_rng.bit_generator.state, resumed.sample_rng.bit_generator.state)
                    self.assertTrue(torch.equal(session.torch_rng, resumed.torch_rng))
                    for key, tensor in session.model.state_dict().items():
                        self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]))
                    for a, b in zip(session.optimizer.state.values(), resumed.optimizer.state.values()):
                        for key in a:
                            self.assertTrue(torch.equal(a[key], b[key]))
                    if mode == "newest_group":
                        self.assertTrue(all(p.game_id >= n for p in resumed.buffer.positions))
                    exported = Path(folder)/"model.json"
                    save_model(exported, resumed.model, provenance=resumed.provenance())
                    loaded = load_model(exported)
                    self.assertEqual(FrozenModel(loaded).identity, continued["model_identity"])
                    self.assertEqual(SelfPlaySession(settings, initial_model=loaded).initial_identity,
                                     continued["model_identity"])
                    from tag_rust.search.neural_mcts import NeuralMctsEvaluator
                    from tag_rust.learning.selfplay.collection import start_search, move_settings
                    with tag.create("loveletter", game_parameters(settings, resumed.next_game_id), players=n) as game:
                        evaluator = NeuralMctsEvaluator(FrozenModel(loaded), turn_context="own_turn")
                        with start_search(game, settings, move_settings(settings, resumed.next_game_id, 0), evaluator) as search:
                            with tag.MctsBatch([search]) as batch:
                                report, = batch.run()
                                self.assertFalse(batch.errors)
                            self.assertIn(report["action_id"], game.observe(game.next_player, ()).offered_action_ids)

    def test_configuration_and_incomplete_collection_never_create_terminal_targets(self):
        for changes in ({"parameters": {"seed": None}}, {"model": {"value_output": "observer"}},
                        {"model": {"value_output": "players", "value_players": [2, 1, 0]}},
                        {"model": {"payloads": ["last_private_history"], "value_output": "players"}}):
            with self.assertRaises(ValueError):
                letter_settings(**changes)
        session = SelfPlaySession(letter_settings(limits={"max_moves": 1}))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"before.pt"
            save_checkpoint(path, session)
            saved = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "max_moves"):
                session.run_cycle()
            self.assertEqual(len(session.buffer), 0)
            self.assertEqual(session.updates, 0)
            self.assertEqual(FrozenModel(session.model).identity, session.initial_identity)
            with self.assertRaises(RuntimeError):
                save_checkpoint(path, session)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(load_checkpoint(path).completed_cycles, 0)

    def test_actual_shared_win_supplies_multiple_positive_targets_and_reloads(self):
        # Recorded fixed-work fixture: an actual four-player game shares victory
        # between seats 1 and 3. The zero model controls predictions, not the deal.
        settings = letter_settings(4, seed=5, games_per_cycle=1, parallel_games=1,
            parameters={"rules": {"winning_tokens": [1, 1, 1]}},
            model={"payloads": ["hands", "players", "context"], "hidden_sizes": [4], "value_output": "players"},
            search={"schema_version": 1, "tree_policy": "puct", "rollout_length": 0,
                    "budgets": {"iterations": 1}, "state_evaluator": {"name": "network_value"},
                    "puct_preferences": {"source": "probabilities", "evaluator": {"name": "network_policy"}}},
            exploration={"root_noise": False, "move_policy": "most_visits"}, limits={"max_moves": 11})
        prototype = SelfPlaySession(settings)
        with torch.no_grad():
            for p in prototype.model.parameters():
                p.zero_()
        session = SelfPlaySession(settings, initial_model=prototype.model)
        report = session.run_cycle()
        self.assertEqual(report["games"][0]["outcomes"], [3, 1, 3, 1])
        self.assertEqual(report["games"][0]["moves"], 11)
        self.assertEqual(report["combined"]["wins"], 2)
        for position in session.buffer.positions:
            np.testing.assert_array_equal(position.terminal_values, [-1, 1, -1, 1])
            self.assertTrue((position.observations[0].results == 0).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"shared.pt"
            save_checkpoint(path, session)
            self.assertEqual(load_checkpoint(path).buffer.digest(), session.buffer.digest())

    def test_checkpoint_replay_rejects_forged_private_inputs_even_with_a_new_digest(self):
        from tag_rust.learning.selfplay.files import _position
        from tag_rust.learning.selfplay.data import ReplayBuffer
        session = SelfPlaySession(letter_settings(data={"capacity": 12}))
        session.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"state.pt"
            save_checkpoint(path, session)
            original = torch.load(path, weights_only=True)
            def bad_starter(data):
                game = data["history"][0]["games"][0]
                game["first_player"] = (game["first_player"]+1) % 3
            def bad_seed(data):
                data["history"][0]["games"][0]["game_seed"] += 1
            def bad_action(data):
                data["history"][0]["games"][0]["actions"][0] = 52
            def bad_outcomes(data):
                data["history"][0]["games"][0]["outcomes"] = [3, 3, 3]
            def bad_settings(data):
                data["settings"]["search"]["budgets"]["iterations"] += 1
            def bad_memory(data):
                data["settings"]["model"]["payloads"] += ("last_private_history",)
            for mutate in (bad_starter, bad_seed, bad_action, bad_outcomes, bad_settings, bad_memory):
                data = copy.deepcopy(original)
                mutate(data)
                torch.save(data, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            data = copy.deepcopy(original)
            hands = data["positions"][0]["observations"][0]["payloads"]["hands"]
            row, column = torch.nonzero(hands == -1)[0].tolist()
            hands[row, column] = 9  # Forge a supposedly known opponent Princess.
            buffer = ReplayBuffer(session.settings.data, session.model.contract)
            buffer.add_group(tuple(_position(p, session.model.contract) for p in data["positions"]))
            data["buffer_digest"] = buffer.digest()
            data["history"][-1]["buffer_digest"] = buffer.digest()
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "permitted real-game replay"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
