"""Test suspended OSLAPython execution with explicit mock values, without a learned model."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import tag_rust as tag

ADAPTER = Path(__file__).resolve().parents[1]


def setup_game(name="tictactoe", *, rows=1024, seed=7, **settings):
    game = tag.create(name)
    game.set_controller(0, tag.OSLAPython(evaluator_id="mock", seed=seed,
                                         payloads=("cells", "board"), max_rows=rows, **settings))
    return game


def finish(game, request, pattern=None):
    while isinstance(request, tag.EvaluationRequest):
        values = np.zeros(request.action_ids.size, dtype=np.float64) if pattern is None else pattern(request)
        request = game.finish_evaluation(request, values)
    return request


class EvaluationTests(unittest.TestCase):
    def test_os_seed_configuration_and_initial_validation(self):
        with patch("tag_rust.randomness.secrets.randbits", side_effect=[42, 43]):
            self.assertEqual(tag.OSLAPython(evaluator_id="mock").seed, 42)
            self.assertEqual(tag.OSLAPython(evaluator_id="mock").seed, 43)
        settings = {"weights": [1, 2], "scale": 0.5}
        agent = tag.OSLAPython(evaluator_id="mock", evaluator_parameters=settings,
                               payloads=(name for name in ("cells", "board")), seed=0)
        settings["weights"][0] = 99
        self.assertEqual(json.loads(agent.checkpoint.evaluator_configuration)["weights"], [1, 2])
        self.assertEqual(agent.checkpoint.payloads, ("cells", "board"))
        for kwargs in ({"max_rows": 0}, {"max_candidates": 0}, {"max_bytes": 0},
                       {"evaluator_version": 0}, {"payloads": ("cells", "cells")}):
            with self.assertRaises(ValueError):
                tag.OSLAPython(evaluator_id="mock", **kwargs)
        with self.assertRaises(ValueError):
            tag.OSLAPython(evaluator_id="mock", evaluator_parameters={"scale": float("nan")})

    def test_pending_reset_and_old_chunks_are_invalidated(self):
        for ready in (False, True):
            game = setup_game()
            work = game.prepare_evaluation()
            if ready:
                work = finish(game, work)
            retained = game.observe(0).payloads["cells"]
            game.reset()
            with self.assertRaises(tag.StalePositionError):
                game.cancel_evaluation(work)
            self.assertEqual(game.controller_checkpoint(0).word_position, 0)
            np.testing.assert_array_equal(retained, np.zeros(9))
        game = setup_game(rows=2)
        old = game.prepare_evaluation()
        current = game.finish_evaluation(old, np.zeros(2))
        with self.assertRaises(tag.StalePositionError):
            game.finish_evaluation(old, np.zeros(2))
        with self.assertRaises(tag.StalePositionError):
            game.cancel_evaluation(old)
        game.cancel_evaluation(current)
        self.assertEqual(game.controller_checkpoint(0).word_position, 0)

    def test_human_ai_flow_and_shared_settings_create_independent_agents(self):
        game = tag.create("connect4")
        agent = tag.OSLAPython(evaluator_id="mock", seed=7)
        game.set_controller(1, agent)
        with self.assertRaises(ValueError):
            game.prepare_evaluation()
        game.apply(2, position=game.position)
        with self.assertRaises(ValueError):
            game.apply(2, position=game.position)
        request = game.prepare_evaluation()
        self.assertEqual(request.actor, 1)
        # The source has an opposing piece, and every candidate has one new own piece.
        for row in request.payloads["cells"]:
            self.assertEqual(np.count_nonzero(row == 1), 1)
            self.assertEqual(row[17], -1)
        game.commit_evaluation(finish(game, request))
        self.assertEqual(game.next_player, 0)
        self.assertIsNone(game.controller_checkpoint(0))
        other = tag.create("connect4")
        other.set_controller(0, agent)
        other.set_controller(1, agent)
        before = other.controller_checkpoint(1)
        other.commit_evaluation(finish(other, other.prepare_evaluation()))
        self.assertGreater(other.controller_checkpoint(0).word_position, 0)
        self.assertEqual(other.controller_checkpoint(1), before)

    def test_chunk_arrays_and_global_best_for_original_player(self):
        game = setup_game(rows=2)
        before = game.controller_checkpoint(0)
        request = game.prepare_evaluation()
        retained = request.payloads["board"]
        original = retained.copy()
        ids = []
        while isinstance(request, tag.EvaluationRequest):
            rows = request.action_ids.size
            self.assertEqual(request.payloads["cells"].shape, (rows, 9))
            self.assertEqual(request.payloads["board"].shape, (rows, 3, 3, 3))
            self.assertEqual(request.actor, 0)
            np.testing.assert_array_equal(request.next_players, np.ones(rows, dtype=np.int64))
            self.assertEqual(request.progress.shape, (rows, 4))
            self.assertEqual(request.results.shape, (rows, 2))
            self.assertEqual(request.results.dtype, np.dtype("int8"))
            for row, action_id in enumerate(request.action_ids):
                self.assertEqual(request.payloads["cells"][row, action_id], 1)
                self.assertEqual(np.count_nonzero(request.payloads["cells"][row]), 1)
            ids.extend(request.action_ids.tolist())
            request = game.finish_evaluation(request, request.action_ids.astype(np.float64))
            self.assertEqual(game.controller_checkpoint(0), before)
            self.assertEqual(game.observe(0, ()).progress[0], 0)
        self.assertEqual(ids, list(range(9)))
        self.assertEqual(request.action_id, 8)
        game.commit_evaluation(request)
        self.assertEqual(game.observe(0, ()).progress[0], 1)
        np.testing.assert_array_equal(retained, original)
        game.close()
        np.testing.assert_array_equal(retained, original)

    def test_invalid_values_metadata_and_duplicate_replies_allow_retry(self):
        game = setup_game()
        request = game.prepare_evaluation()
        checkpoint = game.controller_checkpoint(0)
        for bad in (np.zeros(8), np.full(9, np.nan), np.full(9, np.inf)):
            with self.assertRaises(ValueError):
                game.finish_evaluation(request, bad)
            self.assertEqual(game.controller_checkpoint(0), checkpoint)
        for bad in ([0.] * 9, np.zeros(9, dtype=np.float32), np.zeros((9, 1))):
            with self.assertRaises(TypeError):
                game.finish_evaluation(request, bad)
        for evaluator in (replace(request.evaluator, version=2), replace(request.evaluator, configuration='{"weight":2}')):
            with self.assertRaises(ValueError):
                game.finish_evaluation(request, np.zeros(9), evaluator=evaluator)
        with self.assertRaises(ValueError):
            game.finish_evaluation(request, np.zeros(9), action_ids=request.action_ids[::-1].copy())
        with self.assertRaises(tag.EvaluationError):
            game.prepare_evaluation()
        with self.assertRaises(tag.EvaluationError):
            game.clone()
        with self.assertRaises(ValueError):
            game.apply(0, position=game.position)
        # Read a strided array by copying its values before detaching.
        proposal = game.finish_evaluation(request, np.zeros(18)[::2])
        self.assertEqual(game.controller_checkpoint(0), checkpoint)
        with self.assertRaises(tag.EvaluationError):
            game.finish_evaluation(request, np.zeros(9))
        game.commit_evaluation(proposal)
        with self.assertRaises(tag.StalePositionError):
            game.commit_evaluation(proposal)

    def test_cancel_reset_replacement_close_and_foreign_requests(self):
        game = setup_game()
        initial = game.controller_checkpoint(0)
        request = game.prepare_evaluation()
        game.cancel_evaluation(request)
        with self.assertRaises(tag.StalePositionError):
            game.finish_evaluation(request, np.zeros(9))
        proposal = finish(game, game.prepare_evaluation())
        chosen = proposal.action_id
        game.cancel_evaluation(proposal)
        self.assertEqual(game.controller_checkpoint(0), initial)
        again = finish(game, game.prepare_evaluation())
        self.assertEqual(again.action_id, chosen)
        other = setup_game()
        foreign = other.prepare_evaluation()
        with self.assertRaises(tag.StalePositionError):
            game.finish_evaluation(foreign, np.zeros(9))
        game.reset()
        with self.assertRaises(tag.StalePositionError):
            game.commit_evaluation(again)
        self.assertEqual(game.controller_checkpoint(0), initial)
        pending = game.prepare_evaluation()
        game.set_controller(0, None)
        with self.assertRaises(tag.StalePositionError):
            game.finish_evaluation(pending, np.zeros(9))
        game.apply(0, position=game.position)
        other.close()
        with self.assertRaises(tag.ClosedGameError):
            other.finish_evaluation(foreign, np.zeros(9))

    def test_clone_restore_reset_and_seed_preserve_exact_continuation(self):
        game = setup_game(seed=2**64 - 1)
        first = finish(game, game.prepare_evaluation())
        game.commit_evaluation(first)
        game.apply(int(game.observe(1, ()).offered_action_ids[0]), position=game.position)
        snapshot = game.controller_checkpoint(0)
        self.assertGreater(snapshot.word_position, 0)
        clone = game.clone()
        game.set_controller(0, None)
        game.restore_controller(0, snapshot)
        a = finish(game, game.prepare_evaluation())
        b = finish(clone, clone.prepare_evaluation())
        self.assertEqual(a.action_id, b.action_id)
        game.commit_evaluation(a)
        clone.commit_evaluation(b)
        self.assertEqual(game.controller_checkpoint(0), clone.controller_checkpoint(0))
        for bad in (replace(snapshot, policy_version=2), replace(snapshot, rng_version=2),
                    replace(snapshot, word_position=2**68), replace(snapshot, max_rows=0)):
            before = game.controller_checkpoint(0)
            with self.assertRaises(ValueError):
                game.restore_controller(0, bad)
            self.assertEqual(game.controller_checkpoint(0), before)
        game.reset()
        self.assertEqual(game.controller_checkpoint(0).seed, 2**64 - 1)
        self.assertEqual(game.controller_checkpoint(0).word_position, 0)
        self.assertIsInstance(tag.OSLAPython(evaluator_id="mock").seed, int)

    def test_limits_metadata_only_and_terminal_candidate(self):
        game = setup_game(rows=9, max_bytes=400)
        request = game.prepare_evaluation()
        self.assertEqual(request.action_ids.size, 2)
        game.cancel_evaluation(request)
        for settings in ({"max_candidates": 8}, {"max_cells": 8}):
            game = setup_game(**settings)
            with self.assertRaises(MemoryError):
                game.prepare_evaluation()
            self.assertEqual(game.controller_checkpoint(0).word_position, 0)
        game = tag.create("tictactoe", {"grid_size": 1})
        game.set_controller(0, tag.OSLAPython(evaluator_id="mock", seed=0, payloads=()))
        request = game.prepare_evaluation()
        self.assertEqual(dict(request.payloads), {})
        self.assertEqual(request.next_players.tolist(), [-1])
        self.assertEqual(request.results.tolist(), [[1, 3]])
        proposal = game.finish_evaluation(request, np.array([1.], dtype=np.float64))
        game.commit_evaluation(proposal)
        self.assertIsNone(game.next_player)
        with self.assertRaises(ValueError):
            game.prepare_evaluation()

    def test_all_candidates_chunks_and_seeded_games_match_native_executor(self):
        output = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
            str(ADAPTER / "Cargo.toml"), "--example", "native_reference"], check=True, capture_output=True, text=True)
        records = json.loads(output.stdout)
        decisions = 0
        for record in records:
            case = record["case"]
            game = tag.create(case["game"], case["parameters"])
            for seat in range(2):
                game.set_controller(seat, tag.OSLAPython(evaluator_id="stage4_mock", seed=7+seat,
                    evaluator_parameters={"pattern":2}, payloads=("cells", "board"), max_rows=2))
            probe = game.prepare_evaluation()
            game.cancel_evaluation(probe)
            for decision in record["mock_evaluations"]:
                request = game.prepare_evaluation()
                for chunk in decision["chunks"]:
                    self.assertEqual(request.offset, chunk["offset"])
                    self.assertEqual(request.total, chunk["total"])
                    self.assertEqual(request.actor, chunk["actor"])
                    np.testing.assert_array_equal(request.action_ids, chunk["action_ids"])
                    for field in ("progress", "results", "next_players"):
                        np.testing.assert_array_equal(getattr(request, field).ravel(), chunk[field])
                    for name, tensor in chunk["payloads"].items():
                        self.assertEqual(request.payloads[name].shape, tuple(tensor["shape"]))
                        np.testing.assert_array_equal(request.payloads[name].ravel(), tensor["values"])
                    values = (request.action_ids % 2).astype(np.float64)
                    request = game.finish_evaluation(request, values)
                self.assertIsInstance(request, tag.MoveProposal)
                self.assertEqual(request.action_id, decision["selected"])
                game.commit_evaluation(request)
                self.assertEqual([game.controller_checkpoint(p).word_position for p in range(2)], decision["rng_word_positions"])
                view = game.observe(0)
                for field in ("progress", "results", "action_mask", "offered_action_ids"):
                    np.testing.assert_array_equal(getattr(view, field), decision["after"][field])
                for name, tensor in decision["after"]["payloads"].items():
                    np.testing.assert_array_equal(view.payloads[name].ravel(), tensor["values"])
                decisions += 1
            self.assertIsNone(game.next_player)
        print(f"OSLAPython native/Python mock parity: {len(records)} games, {decisions} decisions.")


if __name__ == "__main__":
    unittest.main()
