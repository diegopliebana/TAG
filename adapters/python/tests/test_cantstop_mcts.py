"""Stochastic native/neural search, suspension and independent game chance."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import unittest

import numpy as np
import tag_rust as tag
from test_cantstop import same_view
from test_mcts import comparison, settings

NAMES = ("permanent", "temporary", "owners", "dice", "context")
ADAPTER = Path(__file__).resolve().parents[1]


class Constant:
    identity = "cantstop-constant-v1"
    payloads = NAMES

    def __init__(self):
        self.requests = []
        self.batch_sizes = []

    def validate(self, spec, kind, observers, completion):
        pass

    def evaluate(self, requests, *, max_bytes):
        self.requests.extend(requests)
        self.batch_sizes.append(len(requests))
        return [tag.MctsResponse.for_request(request,
            np.ones(len(request.action_ids)) if request.kind == "policy" else
            np.arange(1, request.observations[0].identity.player_count + 1) / 8)
            for request in requests]


class CantStopSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
            str(ADAPTER / "Cargo.toml"), "--example", "cantstop_mcts_reference"],
            cwd=ADAPTER, check=True, capture_output=True, text=True)
        cls.cases = json.loads(result.stdout)

    def test_native_and_external_search_match_native_reports_in_every_phase(self):
        phases = set()
        for case in self.cases:
            with self.subTest(players=case["players"], phase=case["phase"], evaluator=case["evaluator"]):
                with tag.create("cantstop", case["parameters"], players=case["players"]) as game:
                    for action in case["moves"]:
                        game.apply(action, position=game.position)
                    configuration = case["settings"]
                    evaluator = Constant()
                    bindings = {}
                    if case["evaluator"] == "constant":
                        configuration["state_evaluator"]["name"] = "external_value"
                        bindings["external_value"] = evaluator
                    with tag.MctsSearch(game, configuration, bindings=bindings) as search:
                        with tag.MctsBatch([search]) as batch:
                            report, = batch.run()
                            self.assertFalse(batch.errors)
                    self.assertEqual(comparison(report), case["report"])
                    self.assertLessEqual(report["counters"]["forward_model_calls"], 80)
                    for request in evaluator.requests:
                        self.assertEqual(request.rows, 1)
                        self.assertEqual(request.observations[0].observer, request.deciding_player)
                        self.assertEqual(request.observations[0].next_player, request.deciding_player)
                    phases.add(case["phase"])
        self.assertEqual(len(phases), 4)

    def test_batching_retry_and_polling_preserve_simulation_continuation_and_live_dice(self):
        for players in (2, 3, 4):
            evaluator = Constant()
            with ExitStack() as stack:
                games = [stack.enter_context(tag.create("cantstop", {"seed": 42}, players=players)) for _ in range(3)]
                untouched = stack.enter_context(games[0].clone())
                searches = [stack.enter_context(tag.MctsSearch(game, settings(40, policy=True),
                    bindings={"external_value": evaluator, "external_policy": evaluator})) for game in games]
                request = searches[0].advance()
                self.assertIs(searches[0].advance(), request)
                reply = evaluator.evaluate([request], max_bytes=searches[0].max_bytes)[0]
                for bad in (replace(reply, request_id=(999, 999)),
                            replace(reply, values=np.zeros(1)),
                            replace(reply, values=np.full(players, np.nan)),
                            replace(reply, scalar_player=0, values=np.zeros(1))):
                    with self.assertRaises(ValueError): searches[0].resume(request, bad)
                    self.assertIs(searches[0].pending, request)
                searches[0].resume(request, reply)
                with tag.MctsBatch(searches[:2]) as batch:
                    grouped = batch.run()
                    self.assertFalse(batch.errors)
                with tag.MctsBatch(searches[2:]) as batch:
                    single, = batch.run()
                    self.assertFalse(batch.errors)
                for report in grouped:
                    self.assertEqual(comparison(report), comparison(single))
                self.assertIn(2, evaluator.batch_sizes)
                # Root Roll must produce varied simulated dice, all using owned public arrays.
                dice = {tuple(r.observations[0].payloads["dice"].ravel()) for r in evaluator.requests}
                self.assertGreater(len(dice), 1)
                self.assertTrue(all(r.observations[0].observer == r.deciding_player for r in evaluator.requests))
                for _ in range(30):
                    view = untouched.observe(untouched.next_player)
                    action = int(view.offered_action_ids[0])
                    for game in [*games, untouched]:
                        game.apply(action, position=game.position)
                    for game in games:
                        same_view(self, untouched.observe(0), game.observe(0))

    def test_limits_cancellation_seed_reproduction_and_stale_positions(self):
        evaluator = Constant()
        with tag.create("cantstop", {"seed": 42}, players=3) as game:
            before = game.observe(0)
            config = settings(8)
            config["seed"] = None
            with tag.MctsSearch(game, config, bindings={"external_value": evaluator}) as search:
                resolved = search.settings["seed"]
                with tag.MctsBatch([search]) as batch:
                    first, = batch.run()
            config["seed"] = resolved
            with tag.MctsSearch(game, config, bindings={"external_value": evaluator}) as search:
                with tag.MctsBatch([search]) as batch:
                    second, = batch.run()
            self.assertEqual(comparison(first), comparison(second))
            with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}) as search:
                request = search.advance()
                search.cancel()
                with self.assertRaises(RuntimeError):
                    search.resume(request, evaluator.evaluate([request], max_bytes=search.max_bytes)[0])
                request.observations[0].payloads["dice"].fill(123)
            same_view(self, before, game.observe(0))
            for options in ({"max_rows": 0}, {"max_bytes": 1}, {"value_observers": "all_players"}):
                with self.assertRaises(ValueError):
                    with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}, **options) as search:
                        search.advance()
            for completion in ("fill_zeros", "negation"):
                config = settings()
                config["value_completion"] = completion
                with self.assertRaises(ValueError): tag.MctsSearch(game, config, bindings={"external_value": evaluator})
            with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}) as search:
                search.advance()
                game.reset()
                with self.assertRaises(tag.StalePositionError): search.advance()

    def test_terminal_vectors_and_exact_transition_boundaries(self):
        for players in (2, 3, 4):
            evaluator = Constant()
            with tag.create("cantstop", {"seed": 42, "rules": {
                    "column_heights": [1] * 11, "columns_to_win": 1}}, players=players) as game:
                game.apply(0, position=game.position)
                game.apply(int(game.observe(0).offered_action_ids[0]), position=game.position)
                self.assertIn(1, game.observe(0).offered_action_ids)  # Stop banks a winning column.
                with tag.MctsSearch(game, settings(40), bindings={"external_value": evaluator}) as search:
                    with tag.MctsBatch([search]) as batch:
                        report, = batch.run()
                        self.assertFalse(batch.errors)
                stop = next(row for row in report["root_actions"] if row["action_id"] == 1)
                self.assertGreater(stop["visits"], 0)
                self.assertEqual(stop["mean_values"], [1.] + [-1.] * (players - 1))
                self.assertLess(report["counters"]["state_evaluations"], 40)
                for limit in (0, 1, 2, 7):
                    config = settings(100)
                    config["budgets"]["forward_model_calls"] = limit
                    with tag.MctsSearch(game, config, bindings={"external_value": evaluator}) as search:
                        with tag.MctsBatch([search]) as batch:
                            report, = batch.run()
                            self.assertFalse(batch.errors)
                    self.assertEqual(report["counters"]["forward_model_calls"], limit)

    def test_expired_pending_evaluation_adds_no_visits(self):
        evaluator = Constant()
        with tag.create("cantstop", {"seed": 42}, players=3) as game:
            config = settings(100)
            config["budgets"]["time_ms"] = 500
            with tag.MctsSearch(game, config, bindings={"external_value": evaluator}) as search:
                request = search.advance()
                self.assertIsInstance(request, tag.MctsRequest)
                response = evaluator.evaluate([request], max_bytes=search.max_bytes)[0]
                time.sleep(.55)
                report = search.resume(request, response)
                self.assertIn("Time", report["stop_reasons"])
                self.assertEqual(report["counters"]["completed_iterations"], 0)
                self.assertTrue(report["fallback"])
                self.assertFalse(any(row["visits"] for row in report["root_actions"]))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
class CantStopNeuralSearch(unittest.TestCase):
    def evaluator(self, game, *, value=None, **kwargs):
        import torch
        from tag_rust.learning.inputs import ModelContract, ValueSpec
        from tag_rust.learning.network import PolicyValueNetwork
        from tag_rust.learning.snapshot import FrozenModel
        from tag_rust.search.neural_mcts import NeuralMctsEvaluator
        players = game.spec().identity.player_count
        contract = ModelContract.from_spec(game.spec(), payloads=NAMES,
            value=value or ValueSpec(players=tuple(range(players)), gamma=1))
        network = PolicyValueNetwork(contract, (2,))
        with torch.no_grad():
            for parameter in network.parameters(): parameter.zero_()
            network.value.bias[:] = torch.atanh(torch.arange(1, contract.value.size + 1) / 8)
        return NeuralMctsEvaluator(FrozenModel(network), turn_context="own_turn", **kwargs)

    def test_neural_policy_and_full_vectors_match_controlled_predictions(self):
        for players in (2, 3, 4):
            with ExitStack() as stack:
                game = stack.enter_context(tag.create("cantstop", {"seed": 42}, players=players))
                # Start from the last seat's allocation: several legal choices,
                # absolute values and a nonzero acting-player perspective.
                while game.next_player != players - 1:
                    game.apply(int(game.observe(game.next_player).offered_action_ids[0]), position=game.position)
                game.apply(0, position=game.position)
                evaluator = self.evaluator(game)
                reports = []
                for binding in (Constant(), evaluator):
                    search = stack.enter_context(tag.MctsSearch(game, settings(24, policy=True),
                        bindings={"external_value": binding, "external_policy": binding}))
                    with tag.MctsBatch([search]) as batch:
                        report, = batch.run()
                        self.assertFalse(batch.errors)
                    reports.append(report)
                self.assertEqual(reports[0]["action_id"], reports[1]["action_id"])
                self.assertEqual(reports[0]["counters"], reports[1]["counters"])
                for left, right in zip(reports[0]["root_actions"], reports[1]["root_actions"]):
                    self.assertEqual(left["visits"], right["visits"])
                    np.testing.assert_allclose(left["mean_values"], right["mean_values"], atol=1e-6)

    def test_model_contract_rejections_before_search(self):
        from tag_rust.learning.inputs import ValueSpec
        with tag.create("cantstop", {"seed": 42}, players=3) as game:
            for value in (ValueSpec(), ValueSpec(players=(0, 1, 2)),
                          ValueSpec(players=(2, 1, 0), gamma=1)):
                evaluator = self.evaluator(game, value=value)
                with self.assertRaises(ValueError): tag.MctsSearch(game, settings(), bindings={"external_value": evaluator})
            evaluator = self.evaluator(game, supported_observers=(0,))
            with self.assertRaises(ValueError): tag.MctsSearch(game, settings(), bindings={"external_value": evaluator})
            evaluator = self.evaluator(game)
            for parameters, players in (({"seed": 42}, 2), ({"seed": 42, "rules": {"markers": 2}}, 3)):
                with tag.create("cantstop", parameters, players=players) as other:
                    with self.assertRaises(ValueError): tag.MctsSearch(other, settings(), bindings={"external_value": evaluator})


if __name__ == "__main__":
    unittest.main()
