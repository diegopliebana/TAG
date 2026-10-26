"""Observation-derived native/neural MCTS and resumable independent-tree batches."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import tag_rust as tag
from test_loveletter import NAMES, same_view
from test_mcts import comparison, settings

SEARCH_NAMES = tuple(name for name in NAMES if name != "last_private_history")


class Constant:
    identity = "loveletter-constant-v1"
    payloads = SEARCH_NAMES

    def __init__(self):
        self.requests = []
        self.batch_sizes = []

    def validate(self, spec, kind, observers, completion):
        pass

    def evaluate(self, requests, *, max_bytes):
        self.requests.extend(requests)
        self.batch_sizes.append(len(requests))
        return [tag.MctsResponse.for_request(r, np.ones(len(r.action_ids)) if r.kind == "policy"
                else np.arange(1, r.observations[0].identity.player_count + 1) / 8)
                for r in requests]


def run_search(game, configuration, evaluator=None, **kwargs):
    bindings = {} if evaluator is None else {"external_value": evaluator}
    if configuration.get("tree_policy") == "puct":
        bindings["external_policy"] = evaluator
    with tag.MctsSearch(game, configuration, bindings=bindings, **kwargs) as search:
        with tag.MctsBatch([search]) as batch:
            report, = batch.run()
            if batch.errors:
                raise AssertionError(batch.errors)
    return report


def paired_worlds(stack):
    left = stack.enter_context(tag.create("loveletter", {"seed": 0}, players=3))
    initial = left.observe(0)
    for seed in range(1, 257):
        right = stack.enter_context(tag.create("loveletter", {"seed": seed}, players=3))
        if (np.array_equal(initial.payloads["hands"][0], right.observe(0).payloads["hands"][0])
                and any(not np.array_equal(left.observe(s).payloads["hands"][s],
                                           right.observe(s).payloads["hands"][s]) for s in (1, 2))):
            return left, right
        right.close()
    raise AssertionError("paired actual-world seed fixture not found")


class LoveLetterSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = Path(__file__).resolve().parents[1]
        result = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
            str(adapter / "Cargo.toml"), "--example", "loveletter_mcts_reference"],
            cwd=adapter, check=True, capture_output=True, text=True)
        cls.cases = json.loads(result.stdout)

    def test_native_and_external_search_match_rust_at_multiple_seats_and_histories(self):
        phases = set()
        for case in self.cases:
            with self.subTest(players=case["players"], phase=case["phase"], evaluator=case["evaluator"]):
                with tag.create("loveletter", case["parameters"], players=case["players"]) as game:
                    for action in case["moves"]:
                        game.apply(action, position=game.position)
                    before = game.observe(game.next_player)
                    evaluator = Constant()
                    config = case["settings"]
                    if case["evaluator"] == "constant":
                        config["state_evaluator"]["name"] = "external_value"
                        report = run_search(game, config, evaluator)
                    else:
                        report = run_search(game, config)
                    self.assertEqual(comparison(report), case["report"])
                    self.assertLessEqual(report["counters"]["forward_model_calls"], 80)
                    same_view(self, before, game.observe(game.next_player))
                    for request in evaluator.requests:
                        self.assertEqual(request.rows, 1)
                        self.assertEqual(request.observations[0].observer, request.deciding_player)
                        self.assertEqual(request.observations[0].next_player, request.deciding_player)
                    phases.add(case["phase"])
        self.assertIn("previous-round", phases)
        self.assertIn("memory", phases)
        self.assertTrue({"seat-0", "seat-1", "seat-2", "seat-3"}.issubset(phases))

    def test_suspended_batches_retry_without_advancing_game_or_simulation(self):
        for n in (2, 3, 4):
            evaluator = Constant()
            with ExitStack() as stack:
                games = [stack.enter_context(tag.create("loveletter", {"seed": 42}, players=n)) for _ in range(3)]
                untouched = stack.enter_context(games[0].clone())
                searches = [stack.enter_context(tag.MctsSearch(g, settings(20, policy=True),
                    bindings={"external_value": evaluator, "external_policy": evaluator})) for g in games]
                request = searches[0].advance()
                self.assertIs(searches[0].advance(), request)
                reply, = evaluator.evaluate([request], max_bytes=searches[0].max_bytes)
                for bad in (replace(reply, request_id=(999, 999)), replace(reply, values=np.array([np.nan])),
                            replace(reply, action_ids=(999,))):
                    with self.assertRaises(ValueError):
                        searches[0].resume(request, bad)
                    self.assertIs(searches[0].pending, request)
                searches[0].resume(request, reply)
                with tag.MctsBatch(searches[:2]) as batch:
                    reports = batch.run()
                    self.assertFalse(batch.errors)
                with tag.MctsBatch(searches[2:]) as batch:
                    single, = batch.run()
                    self.assertFalse(batch.errors)
                for report in reports:
                    self.assertEqual(comparison(report), comparison(single))
                self.assertIn(2, evaluator.batch_sizes)
                self.assertTrue(all(r.rows == 1 and r.observations[0].observer == r.deciding_player
                                    for r in evaluator.requests))
                # Whole subsequent continuations detect accidental changes to live RNG/decks.
                while untouched.next_player is not None:
                    view = untouched.observe(untouched.next_player, ())
                    action = int(view.offered_action_ids[0])
                    for g in [untouched, *games]:
                        g.apply(action, position=g.position)
                    for g in games:
                        same_view(self, untouched.observe(0), g.observe(0))

    def test_full_vector_contract_limits_cancellation_seeds_and_stale_positions(self):
        evaluator = Constant()
        with tag.create("loveletter", {"seed": 42}, players=3) as game:
            before = game.observe(0)
            for observer in ("all_players", "root_player"):
                with self.assertRaisesRegex(ValueError, "deciding_player"):
                    tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}, value_observers=observer)
            for completion in ("fill_zeros", "negation"):
                config = settings()
                config["value_completion"] = completion
                with self.assertRaises(ValueError):
                    tag.MctsSearch(game, config, bindings={"external_value": evaluator})
            # The compiled boundary enforces the contract too, without the Python facade.
            with self.assertRaisesRegex(ValueError, "deciding_player"):
                game._native.start_mcts(json.dumps(settings()), "all_players", list(SEARCH_NAMES),
                                        "external_value", "external_policy", 16, 2**20)
            with self.assertRaisesRegex(ValueError, "previous-round"):
                game._native.start_mcts(json.dumps(settings()), "deciding_player", list(NAMES),
                                        "external_value", "external_policy", 16, 2**20)
            unsupported = Constant()
            unsupported.payloads = NAMES
            with self.assertRaisesRegex(ValueError, "previous-round"):
                tag.MctsSearch(game, settings(), bindings={"external_value": unsupported})
            with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}) as search:
                request = search.advance()
                for bad in (tag.MctsResponse.for_request(request, [0.], scalar_player=0),
                            tag.MctsResponse.for_request(request, [0., 0.])):
                    with self.assertRaises(ValueError):
                        search.resume(request, bad)
                    self.assertIs(search.pending, request)
                search.cancel()
                with self.assertRaises(RuntimeError):
                    search.resume(request, evaluator.evaluate([request], max_bytes=search.max_bytes)[0])
            config = settings(8)
            config["seed"] = None
            first = run_search(game, config, evaluator)
            config["seed"] = first["settings"]["seed"]
            self.assertEqual(comparison(first), comparison(run_search(game, config, evaluator)))
            for limit in (0, 1, 2, 7):
                config = settings(100)
                config["budgets"]["forward_model_calls"] = limit
                report = run_search(game, config, evaluator)
                self.assertEqual(report["counters"]["forward_model_calls"], limit)
            with self.assertRaises(ValueError):
                with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}, max_bytes=1) as limited:
                    limited.advance()
            same_view(self, before, game.observe(0))
            with tag.MctsSearch(game, settings(), bindings={"external_value": evaluator}) as search:
                request = search.advance()
                request.observations[0].payloads["hands"].fill(99)
                game.reset()
                with self.assertRaises(tag.StalePositionError):
                    search.advance()
            same_view(self, before, game.observe(0))

    def test_terminal_rewards_bypass_external_estimates(self):
        # Find a reachable initial own Princess: playing it loses this two-player game.
        seed = next(seed for seed in range(128) if self._princess(seed))
        with tag.create("loveletter", {"seed": seed, "rules": {"winning_tokens": [1, 1, 1]}}) as game:
            report = run_search(game, settings(32), Constant())
            row = next(r for r in report["root_actions"] if r["action_id"] == 7)
            self.assertGreater(row["visits"], 0)
            self.assertEqual(row["mean_values"], [-1., 1.])

    @staticmethod
    def _princess(seed):
        with tag.create("loveletter", {"seed": seed}) as game:
            return 9 in game.observe(0, ("hands",)).payloads["hands"][0]

    def test_actual_hidden_worlds_cannot_change_requests_or_root_statistics(self):
        with ExitStack() as stack:
            left, right = paired_worlds(stack)
            same_view(self, left.observe(0), right.observe(0))
            a, b = Constant(), Constant()
            first = run_search(left, settings(24, policy=True), a)
            second = run_search(right, settings(24, policy=True), b)
            self.assertEqual(comparison(first), comparison(second))
            self.assertEqual(len(a.requests), len(b.requests))
            for x, y in zip(a.requests, b.requests):
                self.assertEqual((x.kind, x.root_player, x.deciding_player, x.action_ids, x.policy_node),
                                 (y.kind, y.root_player, y.deciding_player, y.action_ids, y.policy_node))
                same_view(self, x.observations[0], y.observations[0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
class LoveLetterNeuralSearch(unittest.TestCase):
    def evaluator(self, game, value=None, payloads=SEARCH_NAMES, **kwargs):
        import torch
        from tag_rust.learning.inputs import ModelContract, ValueSpec
        from tag_rust.learning.network import PolicyValueNetwork
        from tag_rust.learning.snapshot import FrozenModel
        from tag_rust.search.neural_mcts import NeuralMctsEvaluator
        n = game.spec().identity.player_count
        contract = ModelContract.from_spec(game.spec(), payloads=payloads,
                    value=value or ValueSpec(players=tuple(range(n)), gamma=1))
        network = PolicyValueNetwork(contract, (2,))
        with torch.no_grad():
            for p in network.parameters():
                p.zero_()
            network.value.bias[:] = torch.atanh(torch.arange(1, contract.value.size + 1) / 8)
        class Traced(NeuralMctsEvaluator):
            def evaluate(self, requests, *, max_bytes):
                self.batch_sizes.append(len(requests))
                return super().evaluate(requests, max_bytes=max_bytes)
        evaluator = Traced(FrozenModel(network), turn_context="own_turn", **kwargs)
        evaluator.batch_sizes = []
        return evaluator

    def test_neural_policy_vectors_and_independent_batches_match_controlled_values(self):
        for n in (2, 3, 4):
            with ExitStack() as stack:
                games = [stack.enter_context(tag.create("loveletter", {"seed": 42, "first_player": n-1}, players=n)) for _ in range(3)]
                evaluator = self.evaluator(games[0])
                config = settings(24, policy=True)
                config["tree_depth"] = 1
                searches = [stack.enter_context(tag.MctsSearch(g, config,
                    bindings={"external_value": evaluator, "external_policy": evaluator})) for g in games]
                with tag.MctsBatch(searches[:2]) as batch:
                    grouped = batch.run()
                    self.assertFalse(batch.errors)
                with tag.MctsBatch(searches[2:]) as batch:
                    single, = batch.run()
                    self.assertFalse(batch.errors)
                self.assertIn(2, evaluator.batch_sizes)
                for report in [*grouped, single]:
                    self.assertEqual(comparison(report), comparison(single))
                    for row in report["root_actions"]:
                        if row["visits"]:
                            np.testing.assert_allclose(row["mean_values"], np.arange(1, n+1)/8, atol=1e-6)
                plain = run_search(games[0], config, Constant())
                self.assertEqual(plain["action_id"], single["action_id"])
                self.assertEqual(plain["counters"], single["counters"])

    def test_neural_contract_rejections_precede_search(self):
        from tag_rust.learning.inputs import ValueSpec
        with tag.create("loveletter", {"seed": 42}, players=3) as game:
            for value in (ValueSpec(), ValueSpec(players=(0, 1, 2)), ValueSpec(players=(2, 1, 0), gamma=1)):
                evaluator = self.evaluator(game, value=value)
                with self.assertRaises(ValueError):
                    tag.MctsSearch(game, settings(), bindings={"external_value": evaluator})
            evaluator = self.evaluator(game, supported_observers=(0,))
            with self.assertRaises(ValueError):
                tag.MctsSearch(game, settings(), bindings={"external_value": evaluator})
            evaluator = self.evaluator(game)
            unsupported = self.evaluator(game, payloads=NAMES)
            with self.assertRaisesRegex(ValueError, "previous-round"):
                tag.MctsSearch(game, settings(), bindings={"external_value": unsupported})
            for parameters, n in (({"seed": 42}, 2), ({"seed": 42, "rules": {"winning_tokens": [2, 2, 2]}}, 3)):
                with tag.create("loveletter", parameters, players=n) as other:
                    with self.assertRaises(ValueError):
                        tag.MctsSearch(other, settings(), bindings={"external_value": evaluator})

    def test_nonconstant_neural_predictions_remain_independent_of_actual_hidden_cards(self):
        import torch
        from tag_rust.learning.inputs import ModelContract, ValueSpec
        from tag_rust.learning.network import PolicyValueNetwork
        from tag_rust.learning.snapshot import FrozenModel
        from tag_rust.search.neural_mcts import NeuralMctsEvaluator
        with ExitStack() as stack:
            left, right = paired_worlds(stack)
            torch.manual_seed(707)
            contract = ModelContract.from_spec(left.spec(), payloads=SEARCH_NAMES,
                                               value=ValueSpec(players=(0, 1, 2), gamma=1))
            network = PolicyValueNetwork(contract, (4,))
            evaluator = NeuralMctsEvaluator(FrozenModel(network), turn_context="own_turn")
            a = run_search(left, settings(20, policy=True), evaluator)
            b = run_search(right, settings(20, policy=True), evaluator)
            self.assertEqual(comparison(a), comparison(b))

    def test_native_and_neural_agents_finish_complete_multiplayer_games(self):
        for n in (2, 3, 4):
            with tag.create("loveletter", {"seed": 42, "rules": {"winning_tokens": [2, 2, 2]}}, players=n) as game:
                evaluator = self.evaluator(game)
                moves = 0
                while game.next_player is not None:
                    config = settings(6, policy=bool(game.next_player % 2))
                    if game.next_player % 2:
                        report = run_search(game, config, evaluator)
                    else:
                        config["state_evaluator"] = {"name": "game_heuristic"}
                        report = run_search(game, config)
                    view = game.observe(game.next_player)
                    self.assertIn(report["action_id"], view.offered_action_ids)
                    game.apply(report["action_id"], position=view.position)
                    moves += 1
                    self.assertLess(moves, 3000)
                final = game.observe(0)
                self.assertGreater(np.count_nonzero(final.results == 1), 0)
                np.testing.assert_array_equal(final.payloads["terminal_values"], np.where(final.results == 1, 1, -1))


if __name__ == "__main__":
    unittest.main()
