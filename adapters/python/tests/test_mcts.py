"""Check external MCTS responses, batching, perspectives and frozen neural inference."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import unittest
import numpy as np
import tag_rust as tag


def settings(iterations=16, *, policy=False):
    value = {"schema_version":1,"seed":42,"rollout_length":0,
             "budgets":{"iterations":iterations},"state_evaluator":{"name":"external_value"}}
    if policy:
        value.update(tree_policy="puct",puct_preferences={"source":"probabilities","evaluator":{"name":"external_policy"}})
    return value


class ConstantEvaluator:
    payloads = ("cells",)
    def __init__(self, *, scalar=False):
        self.identity = "constant-scalar" if scalar else "constant-vector"
        self.scalar = scalar
        self.calls = []

    def validate(self, spec, kind, observers, completion):
        pass

    def evaluate(self, requests, *, max_bytes):
        self.calls.append((len(requests),sum(request.rows for request in requests)))
        replies = []
        for request in requests:
            if request.kind == "policy":
                replies.append(tag.MctsResponse.for_request(request, np.ones(len(request.action_ids))))
            elif self.scalar:
                replies.append(tag.MctsResponse.for_request(request, [.25], scalar_player=request.observations[0].observer))
            else:
                replies.append(tag.MctsResponse.for_request(request, [.25,.75]))
        return replies


def comparison(report):
    return {name:report[name] for name in ("action_id","root_actions","counters","nodes","edges","stop_reasons")}


class ExternalMcts(unittest.TestCase):
    def test_configuration_uses_the_native_parser_before_binding_setup(self):
        with tag.create("tictactoe") as game:
            for configuration in ({}, {"schema_version":1,"state_evaluator":None},
                                  {"schema_version":1,"budgets":{"iterations":None}}, "[]"):
                with self.assertRaises(ValueError):
                    tag.MctsSearch(game,configuration)

    def test_policy_probabilities_and_action_order_are_checked_before_resume(self):
        evaluator = ConstantEvaluator()
        with tag.create("tictactoe") as game:
            with tag.MctsSearch(game,settings(12,policy=True),bindings={"external_value":evaluator,"external_policy":evaluator}) as search:
                while True:
                    request = search.advance()
                    if request is None:
                        continue
                    if request.kind == "policy":
                        break
                    search.resume(request,evaluator.evaluate([request],max_bytes=search.max_bytes)[0])
                for values in (np.zeros(len(request.action_ids)), -np.ones(len(request.action_ids)), np.ones(1)):
                    with self.assertRaises(ValueError):
                        search.resume(request,tag.MctsResponse.for_request(request,values))
                    self.assertIs(search.pending,request)
                response = evaluator.evaluate([request],max_bytes=search.max_bytes)[0]
                with self.assertRaises(ValueError):
                    search.resume(request,replace(response,action_ids=tuple(reversed(response.action_ids))))
                with tag.MctsBatch([search]) as batch:
                    report, = batch.run()
                self.assertEqual(report["counters"]["completed_iterations"],12)

    def test_terminal_values_bypass_external_predictions_and_automatic_seed_is_exposed(self):
        evaluator = ConstantEvaluator()
        with tag.create("tictactoe") as game:
            for action in (0,3,1,4):
                game.apply(action,position=game.position)
            config = settings(30)
            config["seed"] = None
            with tag.MctsSearch(game,config,bindings={"external_value":evaluator}) as search:
                self.assertIsInstance(search.settings["seed"],int)
                with tag.MctsBatch([search]) as batch:
                    report, = batch.run()
                winning = next(row for row in report["root_actions"] if row["action_id"]==2)
                self.assertGreater(winning["visits"],0)
                self.assertEqual(winning["mean_values"],[1.,-1.])
                self.assertLess(report["counters"]["state_evaluations"],30)

    def test_grouping_and_single_search_have_identical_statistics_on_both_games(self):
        for game_name in ("tictactoe","connect4"):
            evaluator = ConstantEvaluator()
            with ExitStack() as stack:
                games = [stack.enter_context(tag.create(game_name)) for _ in range(3)]
                searches = [stack.enter_context(tag.MctsSearch(game, settings(policy=True),
                    bindings={"external_value":evaluator,"external_policy":evaluator})) for game in games]
                snapshots = [game.observe(0).payloads["cells"].copy() for game in games]
                with tag.MctsBatch(searches[:2]) as batch:
                    grouped = batch.run()
                    self.assertFalse(batch.errors)
                with tag.MctsBatch(searches[2:]) as single:
                    isolated = single.run()[0]
                for report in grouped:
                    self.assertEqual(comparison(report), comparison(isolated))
                self.assertTrue(any(count > 1 for count,_ in evaluator.calls))
                for game, before in zip(games,snapshots):
                    np.testing.assert_array_equal(game.observe(0).payloads["cells"],before)

    def test_bad_group_reply_is_atomic_and_retry_then_duplicate_is_checked(self):
        evaluator = ConstantEvaluator()
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create("tictactoe")) for _ in range(2)]
            searches = [stack.enter_context(tag.MctsSearch(g,settings(),bindings={"external_value":evaluator})) for g in games]
            batch = stack.enter_context(tag.MctsBatch(searches))
            group, = batch.advance()
            requests = [request for _,request in group.members]
            replies = evaluator.evaluate(requests,max_bytes=batch.max_bytes)
            bad = replace(replies[1], values=np.array([np.nan,0.]))
            with self.assertRaises(ValueError): batch.resume(group,[replies[0],bad])
            self.assertIs(searches[0].pending,requests[0])
            for bad in [replace(replies[0],request_id=(999,999)),
                        replace(replies[0],evaluator_identity="other"),
                        replace(replies[0],action_ids=(1,)),
                        replace(replies[0],values=np.array([1.,2.],dtype=np.float32))]:
                with self.assertRaises(ValueError): searches[0].resume(requests[0],bad)
            batch.resume(group,replies)
            with self.assertRaises(ValueError): batch.resume(group,replies)
            results = batch.run()
            self.assertTrue(all(report["counters"]["completed_iterations"]==16 for report in results))

    def test_scalar_completion_perspective_and_request_ownership(self):
        evaluator = ConstantEvaluator(scalar=True)
        for policy, expected in (("fill_zeros",[.25,0.]),("negation",[.25,-.25])):
            with tag.create("tictactoe") as game:
                config = settings(4)
                config["value_completion"] = policy
                with tag.MctsSearch(game,config,bindings={"external_value":evaluator},value_observers="root_player") as search:
                    request = search.advance()
                    wrong = tag.MctsResponse.for_request(request,[.25],scalar_player=1)
                    with self.assertRaises(ValueError): search.resume(request,wrong)
                    with tag.MctsBatch([search]) as batch:
                        report, = batch.run()
                    for row in report["root_actions"]:
                        if row["visits"]:
                            self.assertEqual(row["mean_values"],expected)

    def test_limits_incompatible_encodings_and_stale_root(self):
        evaluator = ConstantEvaluator()
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create(name)) for name in ("tictactoe","connect4")]
            searches = [stack.enter_context(tag.MctsSearch(g,settings(),bindings={"external_value":evaluator})) for g in games]
            with tag.MctsBatch(searches,max_group_rows=2) as batch:
                self.assertEqual(len(batch.advance()),2)
            search = stack.enter_context(tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator}))
            with tag.MctsBatch([search],max_group_rows=1) as small:
                with self.assertRaises(MemoryError): small.advance()
            search = stack.enter_context(tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator}))
            search.advance()
            games[0].reset()
            with self.assertRaises(tag.StalePositionError): search.advance()
            with tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator},max_rows=1) as small:
                with self.assertRaisesRegex(ValueError,"max_rows"): small.advance()
            with tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator},max_bytes=1) as small:
                with self.assertRaises(ValueError): small.advance()
            search = stack.enter_context(tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator}))
            with tag.MctsBatch([search],max_bytes=1) as small:
                with self.assertRaises(MemoryError): small.advance()
            search = stack.enter_context(tag.MctsSearch(games[0],settings(),bindings={"external_value":evaluator}))
            before = games[0].observe(0).payloads["cells"].copy()
            request = search.advance()
            search.cancel()
            request.observations[0].payloads["cells"][:] = 99
            np.testing.assert_array_equal(games[0].observe(0).payloads["cells"],before)

    def test_native_path_zero_budget_and_successful_peer_survive_runtime_failure(self):
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create("tictactoe")) for _ in range(2)]
            searches = [stack.enter_context(tag.MctsSearch(g,{"schema_version":1,"seed":42,"budgets":{"iterations":0}})) for g in games]
            games[0].reset()
            with tag.MctsBatch(searches) as batch:
                reports = batch.run()
                self.assertIsNone(reports[0])
                self.assertIn(0,batch.errors)
                self.assertTrue(reports[1]["fallback"])
                self.assertEqual(reports[1]["counters"]["forward_model_calls"],0)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH,"optional PyTorch not installed")
class NeuralMcts(unittest.TestCase):
    def evaluator(self,game,*,vector=False,**kwargs):
        import torch
        from tag_rust.learning.inputs import ModelContract,ValueSpec
        from tag_rust.learning.network import PolicyValueNetwork
        from tag_rust.learning.snapshot import FrozenModel
        from tag_rust.search.neural_mcts import NeuralMctsEvaluator
        contract = ModelContract.from_spec(game.spec(),payloads=("cells",),
                                         value=ValueSpec(players=(1,0)) if vector else ValueSpec())
        network = PolicyValueNetwork(contract,(2,))
        with torch.no_grad():
            for parameter in network.parameters(): parameter.zero_()
            if vector: network.value.bias[:] = torch.atanh(torch.tensor([.75,.25]))
        return NeuralMctsEvaluator(FrozenModel(network),**kwargs)

    def test_frozen_full_vector_order_matches_mock_non_zero_sum_values(self):
        for name in ("tictactoe","connect4"):
            with tag.create(name) as game:
                evaluator = self.evaluator(game,vector=True)
                with tag.MctsSearch(game,settings(4),bindings={"external_value":evaluator},value_observers="deciding_player") as search:
                    with tag.MctsBatch([search]) as batch:
                        report, = batch.run()
                    for row in report["root_actions"]:
                        if row["visits"]:
                            np.testing.assert_allclose(row["mean_values"],[.25,.75],atol=1e-6)

    def test_scalar_all_player_assembly_policy_and_contract_rejections(self):
        with ExitStack() as stack:
            game = stack.enter_context(tag.create("tictactoe"))
            evaluator = self.evaluator(game)
            search = stack.enter_context(tag.MctsSearch(game,settings(policy=True),
                bindings={"external_value":evaluator,"external_policy":evaluator}))
            with tag.MctsBatch([search]) as batch:
                report, = batch.run()
                self.assertFalse(batch.errors)
            self.assertGreater(report["counters"]["action_evaluations"],0)
            for row in report["root_actions"]:
                np.testing.assert_array_equal(row["mean_values"],[0.,0.])
            for options in ({"turn_context":"own_turn"},{"supported_observers":(0,)}):
                restricted = self.evaluator(game,**options)
                with self.assertRaises(ValueError): tag.MctsSearch(game,settings(),bindings={"external_value":restricted})
            with self.assertRaises(ValueError):
                tag.MctsSearch(game,settings(),bindings={"external_value":evaluator},value_observers="root_player")
            other = stack.enter_context(tag.create("connect4"))
            with self.assertRaises(ValueError): tag.MctsSearch(other,settings(),bindings={"external_value":evaluator})


if __name__ == "__main__":
    unittest.main()
