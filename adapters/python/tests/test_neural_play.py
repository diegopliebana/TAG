"""Check frozen inference against real pending requests and controlled reference values."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import unittest
import numpy as np
import tag_rust as tag

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.neural import PolicyValueNetwork
    from tag_rust.neural_inputs import ModelContract, ValueSpec, pack_observations
    from tag_rust.neural_snapshot import FrozenModel
    from tag_rust.neural_play import NeuralValueEvaluator, NeuralPolicy


def network(game, *, value=None):
    model = PolicyValueNetwork(ModelContract.from_spec(game.spec(), payloads=("cells",), value=value), (2,))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class FrozenModels(unittest.TestCase):
    def test_digest_fixture_source_isolation_and_contract_identity(self):
        with tag.create("tictactoe") as game:
            model = network(game)
            state = torch.get_rng_state().clone()
            snapshot = FrozenModel(model)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(snapshot.identity, "2c64c710529a2c178f479af620649a12485d052acfa1b7f0de1fd58ef68785f4")
            self.assertEqual(FrozenModel(model).identity, snapshot.identity)
            batch = pack_observations(snapshot.contract, [game.observe(0)])
            original = snapshot.predict(batch)
            with torch.no_grad():
                model.value.bias[:] = 0.7
            np.testing.assert_array_equal(snapshot.predict(batch).values, original.values)
            self.assertNotEqual(FrozenModel(model).identity, snapshot.identity)
            self.assertTrue(model.training)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            original.values[:] = 99
            self.assertEqual(snapshot.predict(batch).values.item(), 0)
            description = snapshot.description
            description["settings"]["hidden_sizes"][:] = [99]
            self.assertEqual(snapshot.description["settings"]["hidden_sizes"], [2])
            model.contract = replace(model.contract, value=ValueSpec(gamma=0.5))
            self.assertNotEqual(FrozenModel(model).identity, snapshot.identity)

    def test_custom_models_require_declared_code_and_independent_state(self):
        with tag.create("tictactoe") as game:
            contract = network(game).contract
            class Custom(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.contract = contract
                    self.weight = torch.nn.Parameter(torch.zeros(1))
                def forward(self, payloads, context, candidates=None):
                    return torch.zeros((len(context), 9)), self.weight.expand(len(context))
            model = Custom()
            with self.assertRaises(ValueError):
                FrozenModel(model)
            a = FrozenModel(model, implementation="test-custom/1")
            b = FrozenModel(model, implementation="test-custom/2")
            self.assertNotEqual(a.identity, b.identity)
            model.register_buffer("scratch", torch.zeros(1), persistent=False)
            with self.assertRaises(ValueError):
                FrozenModel(model, implementation="test-custom/1")
            class Shared(Custom):
                def __deepcopy__(self, memo):
                    return self
            shared = Shared()
            with self.assertRaises(ValueError):
                FrozenModel(shared, implementation="test-custom/1")
            self.assertTrue(shared.training)
            self.assertTrue(shared.weight.requires_grad)
            class Changed(Custom):
                def __deepcopy__(self, memo):
                    copied = Custom()
                    with torch.no_grad(): copied.weight.fill_(1)
                    return copied
            with self.assertRaises(ValueError):
                FrozenModel(Changed(), implementation="test-custom/1")


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class NeuralOsla(unittest.TestCase):
    def test_synthetic_three_player_values_and_candidate_model_value_only_path(self):
        with tag.create("tictactoe") as game:
            base = network(game).contract
            contract = replace(base, identity=replace(base.identity, game="synthetic-three-player", player_count=3),
                               value=ValueSpec(players=(2,0,1)))
            model = PolicyValueNetwork(contract, (2,))
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
                model.value.bias.copy_(torch.tensor([0.1,0.3,0.5]))
            evaluator = NeuralValueEvaluator(FrozenModel(model))
            request = SimpleNamespace(identity=contract.identity, evaluator=evaluator.identity,
                      observers=np.arange(3,dtype=np.int64), action_ids=np.arange(3,dtype=np.int64),
                      payloads={"cells":np.zeros((3,9),dtype=np.float32)},
                      results=np.zeros((3,3),dtype=np.int8), next_players=np.array([1,2,0],dtype=np.int64))
            np.testing.assert_allclose(evaluator.evaluate(request), np.tanh([0.3,0.5,0.1]), atol=1e-6)
            missing = NeuralValueEvaluator(FrozenModel(PolicyValueNetwork(replace(contract, value=ValueSpec(players=(0,1))), (2,))))
            request.evaluator = missing.identity
            with self.assertRaises(ValueError): missing.evaluate(request)
            candidate_contract = replace(base, actions=tag.ActionSpec(0,"candidates"),
                 candidate_tensor=tag.TensorSpec("action_features",np.dtype("float32"),(2,),("feature",),()))
            candidate = PolicyValueNetwork(candidate_contract,(2,))
            with torch.no_grad():
                for p in candidate.parameters(): p.zero_()
            evaluator = NeuralValueEvaluator(FrozenModel(candidate))
            request = SimpleNamespace(identity=base.identity,evaluator=evaluator.identity,actor=0,
                 action_ids=np.arange(3,dtype=np.int64),payloads={"cells":np.zeros((3,9),dtype=np.float32)},
                 results=np.zeros((3,2),dtype=np.int8),next_players=np.ones(3,dtype=np.int64))
            np.testing.assert_array_equal(evaluator.evaluate(request), [0,0,0])

    def test_complete_real_games_and_controlled_chunk_rng_parity(self):
        for name in ("tictactoe", "connect4"):
            traces = []
            for max_rows in (1, 3, 1024):
                with tag.create(name) as game:
                    evaluator = NeuralValueEvaluator(FrozenModel(network(game)))
                    for seat in (0, 1):
                        game.set_controller(seat, evaluator.controller(seed=7+seat, max_rows=max_rows))
                    trace = []
                    while game.next_player is not None:
                        request = game.prepare_evaluation()
                        while isinstance(request, tag.EvaluationRequest):
                            values = evaluator.evaluate(request)
                            self.assertEqual(values.dtype, np.float64)
                            # A zero model is an independent reference: all ongoing
                            # candidates have value 0, terminal values follow result tags.
                            codes = request.results[:, request.actor]
                            expected = np.choose(codes, [0., 1., 0., -1.])
                            np.testing.assert_array_equal(values, expected)
                            request = game.finish_evaluation(request, values)
                        trace.append(request.action_id)
                        game.commit_evaluation(request)
                    traces.append((trace, game.controller_checkpoint(0), game.controller_checkpoint(1)))
            self.assertEqual(traces[0][0], traces[1][0])
            self.assertEqual(traces[0][0], traces[2][0])
            for seat in (1, 2):
                snapshots = [(t[seat].stream, t[seat].word_position) for t in traces]
                self.assertEqual(snapshots, [snapshots[0]]*3)

    def test_grouped_values_metadata_buffers_and_request_failures(self):
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create("tictactoe")) for _ in range(2)]
            evaluator = NeuralValueEvaluator(FrozenModel(network(games[0])))
            for i, game in enumerate(games):
                if i:
                    game.apply(4, position=game.position)
                game.set_controller(game.next_player, evaluator.controller(seed=22+i))
            batch = stack.enter_context(tag.GameBatch(games))
            result = batch.advance()
            group = result.groups[0]
            self.assertEqual(set(group.observers), {0, 1})
            values = evaluator.evaluate(group)
            expected = []
            for segment in group.segments:
                request = SimpleNamespace(identity=group.identity, evaluator=group.evaluator, actor=segment.actor,
                      action_ids=group.action_ids[segment.start:segment.stop],
                      payloads={k:v[segment.start:segment.stop] for k,v in group.payloads.items()},
                      results=group.results[segment.start:segment.stop], next_players=group.next_players[segment.start:segment.stop])
                expected.extend(evaluator.evaluate(request))
            np.testing.assert_allclose(values, expected, atol=1e-6, rtol=0)
            before = [game.position for game in games]
            with self.assertRaises(ValueError):
                evaluator.evaluate(replace(group, evaluator=replace(group.evaluator, version=99)))
            bad = group.payloads["cells"].copy()
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                evaluator.evaluate(replace(group, payloads={"cells":bad}))
            self.assertEqual([game.position for game in games], before)
            values[:] = 99
            self.assertFalse((evaluator.evaluate(group) == 99).any())
            batch.resume(group, evaluator.evaluate(group))
            with self.assertRaises(tag.StalePositionError):
                batch.resume(group, evaluator.evaluate(group))

    def test_all_terminal_rows_skip_model_and_keep_observer_rewards(self):
        with tag.create("tictactoe") as game:
            evaluator = NeuralValueEvaluator(FrozenModel(network(game)))
            request = SimpleNamespace(identity=game.spec().identity, evaluator=evaluator.identity,
                 actor=1, action_ids=np.arange(3, dtype=np.int64), payloads={"cells":np.zeros((3,9), dtype=np.float32)},
                 results=np.array([[3,1], [2,2], [1,3]], dtype=np.int8), next_players=np.full(3,-1,dtype=np.int64))
            with patch.object(evaluator.snapshot, "predict", side_effect=AssertionError("terminal inference")):
                np.testing.assert_array_equal(evaluator.evaluate(request), [1,0,-1])

    def test_vector_values_select_original_observer_and_reward_mismatch_rejects(self):
        with tag.create("tictactoe") as game:
            model = network(game, value=ValueSpec(players=(1,0)))
            with torch.no_grad():
                model.value.bias.copy_(torch.tensor([0.2, 0.6]))
            evaluator = NeuralValueEvaluator(FrozenModel(model))
            for seat in (0, 1):
                if seat:
                    game.reset()
                    game.set_controller(0, None)
                    game.apply(4, position=game.position)
                game.set_controller(seat, evaluator.controller(seed=1))
                request = game.prepare_evaluation()
                expected = float(np.tanh(0.6 if seat == 0 else 0.2))
                np.testing.assert_allclose(evaluator.evaluate(request), expected, atol=1e-6)
                game.cancel_evaluation(request)
            model.contract = replace(model.contract, value=ValueSpec(units="points"))
            with self.assertRaises(ValueError):
                NeuralValueEvaluator(FrozenModel(model))

    def test_changed_snapshot_and_stale_request_leave_commit_checks_in_rust(self):
        with tag.create("tictactoe") as game:
            model = network(game)
            first = NeuralValueEvaluator(FrozenModel(model))
            game.set_controller(0, first.controller(seed=3))
            request = game.prepare_evaluation()
            with torch.no_grad():
                model.value.bias[:] = 0.4
            second = NeuralValueEvaluator(FrozenModel(model))
            with self.assertRaises(ValueError):
                second.evaluate(request)
            values = first.evaluate(request)
            game.reset()
            with self.assertRaises(tag.StalePositionError):
                game.finish_evaluation(request, values)


if __name__ == "__main__":
    unittest.main()
