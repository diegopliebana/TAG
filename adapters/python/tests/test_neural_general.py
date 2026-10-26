"""Synthetic interface fixtures for varied tabletop data; these are not game engines."""
from dataclasses import replace
from types import SimpleNamespace, MappingProxyType
import importlib.util
import unittest
import numpy as np
import tag_rust as tag
from tag_rust.neural_inputs import ModelContract, ValueSpec, CandidateActions, pack_observations

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.neural import PolicyValueNetwork, infer


def tensor(name, shape, dtype="float32", bounds=None):
    return tag.TensorSpec(name, np.dtype(dtype), shape, tuple(f"axis{i}" for i in range(len(shape))), (), bounds)


def fixture(*, dynamic=False, value=None):
    identity = tag.EncodingIdentity("synthetic-tabletop", 7, 3, player_count=3,
                                   configuration={"deck": "example-v1", "rounds": "5"})
    tensors = (tensor("public_tokens", (3,), "int64", (0, 100)),
               tensor("own_hand", (4, 2)), tensor("hand_valid", (4,), "bool"))
    spec = tag.GameSpec(identity, tuple(tag.PayloadSpec("features", t.name, t, MappingProxyType({})) for t in tensors),
                        (), tag.ActionSpec(0, "candidates") if dynamic else tag.ActionSpec(5, "indexed"))
    contract = ModelContract.from_spec(spec, payloads=tuple(t.name for t in tensors), value=value,
                                      candidate_tensor=tensor("action_features", (2,)) if dynamic else None)
    views = []
    for observer in range(3):
        # Each observer's own permitted hand is encoded separately. There is no
        # full-state argument from which the packer could obtain other hands.
        views.append(SimpleNamespace(identity=identity, observer=observer, next_player=observer,
                     payloads={"public_tokens": np.array([3, 7, 2], dtype=np.int64),
                               "own_hand": np.full((4, 2), observer + 0.25, dtype=np.float32),
                               "hand_valid": np.array([True, True, False, False])},
                     action_mask=np.array([False, True, True, False, True])))
    return contract, views


class GeneralInputs(unittest.TestCase):
    def test_boardless_identity_multiple_types_and_private_views(self):
        contract, views = fixture()
        self.assertEqual(contract.identity.width, 0)
        self.assertEqual(dict(contract.identity.configuration)["deck"], "example-v1")
        batch = pack_observations(contract, views)
        self.assertEqual(batch.context.shape, (3, 7))
        np.testing.assert_array_equal(batch.context[:, :3], np.eye(3))
        np.testing.assert_array_equal(batch.context[:, 3:6], np.eye(3))
        np.testing.assert_array_equal(batch.inputs[:, -1], [1, 1, 1])
        self.assertEqual(batch.payloads["own_hand"].shape, (3, 4, 2))
        self.assertEqual(batch.payloads["public_tokens"].dtype, np.int64)
        views[0].payloads["own_hand"][:] = -100
        self.assertEqual(batch.payloads["own_hand"][0, 0, 0], 0.25)
        batch.payloads["own_hand"][0] = 99
        self.assertEqual(views[1].payloads["own_hand"][0, 0], 1.25)
        with self.assertRaises(MemoryError):
            pack_observations(contract, views, max_bytes=1)

    def test_descriptions_are_part_of_compatibility(self):
        contract, views = fixture()
        with self.assertRaises(ValueError):
            pack_observations(contract, [SimpleNamespace(**{**vars(views[0]), "identity":
                replace(contract.identity, configuration={"deck": "changed"})})])
        altered = replace(contract.tensors[0], axes=("different-order",))
        self.assertNotEqual(contract, replace(contract, tensors=(altered, *contract.tensors[1:])))
        with self.assertRaises(ValueError):
            replace(contract, value=ValueSpec(players=(0, 3)))
        with self.assertRaises(ValueError):
            replace(contract, tensors=(tensor("ragged", (-1, 2)),))
        with self.assertRaises(ValueError):
            replace(contract, value=ValueSpec(players=(1, 1)))
        with self.assertRaises(ValueError):
            replace(contract, identity=replace(contract.identity, player_count=0))

    def test_padded_variable_candidates_and_ids(self):
        contract, views = fixture(dynamic=True)
        choices = [CandidateActions(np.ones((n, 2), dtype=np.float32), np.arange(n, dtype=np.int64) + base)
                   for n, base in ((2, 40), (1, 90), (0, 0))]
        batch = pack_observations(contract, views, candidates=choices)
        np.testing.assert_array_equal(batch.legal_mask, [[True, True], [True, False], [False, False]])
        np.testing.assert_array_equal(batch.action_ids, [[40, 41], [90, -1], [-1, -1]])
        choices[0].features[:] = 99
        self.assertEqual(batch.candidate_features[0, 0, 0], 1)
        choices[0].action_ids[:] = 40
        with self.assertRaises(ValueError):
            pack_observations(contract, views, candidates=choices)

    def test_real_combined_board_inputs(self):
        for game_name in ("tictactoe", "connect4"):
            with tag.create(game_name) as game:
                contract = ModelContract.from_spec(game.spec(), payloads=("board", "cells"))
                batch = pack_observations(contract, [game.observe(0)])
                self.assertEqual(tuple(batch.payloads), ("board", "cells"))
                self.assertEqual(batch.inputs.shape[1], 4 * contract.identity.width * contract.identity.height + 5)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class GeneralInference(unittest.TestCase):
    def test_custom_inference_disables_dropout_and_restores_flags_after_failure(self):
        contract, views = fixture()
        class WithDropout(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.contract = contract
                self.dropout = torch.nn.Dropout(0.75)
                self.fail = False
            def forward(self, payloads, context, candidates=None):
                if self.fail:
                    raise ValueError("controlled model failure")
                return torch.zeros((len(context), 5)), self.dropout(torch.ones(len(context)))
        model = WithDropout()
        batch = pack_observations(contract, views)
        before = torch.get_rng_state().clone()
        np.testing.assert_array_equal(infer(model, batch).values, [1, 1, 1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(model.training)
        self.assertTrue(model.dropout.training)
        model.dropout.eval()
        model.fail = True
        with self.assertRaises(ValueError):
            infer(model, batch)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    def test_multiplayer_values_have_explicit_order_and_no_zero_sum_conversion(self):
        contract, views = fixture(value=ValueSpec(players=(2, 0, 1), bounds=None, units="future-points", gamma=0.8, discount_clock="round"))
        model = PolicyValueNetwork(contract, (4,))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.value.bias.copy_(torch.tensor([5., 20., 8.]))
        output = infer(model, pack_observations(contract, views))
        self.assertEqual(output.value_players, (2, 0, 1))
        np.testing.assert_array_equal(output.values, [[5, 20, 8]] * 3)
        self.assertTrue(np.isin(output.choose(), (1, 2, 4)).all())

    def test_candidate_scores_reference_order_padding_and_empty_batches(self):
        contract, views = fixture(dynamic=True)
        model = PolicyValueNetwork(contract, (2,))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # Ignore state features; each candidate score depends on its own pair.
            model.policy[0].weight[:, -2:] = 0.25
            model.policy[2].weight[:] = 0.5
        choices = [CandidateActions(np.array([[0, 0], [1, 2]], dtype=np.float32), np.array([900, 30], dtype=np.int64)),
                   CandidateActions(np.array([[2, 2]], dtype=np.float32), np.array([70], dtype=np.int64)),
                   CandidateActions(np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64))]
        batch = pack_observations(contract, views, candidates=choices)
        result = infer(model, batch)
        np.testing.assert_allclose(result.logits[0], np.tanh([0, 0.75]), atol=1e-7)
        self.assertEqual(result.choose().tolist(), [30, 70, -1])
        self.assertEqual(result.probabilities[1, 1], 0)
        permuted = [CandidateActions(choices[0].features[::-1], choices[0].action_ids[::-1])]
        other = infer(model, pack_observations(contract, views[:1], candidates=permuted))
        self.assertEqual(other.choose().tolist(), [30])
        np.testing.assert_allclose(other.values, result.values[:1], atol=1e-7)
        empty = infer(model, pack_observations(contract, views, candidates=[choices[2]] * 3))
        self.assertEqual(empty.probabilities.shape, (3, 0))
        self.assertEqual(empty.choose().tolist(), [-1, -1, -1])

    def test_integer_precision_and_external_structured_model(self):
        contract, views = fixture()
        wide = replace(contract, tensors=(replace(contract.tensors[0], bounds=None), *contract.tensors[1:]))
        views[0].payloads["public_tokens"][0] = 2**60 + 1
        batch = pack_observations(wide, views)
        with self.assertRaises(ValueError):
            infer(PolicyValueNetwork(wide), batch)
        class StructuredModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.contract = wide
            def forward(self, payloads, context, candidates=None):
                self.asserted_dtype = payloads["public_tokens"].dtype
                # A structured model receives the declared integer tensor intact.
                assert payloads["public_tokens"][0, 0].item() == 2**60 + 1
                return torch.zeros((len(context), 5)), torch.zeros(len(context))
        structured = StructuredModel()
        infer(structured, batch)
        self.assertEqual(structured.asserted_dtype, torch.int64)

    def test_bounded_values_gradients_and_bad_vector_shape(self):
        contract, views = fixture(value=ValueSpec(players=(1,), bounds=(0, 100), units="points"))
        model = PolicyValueNetwork(contract, (4,))
        batch = pack_observations(contract, views)
        output = infer(model, batch)
        self.assertEqual(output.values.shape, (3, 1))
        self.assertTrue(((output.values >= 0) & (output.values <= 100)).all())
        logits, values = model({k: torch.from_numpy(v.copy()) for k, v in batch.payloads.items()}, torch.from_numpy(batch.context))
        (logits.sum() + values.sum()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        class Bad(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.contract = contract
            def forward(self, payloads, context, candidates=None):
                return torch.zeros((3, 5)), torch.zeros(3)
        with self.assertRaises(ValueError):
            infer(Bad(), batch)


if __name__ == "__main__":
    unittest.main()
