"""Check model inputs against real Rust observations and independent arithmetic."""
from dataclasses import replace
import importlib.util
import subprocess
import sys
import unittest
import numpy as np
import tag_rust as tag
from tag_rust.neural_inputs import ModelContract, NeuralBatch, pack_observations

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.neural import PolicyValueNetwork, infer


class NeuralInputs(unittest.TestCase):
    def test_optional_import(self):
        subprocess.run([sys.executable, "-c", "import tag_rust, sys; assert 'torch' not in sys.modules"], check=True)

    def test_shapes_seats_turns_and_ownership(self):
        for name, size, actions in (("tictactoe", 9, 9), ("connect4", 42, 7)):
            with self.subTest(game=name), tag.create(name) as game:
                contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
                before = game.observe(0)
                game.apply(4 if name == "tictactoe" else 3, position=before.position)
                views = [game.observe(0), game.observe(1)]
                packed = pack_observations(contract, views)
                self.assertEqual(packed.inputs.shape, (2, size + 5))
                self.assertEqual(packed.legal_mask.shape, (2, actions))
                np.testing.assert_array_equal(packed.context, [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1]])
                cell = 4 if name == "tictactoe" else 23
                np.testing.assert_array_equal(packed.inputs[:, cell], [1, -1])
                views[0].payloads["cells"][:] = 0
                self.assertEqual(packed.inputs[0, cell], 1)
                self.assertTrue(packed.inputs.flags.c_contiguous)
                self.assertFalse(np.shares_memory(packed.legal_mask, views[1].action_mask))

    def test_changed_settings_and_bad_inputs(self):
        with tag.create("tictactoe", {"grid_size": 4}) as game:
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            obs = game.observe(0)
            self.assertEqual(pack_observations(contract, [obs]).inputs.shape, (1, 21))
            cases = [replace(obs, identity=replace(obs.identity, rules_version=99)),
                     replace(obs, observer=2),
                     replace(obs, action_mask=np.ones(1, dtype=bool)),
                     replace(obs, payloads={"cells": np.full(16, np.nan, dtype=np.float32)})]
            for invalid in cases:
                with self.assertRaises(ValueError):
                    pack_observations(contract, [invalid])
            with self.assertRaises(TypeError):
                pack_observations(contract, [replace(obs, payloads={"cells": np.zeros(16)})])
            with self.assertRaises(ValueError):
                pack_observations(contract, [])
            bad = replace(game.spec(), actions=replace(game.spec().actions, size=0))
            with self.assertRaises(ValueError):
                ModelContract.from_spec(bad, payloads=("cells",))

    def test_terminal_has_no_policy_choices(self):
        with tag.create("tictactoe") as game:
            for action in (0, 3, 1, 4, 2):
                game.apply(action, position=game.position)
            batch = pack_observations(ModelContract.from_spec(game.spec(), payloads=("cells",)),
                                      [game.observe(0), game.observe(1)])
            self.assertFalse(batch.legal_mask.any())
            np.testing.assert_array_equal(batch.inputs[:, -1], [0, 0])


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class NeuralInference(unittest.TestCase):
    def test_reference_arithmetic_and_batch_equivalence(self):
        for name in ("tictactoe", "connect4"):
            with self.subTest(game=name), tag.create(name) as game:
                contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
                model = PolicyValueNetwork(contract, (2,))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(0.1)
                    model.policy.bias.copy_(torch.arange(contract.actions.size) / 10)
                observations = [game.observe(0), game.observe(1)]
                batch = pack_observations(contract, observations)
                output = infer(model, batch)
                # All hidden weights are 0.1; each hidden unit sees the same sum.
                hidden = np.tanh(0.1 * batch.inputs.sum(axis=1) + 0.1)
                expected_logits = 0.2 * hidden[:, None] + np.arange(contract.actions.size)[None, :] / 10
                np.testing.assert_allclose(output.logits, expected_logits, atol=2e-7)
                np.testing.assert_allclose(output.values, np.tanh(0.2 * hidden + 0.1), atol=2e-7)
                for row, obs in enumerate(observations):
                    one = infer(model, pack_observations(contract, [obs]))
                    self.assertEqual(one.values.shape, (1,))
                    np.testing.assert_allclose(one.values, output.values[row:row+1], atol=1e-7)
                self.assertEqual(output.choose().tolist(), [contract.actions.size-1, -1])
                self.assertEqual(output.probabilities[1].sum(), 0)
                self.assertAlmostEqual(float(output.probabilities[0].sum()), 1, places=6)
                reference = np.exp(expected_logits[0] - expected_logits[0].max())
                reference /= reference.sum()
                np.testing.assert_allclose(output.probabilities[0], reference, atol=1e-7)
                saved = output.values.copy()
                output.values[:] = 99
                np.testing.assert_array_equal(infer(model, batch).values, saved)

    def test_masked_choices_and_seeded_sampling(self):
        with tag.create("tictactoe") as game:
            game.apply(0, position=game.position)
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            model = PolicyValueNetwork(contract)
            batch = pack_observations(contract, [game.observe(1)])
            output = infer(model, batch)
            self.assertEqual(output.probabilities[0, 0], 0)
            a, b = np.random.default_rng(42), np.random.default_rng(42)
            first = [output.choose(a).item() for _ in range(100)]
            self.assertEqual(first, [output.choose(b).item() for _ in range(100)])
            self.assertNotIn(0, first)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_changed_board_and_terminal_inference(self):
        with tag.create("tictactoe", {"grid_size": 4}) as game:
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            model = PolicyValueNetwork(contract)
            output = infer(model, pack_observations(contract, [game.observe(0)]))
            self.assertEqual(output.logits.shape, (1, 16))
            self.assertEqual(model.features[0].in_features, 21)
        with tag.create("tictactoe") as game:
            for action in (0, 3, 1, 4, 2):
                game.apply(action, position=game.position)
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            output = infer(PolicyValueNetwork(contract), pack_observations(
                contract, [game.observe(0), game.observe(1)]))
            self.assertEqual(output.choose().tolist(), [-1, -1])
            self.assertFalse(output.probabilities.any())

    def test_invalid_contract_buffers_outputs_and_devices(self):
        with tag.create("tictactoe") as game:
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            batch = pack_observations(contract, [game.observe(0)])
            model = PolicyValueNetwork(contract)
            for field, value in (("feature_version", 99),
                                 ("actions", replace(contract.actions, mapping="other")),
                                 ("identity", replace(contract.identity, encoding_version=2))):
                with self.assertRaises(ValueError):
                    infer(model, replace(batch, contract=replace(contract, **{field: value})))
            for invalid in (replace(batch, payloads={"cells": np.full((1, 9), np.nan, dtype=np.float32)}),
                            replace(batch, context=batch.context[:, :-1]),
                            replace(batch, legal_mask=batch.legal_mask[:, :-1])):
                with self.assertRaises(ValueError):
                    infer(model, invalid)
            with self.assertRaises(ValueError):
                infer(model.double(), batch)
            model.float()
            with torch.no_grad():
                model.policy.bias.fill_(float("nan"))
            with self.assertRaises(ValueError):
                infer(model, batch)

    def test_malformed_reference_network_outputs(self):
        with tag.create("tictactoe") as game:
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
            batch = pack_observations(contract, [game.observe(0)])
            class Broken(PolicyValueNetwork):
                def forward(self, payloads, context, candidates=None):
                    return torch.zeros((len(context), 8)), torch.zeros(len(context))
            with self.assertRaises(ValueError):
                infer(Broken(contract), batch)


if __name__ == "__main__":
    unittest.main()
