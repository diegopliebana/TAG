"""Shared validation preserves outcome perspectives and checkpoint ownership."""
import importlib.util
import unittest
import numpy as np
from tag_rust.learning.outcomes import outcome_counts, terminal_rewards


class OutcomeTests(unittest.TestCase):
    def test_independent_player_values_and_counts(self):
        # Multiple winners and all-loss outcomes are valid independent values.
        np.testing.assert_array_equal(terminal_rewards((1, 1, 3, 2)), [1., 1., -1., 0.])
        np.testing.assert_array_equal(terminal_rewards((3, 3, 3)), [-1., -1., -1.])
        self.assertEqual(terminal_rewards((2,)).dtype, np.float32)
        for codes in ((), (1, 0), (4,)):
            with self.assertRaises(ValueError):
                terminal_rewards(codes)
        first = outcome_counts(truncated=True)
        second = outcome_counts(truncated=True)
        first["wins"] += 1
        self.assertEqual(second["wins"], 0)
        self.assertIn("episodes", first)
        self.assertNotIn("truncated", outcome_counts(counter="games"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class CheckpointTests(unittest.TestCase):
    def test_optimizer_validation_and_rng_ownership(self):
        import copy
        import torch
        from tag_rust.learning.checkpoint_state import (
            restore_adam_groups, validate_adam_tensors, restored_torch_rng)
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.Adam([parameter])
        parameter.sum().backward()
        optimizer.step()
        arguments = dict(state_error="state", tensor_error="tensor", step_error="step")
        validate_adam_tensors(optimizer, 1, **arguments)
        saved = copy.deepcopy(optimizer.state_dict())
        saved["param_groups"][0]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "settings"):
            restore_adam_groups(optimizer, saved, "settings")
        with self.assertRaisesRegex(ValueError, "step"):
            validate_adam_tensors(optimizer, 2, **arguments)
        state = optimizer.state[parameter]
        state["exp_avg"] = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, "tensor"):
            validate_adam_tensors(optimizer, 1, **arguments)
        state["exp_avg"] = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_adam_tensors(optimizer, 1, finite_error="finite", **arguments)
        before = torch.get_rng_state().clone()
        other = torch.Generator().manual_seed(54321).get_state()
        restored = restored_torch_rng(other)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(other, restored))
        self.assertNotEqual(other.data_ptr(), restored.data_ptr())
