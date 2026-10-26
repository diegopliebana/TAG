"""Checks shared by the two training checkpoint formats.

Each caller still checks its own training schedule and optimizer-state count.
Loading a checkpoint must not change the process-wide Torch random stream.
"""
import torch


def restore_adam_groups(optimizer, saved, message):
    """Require the declared Adam settings before restoring its parameter state."""
    if saved["param_groups"] != optimizer.state_dict()["param_groups"]:
        raise ValueError(message)
    optimizer.load_state_dict(saved)


def validate_adam_tensors(optimizer, updates, *, state_error, tensor_error,
                          step_error, finite_error=None):
    """Check Adam moments against each parameter and its completed update count."""
    for parameter, state in optimizer.state.items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(state_error)
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(finite_error or tensor_error)
            if not isinstance(value, torch.Tensor) or (name != "step" and (
                    value.shape != parameter.shape or value.dtype != parameter.dtype)):
                raise ValueError(tensor_error)
        if state["step"].numel() != 1 or state["step"].item() != updates:
            raise ValueError(step_error)


def restored_torch_rng(saved):
    """Validate a CPU random-generator state and return an independent copy."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(saved)
    return saved.clone()
