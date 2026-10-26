"""Optional model-independent CPU inference and a small fully connected implementation."""
from dataclasses import dataclass
from contextlib import contextmanager
from math import prod
import numpy as np
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ModuleNotFoundError("Optional PyTorch is required; see docs/reference/learning/neural-inference.md") from error
from torch import nn
from tag_rust.learning.inputs import ModelContract, NeuralBatch, flatten_inputs, validate_batch


class PolicyValueNetwork(nn.Module):
    """Flatten named numeric inputs and public context into shared learned features.

    contract fixes input/action/value meanings. hidden_sizes configures the fully
    connected tanh layers. Fixed policies predict permanent IDs; candidate policies
    score each supplied feature vector alongside its observation's shared features.
    Other architectures can implement forward(payloads, context, candidates) with
    the same contract and output shapes, retaining named/structured inputs.
    """
    def __init__(self, contract: ModelContract, hidden_sizes=(64, 64)):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        if not hidden_sizes or any(type(n) is not int or n <= 0 for n in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        self.contract, self.hidden_sizes = contract, hidden_sizes
        layers, previous = [], contract.input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width, device="cpu", dtype=torch.float32), nn.Tanh()))
            previous = width
        self.features = nn.Sequential(*layers)
        if contract.candidate_tensor is None:
            self.policy = nn.Linear(previous, contract.actions.size, device="cpu", dtype=torch.float32)
        else:
            self.policy = nn.Sequential(
                nn.Linear(previous + prod(contract.candidate_tensor.shape), previous, device="cpu", dtype=torch.float32),
                nn.Tanh(), nn.Linear(previous, 1, device="cpu", dtype=torch.float32))
        self.value = nn.Linear(previous, contract.value.size, device="cpu", dtype=torch.float32)

    def forward(self, payloads, context, candidates=None):
        """Return policy scores [B,A/C] and values [B] or [B,V], with gradients."""
        inputs = torch.cat([*(payloads[t.name].float().reshape(len(context), -1)
                              for t in self.contract.tensors), context], dim=1)
        features = self.features(inputs)
        if self.contract.candidate_tensor is None:
            logits = self.policy(features)
        else:
            count = candidates.shape[1]
            actions = candidates.float().reshape(len(context), count, prod(self.contract.candidate_tensor.shape))
            shared = features[:, None, :].expand(-1, count, -1)
            logits = self.policy(torch.cat((shared, actions), dim=2)).squeeze(-1)
        values = self.value(features)
        bounds = self.contract.value.bounds
        if bounds is not None:
            lower, upper = bounds
            values = lower + (upper - lower) * (torch.tanh(values) + 1) / 2
        if self.contract.value.players is None:
            values = values.squeeze(-1)
        return logits, values


@dataclass(frozen=True)
class InferenceResult:
    """Own predictions and action IDs; value_players=None means each row's observer."""
    logits: np.ndarray
    values: np.ndarray
    probabilities: np.ndarray
    policy_rows: np.ndarray
    action_ids: np.ndarray
    value_players: tuple[int, ...] | None

    def choose(self, rng=None):
        """Choose permanent or decision-local IDs; return -1 for inapplicable rows."""
        actions = np.full(len(self.values), -1, dtype=np.int64)
        for row in np.flatnonzero(self.policy_rows):
            p = self.probabilities[row]
            index = np.argmax(p) if rng is None else rng.choice(len(p), p=p.astype(np.float64) / p.sum(dtype=np.float64))
            actions[row] = self.action_ids[row, index]
        return actions


def _tensor(array):
    return torch.from_numpy(np.array(array, order="C", copy=True))


@contextmanager
def _evaluation_mode(model):
    """Disable training-time layers during inference and restore each module's flag."""
    flags = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        yield
    finally:
        for module, training in flags:
            module.training = training


def infer(model, batch: NeuralBatch):
    """Run a compatible CPU model on independently copied named inputs without gradients.

    The model declares its input/output specification in `contract` and implements
    forward(payloads, context, candidates). Observation arrays retain their
    specified number types and shapes when passed to that method.
    Terminal values remain predictions; acting code must use declared outcomes.
    Keep weights fixed while associated external requests remain pending.
    """
    if batch.contract != model.contract:
        raise ValueError("model and batch contracts differ")
    validate_batch(batch)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("inference requires float32 CPU parameters")
    if any(buffer.device.type != "cpu" for buffer in model.buffers()):
        raise ValueError("inference requires CPU buffers")
    if isinstance(model, PolicyValueNetwork):
        # This model explicitly narrows numeric inputs to float32. Validate that
        # conversion before torch sees them. Structured models own preprocessing.
        flatten_inputs(batch.contract, batch.payloads, batch.context)
        if batch.candidate_features is not None:
            raw = batch.candidate_features
            with np.errstate(over="ignore", invalid="ignore"):
                converted = raw.astype(np.float32)
            if not np.isfinite(converted).all() or (raw.dtype.kind in "iu" and not np.array_equal(converted.astype(object), raw.astype(object))):
                raise ValueError("candidate features cannot be represented safely in float32")
    with torch.inference_mode(), _evaluation_mode(model):
        logits, values = model({k: _tensor(v) for k, v in batch.payloads.items()}, _tensor(batch.context),
                               None if batch.candidate_features is None else _tensor(batch.candidate_features))
        expected = (len(batch.context),) if batch.contract.value.players is None else (len(batch.context), batch.contract.value.size)
        if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("model must return policy and value tensors")
        if (logits.shape != batch.legal_mask.shape or values.shape != expected
                or logits.dtype != torch.float32 or values.dtype != torch.float32
                or logits.device.type != "cpu" or values.device.type != "cpu"
                or not torch.isfinite(logits).all() or not torch.isfinite(values).all()):
            raise ValueError("invalid policy/value outputs")
        bounds = batch.contract.value.bounds
        if bounds is not None and ((values < bounds[0]).any() or (values > bounds[1]).any()):
            raise ValueError("values exceed the declared range")
        applicable = batch.legal_mask.any(axis=1)
        probabilities = torch.zeros_like(logits)
        if applicable.any():
            selected = logits[_tensor(applicable)].masked_fill(~_tensor(batch.legal_mask[applicable]), -torch.inf)
            probabilities[_tensor(applicable)] = torch.softmax(selected, dim=1)
        return InferenceResult(logits.numpy().copy(), values.numpy().copy(), probabilities.numpy().copy(),
                               applicable.copy(), batch.action_ids.copy(), batch.contract.value.players)
