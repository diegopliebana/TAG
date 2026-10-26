"""Keep model parameters fixed during evaluation and trajectory collection.

The snapshot identifies the model implementation, input/output specification and
tensor state so replies can be checked against the model used for the request.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import json
import sys
from threading import Lock
import numpy as np
import torch
from tag_rust.learning.network import PolicyValueNetwork, infer


def _record(value):
    if is_dataclass(value):
        return {f.name: _record(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, (tuple, list)):
        return [_record(item) for item in value]
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _frame(digest, data):
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)


class FrozenModel:
    """Own one copied CPU model; changing the source model leaves this snapshot intact.

    Reference networks identify their implementation/settings automatically. Custom
    models require implementation and settings describing all forward behavior;
    they must be stateless in eval mode and keep tensor state in state_dict.
    Unsupported extra state, nonpersistent buffers and shared-copy storage reject.
    The implementation string is a caller declaration, not a hash of Python code.
    """
    def __init__(self, model, *, implementation=None, settings=None):
        if sys.byteorder != "little":
            raise ValueError("snapshot digest version 1 requires a little-endian CPU")
        if type(model) is PolicyValueNetwork:
            if implementation is not None or settings is not None:
                raise ValueError("reference model identity is automatic")
            implementation = "tag_rust.PolicyValueNetwork/2"
            settings = {"hidden_sizes": list(model.hidden_sizes)}
        elif not isinstance(implementation, str) or not implementation:
            raise ValueError("custom models require a declared implementation identity")
        settings = {} if settings is None else settings
        # Round-trip a plain description before copying user model state.
        self._description = json.loads(_json(dict(version=1, implementation=implementation,
                                       settings=settings, contract=_record(model.contract))))
        self._model = deepcopy(model)
        self._contract = deepcopy(model.contract)
        if self._model is model or self._model.contract != self._contract:
            raise ValueError("model copy did not preserve an independent compatible instance")
        source_modules = {id(module) for module in model.modules()}
        if any(id(module) in source_modules for module in self._model.modules()):
            raise ValueError("model copy shares source modules")
        for parameter in self._model.parameters():
            if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
                raise ValueError("snapshot requires float32 CPU parameters")
        state = self._model.state_dict()
        source = model.state_dict()
        if set(state) != set(source):
            raise ValueError("model copy changed tensor state names")
        if any(name not in state for name, _ in self._model.named_buffers()):
            raise ValueError("snapshot does not support nonpersistent buffers")
        digest = hashlib.sha256()
        _frame(digest, _json(self._description))
        for name, tensor in sorted(state.items()):
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or tensor.is_quantized):
                raise ValueError("snapshot state must contain dense, nonquantized CPU tensors")
            original = source[name]
            if (not isinstance(original, torch.Tensor) or original.device != tensor.device
                    or original.dtype != tensor.dtype or original.shape != tensor.shape
                    or not torch.equal(original, tensor)):
                raise ValueError("model copy changed tensor state")
            if tensor.numel() and tensor.untyped_storage().data_ptr() == source[name].untyped_storage().data_ptr():
                raise ValueError("model copy shares source tensor storage")
            if not torch.isfinite(tensor).all():
                raise ValueError("snapshot state contains non-finite entries")
            _frame(digest, _json(dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape))))
            _frame(digest, tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self._identity = digest.hexdigest()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._lock = Lock()

    @property
    def contract(self):
        return self._contract

    @property
    def identity(self):
        """Version-1 SHA-256 identity covering declared code/settings, contract and weights."""
        return self._identity

    @property
    def description(self):
        return deepcopy(self._description)

    def predict(self, batch):
        """Evaluate independent inputs; return no mutable reference to snapshot weights."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("snapshot is already in use")
        try:
            return infer(self._model, batch)
        finally:
            self._lock.release()
