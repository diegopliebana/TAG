"""Portable evaluation models and completed-group PyTorch training checkpoints.

Evaluation JSON contains explicit float32 tensor shapes and flattened values.
Training checkpoints contain only tensors and primitive containers, loaded with
PyTorch's restricted weights-only loader. Neither format serializes model code.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .checkpoint_state import restore_adam_groups, validate_adam_tensors, restored_torch_rng
from ..engine.api import EncodingIdentity, ActionSpec, TensorSpec
from .inputs import ModelContract, ValueSpec
from .network import PolicyValueNetwork
from .snapshot import FrozenModel, _record
from .training import TrainingSession, TrainingSettings, runtime_identity

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _atomic_write(path, writer):
    """Replace the destination only after a complete file was successfully written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _readable(path):
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the 64 MiB file limit")
    return path


def model_document(model, *, provenance=None):
    """Describe the reference architecture and independent numeric weights for export."""
    if type(model) is not PolicyValueNetwork:
        raise ValueError("portable artifacts support the reference PolicyValueNetwork")
    if model.contract.candidate_tensor is not None:
        raise ValueError("portable reference artifacts currently require fixed action IDs")
    snapshot = FrozenModel(model)
    return dict(format="tag-neural-model", version=1, implementation="tag_rust.PolicyValueNetwork/2",
                contract=_record(model.contract), hidden_sizes=list(model.hidden_sizes),
                weights={name:dict(dtype="float32", shape=list(tensor.shape),
                                   values=tensor.detach().reshape(-1).tolist())
                         for name, tensor in model.state_dict().items()},
                model_identity=snapshot.identity, provenance={} if provenance is None else provenance)


def _contract(document):
    """Reconstruct checked descriptions without importing arbitrary Python classes."""
    def tensor(t):
        return TensorSpec(t["name"], np.dtype(t["dtype"]), tuple(t["shape"]), tuple(t["axes"]),
                          tuple(t["components"]), None if t["bounds"] is None else tuple(t["bounds"]))
    return ModelContract(EncodingIdentity(**document["identity"]), ActionSpec(**document["actions"]),
                         tuple(tensor(t) for t in document["tensors"]), ValueSpec(**document["value"]),
                         None if document["candidate_tensor"] is None else tensor(document["candidate_tensor"]),
                         document["feature_version"])


def load_document(document, *, expected_contract=None):
    """Validate all identities, shapes and finite weights before returning a CPU model."""
    if not isinstance(document, dict):
        raise ValueError("evaluation model must be a JSON object")
    if (document.get("format"), document.get("version"), document.get("implementation")) != (
            "tag-neural-model", 1, "tag_rust.PolicyValueNetwork/2"):
        raise ValueError("unsupported evaluation model format/version/implementation")
    try:
        contract = _contract(document["contract"])
        if expected_contract is not None and contract != expected_contract:
            raise ValueError("saved model contract differs from requested game/input/action/value meanings")
        widths = document["hidden_sizes"]
        if not isinstance(widths, list) or not widths or any(type(n) is not int or n <= 0 for n in widths):
            raise ValueError("invalid hidden layer sizes")
        # Bound allocation before constructing layers from file-supplied dimensions.
        dimensions = [contract.input_size, *widths]
        parameters = sum((a+1)*b for a,b in zip(dimensions, dimensions[1:]))
        parameters += (dimensions[-1]+1) * (contract.actions.size + contract.value.size)
        if contract.candidate_tensor is not None:
            raise ValueError("portable reference artifacts currently require fixed action IDs")
        if parameters > MAX_ARTIFACT_BYTES // 4:
            raise ValueError("model parameter allocation exceeds artifact limit")
        with torch.random.fork_rng(devices=[]):
            model = PolicyValueNetwork(contract, widths)
        expected = model.state_dict()
        if set(document["weights"]) != set(expected):
            raise ValueError("weight names differ from the declared architecture")
        state = {}
        for name, template in expected.items():
            weight = document["weights"][name]
            if weight["dtype"] != "float32" or weight["shape"] != list(template.shape):
                raise ValueError(f"weight type/shape mismatch: {name}")
            array = np.asarray(weight["values"], dtype=np.float64)
            if array.shape != (template.numel(),) or not np.isfinite(array).all():
                raise ValueError(f"invalid finite flat weights: {name}")
            converted = array.astype(np.float32)
            if not np.isfinite(converted).all() or not np.array_equal(converted.astype(np.float64), array):
                raise ValueError(f"weights must represent exact float32 numbers: {name}")
            state[name] = torch.from_numpy(converted.reshape(template.shape).copy())
        model.load_state_dict(state, strict=True)
        if FrozenModel(model).identity != document["model_identity"]:
            raise ValueError("model weight/description digest mismatch")
        model.eval()
        return model
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed neural model document") from error


def save_model(path, model, *, provenance=None):
    """Atomically write an evaluation model as versioned, browser-readable JSON."""
    data = json.dumps(model_document(model, provenance=provenance), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError("model JSON exceeds artifact limit")
    _atomic_write(path, lambda stream: stream.write(data))


def load_model(path, *, expected_contract=None):
    """Load a portable evaluation model; leave caller RNG state unchanged."""
    try:
        with _readable(path).open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "expected a UTF-8 evaluation-model JSON file; training checkpoints (.pt) "
            "require load_checkpoint or neural_train.py --resume"
        ) from error
    return load_document(document, expected_contract=expected_contract)


def save_checkpoint(path, session):
    """Save only after a successful complete collection/update group."""
    session.check_boundary()
    checkpoint = dict(format="tag-neural-training", version=1,
                      model=model_document(session.model, provenance=session.provenance()),
                      settings=asdict(session.settings), completed=dict(session.completed),
                      next_episode_id=session.next_episode_id, updates=session.updates,
                      history=session.history, optimizer=session.optimizer.state_dict(),
                      torch_rng=session.torch_rng.clone(), runtime=session.runtime)
    def write(stream):
        torch.save(checkpoint, stream)
        if stream.tell() > MAX_ARTIFACT_BYTES:
            raise ValueError("training checkpoint exceeds artifact limit")
    _atomic_write(path, write)


def _load_checkpoint(path):
    """Resume fresh episodes at the next group under matching recorded CPU conditions.

    No live environments, pending requests or partly consumed group is restored.
    Runtime mismatch fails explicitly rather than claiming exact continuation.
    """
    data = torch.load(_readable(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("training checkpoint must contain a dictionary")
    if (data.get("format"), data.get("version")) != ("tag-neural-training", 1):
        raise ValueError("unsupported training checkpoint format/version")
    if data["settings"]["master_seed"] is None:
        raise ValueError("training checkpoint requires its resolved master seed")
    settings = TrainingSettings(**data["settings"])
    # Check before construction changes global thread/determinism settings.
    current = runtime_identity()
    current.update(threads=settings.threads, deterministic=True)
    if current != data["runtime"]:
        raise ValueError("checkpoint CPU/software runtime differs from this execution")
    session = TrainingSession(settings)
    model = load_document(data["model"], expected_contract=session.model.contract)
    completed = data["completed"]
    if (set(completed) != {"teacher", "actor_critic"}
            or any(type(n) is not int or n < 0 for n in completed.values())
            or completed["teacher"] > settings.teacher_episodes or completed["actor_critic"] > settings.actor_episodes
            or (completed["actor_critic"] and completed["teacher"] != settings.teacher_episodes)
            or type(data["next_episode_id"]) is not int or data["next_episode_id"] != sum(completed.values())
            or type(data["updates"]) is not int or data["updates"] != len(data["history"])):
        raise ValueError("invalid completed-group counters")
    for phase, limit in (("teacher", settings.teacher_episodes), ("actor_critic", settings.actor_episodes)):
        if completed[phase] != limit and completed[phase] % settings.group_size:
            raise ValueError("checkpoint is not at a completed group boundary")
    seen = dict(teacher=0, actor_critic=0)
    next_id, previous_identity = 0, None
    for index, report in enumerate(data["history"]):
        phase = "teacher" if seen["teacher"] < settings.teacher_episodes else "actor_critic"
        limit = settings.teacher_episodes if phase == "teacher" else settings.actor_episodes
        count = min(settings.group_size, limit - seen[phase])
        if (count <= 0 or report["phase"] != phase or report["episodes"] != count
                or report["first_episode_id"] != next_id or report["update"] != index + 1
                or previous_identity is not None and report["before_identity"] != previous_identity):
            raise ValueError("checkpoint history differs from its completed group schedule")
        previous_identity = report["model_identity"]
        seen[phase] += count
        next_id += count
    if seen != completed or (previous_identity is not None and previous_identity != data["model"]["model_identity"]):
        raise ValueError("checkpoint model/counters differ from its update history")
    session.model.load_state_dict(model.state_dict())
    restore_adam_groups(session.optimizer, data["optimizer"],
                        "optimizer settings differ from the declared Adam configuration")
    validate_adam_tensors(session.optimizer, data["updates"],
        state_error="unexpected Adam state",
        tensor_error="optimizer tensor type/shape mismatch",
        finite_error="non-finite optimizer state",
        step_error="Adam update count differs from completed groups")
    if data["updates"] and len(session.optimizer.state) != len(list(session.model.parameters())):
        raise ValueError("missing Adam parameter state")
    session.torch_rng = restored_torch_rng(data["torch_rng"])
    session.completed = completed.copy()
    session.next_episode_id = data["next_episode_id"]
    session.updates = data["updates"]
    session.history = data["history"]
    return session


def load_checkpoint(path):
    """Load a checked completed-group checkpoint under its recorded CPU conditions."""
    try:
        return _load_checkpoint(path)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed training checkpoint") from error
