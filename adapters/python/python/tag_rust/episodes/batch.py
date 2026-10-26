"""Bulk learner transitions with explicit resets and independent native episodes."""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from threading import Lock
import numpy as np
from .. import _engine
from tag_rust.engine.api import _integer
from tag_rust.episodes.gym import GameEnv


@dataclass(frozen=True)
class EpisodeBatchResult:
    """Full-size arrays. Only valid rows are new usable observations/transitions.

    Invalid or inactive rows have zero numeric data and info=None. Check errors
    and statuses separately. These zeros never represent an automatic reset.
    Intermediate arrays/infos have their own validity mask; all three fields are
    None when capture is disabled throughout the batch. Reset rows are invalid.
    """
    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid: np.ndarray
    infos: tuple[dict | None, ...]
    errors: tuple[str | None, ...]
    statuses: tuple[str, ...]
    intermediate_observations: dict[str, np.ndarray] | None = None
    intermediate_infos: tuple[dict | None, ...] | None = None
    intermediate_valid: np.ndarray | None = None


class GameEnvBatch:
    """Retain 1-256 compatible original GameEnv objects and run them in Rust.

    Compatible members have identical encoding/action layouts and payload order.
    Learner seats, opponents, seeds and horizons can differ. Closing releases
    its references; the original environments remain open.
    """
    def __init__(self, environments, *, max_bytes=64*1024*1024):
        envs = tuple(environments)
        if not 1 <= len(envs) <= 256 or any(not isinstance(e, GameEnv) for e in envs):
            raise ValueError("provide 1 to 256 GameEnv objects")
        if len({id(e) for e in envs}) != len(envs):
            raise ValueError("duplicate environment")
        self._lock = Lock()
        self._envs = envs
        self._closed = False
        with self._exclusive():
            self._native = _engine.EpisodeBatch([e._native for e in envs], _integer(max_bytes, "max_bytes"))
        self.single_action_space = envs[0].action_space
        self.single_observation_space = envs[0].observation_space

    def __len__(self):
        return len(self._envs)

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("another operation is using this batch")
        try:
            if self._closed:
                raise RuntimeError("episode batch is closed")
            with ExitStack() as stack:
                for env in sorted(self._envs, key=id):
                    stack.enter_context(env._exclusive())
                yield
        finally:
            self._lock.release()

    @staticmethod
    def _sequence(value, label):
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{label} must be a sequence")
        return list(value)

    def _indices(self, indices):
        values = list(range(len(self))) if indices is None else [
            _integer(i, "member index") for i in self._sequence(indices, "indices")]
        if len(set(values)) != len(values) or any(i >= len(self) for i in values):
            raise ValueError("indices must be distinct batch member indices")
        return values

    @staticmethod
    def _invalidate(value, index, error, status):
        for array in value["observations"].values():
            array[index] = 0
        for name in ("rewards", "terminated", "truncated", "valid"):
            value[name][index] = 0
        value["infos"][index] = None
        value["errors"][index] = f"{type(error).__name__}: {error}"
        value["statuses"][index] = status
        if value.get("intermediate_valid") is not None:
            value["intermediate_valid"][index] = False
            value["intermediate_infos"][index] = None
            for array in value["intermediate_observations"].values():
                array[index] = 0

    def _format(self, value, seed_states=None, *, stepping=False):
        for index in np.flatnonzero(value["valid"]):
            index = int(index)
            env = self._envs[index]
            seed_info = env._seed_info if seed_states is None else seed_states[index][3]
            try:
                value["infos"][index] = env._format_info(value["infos"][index], seed_info)
                # Bulk intermediate views have dedicated stacked fields.
                value["infos"][index].pop("intermediate", None)
            except Exception as error:
                if stepping:
                    self._native.fail([index])
                self._invalidate(value, index, error, env.status)
        return EpisodeBatchResult(
            value["observations"], value["rewards"], value["terminated"], value["truncated"],
            value["valid"], tuple(value["infos"]), tuple(value["errors"]), tuple(value["statuses"]),
            value.get("intermediate_observations"),
            None if value.get("intermediate_infos") is None else tuple(value["intermediate_infos"]),
            value.get("intermediate_valid"))

    def step(self, actions, *, active=None):
        """Validate all active actions, then execute once in Rust without rollback.

        actions has one entry per member. None means inactive when active is
        omitted. With an explicit Boolean active mask, inactive entries are ignored.
        Runtime/output failures invalidate only their row and require its reset.
        """
        with self._exclusive():
            actions = self._sequence(actions, "actions")
            if len(actions) != len(self):
                raise ValueError("actions must have one entry per member")
            if active is None:
                mask = [a is not None for a in actions]
            else:
                mask = self._sequence(active, "active")
                if len(mask) != len(self) or any(not isinstance(a, (bool, np.bool_)) for a in mask):
                    raise ValueError("active must be a Boolean mask with one entry per member")
            ids = [_integer(a, "action") if enabled else None for a, enabled in zip(actions, mask)]
            value = self._native.step(ids)
            return self._step_return(value, mask)

    def step_teacher(self, *, active=None):
        """Choose configured Rust teacher actions and step selected members in bulk."""
        with self._exclusive():
            mask = [True]*len(self) if active is None else self._sequence(active, "active")
            if len(mask) != len(self) or any(not isinstance(a, (bool, np.bool_)) for a in mask):
                raise ValueError("active must be a Boolean mask with one entry per member")
            value = self._native.step_teacher([bool(v) for v in mask])
            return self._step_return(value, mask)

    def _step_return(self, value, mask):
        """Format one executed bulk step and mark missing outputs failed."""
        try:
            return self._format(value, stepping=True)
        except BaseException:
            self._native.fail([i for i, enabled in enumerate(mask) if enabled])
            raise

    def reset(self, *, indices=None, seeds=None, options=None):
        """Reset selected indices, with one optional explicit seed per selected index.

        All inputs are checked first. Fresh candidates and seed states are committed
        only for successfully formatted rows; failed rows retain their old episode.
        """
        with self._exclusive():
            indices = self._indices(indices)
            options = [None]*len(indices) if options is None else self._sequence(options, "options")
            if len(options) != len(indices):
                raise ValueError("provide one options dictionary per selected index")
            teacher_seeds = [self._envs[i]._teacher_seed(option) for i, option in zip(indices, options)]
            if seeds is None:
                seeds = [None]*len(indices)
            else:
                seeds = self._sequence(seeds, "seeds")
                if len(seeds) != len(indices):
                    raise ValueError("provide one seed per selected index")
                seeds = [None if s is None else _integer(s, "seed") for s in seeds]
            states, seed_errors = {}, {}
            for i, seed, teacher_seed in zip(indices, seeds, teacher_seeds):
                try:
                    states[i] = self._envs[i]._prepare_seed(seed)
                    if teacher_seed is not None:
                        states[i][3]["teacher_seed"] = teacher_seed
                except Exception as error:
                    seed_errors[i] = f"{type(error).__name__}: {error}"
            candidate = self._native.prepare_resets(indices, [
                states[i][3]["opponent_seed"] if i in states else None for i in indices], teacher_seeds)
            value = candidate.output
            for i, error in seed_errors.items():
                value["errors"][i] = error
            result = self._format(value, states)
            accepted = [int(i) for i in np.flatnonzero(result.valid)]
            old = {i: self._envs[i]._seed_state() for i in accepted}
            try:
                for i in accepted:
                    self._envs[i]._restore_seed(states[i])
                self._native.commit_resets(candidate, accepted)
            except BaseException:
                for i in accepted:
                    self._envs[i]._restore_seed(old[i])
                raise
            return result

    @property
    def statuses(self):
        with self._exclusive():
            return tuple(self._native.statuses())

    def close(self):
        """Release this coordinator, leaving retained original environments open."""
        if self._closed:
            return
        with self._exclusive():
            self._native.close()
            self._envs = ()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
