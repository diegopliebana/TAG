"""Single-learner Gymnasium environments backed by Rust rules and native opponents.

Import this optional module explicitly. Ordinary tag_rust imports need no Gymnasium.
"""
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .. import _engine
from tag_rust.engine.api import _integer, _payload


class GameEnv(gym.Env):
    """Own one episode. A step includes the learner move and native opponent reply.

    Opponents: first_available, random, osla, osla_python. Only osla_python accepts
    settings: max_rows and max_candidates. reset(seed=...) owns opponent seeding.
    Invalid actions preserve play. Unexpected runtime step failures require reset.
    """
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, game, parameters=None, *, learner=0, opponent="osla_python",
                 opponent_settings=None, payloads=("board",), max_episode_steps=None,
                 max_cells=4096, max_bytes=64*1024*1024, include_intermediate=False, teacher=None):
        super().__init__()
        self._lock = Lock()
        if type(include_intermediate) is not bool or teacher not in (None, "osla"):
            raise ValueError("include_intermediate must be Boolean; teacher must be None or 'osla'")
        self._include_intermediate = include_intermediate
        self._teacher = teacher
        if not isinstance(game, str) or not isinstance(opponent, str):
            raise TypeError("game and opponent must be strings")
        if isinstance(payloads, (str, bytes)):
            raise TypeError("payloads must be a sequence of names")
        payloads = tuple(payloads)
        if any(not isinstance(n, str) for n in payloads):
            raise TypeError("payload names must be strings")
        def settings(value):
            if value is None:
                return {}
            if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
                raise TypeError("settings must be a dictionary with string keys")
            return {k: _integer(v, k) for k, v in value.items()}
        horizon = None if max_episode_steps is None else _integer(max_episode_steps, "max_episode_steps")
        config = dict(game=game, parameters=settings(parameters), learner=_integer(learner,"learner"),
                      opponent=opponent, opponent_settings=settings(opponent_settings),
                      payloads=payloads, max_episode_steps=horizon,
                      include_intermediate=include_intermediate, teacher=teacher is not None,
                      max_cells=_integer(max_cells,"max_cells"), max_bytes=_integer(max_bytes,"max_bytes"))
        self._native = _engine.Episode(config)
        description = self._native.spec()
        self._identity = dict(description["identity"])
        available = {p.name:p for p in map(_payload, description["payloads"])}
        tensors = {}
        for name in payloads:
            spec = available[name].tensor
            if spec is None or spec.dtype != np.dtype("float32") or spec.bounds is None:
                raise ValueError("Gym requires float32 payloads with declared bounds")
            low, high = spec.bounds
            if low > high:
                raise ValueError("invalid declared payload bounds")
            tensors[name] = spaces.Box(low, high, shape=spec.shape, dtype=spec.dtype)
        self.action_space = spaces.Discrete(description["actions"]["size"])
        tensors["action_mask"] = spaces.MultiBinary(int(self.action_space.n))
        self.observation_space = spaces.Dict(tensors)
        self._opponent = opponent
        self._settings = dict(config["opponent_settings"])
        self._episode_index = None
        self._seed_info = None

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("another operation is using this environment")
        try:
            yield
        finally:
            self._lock.release()

    @property
    def include_intermediate(self):
        """Whether the fixed native episode configuration captures after-action views."""
        return self._include_intermediate

    @property
    def teacher(self):
        """The fixed learner teacher identity, or None for external learner actions."""
        return self._teacher

    @property
    def status(self):
        """new, ready, failed or closed; ended episodes reject further step calls."""
        return self._native.status

    def _format(self, value, seed_info):
        observation = dict(value["payloads"])
        observation["action_mask"] = value["action_mask"].astype(np.int8)
        return observation, self._format_info(value, seed_info)

    def _format_info(self, value, seed_info):
        info = dict(identity=dict(value["identity"]), observer=value["observer"],
                    next_player=value["next_player"], progress=value["progress"],
                    results=value["results"], learner_steps=value["learner_steps"],
                    underlying_moves=value["underlying_moves"], moves_this_call=value["moves_this_call"],
                    opponent=self._opponent, opponent_settings=dict(self._settings),
                    opponent_rng=tuple(value["opponent_rng"]), **dict(seed_info))
        info["action"] = value.get("action")
        info["teacher"] = self.teacher
        info["teacher_rng"] = value.get("teacher_rng")
        if self.include_intermediate:
            raw = value.get("intermediate")
            info["intermediate"] = None if raw is None else dict(
                observation={**raw["payloads"], "action_mask": raw["action_mask"].astype(np.int8)},
                info={name:raw[name] for name in ("identity", "observer", "next_player", "progress", "results")})
        return info

    def _teacher_seed(self, options):
        options = {} if options is None else options
        if not isinstance(options, dict) or set(options) - {"teacher_seed"}:
            raise ValueError("reset accepts only teacher_seed options")
        seed = options.get("teacher_seed")
        if (seed is not None) != (self.teacher is not None):
            raise ValueError("supply teacher_seed exactly when a teacher is configured")
        if seed is not None:
            seed = _integer(seed, "teacher_seed")
            if seed >= 2**64:
                raise OverflowError("teacher_seed exceeds uint64")
        return seed

    def reset(self, *, seed=None, options=None):
        """Prepare a fresh episode and commit only after the entire return is built.

        Explicit seeds restart PCG64 and episode numbering. None continues them.
        A failed reset preserves the old episode and seed-generator state.
        A configured teacher requires options={"teacher_seed": uint64}; its stream
        is independent of the existing two-word environment/opponent derivation.
        """
        with self._exclusive():
            teacher_seed = self._teacher_seed(options)
            seed = None if seed is None else _integer(seed,"seed")
            state = self._prepare_seed(seed)
            if teacher_seed is not None:
                state[3]["teacher_seed"] = teacher_seed
            candidate = self._native.prepare_reset(state[3]["opponent_seed"], teacher_seed)
            result = self._format(candidate.output, state[3])
            old = self._seed_state()
            try:
                self._restore_seed(state)
                self._native.commit_reset(candidate)
            except BaseException:
                self._restore_seed(old)
                raise
            return result

    def _seed_state(self):
        """Capture references to the current committed wrapper seed state."""
        return self._np_random, self._np_random_seed, self._episode_index, self._seed_info

    def _restore_seed(self, state):
        self._np_random, self._np_random_seed, self._episode_index, self._seed_info = state

    def _prepare_seed(self, seed):
        """Stage the ordinary reset seed operation without changing committed state."""
        old = self._seed_state()
        try:
            if seed is None and self._np_random is not None:
                self._np_random = deepcopy(self._np_random)
            super().reset(seed=seed)
            if not isinstance(self.np_random.bit_generator, np.random.PCG64):
                raise ValueError("episode seed derivation requires PCG64")
            environment_seed, opponent_seed = map(int, self.np_random.bit_generator.random_raw(2))
            index = 0 if seed is not None or self._episode_index is None else self._episode_index + 1
            seed_info = dict(environment_seed=environment_seed, opponent_seed=opponent_seed,
                             seed_derivation_version=1, seed_generator="PCG64", numpy_version=np.__version__,
                             episode_index=index)
            return self._np_random, self._np_random_seed, index, seed_info
        finally:
            self._restore_seed(old)

    def step(self, action):
        """Apply one legal learner action and run the native opponent.

        Runtime/output errors after execution mark the episode failed. No partial
        transition is returned; reset before continuing. Invalid input is retryable.
        """
        with self._exclusive():
            action = _integer(action,"action")
            value = self._native.step(action)
            return self._step_return(value)

    def step_teacher(self):
        """Choose with the independent Rust OSLA teacher and execute one learner step."""
        with self._exclusive():
            value = self._native.step_teacher()
            return self._step_return(value)

    def _step_return(self, value):
        """Format an already executed step, requiring reset if output fails."""
        try:
            observation,info = self._format(value,self._seed_info)
            return observation,float(value["reward"]),bool(value["terminated"]),bool(value["truncated"]),info
        except BaseException:
            self._native.fail()
            raise

    def close(self):
        """Release the owned Rust episode. Repeated close calls are harmless."""
        with self._exclusive():
            self._native.close()
