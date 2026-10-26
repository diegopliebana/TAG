"""Collect groups of training trajectories for imitation or actor-critic learning."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from ..randomness import resolve_seed
from .outcomes import REWARD_LABELS, outcome_counts
import numpy as np
from tag_rust.engine.api import _integer
from tag_rust.episodes.collection import Transition, collect_episodes
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch
from tag_rust.learning.play import NeuralPolicy
from tag_rust.learning.seeds import episode_seed


@dataclass(frozen=True)
class CollectedStep:
    """Store a transition and optional policy statistics from collection.

    The value prediction is the baseline used to calculate the advantage. The
    selected action's log-probability records the behaviour policy. Arrays in
    the transition have independent storage.
    """
    index: int
    transition: Transition
    policy_statistics: dict | None


@dataclass(frozen=True)
class EpisodeRecord:
    """A trajectory with its learner seat, training phase, policy settings and seeds.

    Steps retain their order. Separate seeds identify environment reset, teacher
    action selection and neural policy sampling.
    """
    logical_episode_id: int
    seat: int
    phase: str
    behavior: dict
    reset_seed: int
    teacher_seed: int | None
    policy_seed: int | None
    steps: tuple[CollectedStep, ...]


@dataclass(frozen=True)
class EpisodeGroup:
    """A group of completed or horizon-truncated episodes.

    Outcomes are reported by learner seat and combined. Collection returns these
    records before the training code updates or saves the model.
    """
    episodes: tuple[EpisodeRecord, ...]
    next_episode_id: int
    outcomes: dict
    master_seed: int | None = None


class GroupCollectionError(RuntimeError):
    """Report collection failure and retain episodes that completed successfully.

    An incomplete group cannot be passed to the normal training update.
    """
    def __init__(self, episodes, cause):
        self.partial_episodes = episodes
        self.cause = cause
        super().__init__(f"episode group is incomplete: {cause}")


def _outcomes(episodes):
    def counts(rows):
        tally = outcome_counts(truncated=True)
        tally["episodes"] = len(rows)
        for episode in rows:
            final = episode.steps[-1].transition
            if final.truncated:
                tally["truncated"] += 1
            elif final.terminated:
                tally[REWARD_LABELS[final.reward]] += 1
        return tally
    return dict(combined=counts(episodes), per_seat={seat:counts([e for e in episodes if e.seat == seat]) for seat in (0, 1)})


def collect_episode_group(game, *, episode_count, master_seed=None, first_episode_id=0, phase="actor_critic",
                          snapshot=None, learner=None, opponent=None, opponent_settings=None,
                          parameters=None, payloads=None, include_intermediate=True,
                          max_episode_steps=None, max_bytes=64*1024*1024):
    """Collect one ended/truncated episode per scheduled member, with no weight updates.

    Training seat is logical_episode_id modulo two, unless learner fixes a seat.
    phase='teacher' selects Rust OSLA; actor_critic/evaluation use a frozen snapshot.
    Evaluation has its own phase/ID stream. Continue a training schedule by passing
    the previous group's next_episode_id, including across teacher/actor phases.
    Seed purposes and behavior metadata remain in every returned EpisodeRecord.
    """
    master_seed = resolve_seed(master_seed)
    count = _integer(episode_count, "episode_count")
    first = _integer(first_episode_id, "first_episode_id")
    if not 1 <= count <= 256 or phase not in ("teacher", "actor_critic", "evaluation"):
        raise ValueError("provide 1-256 episodes and teacher/actor_critic/evaluation phase")
    if learner is not None and _integer(learner, "learner") not in (0, 1):
        raise ValueError("current episode games support learner 0 or 1")
    teacher = phase == "teacher"
    if opponent is None:
        opponent = "osla" if teacher else "random"
    if (snapshot is None) != teacher:
        raise ValueError("teacher phase uses native OSLA; other phases require a snapshot")
    if payloads is None:
        if teacher:
            raise ValueError("teacher collection requires explicitly selected payloads")
        payloads = tuple(t.name for t in snapshot.contract.tensors)
    else:
        payloads = tuple(payloads)
        if snapshot is not None and payloads != tuple(t.name for t in snapshot.contract.tensors):
            raise ValueError("collection payload order differs from model")
    ids = [first + i for i in range(count)]
    seats = [logical % 2 if learner is None else learner for logical in ids]
    resets = [episode_seed(master_seed, phase, logical, "reset") for logical in ids]
    teachers = [episode_seed(master_seed, phase, logical, "teacher_learner") if teacher else None for logical in ids]
    policies = [None if teacher else episode_seed(master_seed, phase, logical, "policy_sampling") for logical in ids]
    behavior = (dict(teacher="osla", settings={}, seed_derivation_version=1) if teacher else
                dict(model_identity=snapshot.identity, model=snapshot.description,
                     policy_rng="PCG64", seed_derivation_version=1))
    steps = [[] for _ in ids]
    def records():
        return tuple(EpisodeRecord(logical, seat, phase, deepcopy(behavior), reset, teach, policy, tuple(member_steps))
                     for logical,seat,reset,teach,policy,member_steps in zip(ids,seats,resets,teachers,policies,steps))
    with ExitStack() as stack:
        envs = [stack.enter_context(GameEnv(game, parameters, learner=seat, opponent=opponent,
                  opponent_settings=opponent_settings, payloads=payloads, include_intermediate=include_intermediate,
                  teacher="osla" if teacher else None, max_episode_steps=max_episode_steps, max_bytes=max_bytes)) for seat in seats]
        batch = stack.enter_context(GameEnvBatch(envs, max_bytes=max_bytes))
        policy = None if teacher else NeuralPolicy(snapshot, {i:np.random.Generator(np.random.PCG64(seed)) for i,seed in enumerate(policies)})
        try:
            stream = collect_episodes(batch, seeds=resets, options=[{"teacher_seed":s} if teacher else None for s in teachers],
                                      teacher=teacher, choose_batch=None if teacher else policy.choose_batch)
            for transition in stream:
                member = transition.member
                stats = None if teacher else deepcopy(policy.last_statistics[member])
                steps[member].append(CollectedStep(len(steps[member]), transition, stats))
        except Exception as error:
            raise GroupCollectionError(records(), error) from error
    episodes = records()
    if any(not e.steps or not (e.steps[-1].transition.terminated or e.steps[-1].transition.truncated) for e in episodes):
        raise GroupCollectionError(episodes, RuntimeError("collection ended before every episode"))
    return EpisodeGroup(episodes, first + count, _outcomes(episodes), master_seed)
