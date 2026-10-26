"""Learn from complete teacher or neural-policy episodes on the CPU.

Each update uses a group of trajectories collected with fixed policy parameters.
Compute returns and advantages first, then use Adam to update the shared
policy/value network.
"""
from dataclasses import asdict, dataclass, replace
import math
import platform
import sys
import numpy as np
import torch
from ..engine.api import create, registered_games
from ..randomness import resolve_seed
from .collection import collect_episode_group
from .inputs import ModelContract
from .network import PolicyValueNetwork
from .play import pack_episode_rows
from .seeds import episode_seed
from .snapshot import FrozenModel

OPPONENTS = ("first_available", "random", "osla", "osla_python")


@dataclass(frozen=True)
class TrainingSettings:
    """Configure imitation-learning and actor-critic episode counts, random seeds,
    network architecture, learning rate and loss coefficients."""
    game: str = "tictactoe"
    teacher_episodes: int = 256
    actor_episodes: int = 1024
    group_size: int = 16
    master_seed: int | None = None
    learner: int | None = None
    teacher_opponent: str = "osla"
    actor_opponent: str = "random"
    max_episode_steps: int | None = None
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = .001
    decision_value_weight: float = .5
    intermediate_value_weight: float = .5
    entropy_weight: float = .01
    threads: int = 1

    def __post_init__(self):
        if self.master_seed is not None:
            resolve_seed(self.master_seed)
        for name in ("teacher_episodes", "actor_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.teacher_episodes + self.actor_episodes == 0:
            raise ValueError("select at least one training episode")
        if type(self.group_size) is not int or not 1 <= self.group_size <= 256:
            raise ValueError("group_size must be 1-256")
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("threads must be positive")
        if self.learner is not None and (type(self.learner) is not int or self.learner not in (0, 1)):
            raise ValueError("learner must be None, 0 or 1")
        if self.max_episode_steps is not None and (type(self.max_episode_steps) is not int or self.max_episode_steps < 1):
            raise ValueError("max_episode_steps must be positive")
        if self.game not in registered_games():
            raise ValueError(f"unknown game {self.game!r}; registered games: {', '.join(registered_games())}")
        if any(o not in OPPONENTS for o in (self.teacher_opponent, self.actor_opponent)):
            raise ValueError("select registered opponents")
        sizes = tuple(self.hidden_sizes)
        if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        object.__setattr__(self, "hidden_sizes", sizes)
        for name in ("learning_rate", "decision_value_weight", "intermediate_value_weight", "entropy_weight"):
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def runtime_identity():
    """Describe the tested execution conditions required for exact CPU continuation."""
    return dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
                platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                adapter_version="0.1.0", training_format=1, seed_derivation_version=1)


def discounted_returns(rewards, *, gamma=.99, continuation=0.):
    """Compute discounted returns backwards using G_t = r_t + gamma * G_(t+1).

    Start with zero at game termination. At a training horizon, bootstrap from
    the final observation's value predicted by the policy used for collection.

    For rewards [0,0,1], gamma=.99 gives [.9801,.99,1]. No additional discount
    is introduced for the intermediate observation within a learner step.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError("provide nonempty finite scalar rewards")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not math.isfinite(continuation):
        raise ValueError("invalid discount or continuation")
    returns = np.empty(len(rewards), dtype=np.float32)
    value = float(continuation)
    for t in range(len(rewards)-1, -1, -1):
        value = float(rewards[t]) + gamma * value
        returns[t] = value
    if not np.isfinite(returns).all():
        raise ValueError("returns exceed float32 range")
    return returns


def _forward(model, batch):
    return model({name: torch.from_numpy(value.copy()) for name, value in batch.payloads.items()},
                 torch.from_numpy(batch.context.copy()))


@dataclass(frozen=True)
class LearningTargets:
    """Returns, advantages and loss weights, ordered like the training observations.

    These arrays are fixed targets for one update and have independent storage."""
    batch: object
    actions: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    actor_weights: np.ndarray
    intermediate_batch: object | None
    intermediate_returns: np.ndarray
    episode_count: int


def make_targets(group, snapshot):
    """Compute return targets and advantages for the observing player before updating.

    Bootstrap a truncated trajectory from the fixed model's final value estimate.

    Intermediate terminal positions are omitted: OSLA already uses their exact
    results. Other intermediate positions receive the same target as their step.
    """
    if not group.episodes:
        raise ValueError("a training group must contain episodes")
    contract = snapshot.contract
    if contract.value.players is not None or contract.candidate_tensor is not None:
        raise ValueError("this training workflow requires observer values and fixed action IDs")
    observations, infos, actions, targets, weights = [], [], [], [], []
    mids, mid_infos, mid_returns = [], [], []
    for episode in group.episodes:
        if episode.phase not in ("teacher", "actor_critic"):
            raise ValueError("only teacher/actor-critic groups are training data")
        if episode.phase == "actor_critic" and episode.behavior.get("model_identity") != snapshot.identity:
            raise ValueError("actor-critic data came from different collection weights")
        if not episode.steps:
            raise ValueError("cannot learn from an empty episode")
        transitions = [s.transition for s in episode.steps]
        last = transitions[-1]
        if not (last.terminated or last.truncated) or any(t.terminated or t.truncated for t in transitions[:-1]):
            raise ValueError("episodes must end exactly once at their final transition")
        if any(t.info["observer"] != episode.seat or t.next_info["observer"] != episode.seat for t in transitions):
            raise ValueError("episode value perspective changed")
        continuation = 0.
        if not last.terminated:
            final = pack_episode_rows(contract, [last.next_observation], [last.next_info])
            continuation = float(snapshot.predict(final).values[0])
        returns = discounted_returns([t.reward for t in transitions], gamma=contract.value.gamma, continuation=continuation)
        for index, (transition, target) in enumerate(zip(transitions, returns)):
            observations.append(transition.observation)
            infos.append(transition.info)
            actions.append(transition.action)
            targets.append(target)
            weights.append(contract.value.gamma ** index)
            mid = transition.intermediate
            if mid is not None and mid["info"]["next_player"] is not None:
                if mid["info"]["observer"] != episode.seat:
                    raise ValueError("intermediate value perspective changed")
                mids.append(mid["observation"])
                mid_infos.append(mid["info"])
                mid_returns.append(target)
    batch = pack_episode_rows(contract, observations, infos)
    if not batch.legal_mask.any(axis=1).all():
        raise ValueError("training decisions require at least one legal action")
    actions = np.asarray(actions, dtype=np.int64)
    if (actions < 0).any() or (actions >= contract.actions.size).any() or not batch.legal_mask[np.arange(len(actions)), actions].all():
        raise ValueError("training labels must be legal permanent action IDs")
    returns = np.asarray(targets, dtype=np.float32)
    advantages = returns - snapshot.predict(batch).values
    return LearningTargets(batch, actions, returns, advantages.copy(), np.asarray(weights, dtype=np.float32),
                           pack_episode_rows(contract, mids, mid_infos) if mids else None,
                           np.asarray(mid_returns, dtype=np.float32), len(group.episodes))


def loss_terms(model, targets, *, phase, settings):
    """Recompute differentiable predictions; fixed targets receive no gradients.

    Actor loss averages sums over episodes. Value losses and policy entropy
    average their applicable rows. Teacher cross-entropy averages decisions.
    """
    if phase not in ("teacher", "actor_critic"):
        raise ValueError("unknown training phase")
    logits, values = _forward(model, targets.batch)
    legal = torch.from_numpy(targets.batch.legal_mask.copy())
    log_policy = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
    safe_logs = torch.where(legal, log_policy, 0.)
    chosen = log_policy[torch.arange(len(values)), torch.from_numpy(targets.actions.copy())]
    entropy = -(log_policy.exp() * safe_logs).sum(dim=1).mean()
    if phase == "teacher":
        policy_loss = -chosen.mean()
    else:
        scale = torch.from_numpy((targets.actor_weights * targets.advantages).copy())
        policy_loss = -(scale * chosen).sum() / targets.episode_count
    decision = (values - torch.from_numpy(targets.returns.copy())).square().mean()
    intermediate = values.new_zeros(())
    if targets.intermediate_batch is not None:
        _, mid_values = _forward(model, targets.intermediate_batch)
        intermediate = (mid_values - torch.from_numpy(targets.intermediate_returns.copy())).square().mean()
    total = (policy_loss + settings.decision_value_weight * decision
             + settings.intermediate_value_weight * intermediate
             - (settings.entropy_weight if phase == "actor_critic" else 0.) * entropy)
    return dict(total=total, policy=policy_loss, decision_value=decision,
                intermediate_value=intermediate, entropy=entropy)


class TrainingSession:
    """Own a model, Adam state and a schedule advanced only at completed groups.

    Construction selects the configured process-wide CPU thread count and enables
    deterministic PyTorch algorithms. Random model initialization uses a temporary
    RNG scope, leaving the caller's global generator unchanged. Evaluation has
    separate streams. A failed update marks the session unusable; reload its last
    completed checkpoint to continue.
    """
    def __init__(self, settings=TrainingSettings()):
        if settings.master_seed is None:
            settings = replace(settings, master_seed=resolve_seed())
        self.settings = settings
        torch.set_num_threads(settings.threads)
        torch.use_deterministic_algorithms(True)
        with create(settings.game) as game:
            contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(episode_seed(settings.master_seed, "initialization", 0, "model"))
            self.model = PolicyValueNetwork(contract, settings.hidden_sizes)
            self.torch_rng = torch.get_rng_state().clone()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.learning_rate, foreach=False)
        self.completed = dict(teacher=0, actor_critic=0)
        self.next_episode_id = 0
        self.updates = 0
        self.history = []
        self.runtime = runtime_identity()
        self.failed = False
        self._busy = False

    @property
    def phase(self):
        if self.completed["teacher"] < self.settings.teacher_episodes:
            return "teacher"
        if self.completed["actor_critic"] < self.settings.actor_episodes:
            return "actor_critic"
        return "complete"

    def check_boundary(self):
        """Require a successful completed group and unchanged runtime conditions."""
        if self.failed or self._busy:
            raise RuntimeError("session requires a completed successful group boundary")
        if runtime_identity() != self.runtime:
            raise ValueError("training runtime changed; restore its recorded CPU settings")

    def step(self):
        """Collect one frozen group and perform one full-group Adam update."""
        self.check_boundary()
        phase = self.phase
        if phase == "complete":
            raise RuntimeError("training schedule is complete")
        limit = self.settings.teacher_episodes if phase == "teacher" else self.settings.actor_episodes
        count = min(self.settings.group_size, limit - self.completed[phase])
        self._busy = True
        try:
            snapshot = FrozenModel(self.model)
            group = collect_episode_group(self.settings.game, episode_count=count,
                master_seed=self.settings.master_seed, first_episode_id=self.next_episode_id,
                phase=phase, snapshot=None if phase == "teacher" else snapshot,
                learner=self.settings.learner, payloads=("cells",), include_intermediate=True,
                opponent=self.settings.teacher_opponent if phase == "teacher" else self.settings.actor_opponent,
                max_episode_steps=self.settings.max_episode_steps)
            targets = make_targets(group, snapshot)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.torch_rng)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                losses = loss_terms(self.model, targets, phase=phase, settings=self.settings)
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise ValueError("non-finite training loss")
                losses["total"].backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError("missing or non-finite parameter gradient")
                self.optimizer.step()
                self.torch_rng = torch.get_rng_state().clone()
            after = FrozenModel(self.model)
            report = dict(update=self.updates + 1, phase=phase, first_episode_id=self.next_episode_id,
                          episodes=count, decisions=len(targets.actions), intermediates=len(targets.intermediate_returns),
                          losses={key:float(value.detach()) for key, value in losses.items()},
                          before_identity=snapshot.identity, model_identity=after.identity, outcomes=group.outcomes)
            self.completed[phase] += count
            self.next_episode_id = group.next_episode_id
            self.updates += 1
            self.history.append(report)
            return report
        except Exception:
            self.failed = True
            raise
        finally:
            self._busy = False

    def run(self, *, max_groups=None):
        """Yield completed update reports, optionally stopping for a resumable save."""
        if max_groups is not None and (type(max_groups) is not int or max_groups < 0):
            raise ValueError("max_groups must be a nonnegative integer")
        groups = 0
        while self.phase != "complete" and (max_groups is None or groups < max_groups):
            yield self.step()
            groups += 1

    def provenance(self):
        return dict(settings=asdict(self.settings), completed=dict(self.completed), updates=self.updates,
                    next_episode_id=self.next_episode_id, runtime=self.runtime)
