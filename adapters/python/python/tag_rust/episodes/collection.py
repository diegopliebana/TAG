"""Collect one complete episode per batch member as independent transitions."""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    """One learner action, its reward and the observation after opponent play.

    member identifies the original batch row. info describes the starting
    decision; next_info describes the returned decision or ending. Arrays and
    dictionaries belong to this record and remain available after reset/close.
    """
    member: int
    observation: dict
    action: int
    reward: float
    next_observation: dict
    terminated: bool
    truncated: bool
    info: dict
    next_info: dict
    intermediate: dict | None = None


class CollectionError(RuntimeError):
    """A batch member failed; errors and statuses retain the batch diagnostics."""
    def __init__(self, result):
        self.errors = result.errors
        self.statuses = result.statuses
        super().__init__(f"episode collection failed: {self.errors}")


def _observation(result, member):
    return {name: values[member].copy() for name, values in result.observations.items()}


def collect_episodes(batch, choose_action=None, *, seeds=None, options=None, choose_batch=None, teacher=False):
    """Reset all members, then yield their learner transitions in batch-row order.

    choose_action(member, observation, info) returns a legal action ID. It receives
    independent inputs. seeds follows GameEnvBatch.reset's per-member convention.
    Alternatively choose_batch(members, observations, infos) receives active rows
    together and returns one action per member, or teacher=True selects native
    teacher steps. Exactly one selection mode is required. options supplies reset
    options per member, including an explicit teacher_seed when configured.
    Each member stops at termination or truncation. Consume the iterator before
    using the batch or its original environments elsewhere. The caller owns them.

    On step failure, successful peer transitions are yielded before CollectionError.
    A reset failure raises immediately. An exception from choose_action propagates
    before that round's step. Closing or abandoning the iterator leaves members
    at their current positions; a later collection starts with an explicit reset.
    """
    if type(teacher) is not bool or sum((choose_action is not None, choose_batch is not None, teacher)) != 1:
        raise ValueError("select exactly one per-member policy, batch policy or teacher")
    current = batch.reset(seeds=seeds, **({} if options is None else {"options":options}))
    if any(current.errors):
        raise CollectionError(current)
    active = current.valid.copy()
    while active.any():
        before = {i: (_observation(current, i), deepcopy(current.infos[i]))
                  for i, enabled in enumerate(active) if enabled}
        if teacher:
            actions = None
            result = batch.step_teacher(active=active)
        else:
            if choose_batch is not None:
                members = list(before)
                chosen = list(choose_batch(members, [deepcopy(before[i][0]) for i in members],
                                          [deepcopy(before[i][1]) for i in members]))
                if len(chosen) != len(members):
                    raise ValueError("batch policy must return one action per active member")
                actions = [None]*len(active)
                for i, action in zip(members, chosen):
                    actions[i] = action
            else:
                actions = [choose_action(i, deepcopy(before[i][0]), deepcopy(before[i][1]))
                           if enabled else None for i, enabled in enumerate(active)]
            result = batch.step(actions, active=active)
        for i, enabled in enumerate(active):
            if enabled and result.valid[i]:
                observation, info = before[i]
                intermediate = None
                if result.intermediate_valid is not None and result.intermediate_valid[i]:
                    intermediate = dict(observation={k:v[i].copy() for k,v in result.intermediate_observations.items()},
                                        info=deepcopy(result.intermediate_infos[i]))
                yield Transition(
                    i, observation, int(result.infos[i]["action"] if teacher else actions[i]), float(result.rewards[i]),
                    _observation(result, i), bool(result.terminated[i]),
                    bool(result.truncated[i]), info, deepcopy(result.infos[i]), intermediate)
        if any(result.errors):
            raise CollectionError(result)
        active &= ~(result.terminated | result.truncated)
        current = result
