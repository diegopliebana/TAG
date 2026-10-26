"""Evaluate frozen learned models using separate, reproducible episode streams."""
from ..randomness import resolve_seed
from .outcomes import RESULT_LABELS, outcome_counts
from ..engine.api import create
from ..search.evaluation import OSLAPython
from .collection import collect_episode_group
from .play import NeuralValueEvaluator
from .seeds import episode_seed
from .snapshot import FrozenModel


def empty_outcomes():
    def counts():
        return outcome_counts(truncated=True)
    return dict(combined=counts(), per_seat={0:counts(), 1:counts()})


def merge_outcomes(total, addition):
    for key in total["combined"]:
        total["combined"][key] += addition["combined"][key]
        for seat in (0, 1):
            total["per_seat"][seat][key] += addition["per_seat"][seat][key]


def evaluate_model(model, *, episodes=100, seed=None, mode="policy", opponent=None,
                   learner=None, group_size=16):
    """Play exactly N matches, alternating seats independently of training.

    policy samples legal actions from the frozen network against any registered
    episode opponent. osla uses learned successor values against native-scored
    OSLAPython, using the existing Rust-owned game execution interface.
    """
    seed = resolve_seed(seed)
    if type(episodes) is not int or episodes < 1 or type(group_size) is not int or not 1 <= group_size <= 256:
        raise ValueError("episodes must be positive and group_size 1-256")
    if type(seed) is not int or seed < 0 or learner is not None and (type(learner) is not int or learner not in (0, 1)):
        raise ValueError("provide a nonnegative seed and optional learner 0/1")
    if mode not in ("policy", "osla"):
        raise ValueError("mode must be policy or osla")
    snapshot = FrozenModel(model)
    game_name = snapshot.contract.identity.game
    with create(game_name) as game:
        from .inputs import ModelContract
        expected = ModelContract.from_spec(game.spec(), payloads=tuple(t.name for t in snapshot.contract.tensors))
        if expected != snapshot.contract:
            raise ValueError("evaluation task differs from saved model contract")
    outcomes = empty_outcomes()
    selected = opponent or ("random" if mode == "policy" else "osla_python")
    if mode == "policy":
        for first in range(0, episodes, group_size):
            group = collect_episode_group(game_name, episode_count=min(group_size, episodes-first),
                        master_seed=seed, first_episode_id=first, phase="evaluation", snapshot=snapshot,
                        learner=learner, opponent=selected, include_intermediate=False)
            merge_outcomes(outcomes, group.outcomes)
    else:
        if selected != "osla_python":
            raise ValueError("OSLA evaluation currently supports the native-scored osla_python opponent")
        evaluator = NeuralValueEvaluator(snapshot)
        for logical in range(episodes):
            seat = logical % 2 if learner is None else learner
            with create(game_name) as game:
                game.set_controller(seat, evaluator.controller(seed=episode_seed(seed,"evaluation",logical,"neural_osla")))
                game.set_controller(1-seat, OSLAPython.rust_heuristic(seed=episode_seed(seed,"evaluation",logical,"opponent")))
                stop = game.advance()
                while stop.reason == "evaluation":
                    stop = game.resume(stop.request, evaluator.evaluate(stop.request))
                if stop.reason != "finished":
                    raise RuntimeError(f"unexpected execution stop: {stop.reason}")
                result = int(game.observe(seat, payloads=()).results[seat])
                for counts in (outcomes["combined"], outcomes["per_seat"][seat]):
                    counts["episodes"] += 1
                    counts[RESULT_LABELS[result]] += 1
    for counts in (outcomes["combined"], *outcomes["per_seat"].values()):
        count = counts["episodes"]
        counts["win_rate"] = counts["wins"] / count if count else 0.
        counts["draw_rate"] = counts["draws"] / count if count else 0.
        counts["loss_rate"] = counts["losses"] / count if count else 0.
    return dict(game=game_name, model_identity=snapshot.identity, mode=mode, opponent=selected,
                seed=seed, seat_schedule="alternating" if learner is None else f"fixed:{learner}", **outcomes)
