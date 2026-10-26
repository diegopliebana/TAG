"""Run seeded, untrained neural/teacher collection and neural-value OSLA examples."""
from tag_rust.randomness import resolve_seed, seed_argument
import argparse
import json
import torch
import tag_rust as tag
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.inputs import ModelContract
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.play import NeuralValueEvaluator
from tag_rust.learning.collection import collect_episode_group
from tag_rust.learning.seeds import episode_seed


def empty_counts():
    return dict(episodes=0, wins=0, draws=0, losses=0, truncated=0)


def empty_report():
    return dict(combined=empty_counts(), per_seat={0:empty_counts(), 1:empty_counts()})


def merge(report, outcomes):
    for key in report["combined"]:
        report["combined"][key] += outcomes["combined"][key]
        for seat in (0, 1):
            report["per_seat"][seat][key] += outcomes["per_seat"][seat][key]


def run(args):
    args.seed = resolve_seed(args.seed)
    if args.episodes <= 0 or not 1 <= args.batch_size <= 256 or args.seed < 0:
        raise ValueError("episodes must be positive, batch-size 1-256 and seed nonnegative")
    modes = ("teacher", "policy", "osla") if args.mode == "all" else (args.mode,)
    if "osla" in modes and args.opponent not in (None, "osla_python"):
        raise ValueError("OSLA example supports the native osla_python opponent; use teacher/policy for other opponents")
    initialization = episode_seed(args.seed, "initialization", 0, "model")
    torch.manual_seed(initialization)
    with tag.create(args.game) as game:
        model = PolicyValueNetwork(ModelContract.from_spec(game.spec(), payloads=("cells",)))
    snapshot = FrozenModel(model)
    report = dict(game=args.game, seed=args.seed, model_initialization_seed=initialization,
                  model_identity=snapshot.identity, scope="untrained execution/collection; no weight updates", runs={})
    next_training_id = 0
    for mode in modes:
        outcomes = empty_report()
        opponent = args.opponent or {"teacher":"osla", "policy":"random", "osla":"osla_python"}[mode]
        if mode != "osla":
            remaining = args.episodes
            while remaining:
                count = min(remaining, args.batch_size)
                group = collect_episode_group(args.game, episode_count=count, master_seed=args.seed,
                     first_episode_id=next_training_id, phase="teacher" if mode=="teacher" else "actor_critic",
                     snapshot=None if mode=="teacher" else snapshot, learner=args.learner,
                     opponent=opponent, payloads=("cells",), max_episode_steps=args.max_steps)
                merge(outcomes, group.outcomes)
                next_training_id = group.next_episode_id
                remaining -= count
        else:
            if args.max_steps is not None:
                raise ValueError("max-steps applies to teacher/policy learner episodes")
            evaluator = NeuralValueEvaluator(snapshot)
            for logical in range(args.episodes):
                seat = logical % 2 if args.learner is None else args.learner
                with tag.create(args.game) as game:
                    game.set_controller(seat, evaluator.controller(seed=episode_seed(args.seed,"evaluation",logical,"neural_osla")))
                    game.set_controller(1-seat, tag.OSLAPython.rust_heuristic(seed=episode_seed(args.seed,"evaluation",logical,"opponent")))
                    stop = game.advance()
                    while stop.reason == "evaluation":
                        stop = game.resume(stop.request, evaluator.evaluate(stop.request))
                    if stop.reason != "finished":
                        raise RuntimeError(f"unexpected execution stop: {stop.reason}")
                    code = int(game.observe(seat, payloads=()).results[seat])
                    for counts in (outcomes["combined"], outcomes["per_seat"][seat]):
                        counts["episodes"] += 1
                        counts[{1:"wins", 2:"draws", 3:"losses"}[code]] += 1
        report["runs"][mode] = dict(opponent=opponent, **outcomes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="registered Rust game ID (see tag_rust.registered_games())", default="tictactoe")
    parser.add_argument("--mode", choices=("all", "teacher", "policy", "osla"), default="all")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--learner", type=int, choices=(0,1), default=None)
    parser.add_argument("--opponent", choices=("first_available", "random", "osla", "osla_python"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    try:
        result = run(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
