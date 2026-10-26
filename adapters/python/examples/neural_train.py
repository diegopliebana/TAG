"""Train with an OSLA teacher, then actor-critic; save evaluation and training files."""
from tag_rust.randomness import seed_argument
import argparse
import json
from pathlib import Path
from tag_rust.learning.training import TrainingSettings, TrainingSession, OPPONENTS
from tag_rust.learning.artifacts import load_checkpoint, save_checkpoint, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="registered Rust game ID (see tag_rust.registered_games())", default=None)
    parser.add_argument("--teacher-episodes", type=int, default=None)
    parser.add_argument("--actor-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--learner", type=int, choices=(0,1), default=None)
    parser.add_argument("--teacher-opponent", choices=OPPONENTS, default=None)
    parser.add_argument("--opponent", choices=OPPONENTS, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--decision-value-weight", type=float, default=None)
    parser.add_argument("--intermediate-value-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-groups", type=int, help="stop after this many additional groups")
    parser.add_argument("--output", type=Path, required=True, help="portable evaluation JSON")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    supplied = dict(game=args.game, teacher_episodes=args.teacher_episodes, actor_episodes=args.actor_episodes,
                    group_size=args.batch_size, master_seed=args.seed, learner=args.learner,
                    teacher_opponent=args.teacher_opponent, actor_opponent=args.opponent,
                    max_episode_steps=args.max_steps, learning_rate=args.learning_rate, threads=args.threads,
                    hidden_sizes=args.hidden_sizes, decision_value_weight=args.decision_value_weight,
                    intermediate_value_weight=args.intermediate_value_weight, entropy_weight=args.entropy_weight)
    supplied = {key:value for key,value in supplied.items() if value is not None}
    try:
        if args.output.resolve() == args.checkpoint.resolve():
            raise ValueError("evaluation model and training checkpoint require separate paths")
        if args.resume:
            if supplied:
                raise ValueError("resume uses saved training settings; omit training-setting flags")
            session = load_checkpoint(args.resume)
        else:
            session = TrainingSession(TrainingSettings(**supplied))
        for report in session.run(max_groups=args.max_groups):
            save_checkpoint(args.checkpoint, session)
            print(json.dumps(report), flush=True)
        save_checkpoint(args.checkpoint, session)
        save_model(args.output, session.model, provenance=session.provenance())
        print(json.dumps(dict(phase=session.phase, model=str(args.output), checkpoint=str(args.checkpoint),
                              **session.provenance()), indent=2))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
