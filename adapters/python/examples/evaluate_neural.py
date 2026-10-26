"""Load a trained model and play exact-N direct-policy or neural-value OSLA matches."""
from tag_rust.randomness import resolve_seed, seed_argument
import argparse
import json
from pathlib import Path
from tag_rust.learning.artifacts import load_model
from tag_rust.learning.evaluation import evaluate_model
from tag_rust.learning.training import OPPONENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--mode", choices=("policy", "osla"), default="policy")
    parser.add_argument("--opponent", choices=OPPONENTS)
    parser.add_argument("--learner", type=int, choices=(0,1))
    args = parser.parse_args()
    try:
        result = evaluate_model(load_model(args.model), episodes=args.episodes, seed=args.seed,
                     mode=args.mode, opponent=args.opponent, learner=args.learner, group_size=args.batch_size)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
