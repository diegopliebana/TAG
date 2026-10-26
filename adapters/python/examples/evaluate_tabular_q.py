"""Evaluate a saved Tic-Tac-Toe Q table over an exact number of games."""
from tag_rust.randomness import resolve_seed, seed_argument
import argparse
import json
from pathlib import Path

from tabular_q import NATIVE_OPPONENTS, QTable, run


def evaluate(model, *, games=1000, seed=None, opponent=None, learner=None):
    """Load model and play games with greedy legal actions and seeded tie-breaking.

    opponent selects a registered Rust agent, defaulting to the saved training
    opponent. Seats alternate by default; learner=0/1 fixes the evaluation seat.
    The collection horizon comes from the saved task.
    Combined rates use all games; per-seat rates use that seat's episode count.
    Both include truncated episodes in their denominator.
    """
    seed = resolve_seed(seed)
    model = Path(model)
    table = QTable.load(model)
    selected = table.task["opponent"] if opponent is None else opponent
    counts = run(table, rounds=games, members=1, seed=seed, epsilon=0,
                 learning=False, opponent=selected, learner=learner)
    return dict(model=str(model), training_task=table.task,
                evaluation_task=dict(table.task, opponent=selected, learner=learner), seed=seed,
                games=games, epsilon=0, **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=seed_argument, default=None)
    parser.add_argument("--learner", type=int, choices=(0, 1),
                        help="fixed learner seat; default alternates first/second player")
    parser.add_argument("--opponent", choices=NATIVE_OPPONENTS,
                        help="Rust opponent; defaults to the saved training opponent")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    if args.games < 1 or args.seed < 0:
        parser.error("games must be positive and seed nonnegative")
    print(json.dumps(evaluate(args.model, games=args.games, seed=args.seed,
                              opponent=args.opponent, learner=args.learner), indent=2))


if __name__ == "__main__":
    main()
