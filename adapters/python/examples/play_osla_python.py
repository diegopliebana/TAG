"""Coordinate Rust lookahead from Python; Rust scoring is the default."""
from tag_rust.randomness import resolve_seed, seed_argument
from open_lines import evaluate_open_lines
import tag_rust


def play(name, *, scoring="rust", seed=None):
    seed = resolve_seed(seed)
    print(f"{name}: master seed {seed}")
    if scoring not in ("python", "rust"):
        raise ValueError("scoring must be 'python' or 'rust'")
    # Create the Rust game. Leaving the with block closes this Python game handle.
    with tag_rust.create(name) as game:
        # Both modes receive candidate arrays in Python. In rust mode Python
        # passes those arrays back to Rust for scoring, then receives float64 values.
        if scoring == "rust":
            evaluate = game.open_lines_evaluator().evaluate
        else:
            heuristic = game.spec().heuristic
            lines = heuristic.indices.reshape(-1, heuristic.window_length)
            evaluate = lambda request: evaluate_open_lines(request, lines)
        # Give each of the two players its own Rust-owned agent and random stream.
        for seat in range(2):
            game.set_controller(seat, tag_rust.OSLAPython(
                # Identify the evaluator and give each player a reproducible seed.
                evaluator_id=f"{scoring}-open-lines", seed=(seed + seat) % 2**64,
                # Request only the cells encoding used by this heuristic. The
                # default batch limit fits every candidate on these example boards.
                payloads=("cells",),
            ))
        # next_player becomes None when the game has finished.
        while game.next_player is not None:
            # Rust lists the available moves and prepares the first batch:
            # it tries each move in that batch on a separate copy of the position.
            # It returns those candidate positions for Python to score.
            # The live board and the agent's random stream remain unchanged.
            request = game.prepare_evaluation()
            while isinstance(request, tag_rust.EvaluationRequest):
                # Choose where scoring runs; the prepare/finish/commit flow is shared.
                values = evaluate(request)
                # Rust checks and stores these scores. If candidates remain,
                # it returns another EvaluationRequest for the next batch.
                # After the final batch, Rust compares ALL candidates, resolves
                # any near-ties using the agent's RNG, and returns a MoveProposal.
                # That RNG update is held with the proposal until it is committed.
                request = game.finish_evaluation(request, values)
            # Apply the proposed move to the live Rust game and accept its RNG
            # update. This is the call that actually advances the game by one move.
            game.commit_evaluation(request)
        # Read the final counters and per-player results from Player 0's view.
        # An empty payloads tuple skips the cells/board arrays, which we don't use.
        result = game.observe(0, payloads=())
        print(f"{name}: {result.progress[0]} moves, results={result.results.tolist()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", choices=("python", "rust"), default="rust")
    parser.add_argument("--seed", type=seed_argument, default=None)
    args = parser.parse_args()
    play("tictactoe", scoring=args.scoring, seed=args.seed)
    play("connect4", scoring=args.scoring, seed=args.seed)
