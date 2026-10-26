"""Configure agents in Python and let Rust run. Use --evaluation python for external values."""
from tag_rust.randomness import resolve_seed, seed_argument
import argparse
import tag_rust
from open_lines import evaluate_open_lines


def play(name, *, evaluation="rust", seed=None):
    seed = resolve_seed(seed)
    print(f"{name}: master seed {seed}")
    with tag_rust.create(name) as game:
        for seat in range(2):
            if evaluation == "rust":
                controller = tag_rust.OSLAPython.rust_heuristic(seed=(seed+seat) % 2**64)
            else:
                controller = tag_rust.OSLAPython(evaluator_id="python-open-lines",seed=(seed+seat) % 2**64)
            game.set_controller(seat,controller)

        # With Rust evaluators this call runs the complete match. Python gets no
        # candidate arrays and performs no per-move loop or heuristic calculation.
        result = game.advance(max_moves=4096)
        if evaluation == "python":
            support = game.spec().heuristic
            lines = support.indices.reshape(-1,support.window_length)
            while result.reason == "evaluation":
                request = result.request
                values = evaluate_open_lines(request,lines)
                # Rust accepts the scores, commits a completed decision and runs
                # on until another batch is needed or the match finishes.
                result = game.resume(request,values,max_moves=4096)
        if result.reason != "finished":
            raise RuntimeError(f"Match stopped for {result.reason}")
        final = game.observe(0,payloads=())
        print(f"{name}: {final.progress[0]} moves; results={final.results.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",choices=("rust","python"),default="rust")
    parser.add_argument("--seed", type=seed_argument, default=None)
    options = parser.parse_args()
    play("tictactoe",evaluation=options.evaluation, seed=options.seed)
    play("connect4",evaluation=options.evaluation, seed=options.seed)
