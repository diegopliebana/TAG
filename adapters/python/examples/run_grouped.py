"""Run independent games in Rust and score their combined candidates in Python."""
import argparse
from tag_rust.randomness import resolve_seed, seed_argument
from collections import deque
from contextlib import ExitStack
import tag_rust as tag
from open_lines import evaluate_open_lines


def play(seed=None):
    seed = resolve_seed(seed)
    print(f"Grouped games master seed: {seed}")
    with ExitStack() as stack:
        # Each pair shares an input format. Rust produces separate groups for
        # Tic-Tac-Toe and Connect 4, then combines the two games within each pair.
        games = [stack.enter_context(tag.create(name)) for name in
                 ("tictactoe", "tictactoe", "connect4", "connect4")]
        lines = {}
        for index, game in enumerate(games):
            spec = game.spec()
            lines[spec.identity] = spec.heuristic.indices.reshape(-1, spec.heuristic.window_length)
            for seat in range(2):
                game.set_controller(seat, tag.OSLAPython(
                    evaluator_id="python-open-lines", seed=(seed + 2 * index + seat) % 2**64))
        batch = stack.enter_context(tag.GameBatch(games))
        pending = deque(batch.advance().groups)
        evaluations = 0
        while pending:
            group = pending.popleft()
            # One evaluator call handles all rows, potentially from several games.
            # The example reads group.observers[row] for each row's player viewpoint.
            values = evaluate_open_lines(group, lines[group.identity])
            result = batch.resume(group, values)
            for stop in result.stops:
                if stop.error is not None:
                    raise RuntimeError(f"game {stop.game_index}: {stop.error}")
            pending.extend(result.groups)
            evaluations += 1
        for index, game in enumerate(games):
            result = game.observe(0, payloads=())
            print(f"game {index}: {result.progress[0]} moves, results={result.results.tolist()}")
        print(f"Combined Python evaluation calls: {evaluations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=seed_argument, default=None)
    play(parser.parse_args().seed)
