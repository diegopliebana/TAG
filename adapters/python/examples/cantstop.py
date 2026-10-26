"""Play Can't Stop through Python's basic Rust game API (first offered action)."""
import argparse
import tag_rust as tag


def seed_value(text):
    if text == "auto":
        return None
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be auto or a nonnegative integer") from error
    if not 0 <= value <= 2**64-1:
        raise argparse.ArgumentTypeError("seed must fit an unsigned 64-bit integer")
    return value


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("action limit must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Basic Python game-API demonstration. Every seat uses First Available "
                    "(the first offered action); agents are fixed in this example.",
        epilog="For configurable Random/OSLA/MCTS opponents, edit the players list in "
               "examples/matches/cantstop.json, then run:\n"
               "  cargo run --offline --release --features mcts-json,replay "
               "--example match_play -- examples/matches/cantstop.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2,
                        help="number of First Available players (default: 2); does not select agents")
    parser.add_argument("--seed", type=seed_value, default=None, help="integer or auto (default)")
    parser.add_argument("--max-actions", type=positive, default=None, help="omit to disable the action limit")
    args = parser.parse_args()
    with tag.create("cantstop", {"seed": args.seed}, players=args.players) as game:
        print(f"game seed: {game.seed}")
        print("agents (fixed in this example; scores/outcomes use seat order):")
        for seat in range(args.players):
            print(f"  seat {seat}: First Available (first offered action)")
        print("For configurable agents, use examples/matches/cantstop.json with match_play; see --help.")
        actions = 0
        while game.next_player is not None:
            if args.max_actions is not None and actions >= args.max_actions:
                break
            view = game.observe(game.next_player, ())
            # This demonstration always takes the first offered action. Stop is
            # first after allocating, so it banks immediately when possible.
            game.apply(int(view.offered_action_ids[0]), position=view.position)
            actions += 1
        view = game.observe(0, ("owners",))
        scores = view.payloads["owners"][:, 1:].sum(axis=0).astype(int).tolist()
        labels = ("ongoing", "win", "draw", "loss")
        print(f"actions: {actions}; truncated: {game.next_player is not None}")
        print(f"claimed-column scores: {scores}")
        print("outcomes:", [labels[int(code)] for code in view.results])


if __name__ == "__main__":
    main()
