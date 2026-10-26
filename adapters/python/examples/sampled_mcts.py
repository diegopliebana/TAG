"""Calculate observation-derived MCTS moves with native or neural values."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import tag_rust as tag
from tag_rust.randomness import resolve_seed, seed_argument


def derived_seed(master, role, index=0, game="cantstop"):
    """Game-specific demo-v1 recipe: first eight SHA-256 bytes, little-endian."""
    text = f"{game}-mcts-demo-v1:{master}:{role}:{index}"
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "little")


def run(args):
    game_name = getattr(args, "game", "cantstop")
    master = resolve_seed(args.seed)
    def seed_for(role, index=0):
        return derived_seed(master, role, index, game_name)
    if not 0 <= master < 2**64 or not 1 <= args.searches <= 64 or args.iterations < 0:
        raise ValueError("seed must fit u64, searches must be 1-64 and iterations nonnegative")
    if args.native and args.model:
        raise ValueError("--model requires neural mode")
    with ExitStack() as stack:
        games, seats = [], []
        for index in range(args.searches):
            seat = index % args.players
            parameters = {"seed": seed_for("game", index)}
            if game_name == "loveletter":
                parameters["first_player"] = seat
            game = stack.enter_context(tag.create(game_name, parameters, players=args.players))
            # Can't Stop's established demo reaches the requested seat by play;
            # Love Letter can select the starter directly without changing rules.
            while game.next_player != seat:
                view = game.observe(game.next_player, ())
                game.apply(int(view.offered_action_ids[0]), position=view.position)
            games.append(game)
            seats.append(seat)
        bindings = {}
        model_identity = None
        if not args.native:
            import torch
            from tag_rust.learning.inputs import ModelContract, ValueSpec
            from tag_rust.learning.network import PolicyValueNetwork
            from tag_rust.learning.snapshot import FrozenModel
            from tag_rust.search.neural_mcts import NeuralMctsEvaluator
            torch.manual_seed(seed_for("model"))
            if args.model:
                from tag_rust.learning.artifacts import load_model
                network = load_model(args.model)
            else:
                spec = games[0].spec()
                payloads = tuple(p.name for p in spec.payloads
                    if game_name != "loveletter" or p.name != "last_private_history")
                contract = ModelContract.from_spec(spec, payloads=payloads,
                    value=ValueSpec(players=tuple(range(args.players)), gamma=1))
                network = PolicyValueNetwork(contract, (32, 32))
            snapshot = FrozenModel(network)
            model_identity = snapshot.identity
            evaluator = NeuralMctsEvaluator(snapshot, turn_context="own_turn")
            bindings = {"network_value": evaluator, "network_policy": evaluator}
        searches = []
        for index, game in enumerate(games):
            configuration = {"schema_version": 1, "seed": seed_for("search", index),
                "rollout_length": 0, "budgets": {"iterations": args.iterations}}
            if bindings:
                configuration.update(tree_policy="puct", state_evaluator={"name": "network_value"},
                    puct_preferences={"source": "probabilities", "evaluator": {"name": "network_policy"}})
            searches.append(stack.enter_context(tag.MctsSearch(game, configuration, bindings=bindings)))
        with tag.MctsBatch(searches) as batch:
            reports = batch.run()
            errors = dict(batch.errors)
        return {"game": game_name, "seed": master, "seed_recipe": f"{game_name}-mcts-demo-v1", "players": args.players,
            "mode": "native" if args.native else "neural", "untrained_model": not args.native and args.model is None,
            "model_identity": model_identity, "errors": errors,
            "information": "Research output includes seeds; do not pass seeds or game handles to policies.",
            "decisions": [{"seat": seat, "game_seed": game.seed, "report": report}
                          for seat, game, report in zip(seats, games, reports)]}


def main(default_game="cantstop"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("cantstop", "loveletter"), default=default_game)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--searches", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--native", action="store_true", help="use native game heuristic without PyTorch")
    parser.add_argument("--model", type=Path, help="compatible portable model; omitted uses an untrained network")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
