"""Run independent neural MCTS decisions; report measured batching and search costs."""
from tag_rust.randomness import resolve_seed, seed_argument
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import torch
import tag_rust as tag
from tag_rust.learning.inputs import ModelContract
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.search.neural_mcts import NeuralMctsEvaluator


def run(args):
    args.seed = resolve_seed(args.seed)
    if not 1 <= args.searches <= 64 or not 0 <= args.seed < 2**64:
        raise ValueError("searches must be 1-64 and seed an unsigned 64-bit integer")
    settings = json.loads(args.config.read_text())
    torch.manual_seed(args.seed)
    with ExitStack() as stack:
        games = [stack.enter_context(tag.create(args.game)) for _ in range(args.searches)]
        # Alternate the root's acting seat across otherwise independent searches.
        for index, game in enumerate(games):
            if index % 2:
                obs = game.observe(0, ())
                game.apply(int(obs.offered_action_ids[0]), position=obs.position)
        if args.model is None:
            snapshot = FrozenModel(PolicyValueNetwork(ModelContract.from_spec(games[0].spec(), payloads=("cells",)), (32, 32)))
        else:
            from tag_rust.learning.artifacts import load_model
            snapshot = FrozenModel(load_model(args.model))
        evaluator = NeuralMctsEvaluator(snapshot)
        searches = []
        start = time.perf_counter()
        for index, game in enumerate(games):
            config = dict(settings)
            # Explicit per-search seeds make this fixed-work demonstration reproducible.
            config["seed"] = (args.seed + index) % 2**64
            searches.append(stack.enter_context(tag.MctsSearch(game, config,
                bindings={"network_value":evaluator,"network_policy":evaluator})))
        batch = stack.enter_context(tag.MctsBatch(searches, max_group_rows=args.max_group_rows))
        rows, inference_ms = [], 0.0
        while True:
            groups = batch.advance()
            if not groups and all(search.report is not None or index in batch.errors
                                  for index, search in enumerate(searches)):
                break
            for group in groups:
                rows.append(group.rows)
                before = time.perf_counter()
                replies = group.evaluator.evaluate(tuple(request for _, request in group.members),
                                                   max_bytes=batch.max_bytes)
                inference_ms += (time.perf_counter() - before) * 1000
                batch.resume(group, replies)
        return {"seed":args.seed,"game":args.game,"scope":"neural search decisions; no training or tournament integration",
                "untrained_model":args.model is None,"model_identity":snapshot.identity,
                "elapsed_ms":(time.perf_counter()-start)*1000,"inference_ms":inference_ms,
                "inference_calls":len(rows),"inference_rows":sum(rows),
                "max_batch_rows":max(rows,default=0),"mean_batch_rows":sum(rows)/len(rows) if rows else 0,
                "errors":batch.errors,"searches":[search.report for search in searches]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="registered Rust game ID (see tag_rust.registered_games())", default="tictactoe")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3]/"examples/mcts/neural.json")
    parser.add_argument("--model", type=Path, help="existing portable evaluation-model JSON; omitted creates an untrained model")
    parser.add_argument("--searches", type=int, default=4)
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--max-group-rows", type=int, default=4096)
    arguments = parser.parse_args()
    result = run(arguments)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["errors"]:
        raise SystemExit(1)
