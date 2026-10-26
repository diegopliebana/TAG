# Run Rust MCTS from Python

Use this interface when a Python script should choose moves with the Rust MCTS
engine: for example, to control a game, collect search statistics or compare
models across independent positions. Python starts each search and decides what
to do with its recommendation. Rust owns simulations, tree traversal, selection
and back-propagation; it is not a Python implementation of MCTS.

Complete [Python setup](../setup.md) and read [game operations](../running/game-operations-from-python.md).
Native heuristics need no PyTorch. Neural search additionally uses Python/PyTorch
to evaluate batches of permitted observations while Rust waits for the results.

[Neural evaluation diagram](../../reference/learning/neural-mcts.md#advance-evaluate-and-resume) shows how Rust and Python share the work.

## Run and apply native Rust search

Save this as a Python script and run it from the repository root. It plays a
complete game with Rust-evaluated MCTS for every seat:

```python
import tag_rust as tag

with tag.create("connect4") as game:
    moves = 0
    while game.next_player is not None:
        settings = {"schema_version": 1, "seed": 7 + moves,
                    "budgets": {"iterations": 32}}
        with tag.MctsSearch(game, settings) as search:
            with tag.MctsBatch([search]) as batch:
                report, = batch.run()
                if batch.errors:
                    raise RuntimeError(batch.errors)
            game.apply(report["action_id"], position=search.position)
        moves += 1
    print("Moves:", moves)
    print("Results:", game.observe(0, payloads=()).results.tolist())
```

Each `MctsSearch` captures one decision position and creates a fresh agent/tree.
`batch.run()` returns a report; it does not apply the move. `game.apply` validates
the recommended action ID against the captured position. Read `next_player`
after each accepted move rather than alternating seat numbers yourself.
Result labels are 0=ongoing, 1=win, 2=draw and 3=loss.

The report contains root visits, mean values in absolute seat order, counters and
the resolved search seed. Settings use the [MCTS configuration](mcts.md), including
work or time budgets. Set the search seed to `None` or omit it for OS randomness;
retain `search.settings["seed"]` when reproducing that search. The example instead
uses fixed, distinct seeds for successive decisions. Game chance has a separate
seed where applicable; see [reproducibility](../running/README.md#reproducibility).

## Run independent searches

You can calculate recommendations without applying them, or create searches for
several independent game positions. The supplied demonstrations accept a native
or neural evaluator; neural demonstrations use untrained weights unless `--model`
supplies a compatible export. They perform no training or tournaments.

The [game manuals](../README.md#game-manuals) contain their exact demonstration
commands, player-count options and observation restrictions. Native game/search
capability registration is independent of discovery through
`tag.registered_games()`: basic game access alone does not enable search.

## Use a saved neural evaluator

After [MCTS self-play](../learning/mcts-selfplay.md) has produced a matching model:

```python
import tag_rust as tag
from tag_rust.learning.artifacts import load_model
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.search.neural_mcts import NeuralMctsEvaluator

model = FrozenModel(load_model("examples/selfplay/results/connect4-model.json"))
evaluator = NeuralMctsEvaluator(model)
with tag.create("connect4") as game:
    with tag.MctsSearch(game, "examples/mcts/neural.json", bindings={
        "network_value": evaluator, "network_policy": evaluator
    }) as search:
        with tag.MctsBatch([search]) as batch:
            report, = batch.run()
            if batch.errors:
                raise RuntimeError(batch.errors)
        game.apply(report["action_id"], position=search.position)
```

The binding names connect the search JSON's value and policy roles to frozen
PyTorch predictions. Models must match rules, player count, inputs, action mapping
and value convention. An untrained model can test execution but does not establish
playing strength. This workflow uses an existing model; it never updates weights.

## Batch independent searches

Pass several searches to `MctsBatch` to evaluate compatible observations together.
Each tree has at most one pending evaluation. Rust retains its simulated state
and RNG progress while waiting; Python supplies predictions and Rust resumes.
These are independent trees, not several unevaluated leaves from one tree.

Closing a batch closes its searches; cancellation applies no move. Python search
checkpoint export and continuation across separate decisions are not implemented.
A real move/reset invalidates searches captured from the previous position.

## Reference

[Neural request/resume and custom evaluator API](../../reference/learning/neural-mcts.md),
[native MCTS API](../../reference/agents/mcts-api.md), [model loading](models.md),
[game information/value contracts](../../reference/games/README.md).
