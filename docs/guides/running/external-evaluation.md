# Run Rust agents with external evaluation

Use this mode when Python supplies position estimates while Rust simulates
actions and owns the game loop. An OSLAPython controller pauses with a batch of
candidate observations; Python scores them and Rust selects and commits a move.
Complete [Python setup](../setup.md). Attached-controller support is narrower
than basic game access: check the [capability matrix](../../reference/framework/overview.md#supported-workflows).

## Choose who computes the estimates

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/run_agents.py --evaluation rust --seed 42
.\.venv\Scripts\python.exe adapters/python/examples/run_agents.py --evaluation python --seed 42
```

Both run complete example games and print master seeds, move counts and seat-order
results. `rust` keeps scoring inside Rust. `python` returns candidate batches to
the supplied Python open-line heuristic. This compares execution routes, not
different game rules. Omit `--seed` or use `auto` for OS randomness.

For explicit prepare/evaluate/commit calls, use:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/play_osla_python.py --scoring python --seed 42
```

`--scoring rust` runs the same external-evaluation interface with a Rust scorer.
All options/defaults are in the [CLI reference](../../reference/python/command-line.md).

## Supply your own estimates

This small evaluator averages the signed occupied-cell features. It demonstrates
the interface; it does not measure tactical strength.

```python
import numpy as np
import tag_rust as tag

with tag.create("tictactoe") as game:
    for seat in range(2):
        game.set_controller(seat, tag.OSLAPython(
            evaluator_id="piece-balance", evaluator_version=1,
            evaluator_parameters={"scale": 0.5}, seed=7 + seat,
            payloads=("cells",), max_rows=3,
        ))
    stop = game.advance()
    while stop.reason == "evaluation":
        request = stop.request
        scale = request.evaluator.parameters["scale"]
        values = scale * request.payloads["cells"].mean(axis=1, dtype=np.float64)
        outcomes = request.results[:, request.actor]
        values[outcomes == 1] = 1.0
        values[outcomes == 2] = 0.0
        values[outcomes == 3] = -1.0
        stop = game.resume(request, values)
    if stop.reason != "finished":
        raise RuntimeError(stop.reason)
    print(game.observe(0, payloads=()).results.tolist())
```

One returned value belongs to each candidate row, from the deciding player's
perspective. Terminal outcomes override ongoing estimates. Rust validates the
request identity, row count, finite values and position before accepting a reply.
Invalid replies can be corrected while the request remains pending; do not
substitute values from another position. Earlier accepted moves remain accepted
if later work fails.

## Group independent games

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/run_grouped.py --seed 42
```

This fixed four-game example combines compatible candidate rows into larger
Python calls. It prints each game's final results and the number of evaluation
calls. `GameBatch` owns coordination, not game rules; members retain their own
controllers, positions and RNGs. Grouped rows carry observer IDs, so a scorer
must use each row's perspective rather than one fixed seat.

Grouping is sequential native execution with batched evaluation; it does not
promise parallel simulation. Neural MCTS uses a separate
[request/batching interface](../agents/rust-mcts-from-python.md).

## Errors and continuation

Only compatible identities, shapes, settings and observers may share an
evaluation group. Handle each reported member error. A request cannot be reused
after commitment, reset or cancellation. Checkpoints restore supported controller
and RNG progress between decisions; pending evaluation handles are in-memory
objects, not portable saved requests. See [reproducibility](README.md#reproducibility).

## Reference

[External OSLA API](../../reference/python/external-osla.md),
[Rust-owned execution](../../reference/python/rust-owned-execution.md),
[grouped requests and limits](../../reference/python/grouped-evaluation.md),
[add an evaluator](../agents/new-evaluator.md).
