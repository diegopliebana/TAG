# Add an evaluator for OSLAPython

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

An evaluator assigns a number to each possible next position. OSLAPython uses
those numbers to choose a move. Larger values mean better positions for the
player choosing the move. Rust owns simulation, action order, selection and the
agent's RNG; your Python function supplies the values.

Start with [Rust-owned execution](../python/rust-owned-execution.md). The
[manual batch guide](../python/external-osla.md) explains how to inspect a single decision,
and [grouped evaluation](../python/grouped-evaluation.md) combines independent games.

## Decide what the values mean

Write down the player's viewpoint, value range, treatment of terminal positions,
required input arrays, and settings that affect the calculation. The existing
open-line heuristic uses values between -1 and +1: win +1, draw 0, loss -1;
ongoing positions receive estimates strictly within those terminal extremes.

A single-game request's `actor` is the player choosing the action. All its rows
use that player's viewpoint, even when the simulated move passes the turn to
someone else. A grouped request instead supplies `observers[row]`, allowing
rows from different players in one evaluation call. Score for that observer;
`next_players[row]` describes whose turn follows the simulated move.

Results use absolute seat indices. Read `results[row, observer]` to find the
outcome for the player whose value you are estimating. The current result codes
are ongoing 0, win 1, draw 2 and loss 3. Outcome codes and heuristic values serve
different purposes; your function translates the codes onto its declared scale.

OSLAPython accepts one finite float64 value per candidate, in exactly the request's
row order. Its near-best selection includes every action within 0.001 of the
maximum value, so the scale of an evaluator affects which actions can tie.
Values from all chunks participate in the same final comparison.

## A small Python evaluator

This example supports the current two-player cells format: own pieces +1,
opposing pieces -1 and empty cells 0. It uses the difference in piece counts as
an ongoing-position estimate, scaled to stay below terminal win/loss values.
Counting pieces gives many ties in these games because every move adds one piece;
the example teaches the interface. The supplied
[open-line heuristic](../../../adapters/python/examples/open_lines.py) provides a
more useful example of examining board structure.

```python
import numpy as np
import tag_rust


def evaluate_piece_balance(request):
    cells = request.payloads["cells"]
    rows = len(request.action_ids)
    scale = float(request.evaluator.parameters["scale"])
    if not 0 < scale < 1:
        raise ValueError("scale must be between 0 and 1")

    # Each row is a possible next board. The mean stays between -1 and +1.
    values = scale * cells.mean(axis=1, dtype=np.float64)
    if isinstance(request, tag_rust.GroupedEvaluation):
        observers = request.observers
    else:
        observers = np.full(rows, request.actor, dtype=np.int64)

    # Each row's observer may occupy a different absolute player seat.
    results = request.results[np.arange(rows), observers]
    values[results == 1] = 1.0
    values[results == 2] = 0.0
    values[results == 3] = -1.0
    return values
```

For nine candidates on a 3 by 3 board, `cells` has shape `(9, 9)` and `values`
has shape `(9,)`. For seven Connect 4 candidates, the shapes are `(7, 42)` and
`(7,)`. The same function reads each request's dimensions and observer information.

## Attach the evaluator and play

```python
with tag_rust.create("tictactoe") as game:
    for seat in range(2):
        game.set_controller(seat, tag_rust.OSLAPython(
            evaluator_id="piece-balance", evaluator_version=1,
            evaluator_parameters={"scale": 0.5}, seed=7 + seat,
            payloads=("cells",), max_rows=3,
        ))
    result = game.advance()
    while result.reason == "evaluation":
        values = evaluate_piece_balance(result.request)
        result = game.resume(result.request, values)
    assert result.reason == "finished"
```

Rust returns at most three candidates from a game in each request here. `resume`
accepts a valid chunk, requests further candidates when needed, and commits once
the whole decision has values. If your evaluator raises an exception, its current
request stays pending. Retry it or call `cancel_evaluation(request)` before
preparing new work. Game context management releases the game on exit.

No evaluator registry or game-specific Python adapter is required for this
external route. Your loop receives a request and calls the appropriate function.
When using several evaluator identities, choose the function from the request's
identity; Rust checks the returned identity but cannot inspect which code ran.

The same function works with `GameBatch`:

```python
from collections import deque
from contextlib import ExitStack

with ExitStack() as stack:
    games = [stack.enter_context(tag_rust.create(name))
             for name in ("tictactoe", "tictactoe", "connect4", "connect4")]
    for index, game in enumerate(games):
        for seat in range(2):
            game.set_controller(seat, tag_rust.OSLAPython(
                evaluator_id="piece-balance", evaluator_version=1,
                evaluator_parameters={"scale": 0.5}, seed=100 + 2*index + seat,
                payloads=("cells",),
            ))
    batch = stack.enter_context(tag_rust.GameBatch(games))
    pending = deque(batch.advance().groups)
    while pending:
        group = pending.popleft()
        result = batch.resume(group, evaluate_piece_balance(group))
        for stop in result.stops:
            if stop.error is not None:
                raise RuntimeError(f"game {stop.game_index}: {stop.error}")
        pending.extend(result.groups)
    assert all(game.next_player is None for game in games)
```

The Tic-Tac-Toe and Connect 4 requests form separate groups because their encoding
identities and array dimensions differ. One evaluator function handles both.

## Versions, settings and reproducibility

Choose an evaluator name and version for the calculation you implement. Record all
settings affecting its values in `evaluator_parameters`. The adapter stores these
as consistently ordered JSON. Change the declared version when the meaning or
calculation changes. A loaded network would also need its model/weight identity,
preprocessing and output interpretation recorded by its neural design. The
implemented [frozen neural evaluator](neural-play.md) binds these to an owned
model snapshot and checks individual/grouped requests before scoring.

The controller checkpoint preserves the declared evaluator identity/settings and
Rust agent RNG progress. Python function code, Python-side mutable memory, external
RNGs and neural weights require their own persistence. Restoring the controller
assumes you supply matching evaluation code and external state. MCTS uses the [search evaluator interfaces](../agents/mcts-api.md); neural execution
uses the [model snapshot/evaluator interface](neural-inference.md). Multiple
simultaneous evaluation requests within one tree are not supported.

For reproducible OSLA choices, fix the observation encoding, row values, evaluator
settings, offered-action order and starting agent checkpoint. A seed alone only
specifies the beginning of an RNG stream. Finish or cancel pending decisions
before copying the game; checkpoints save the live agent's state between decisions.

## Rust backends and Python-callable Rust scoring

For fully native execution, implement `BatchEvaluator::evaluate(&mut self, batch)`
on a Rust type. It receives a borrowed CandidateBatch and returns a
`Result<Vec<f64>, OslaPythonError>`: Ok contains the ordered values and Err
explains a failure. Return one finite value per row. Use
`batch.observer` as the viewpoint and retain candidate order. Construct
`OslaPythonAgent::new(backend, settings, seed)` to use it through ordinary `choose()`.
See [OpenLinesBatchEvaluator](../../../src/agents/osla_python/open_lines.rs) and
the [native example](../../../examples/osla_python.rs).

If the backend supports exact continuation, implement `Checkpoint` so its snapshot
includes every setting and changing field that affects future values. Mutable
snapshot data must restore independently. A stateful backend's own updates can
remain after a failed ordinary evaluation; design and document its failure behavior.

The Python adapter currently exposes one native execution choice:
`OSLAPython.rust_heuristic()`. Adding another backend to Rust does not automatically
register a new Python native mode. Extend explicit mode/configuration validation,
construction and checkpoint restoration in
[engine.rs](../../../adapters/python/src/engine.rs), add a public setup factory,
and test advance/resume, copy/reset and restoration. Python-callable scoring is
a separate binding, exemplified by [scoring.rs](../../../adapters/python/src/scoring.rs).
It accepts NumPy buffers and returns values; it should retain computation outside
game rules and release Python's interpreter lock for suitable Rust work.

## What to test

Check the function on both observer seats, ongoing and terminal candidates,
changed board sizes and chunks split at different offsets. Verify exact row order,
finite float64 outputs and a declared value range. If using fixed weights or
preprocessing for one format, reject incompatible encoding identities explicitly.

Compare individual and grouped values for the same rows, including different
observers in one group. Check that changing chunk size leaves complete-game choices
and exact agent RNG progress unchanged when values are unchanged. Include retry,
cancellation and evaluator-version/settings mismatches. The existing
[evaluator tests](../../../adapters/python/tests/test_heuristic.py) and
[grouping tests](../../../adapters/python/tests/test_grouping.py) provide examples.
