# Let Rust run games from Python

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](../learning/overview.md) · [Manual batch operations](external-osla.md)

Python configures the game and controllers, then calls `game.advance()`. Rust
runs decisions until it reaches a point that needs the caller's attention.
Native evaluators keep simulation, scoring, move selection and the game loop
inside Rust. External evaluators return candidate arrays to Python; after Python
supplies values, `game.resume()` continues the same Rust-owned loop.

## Run a complete match with a Rust heuristic

```python
import tag_rust

with tag_rust.create("connect4") as game:
    for seat in range(2):
        game.set_controller(seat, tag_rust.OSLAPython.rust_heuristic(seed=7 + seat))

    result = game.advance(max_moves=4096)
    assert result.reason == "finished"
    final = game.observe(0, payloads=())
    print(final.progress[0], final.results.tolist())
```

The call runs both agents in Rust, including all their candidate batches. Python
receives the stop result when the match finishes. It receives no candidate
NumPy arrays during that run. Rust releases Python's interpreter lock while
running the decisions.

`OSLAPython.rust_heuristic()` selects the existing batched OSLA policy with the
Rust open-line heuristic. The policy remains separate from the original
`OslaAgent`; its scores, moves and seeded RNG continuation are checked against
the existing paths. The factory accepts `seed`, `max_rows`, `max_candidates`,
`max_cells` and `max_bytes`. An omitted seed derives from current time.
The implementation currently supports Tic-Tac-Toe and Connect 4's two-player
cells encoding. Each game supplies its line groups through its Rust description.

## Understand each stop

`advance()` and `resume()` return an `ExecutionResult`:

| Field | Meaning |
|---|---|
| `reason` | One of the four reasons below. |
| `moves` | Number of actual moves accepted during this call. For example, trying nine candidate moves and choosing one adds 1. |
| `position` | Token after all moves accepted in this call. Old tokens become stale after accepted moves. |
| `actor` | Player whose attention or next decision is needed; `None` when finished. |
| `request` | An `EvaluationRequest` for an external evaluator; present when `reason == "evaluation"`. |

| Reason | What happens next |
|---|---|
| `finished` | Read the final results or reset the game. |
| `human` | Observe `actor`'s view and submit a human action with the current position token. Call `advance()` again when ready. |
| `evaluation` | Evaluate the supplied request, then call `resume(request, values)`. |
| `move_limit` | Call `advance()` again to allow more decisions, or inspect or copy the position between decisions. |

`max_moves` is a positive integer, default 1,024. It limits accepted moves in
this call. Finished games and human turns are reported before the move-limit
stop, even when the last permitted move reached that condition. If the next
player is an AI and the limit has been reached, no new request is prepared.
The adapter checks that the requested budget fits the remaining position-token
counter before running. This move limit is separate from an MCTS budget
for simulations or network evaluations.

## Supply values from Python

An ordinary `OSLAPython(...)` controller expects external values. For example:

```python
import numpy as np
import tag_rust

with tag_rust.create("tictactoe") as game:
    for seat in range(2):
        game.set_controller(seat, tag_rust.OSLAPython(
            evaluator_id="constant-example", seed=7 + seat, max_rows=3,
        ))

    result = game.advance(max_moves=4096)
    while result.reason == "evaluation":
        request = result.request
        # Demonstration values: all positions are equally good. A heuristic or
        # network would return one float64 value for each candidate in this batch.
        values = np.zeros(request.action_ids.size, dtype=np.float64)
        result = game.resume(request, values, max_moves=4096)
    assert result.reason == "finished"
```

Rust receives each chunk, checks its IDs, evaluator identity and finite values,
and returns another request if candidates remain. Once all candidates are scored,
it chooses and commits the move, installs the next RNG checkpoint and continues.
The resumed move counts toward that call's move limit. An intermediate chunk
can therefore return `moves == 0` with another evaluation request.

Python may also call `game.open_lines_evaluator().evaluate(request)` in this loop.
That sends received arrays back into Rust for scoring. For a fully native
heuristic, the `rust_heuristic()` controller above avoids those array round trips.

Controllers can be mixed per seat: human (`None`), external evaluator, or Rust
heuristic. Rust runs through native seats and stops at whichever human or external
seat needs input. The evaluator always scores for the request's original actor.

## Cancellation, errors and saved controllers

`game.cancel_evaluation(result.request)` discards outstanding work and preserves
the live RNG. Calling `advance()` with outstanding work raises `EvaluationError`;
resume or cancel that work first. Invalid values and metadata preserve the current
chunk for retry. Old, duplicate and foreign requests fail the existing position
and session checks. Reset, controller replacement and closing invalidate work.

Earlier accepted moves remain accepted if a later decision fails. For example,
if a native player makes one move and preparing the next player's request exceeds
its resource limit, the call raises an error and the position token reflects the
accepted move. Observe the position and correct the controller settings before
continuing. A failing current decision keeps its live RNG unchanged.

Controller checkpoints include `execution`: `"external"` or `"rust_open_lines"`.
Restoring a native checkpoint reconstructs the scorer from the target game's
description and validates its supported code version/settings. Legacy checkpoints
that omit this field select external evaluation. A game copied with no pending evaluation preserves
execution choices and independent RNGs; reset restarts their original seeds.
Copies with outstanding requests are rejected until that work finishes or is
cancelled. Existing manual prepare/finish/commit calls remain available for
external controllers; native-scoring controllers use `advance()`.

## How neural MCTS uses Rust-owned execution

Rust owns the tree, traversal, copied simulations, selection and back-propagation
of values. It suspends when leaf positions need neural
evaluation. Python/PyTorch evaluates batches and returns predicted action
preferences (policy outputs) and position values; Rust resumes the searches.
Python configures the experiment, network and training loop.

The implemented batching groups requests from independent searches.
Taking several unevaluated leaves from one tree requires additional accounting
for work still waiting for results. The [neural MCTS API](../learning/neural-mcts.md) suspends each independent tree
for one evaluation request and resumes it when values arrive. The OSLA API uses
value batches and [grouping across independent games](grouped-evaluation.md).

Run the [complete example](../../../adapters/python/examples/run_agents.py):

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/run_agents.py
.\.venv\Scripts\python.exe adapters/python/examples/run_agents.py --evaluation python
```

The [five-route benchmark](../../benchmarks/osla-python/README.md) includes this
entry point under **Python starts, Rust runs and scores**. It measures complete
matches with setup included and checks game traces separately from timing.

For several games, use [grouped evaluation](grouped-evaluation.md). Rust can join
compatible candidate chunks so Python evaluates several decisions in one call.
