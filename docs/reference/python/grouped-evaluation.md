# Evaluate candidates from several games together

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

Start with [Rust-owned execution](rust-owned-execution.md) for one game. `GameBatch`
extends that approach to several independent games: Rust advances each game, joins
compatible candidate arrays, and returns groups for Python to score. Python calls
an evaluator once per group. Rust distributes the returned values to their original
decisions, selects moves with each agent's own RNG, and continues the games.

This is useful when a neural network can evaluate many positions in one call.
The original integration tests use the Python heuristic on CPU.
[Neural play](../learning/neural-play.md) also supplies batched PyTorch value evaluation for
OSLA. [Neural MCTS](../learning/neural-mcts.md) provides a separate request/resume API
for batching leaf evaluations across independent searches.

## A working example

Run [run_grouped.py](../../../adapters/python/examples/run_grouped.py) with the
project interpreter, or select it in your Python debugger. It creates two
Tic-Tac-Toe games and two Connect 4 games. The central loop is:

```python
with tag_rust.GameBatch(games) as batch:
    pending = deque(batch.advance().groups)
    while pending:
        group = pending.popleft()
        values = evaluate_open_lines(group, lines[group.identity])
        result = batch.resume(group, values)
        pending.extend(result.groups)
```

The full example imports `deque`, creates the original games and controllers,
looks up each game's heuristic line indices, and checks execution errors.
`games` contains the original `Game` objects. Inspecting one of them after a call
shows its accepted moves. Each game keeps its own state, controllers and RNG.

`advance()` visits every game. `resume()` visits only the games represented in
the supplied group. Other outstanding groups stay pending: keep them in the queue.
Each result contains newly produced `groups` and `stops` for the games visited by
that call. A stop gives the game index, reason, accepted move count, position token,
next actor where applicable, and an optional error message.

The reasons are `finished`, `human`, `evaluation`, `move_limit` and `error`.
The default move limit is 1,024 accepted moves per game per call. A resumed move
counts towards that limit. Completion and human input take precedence over it.
After every pending group is resolved or cancelled, call `advance()` again to
continue games that reached their move limit or received human input.

For hand-written Rust scoring, attach `OSLAPython.rust_heuristic()` as in the
single-game guide. The coordinator runs those seats entirely in Rust; they produce
no external evaluation groups. Human, native-scoring and external-scoring seats
can be mixed.

## Which rows belong together?

Rust combines chunks only when all of these agree:

- Game, configuration, rules version and encoding version.
- Evaluator name, version and canonical JSON settings.
- Ordered input names and their dimensions after the leading row dimension.

For example, two default Tic-Tac-Toe instances can share a group. A 5 by 5 board,
a Connect 4 board, or another evaluator setting produces a separate group.
A chunk stays together even if it would otherwise fit partly into an earlier group.
A newly returned chunk can join other newly returned compatible chunks in that call.

A `GroupedEvaluation` contains the usual candidate arrays, with a leading dimension
`N` for the total number of rows. `cells` has shape `[N, width*height]`; `board`, when
requested, has shape `[N, 3, height, width]`. Arrays own their storage and remain
usable after games change or close.

Each row has two additional int64 entries: `game_indices[row]` identifies its index
in `batch.games`, and `observers[row]` identifies the player whose viewpoint was
used to encode it. An evaluator returns one float64 value per row, measuring the
position for that row's observer. Results still use absolute player seats, so read
`group.results[row, group.observers[row]]` to find that observer's terminal result.

`segments` gives the contiguous row range for each original request. For example:

| Game | Player | Row range | Meaning |
|---|---|---|---|
| 0 | 0 | `[0:9]` | Nine candidates for game 0's first move. |
| 1 | 1 | `[9:17]` | Eight candidates for game 1, where Player 0 already moved. |

A segment also records the position, original request ID, candidate offset and
total candidates in that decision. Action IDs can repeat across games. Return
values in the exact combined row order; Rust retains the original mapping.

## Validation, errors and cancellation

`batch.resume(group, values)` requires a one-dimensional float64 NumPy array.
Before accepting any member's scores, Rust checks the entire response: row count,
finite values, evaluator identity, ordered action IDs, original request handles,
positions and controllers. A failure raises an exception and retains the group
for retry or cancellation. No member accepts scores or consumes RNG in this case.
The optional `action_ids=` and `evaluator=` arguments allow a service to return
its own response metadata for these checks.

Once a valid response is accepted, games run independently. A later simulation or
resource error becomes that game's `error` stop; earlier accepted moves remain,
and its position token includes those moves. Other games can still progress.
Always inspect `result.stops` for errors, as the example does.

`batch.cancel(group)` discards its still-matching pending calculations, preserving
live boards and agent RNG progress. `cancel_all()` does this for all its groups.
Cancellation leaves earlier accepted moves in place. If a game has been reset,
closed or given another controller, cancellation skips its obsolete request and
preserves any newer work. A response to a cancelled or consumed group is stale.
Cancelling a request discards its bookkeeping; an already running Python or GPU
calculation finishes according to that library's own execution rules.

Use the coordinator as a context manager or call `close()` explicitly. Closing
cancels its pending requests and releases its references to the games. The original
games remain open. Finish or cancel manually prepared game work before creating a
coordinator. Resolve its pending work before using `advance()` again. Attaching one
game to several coordinators provides no scheduling benefit; each game's existing
single-pending-decision checks still apply.

## Limits and execution ownership

`GameBatch(games, max_group_rows=4096, max_bytes=64*1024*1024)` accepts 1 to 256
distinct original games. Row and byte limits must be positive.

Before running games, Rust reads the retained format description and computes a
conservative upper bound from their
configured external seats: action counts, requested arrays and controller chunk
limits. An individual chunk must fit `max_group_rows`. The sum of possible numeric
output bytes for all games visited in that call must fit `max_bytes`. This includes
IDs, progress, results and row ownership arrays. Reduce controller `max_rows` or
the number of games when the bound is too large. The bound can exceed the actual
rows needed late in a match. It limits returned numeric data per call; game states,
Python objects, temporary buffers and previously retained arrays use extra memory.

Rust runs the games sequentially while detached from Python's interpreter lock.
It combines owned candidate arrays and creates NumPy outputs after reattaching.
Replies are copied into owned Rust buffers before games resume. This implementation
provides larger evaluation groups while retaining deterministic per-game order;
it introduces no worker threads or GPU transfers.

## How chunk sizes affect copying

A controller's `max_rows` (default 1,024) caps candidate states in one game request.
The coordinator's `max_group_rows` (default 4,096) caps combined rows sent for one
Python evaluation. For example, eight games with seven candidates each can
contribute 56 rows to one group. With a group cap of 14, those eight complete
chunks produce four groups. A controller chunk must itself fit the group cap.

The controller's `max_bytes` also caps numeric data in its chunk; remaining
candidates are returned in later chunks. The coordinator's `max_bytes` checks the
conservative bound described above and rejects excessive output before running
its games. Both default to 64 MiB, but apply at these different levels.

Smaller chunks allocate smaller arrays per request, while requiring more requests
and response handling. Larger groups combine more rows into larger arrays and
reduce Python calls. Rust combines candidate arrays before returning them, and
copies response values into owned buffers before resuming. The ordinary path
still simulates every candidate using an independent state copy. Chunk/group size
changes do not inherently remove those simulations or their state copies.

`max_candidates` (default 4,096) rejects decisions with too many available actions;
`max_cells` (default 4,096) restricts board size. These are separate from request
chunking. `max_moves_per_game` bounds accepted moves in one execution call, with a
default of 1,024. The coordinator accepts at most 256 distinct games.

## Where to read the implementation

- [grouping.py](../../../adapters/python/python/tag_rust/search/grouping.py): public result
  classes and thin `GameBatch` wrapper.
- [grouping.rs](../../../adapters/python/src/grouping.rs): original-game references,
  whole-group validation, execution, cancellation and Python result conversion.
- [group_arrays.rs](../../../adapters/python/src/group_arrays.rs): compatibility
  checks, concatenation and named records connecting row ranges to original requests.
- [group_limits.rs](../../../adapters/python/src/group_limits.rs): output limits and
  dispatch to each session's read-only response validation.
- [evaluation.rs](../../../src/session/evaluation.rs): `validate_response()` checks
  a chunk without accepting it; `finish()` reuses that same validation.
- [test_grouping.py](../../../adapters/python/tests/test_grouping.py): move-by-move
  parity, exact RNG progress, mixed players, malformed replies and lifecycle checks.
