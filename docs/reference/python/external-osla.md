# Evaluate OSLAPython candidates in batches

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

For ordinary execution, start with [Rust-owned execution](rust-owned-execution.md).
It keeps move commitment and continuation inside Rust. This page explains the
manual prepare/finish/commit operations for inspecting individual decisions.

[Learning guide entry point](../learning/overview.md) · [Python game operations](game-api.md)

OSLAPython tries each offered action on an independent state copy. Rust prepares
the resulting positions, Python supplies one value per candidate, and Rust chooses
a near-best move using the agent's own RNG. A separate commit call applies the
move and installs the new RNG progress.

This page explains the external-evaluation mechanism using a simple test evaluator.
The original OSLA uses its existing Rust evaluator. The runnable example
implements that same heuristic in Python. The
[benchmark report](../../benchmarks/osla-python/README.md) compares execution costs.

## Run the batch heuristic in Rust

`OslaPythonAgent` also accepts a Rust `BatchEvaluator`. The supplied
`OpenLinesBatchEvaluator` implements the same scoring formula as the Python
example. Construct it from `model.learning_spec(...)`, which supplies the ordered
winning-line groups for this board configuration. Pass it as the first argument
to `OslaPythonAgent::new(backend, settings, seed)`.

The agent's ordinary `choose()` prepares candidate arrays, calls the Rust backend
for each batch and selects a move. This route runs entirely in Rust. The
Python-facing `tag_rust.OSLAPython(...)` obtains values through external requests;
`OSLAPython.rust_heuristic()` selects native scoring inside `advance()`.
The runnable [native example](../../../examples/osla_python.rs) uses the Rust
backend on both games; run `cargo run --release --example osla_python`.
Its checkpoint contains the board-encoding identity and ordered groups, while
the surrounding agent checkpoint preserves settings and RNG progress.

This backend expects the current two-player `cells` format: +1 for our pieces,
-1 for opposing pieces and 0 for empty cells. It checks batch identity, array
dimensions, cell values and result labels before scoring. The
[five-route benchmark](../../benchmarks/osla-python/README.md) compares it with
original OSLA and Python scoring.

## Receive batches in Python and call the Rust heuristic

This route keeps Python in the decision loop. Rust prepares a candidate batch;
Python receives its arrays, calls a compiled Rust scorer, receives values, then
passes those values to `finish_evaluation()`:

```python
import tag_rust

with tag_rust.create("tictactoe") as game:
    scorer = game.open_lines_evaluator()
    game.set_controller(0, tag_rust.OSLAPython(
        evaluator_id="rust-open-lines", seed=7, payloads=("cells",), max_rows=3,
    ))
    request = game.prepare_evaluation()
    while isinstance(request, tag_rust.EvaluationRequest):
        values = scorer.evaluate(request)  # Python calls Rust with the received arrays.
        request = game.finish_evaluation(request, values)
    game.commit_evaluation(request)  # Accept this one move and its RNG update.
```

`game.open_lines_evaluator()` creates a `RustOpenLinesEvaluator` using that game's
configuration. It owns the fixed description independently of the live game and
can score requests with the same encoding identity from other instances. Calling
`evaluate()` changes neither the live board nor the request or controller. The
session checks the request handle and live position when the caller submits the
scores. Scores can also be recomputed after closing the original game because
the scorer only needs its saved description and the supplied arrays.

Each call copies `cells` (float32), `results` (int8) and action IDs (int64) into
Rust-owned buffers, releases Python's lock during scoring, then returns an
independently owned float64 array. This adds input-copy and call costs to the
Rust-only batch evaluator. Strided arrays are supported; incorrect array types,
dimensions, encoding identities and cell/result values raise errors. Each scoring
call accepts at most 4,096 rows and 64 MiB of numeric input/output buffers.

The full-game example supports both scoring choices:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/play_osla_python.py --scoring rust
.\.venv\Scripts\python.exe adapters/python/examples/play_osla_python.py --scoring python
```

The [benchmark](../../benchmarks/osla-python/README.md) labels these two routes
as **Python loop, Rust scoring** and **Python loop, Python scoring**. Both pass the
candidate arrays through Python and use the same finish/commit flow.

## Attach an independently seeded controller

Create a game and attach OSLAPython to Player 0:

```python
import numpy as np
import tag_rust

game = tag_rust.create("tictactoe")
settings = tag_rust.OSLAPython(
    evaluator_id="constant-demo", evaluator_version=1,
    evaluator_parameters={"value": 0}, seed=7,
    payloads=("cells", "board"), max_rows=3,
)
game.set_controller(0, settings)
```

`settings` is a reusable setup value. Attaching it creates a separate Rust agent
with its own RNG. Attaching the same settings to another seat creates another
independent agent. Omitting `seed` derives a uint64 seed from the current time;
an explicit seed can be any integer from 0 through `2**64 - 1`. The chosen seed
can be read through `settings.seed` and is recorded in the attached controller's
checkpoint.

The evaluator identity records its name, version and settings. Python stores the
settings as a JSON string with consistently ordered keys. The identity lets Rust
reject responses declaring a different evaluator or settings. The caller remains
responsible for running the matching evaluation code and changing its declared
version when its behavior changes.

Player 0 is now an AI seat. `game.apply(...)` accepts human actions only, so this
seat advances through evaluation and commit. Player 1 remains human until another
controller is attached. `game.set_controller(0, None)` returns Player 0 to human
input and invalidates outstanding work. Controller changes also invalidate old
position tokens.

## Prepare the first chunk

```python
before = game.controller_checkpoint(0)
request = game.prepare_evaluation()
assert request.actor == 0
assert request.offset == 0
assert request.total == 9
assert request.action_ids.tolist() == [0, 1, 2]
assert request.payloads["cells"].shape == (3, 9)
assert request.payloads["board"].shape == (3, 3, 3, 3)
assert game.observe(0, payloads=()).progress[0] == 0
```

The empty board offers nine moves. `max_rows=3` limits this request to the first
three. `offset=0` says where the chunk begins in the original action list, and
`total=9` is the number of candidates for the whole decision.

`cells` has three rows of nine values. Row 0 is the position after action ID 0,
row 1 after ID 1, and row 2 after ID 2. Each row starts from the same empty board
and contains one X. It is +1 because Player 0 is the observer for **every row**.
The next player is Player 1, but the scoring viewpoint stays with Player 0.

The four dimensions of `board` are candidate, channel, row and column. Thus
`(3, 3, 3, 3)` means three candidates, each containing three 3-by-3 channels.
For default Connect 4, a three-candidate chunk instead has `cells` shape `(3, 42)`
and `board` shape `(3, 3, 6, 7)`.

Preparing creates temporary work. The live board and `before` checkpoint retain
their values. A second preparation on this game raises `EvaluationError` until
the existing work is committed or cancelled.

## Return values, then commit the proposal

Our demonstration assigns zero to every candidate. This lets us inspect batching
and seeded tie selection before implementing a meaningful evaluator:

```python
while isinstance(request, tag_rust.EvaluationRequest):
    values = np.zeros(request.action_ids.size, dtype=np.float64)
    request = game.finish_evaluation(request, values)

proposal = request
assert isinstance(proposal, tag_rust.MoveProposal)
assert game.controller_checkpoint(0) == before
assert game.observe(0, payloads=()).progress[0] == 0

game.commit_evaluation(proposal)
assert game.observe(0, payloads=()).progress[0] == 1
```

`finish_evaluation` checks the returned values. If more candidates remain, it
returns the next `EvaluationRequest`, with a larger offset and the next action IDs.
After all nine values arrive, it returns a `MoveProposal` containing the actor
and selected action ID.

Selection compares all values with the true global maximum, across every chunk.
Candidates at most 0.001 below that maximum qualify. The temporary agent chooses
uniformly among them using its RNG. A unique qualifying action consumes no random
draw. Changing chunk sizes preserves the choice when settings, values, action
order and starting RNG progress are otherwise the same.

Commit checks the session, position and prior agent checkpoint again, then applies
one action and installs the temporary agent's new progress. A second commit of the
same proposal is rejected. The selected action belongs to this game's AI seat.

Player 1 can now make a human move:

```python
human = game.observe(1, payloads=())
game.apply(int(human.offered_action_ids[0]), position=human.position)
```

The runnable [play_osla_python.py](../../../adapters/python/examples/play_osla_python.py)
attaches both seats and runs complete games with Rust scoring by default. The
optional `--scoring python` mode uses
[open_lines.py](../../../adapters/python/examples/open_lines.py).
Its `evaluate_open_lines(request, lines)` reads the winning-line
groups once obtained from `game.spec().heuristic`. For each candidate it checks
the acting player's result, then scores the lines in Rust's traversal order.
See [the scoring explanation](../learning/heuristic-lines.md). The example requests only
`cells`, which contains all the board information this heuristic needs.

The comparison test runs the real native `OslaAgent` and game evaluators in
release mode. Across six game configurations and three batch sizes, it checks
every candidate score with exact float64 equality and every seeded move choice.
This establishes matching behavior for those games. The
[speed comparison](../../benchmarks/osla-python/README.md) measures full games and
scoring separately. Python currently uses explicit loops over
candidates, lines and cells to make the arithmetic easy to follow.

## Read the request fields

| Field | Meaning |
|---|---|
| `id` | Diagnostic request number within this game. The internal handle also binds it to the owning Rust session. |
| `position` | Live game position from which the candidates were prepared. |
| `actor` | Player choosing the action and observing every candidate. |
| `identity` | Game/configuration/encoding identity for the candidate arrays. |
| `evaluator` | Declared evaluator name, version and serialized settings; `.parameters` decodes a fresh copy of the settings. |
| `offset`, `total` | First candidate index in this chunk, and candidate count for the complete decision. |
| `action_ids` | int64 array of fixed action IDs in candidate-row order. |
| `payloads` | Requested float32 arrays, each starting with a candidate dimension. |
| `progress` | int64 array `(N, 4)` of counters after each simulated move. |
| `results` | int8 array `(N, player_count)` of result labels in absolute player order. |
| `next_players` | int64 array `(N,)`, containing the next player or -1 at completion. |

Terminal candidates still need an evaluator value. Their `results` allow an
evaluator to handle a win, loss or draw before calculating any ongoing-position
estimate. `results` are outcome codes; the evaluator defines its value scale.
The fixed line descriptions in `game.spec().heuristic` support the supplied
hand-written Python heuristic. The observation arrays in `payloads` are the
inputs intended for learning models.

Returned arrays own independent output data. A caller may keep them after
cancellation, commitment, reset or close. Editing candidate arrays leaves the
live game unchanged. Python scores must be a one-dimensional float64 NumPy array;
the adapter copies them into Rust-owned storage before detached validation.

## Retry and cancel

A wrong value count, NaN, infinity or mismatched response identity leaves the
current request available for retry. A float32 array or a wrong array shape is
rejected explicitly. For an evaluator returning metadata, supply it with the
response using `action_ids=` and `evaluator=`; Rust checks their exact order and
identity against the prepared work.

For example, cancel a new evaluation before committing:

```python
checkpoint = game.controller_checkpoint(0)
pending = game.prepare_evaluation()
game.cancel_evaluation(pending)
assert game.controller_checkpoint(0) == checkpoint
```

The same cancellation call accepts a ready `MoveProposal`. In both cases the live
RNG is unchanged. Repeating evaluation with the same values therefore selects the
same action. Once a chunk has been accepted, use the returned next request; an old
chunk handle is rejected.

Reset, close and controller replacement invalidate outstanding work. Late or
cross-game responses raise `StalePositionError` (or `ClosedGameError` after close).
If evaluation code raises a Python exception, the request remains pending; the
caller can retry or cancel it. The engine reports failures directly to the caller.

## Copies, reset and checkpoints

`game.clone()` copies state with no pending evaluation, settings and exact controller checkpoints.
Finish/commit or cancel pending work before cloning; cloning while a request or
proposal is pending raises `EvaluationError`. Each copy receives a fresh session
identity and makes independent choices from its copied RNG progress.

`game.controller_checkpoint(player)` returns either `None` for a human or a
`ControllerCheckpoint` with policy version, evaluator identity/settings, requested
arrays, limits and complete RNG seed/stream/word position. Use
`game.restore_controller(player, checkpoint)` to restore a separate controller.
These checkpoints are Python/native continuation data. Browser replay files have
their own supported-agent format; external-evaluation checkpoints do not define a browser controller format.

Reset restarts the board with its original settings and each attached agent with
its configured initial seed. Checkpoint restoration preserves exact current RNG
progress. The two operations serve different purposes when repeating experiments.

## Limits and the native interface

Defaults allow at most 1,024 rows per returned chunk and 4,096 candidate actions
per decision. The per-call cell/byte limits default to the same 4,096 cells and
64 MiB as ordinary encoding. Returned numeric bytes include candidate arrays,
action IDs, next-player labels, progress and results. The row limit and byte
limit both constrain chunk size; a failure is returned if one row cannot fit.
Source observation encoding is checked against the limits as well.

Rust retains one reconstructed source, ordered actions and accumulated scores,
and creates one temporary successor at a time. Each game holds one pending
decision. [Grouped evaluation](grouped-evaluation.md) combines compatible requests
from independent games. These bounds
account for numeric outputs; game states and temporary allocations have their
own memory costs.

For native code, `OslaPythonAgent<E>` implements the same `Agent<F>` inputs as the
other agents and also implements `PreparedEvaluationAgent<F>`. A Rust
`BatchEvaluator` can supply chunk values during ordinary `choose()`. An
`ExternalEvaluation` backend returns the typed `ExternalEvaluationRequired`
error from ordinary `choose()` and is serviced through `EvaluationSession`.
The existing `OslaAgent` and its evaluators retain their original code path.

To implement a different scoring function, follow
[Add an evaluator](../learning/adding-evaluator.md).
