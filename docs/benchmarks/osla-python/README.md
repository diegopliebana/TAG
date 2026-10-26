# Native OSLA and OSLA with a Python heuristic

This benchmark compares five routes using the same open-line scoring formula.
Rust copies positions and tries moves in every route.

| Route | How positions are scored | How moves are selected |
|---|---|---|
| Original Rust | The game's existing evaluator reads a Rust state. | `OslaAgent::choose()` scores each candidate and chooses a near-best move. |
| Batched Rust | `OpenLinesBatchEvaluator` reads Rust-owned candidate arrays. | `OslaPythonAgent::choose()` prepares batches, scores them in Rust and chooses a near-best move. |
| Batched Python | `evaluate_open_lines()` reads NumPy candidate arrays. | The Python loop prepares and finishes requests, then commits Rust's chosen proposal. |
| Python calls Rust scoring | Python receives NumPy arrays and passes them to `RustOpenLinesEvaluator.evaluate()`. Rust copies and scores those inputs and returns a NumPy float64 array. | The same Python prepare/finish/commit loop as the Python heuristic route. |
| Python starts, Rust runs and scores | Python creates two `OSLAPython.rust_heuristic()` controllers. Rust prepares and scores their batches internally. | One `advance(max_moves=4096)` call runs the complete match. No per-move Python loop or candidate-array conversion to NumPy. |

The batched Rust route includes array encoding and batched selection. It runs
headlessly in Rust, through the same agent interface as original OSLA. The
Ordinary `tag_rust.OSLAPython(...)` controllers request external values;
`OSLAPython.rust_heuristic()` selects native owned execution.
The two manual Python-loop rows share the external-evaluation session and transfer of candidate
arrays into Python. Only the scoring call differs; the Python-calls-Rust route
also constructs its Rust scorer during setup. That route includes another call
into Rust, input copies for safe detached scoring, and the returned score array.

The measured Python function is the educational example's explicit loop over
candidates, lines and cells. Its arithmetic order matches Rust. A NumPy operation
that processes many cells together, or a neural network evaluated by PyTorch,
would have a different cost. These measurements describe the current implementations on this machine.

## Reproduce the comparison

Run from the repository root, outside the debugger. The configured Python environment
and dependencies are required; see [Python setup](../../reference/python/setup.md).

```powershell
$env:PYO3_PYTHON = (Resolve-Path '.venv/Scripts/python.exe').Path
$env:VIRTUAL_ENV = (Resolve-Path '.venv').Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
.\.venv\Scripts\python.exe tools/benchmarks/osla/run.py --output docs/benchmarks/osla-python/results.json
```

The Python driver builds and runs both Rust routes in release mode. Build
time, process startup, Python imports and loading NumPy are outside the timings.
The run takes a few minutes. It overwrites the selected JSON output file.

## What is timed

**Complete game:** both players use OSLA, from an empty board until the rules
declare the game finished. Each route creates its game and seeded controllers,
observes positions, lists actions, copies and simulates candidates, scores them,
selects moves and applies them. Action-history collection is disabled during
all complete-game timing. The native routes return their final Rust state to
`black_box` so its computation remains observable to the compiler. History and
intermediate observations are checked in separate validation runs. The
Python route includes reading the game description and configuring its controllers.
It includes Python object/array creation and the calls into the compiled adapter.
The two Rust routes use their agents directly; they have no Python adapter or
pending session requests. The batched Rust route also obtains the game description
and configures the batch evaluator. Consequently this is the cost of each complete
route, including its setup, representation and control machinery.

The new owned-execution route includes Python game/controller setup, native
backend construction, one complete-match `advance` call, conversion of its small
stop result and closing the game. It excludes final-state observation and does
not collect intermediate positions or candidate arrays in Python. Its Rust
loop uses the checked session prepare/finish/commit path internally, so it also
differs from the direct Rust-only `choose()` runner. All measured matches finish
within the explicit 4,096-move call budget.

**Heuristic alone:** the benchmark prepares a corpus containing every candidate
position encountered in the same games. Rust scores already constructed Rust
states; Python scores already constructed NumPy arrays. Creating those states,
encoding them, and transferring their data are outside this measurement. The Rust
batch scorer reads the same encoded cells and line groups as Python. Its input
checks and output vector allocation are included. Returning
the Python function's score arrays is included. Timing is divided by the total
number of candidates, including terminal candidates, which use a cheaper result
lookup. In the **Python calls Rust** scoring column, candidates are already in
NumPy arrays, but each call still includes the Python wrapper, copying those
inputs back into Rust, Rust scoring and returning the score array. This column
measures the scoring call as Python uses it, including the language boundary.
The corpus retains many positions at once, so its memory-access pattern
differs from scoring one freshly simulated state in the full-game loop.

The original evaluators generate the line traversal each time they score a state.
Both scoring implementations for candidate arrays read line indices obtained once
from the game description.
They also score a flat cells array rather than the game's state representation.
Thus the first two routes share scoring logic, but differ in representation,
line traversal and input checks. Their full-game difference measures their combined
cost; it cannot be assigned entirely to batching overhead. The isolated scoring
columns help show how much the evaluator implementations themselves differ.

For each board configuration, the player seed pairs are `(100,101)`, `(102,103)`,
through `(130,131)`: 16 complete games. Before timing, Python checks that its full
move sequences match native OSLA for all three batch settings. The native program
also checks every batched-Rust game trace against original OSLA for all settings.
Every score in the separate corpus is checked for exact agreement across the
four scoring call paths. The owned-execution route reuses the validated native
batch scorer.

For the owned route, validation first calls `advance(max_moves=1)` and compares
each resulting position with the recorded original-OSLA action sequence. A second
game runs in one complete-match call; its final board, results, move count and
both controller checkpoints must match the stepped run. All these observations
and history checks happen outside the timing regions.

Each timing has seven samples after a warm-up. A sample repeats all 16 games,
or all scoring batches, until at least 500 ms have elapsed. Reported game times
are divided by the number of complete games. Batch sizes alternate in a shuffled,
reproducible order between Python samples, interleaving both scoring choices.
The owned-execution setting is interleaved with them at every batch limit.
The native routes run in separate
timing blocks. Games run sequentially on one thread;
there is no cross-game batching or graphical interface.

A batch limit of **1,024** fits all available moves for these boards, giving one
scoring call per actual move, in all four batch-based routes. Limits of **3** and **1**
split those same candidates into smaller batches. These settings preserve moves
and scores.

**Call breakdown:** a separate pass measures `prepare_evaluation`, the Python
call to either heuristic, `finish_evaluation` and `commit_evaluation`. These measurements include
the overhead of their timers and are approximate. Prepare includes candidate
simulation, encoding and conversion to Python arrays. Finish includes validation,
selection, and preparation of the next chunk when one is needed. These call times
therefore combine several kinds of work; they cannot individually identify the
cost of copying, encoding or crossing the language boundary.

## Files and checks

- [Raw samples and machine information](results.json).
- [Native benchmark](../../../tools/benchmarks/osla/native.rs): native
  games, candidate corpus, native scoring and release-mode timing.
- [Python driver](../../../tools/benchmarks/osla/run.py): exact
  comparisons, Python timings, batch settings and call breakdown.
- [Python heuristic](../../../adapters/python/examples/open_lines.py):
  the function being measured, with explanatory comments.
- [Table generation](../../../tools/benchmarks/osla/report.py) renders stored
  samples independently of game execution. The benchmark's
  [instrumented workload](../../../tools/benchmarks/osla/workloads.py) is kept
  separate from the readable play example.
- [Regression test](../../../adapters/python/tests/test_heuristic.py): separate
  native comparison on six configurations and three batch sizes.

The comparison uses the locally rebuilt release extension. All 25 Python tests
and 149 root Rust tests pass in the execution API validation, including native
execution and the Python-to-Rust scoring boundary. Wasm compilation passes.
The benchmark changes are confined to tools and documentation; adapter formatting
and strict Clippy pass, including the native benchmark target.
The benchmark checks 192 complete game traces
against original OSLA for each batched route (768 comparisons in total), and
7,643 candidate scores for each added evaluator. Native backend tests cover
scores, choices, RNG progress, checkpoint restoration, malformed inputs and
terminal-result perspective. No game rules or agent selection behavior changed.
The added route also checks 192 one-call versus stepped-run final-state/RNG
comparisons. The benchmark changes action recording during timing only; recorded
move sequences are still required in the separate validation pass.

## Measurement results

The [generated result tables](results.md) contain the complete-game averages,
heuristic-only timings, all three batch limits, sample medians/ranges and Python
call breakdown. [The raw JSON](results.json) retains every sample and game trace.
The benchmark driver generates both files on each run.

These results replace the earlier measurements with a fresh run of all five
routes. Comparisons should use values from the same run because machine
load and CPU frequency can change between runs.

## Interpretation of this run

The new **Python starts, Rust runs and scores** route averages 0.468 ms per 3x3
Tic-Tac-Toe game, 1.742 ms for 5x5, 1.061 ms for default Connect 4 and 0.318 ms
for small Connect 4. These include game/controller setup, the complete native run
and closing the game. One `advance()` call runs each match; Python receives no
candidate arrays or intermediate positions during timing.

Python coordinating batches and calling Rust scoring takes about 2.3 to 3.1
times as long as owned execution in this run. On default
Connect 4 the mean drops from 2.408 to 1.061 ms. Compared with the current
Python heuristic route, owned execution takes about one sixth to one thirty-fifth
as long across the four boards. For default Connect 4 it drops from 37.327 to 1.061 ms.

The direct Rust-only batched runner remains faster than the owned Python entry
point. For default Connect 4 it averages 0.636 ms. It directly calls `choose()`
and the forward model. Owned execution also includes Python-side configuration,
the adapter and its checked session path, staged selection/commitment and result
conversion. Their difference therefore includes more than the cost of one Python
function call. Original Rust OSLA averages 0.743 ms for this board; its evaluator
generates line traversal during scoring, while the batch scorer reads precomputed
indices. Original OSLA is fastest in both Tic-Tac-Toe configurations and small
Connect 4.

Batching still helps when everything runs in Rust, because fewer chunks require
less allocation and result handling. Its effect is much greater for manual Python
coordination. For 5x5 Tic-Tac-Toe, changing the limit from 1 to 1,024 reduces the
owned route from 2.483 to 1.742 ms, while manual coordination calling Rust drops
from 27.970 to 4.442 ms. Default Connect 4's owned limits 3 and 1,024 are close
(1.045 and 1.061 ms); their small ordering difference should be read with the
sample spread rather than treated as a general advantage of limit 3.

The [result tables](results.md) include scoring-only costs, medians and ranges.
Default Connect 4's owned sample means span approximately 0.891 to 1.158 ms.
These wall-clock ratios describe this machine, implementation and seed set.
The benchmark intentionally excludes intermediate history collection from every
timed route; earlier reports included it, so cross-run differences also reflect
that method change. Use the five routes from this run for comparisons.


## Group requests across independent games

A separate [CPU grouping measurement](grouped-results.md) compares 16 individual
games with `GameBatch`, using the actual Python heuristic and a constant NumPy
scorer. It records evaluation-call counts, exact final state/RNG agreement, raw
samples and workload timing. Its routes differ from the five single-game routes
above. The [grouping guide](../../reference/python/grouped-evaluation.md) explains
the public API and per-row player viewpoints.

The follow-up [scaling experiment](scaling-results.md) varies the number of games
from 1 to 128, holding board/input/chunk settings fixed. It records per-game time,
evaluation calls and exact final-state/RNG agreement for both existing evaluators.

The saved five-route, grouped and scaling measurements precede the final external-evaluation
cleanup. [Targeted before/after measurements](cleanup-results.md) document that
refactoring's mixed CPU timing results. Retained samples describe their measured
implementation; fresh runs use the current code.
