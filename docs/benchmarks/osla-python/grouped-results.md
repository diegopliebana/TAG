# CPU measurement: group requests from independent games

Measured 16-10-2026 19:56 (Europe/Paris), Windows CPU, release Rust extension,
CPython 3.14.8 and NumPy 2.5.4. [Raw samples](grouped-results.json) record timings
and repetition counts. Reproduce with
`.venv/Scripts/python.exe tools/benchmarks/osla/grouped.py`.

These samples precede the final external-evaluation cleanup; see the
[targeted before/after comparison](cleanup-results.md).

Each workload configures 16 default games with seed pairs (100,101) through
(130,131). The individual route runs each game with `advance`/`resume`; the grouped
route lets Rust coordinate all 16 games with `GameBatch`. Both use cells inputs,
controller chunk limit 1,024 and default numeric output limits.

The real evaluator is the educational Python open-line heuristic. The constant
NumPy evaluator returns zero for every candidate; it helps show the effect of
reducing calls when scoring itself is cheap. It changes move choices, so compare
routes within each evaluator row. Neither workload runs a neural network.

| Game | Evaluator | Individual calls | Grouped calls | Individual ms / 16 games | Grouped ms / 16 games |
|---|---|---:|---:|---:|---:|
| tictactoe | python-open-lines | 144 | 9 | 10.882 | 9.886 |
| tictactoe | constant | 129 | 9 | 15.991 | 10.630 |
| connect4 | python-open-lines | 206 | 28 | 555.350 | 546.280 |
| connect4 | constant | 313 | 30 | 38.393 | 28.325 |

The call columns count invocations of the Python evaluator to complete all 16
games once. Each call can score many candidate states. For example, 144 individual
calls versus 9 grouped calls for Tic-Tac-Toe means that combining candidates from
16 games reduces evaluator invocations; both routes still score the same total
number of candidate states. Calls exclude repetitions used to measure time.

Both routes score exactly the same number of candidates within each workload.
Every repetition checks final boards, terminal results and complete controller
checkpoints, including exact RNG progress, against the individually executed
reference. The separate grouping tests compare every accepted move.

Seven samples follow warm-up/validation runs; route order alternates. Each sample
repeats the workload until at least 200 ms of measured time has accumulated, then
reports mean milliseconds per 16-game workload. The table averages those seven
sample means. Timed work includes game/controller creation, heuristic description
lookup, complete play, final fact/checkpoint extraction and context cleanup.
Equality comparisons happen outside the measured interval. The raw samples show
variation; these single-machine CPU measurements establish no GPU throughput or
universal speed guarantee.

Grouping strongly reduces Python evaluation calls. The open-line Python scorer
still loops over every candidate and line, so fewer calls yield a much smaller
change in whole-workload time than in call count. A future neural evaluator can
use the larger leading array dimension in one model call; its CPU/GPU costs need
their own measurements when that feature is implemented.

This comparison is separate from the [five-route single-game benchmark](README.md).
Its routes, workload sizes and timed bookkeeping differ; compare timings inside
each report rather than treating their numbers as interchangeable.
