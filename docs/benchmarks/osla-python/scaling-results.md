# Independent-game grouping: CPU scaling experiment

Measured 16-10-2026 (Europe/Paris). [Raw samples and environment](scaling-results.json).

These samples precede the final external-evaluation cleanup. The [cleanup measurement](cleanup-results.md)
records a targeted before/after comparison; saved scaling samples are retained.

The number of games varies through 1, 2, 4, 8, 16, 32, 64 and 128. All other
settings stay fixed: default boards, cells inputs, controller max_rows=1,024,
coordinator max_group_rows=4,096 and max_bytes=64 MiB. Each game has independently
seeded controllers: game i uses seeds 100+2*i and 101+2*i. Larger workloads
retain the same first games and add more. Rust runs games sequentially; the
experiment measures grouping efficiency on CPU, without worker threads or a neural model.

The individual route runs complete games one after another with advance/resume.
The grouped route uses GameBatch to advance all games and combine compatible requests.
Both evaluate the same candidates and produce identical final boards, results and
complete controller/RNG checkpoints at every repetition.

## How to read the tables

Each row compares two ways of completing the same number of games. The time
columns show mean milliseconds per completed game: total workload time divided
by the number of games. The time ratio is individual time divided by grouped time.
It compares the routes at that row's fixed game count; it describes no linear
growth rate as more games are added. A ratio of 1.22 means the individual route
took 1.22 times as long: grouping used about 82% of that time, saving about 18%.
A ratio below 1 means the grouped route took longer.

The last two columns count calls to the Python evaluator while completing the
entire set of games once. They count function calls, with many candidate states
potentially evaluated in each call. They exclude repetitions used for timing.

For example, the 128-game Tic-Tac-Toe heuristic row has 1,152 individual calls
and 9 grouped calls. Individually, each of 128 nine-move games requests evaluation
nine times: 128 * 9 = 1,152. With grouping, the first call scores candidates for
the first move of all 128 games, the second handles their second moves, and so
on for nine calls. The first grouped call contains 128 * 9 = 1,152 candidate
states; later calls contain fewer candidates as the boards fill. Both routes
evaluate the same total candidate states. The saving comes from handling them
in fewer calls. The summary notation 1,152 -> 9 means those two call counts.

## tictactoe: python-open-lines

| Games | Individual ms/game | Grouped ms/game | Time ratio: individual / grouped | Python calls, individual (all games) | Python calls, grouped (all games) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6940 | 0.8546 | 0.81 | 9 | 9 |
| 2 | 3.3681 | 3.1576 | 1.07 | 18 | 9 |
| 4 | 3.2382 | 2.8163 | 1.15 | 36 | 9 |
| 8 | 3.0108 | 2.8301 | 1.06 | 72 | 9 |
| 16 | 3.4223 | 2.7685 | 1.24 | 144 | 9 |
| 32 | 3.3718 | 2.6282 | 1.28 | 288 | 9 |
| 64 | 2.9533 | 2.3399 | 1.26 | 576 | 9 |
| 128 | 2.7186 | 2.2305 | 1.22 | 1152 | 9 |

## tictactoe: constant

| Games | Individual ms/game | Grouped ms/game | Time ratio: individual / grouped | Python calls, individual (all games) | Python calls, grouped (all games) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0379 | 1.1234 | 0.92 | 7 | 7 |
| 2 | 0.9285 | 0.8909 | 1.04 | 15 | 8 |
| 4 | 0.9597 | 0.7148 | 1.34 | 30 | 8 |
| 8 | 0.9613 | 0.6229 | 1.54 | 61 | 9 |
| 16 | 0.9681 | 0.6374 | 1.52 | 129 | 9 |
| 32 | 0.9526 | 0.5951 | 1.60 | 250 | 9 |
| 64 | 0.8352 | 0.4954 | 1.69 | 503 | 9 |
| 128 | 0.9650 | 0.5684 | 1.70 | 997 | 9 |

## connect4: python-open-lines

| Games | Individual ms/game | Grouped ms/game | Time ratio: individual / grouped | Python calls, individual (all games) | Python calls, grouped (all games) |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.8944 | 72.1516 | 0.98 | 28 | 28 |
| 2 | 42.9338 | 45.0709 | 0.95 | 35 | 28 |
| 4 | 40.8481 | 39.9341 | 1.02 | 60 | 28 |
| 8 | 37.7474 | 36.0340 | 1.05 | 110 | 28 |
| 16 | 36.0477 | 34.1688 | 1.05 | 206 | 28 |
| 32 | 37.5157 | 36.2632 | 1.03 | 429 | 37 |
| 64 | 37.7754 | 33.9507 | 1.11 | 862 | 37 |
| 128 | 36.7882 | 35.9539 | 1.02 | 1734 | 41 |

## connect4: constant

| Games | Individual ms/game | Grouped ms/game | Time ratio: individual / grouped | Python calls, individual (all games) | Python calls, grouped (all games) |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.5201 | 4.0264 | 0.87 | 26 | 26 |
| 2 | 3.6283 | 3.5953 | 1.01 | 55 | 29 |
| 4 | 2.8721 | 2.5631 | 1.12 | 88 | 29 |
| 8 | 2.8668 | 2.1515 | 1.33 | 180 | 30 |
| 16 | 2.5524 | 1.8426 | 1.39 | 313 | 30 |
| 32 | 2.5708 | 1.7322 | 1.48 | 623 | 34 |
| 64 | 2.4725 | 1.6927 | 1.46 | 1225 | 38 |
| 128 | 2.5756 | 1.7448 | 1.48 | 2558 | 42 |

At 128 games, the first combined requests contain 1,152 candidate rows for
Tic-Tac-Toe and 896 for Connect 4, both below the fixed group cap of 4,096.
This experiment therefore varies game count while keeping the row cap inactive;
it does not measure the effect of splitting groups at a smaller cap.

## Method and limits of the result

Seven samples follow validation/warm-up runs. Route order alternates within each
game-count case. Each sample repeats complete workloads until at least 200 ms of
measured time accumulates; long workloads can exceed that minimum in one run.
The table averages the seven sample means. Timed work includes setup, complete play,
final observation/checkpoint extraction and cleanup. Equality comparisons happen
outside timing. Raw samples retain repetition counts and variation.

Game counts run in ascending order, with the real heuristic before the constant
evaluator. CPU load, temperature and interpreter behavior can vary during the run.
Treat small timing differences cautiously, particularly when locating a crossover
between routes. Call counts and exact-result comparisons are deterministic for
these seeds and versions. These timings describe whole workloads rather than
isolating copying, Rust preparation or NumPy conversion.

The real Python heuristic still loops through each candidate and winning line.
The constant NumPy evaluator returns zero, giving a cheap-scoring comparison.
Compare routes within an evaluator: its scores affect chosen moves and game length.
Larger counts amortize coordination calls over more games, but also allocate larger
arrays. Fewer calls alone do not guarantee improved total runtime. There is no
measurement of peak memory, GPU transfer, PyTorch inference or within-tree MCTS.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/osla/grouped.py --scaling
.\.venv\Scripts\python.exe tools/benchmarks/osla/scaling_report.py
```

The earlier [16-game measurement](grouped-results.md) is retained. Compare the
scaling cases within this fresh run; earlier measurements can have different load.
