# Complete all-MCTS games: API costs and process memory

Both implementations completed **250 games per game/player setting**:
4,000 collected games in total, of which **3,998 are included in
statistics**. Explicitly identified interrupted games are retained in the raw
records and excluded from all game/API/memory summaries without replacements.
The per-setting counts below identify the actual sample sizes. Every searched decision completed
5,000 iterations, with random rollouts limited to 10 actions.
Forced actions execute directly without search. All players use native MCTS,
with UCB exploration sqrt(2), final selection by visits and tree depth 1,000.
Configurable rules are aligned; Java Love Letter uses its information-set mode,
Rust its existing observation-derived forks. Neither game trajectories nor
search algorithms/heuristics are claimed equivalent.

Each engine/profile uses 3 sequential fresh processes, with batch sizes
100, 100, 50 games,
with 6 untimed search decisions per process. Engines
do not run concurrently. Java uses a 64 MiB initial/512 MiB maximum heap.
The recorded master seed is 42. Parameters, seed derivations,
measurement hooks and reproducibility checks are described in the
[tool methodology](../../../../tools/benchmarks/mcts-games/README.md).

## Included samples

| Game | Players | Rust included | Java included | Rust excluded | Java excluded |
|---|---|---|---|---|---|
| Tic-Tac-Toe | 2 | 250 | 250 | 0 | 0 |
| Connect 4 | 2 | 250 | 250 | 0 | 0 |
| Can't Stop | 2 | 250 | 250 | 0 | 0 |
| Can't Stop | 3 | 250 | 250 | 0 | 0 |
| Can't Stop | 4 | 250 | 250 | 0 | 0 |
| Love Letter | 2 | 250 | 250 | 0 | 0 |
| Love Letter | 3 | 250 | 248 | 0 | 2 |
| Love Letter | 4 | 250 | 250 | 0 | 0 |

[Exclusion records and reasons](exclusions.json).

## Real execution APIs

Values are microseconds per call: total sampled nanoseconds divided by the number
of timed calls, pooled across batches. Batch means are not given equal weight.
The first column is Java/Rust time; above 1 means Rust was faster in these
encountered positions. Real calls are timed throughout each game. Exact-copy
and copy+apply rows are independent shadow probes at pre-transition positions,
not extra actions or calls attributed to the actual agent. Initialization includes
state construction/setup, excluding parameter construction in both engines.

| Java/Rust time | Game | Players | Operation | Rust µs | Java µs |
|---|---|---|---|---|---|
| 31.79 | Tic-Tac-Toe | 2 | Initialize | 1.3016 | 41.3748 |
| 4.36 | Tic-Tac-Toe | 2 | Actions | 0.3102 | 1.3532 |
| 78.03 | Tic-Tac-Toe | 2 | Exact copy probe | 0.0846 | 6.6028 |
| 39.42 | Tic-Tac-Toe | 2 | Observation/player copy | 0.1536 | 6.0548 |
| 21.30 | Tic-Tac-Toe | 2 | Copy + apply probe | 0.3560 | 7.5826 |
| 32.26 | Connect 4 | 2 | Initialize | 1.4564 | 46.9812 |
| 8.17 | Connect 4 | 2 | Actions | 0.1898 | 1.5511 |
| 44.45 | Connect 4 | 2 | Exact copy probe | 0.2057 | 9.1439 |
| 34.13 | Connect 4 | 2 | Observation/player copy | 0.2444 | 8.3390 |
| 11.75 | Connect 4 | 2 | Copy + apply probe | 1.0697 | 12.5701 |
| 46.37 | Can't Stop | 2 | Initialize | 1.1320 | 52.4924 |
| 14.33 | Can't Stop | 2 | Actions | 0.1452 | 2.0807 |
| 25.61 | Can't Stop | 2 | Exact copy probe | 0.4382 | 11.2215 |
| 73.82 | Can't Stop | 2 | Observation/player copy | 0.1237 | 9.1304 |
| 11.22 | Can't Stop | 2 | Copy + apply probe | 1.0511 | 11.7914 |
| 29.66 | Can't Stop | 3 | Initialize | 1.8700 | 55.4568 |
| 10.15 | Can't Stop | 3 | Actions | 0.1901 | 1.9290 |
| 20.06 | Can't Stop | 3 | Exact copy probe | 0.6150 | 12.3397 |
| 79.33 | Can't Stop | 3 | Observation/player copy | 0.1429 | 11.3353 |
| 9.08 | Can't Stop | 3 | Copy + apply probe | 1.4304 | 12.9946 |
| 34.36 | Can't Stop | 4 | Initialize | 1.9356 | 66.5004 |
| 13.30 | Can't Stop | 4 | Actions | 0.2408 | 3.2019 |
| 23.33 | Can't Stop | 4 | Exact copy probe | 0.7265 | 16.9470 |
| 96.39 | Can't Stop | 4 | Observation/player copy | 0.1665 | 16.0494 |
| 12.07 | Can't Stop | 4 | Copy + apply probe | 1.4799 | 17.8652 |
| 11.92 | Love Letter | 2 | Initialize | 6.7604 | 80.5820 |
| 9.92 | Love Letter | 2 | Actions | 0.3666 | 3.6360 |
| 6.39 | Love Letter | 2 | Exact copy probe | 4.4306 | 28.3247 |
| 12.84 | Love Letter | 2 | Observation/player copy | 1.7577 | 22.5750 |
| 4.42 | Love Letter | 2 | Copy + apply probe | 7.0524 | 31.1861 |
| 12.07 | Love Letter | 3 | Initialize | 7.1512 | 86.3488 |
| 6.61 | Love Letter | 3 | Actions | 0.4641 | 3.0699 |
| 4.95 | Love Letter | 3 | Exact copy probe | 4.4186 | 21.8685 |
| 14.33 | Love Letter | 3 | Observation/player copy | 1.7980 | 25.7628 |
| 3.62 | Love Letter | 3 | Copy + apply probe | 6.7202 | 24.3545 |
| 9.20 | Love Letter | 4 | Initialize | 9.4976 | 87.3744 |
| 6.17 | Love Letter | 4 | Actions | 0.6725 | 4.1511 |
| 3.78 | Love Letter | 4 | Exact copy probe | 7.2016 | 27.2488 |
| 11.14 | Love Letter | 4 | Observation/player copy | 2.6383 | 29.3959 |
| 2.89 | Love Letter | 4 | Copy + apply probe | 10.2669 | 29.7052 |

## Search-position APIs

Search API calls are counted throughout. Latencies sample the first 16 calls
per game/category, then every 1024th call. Copy probes sample
pre-transition search positions; actual copy-call timings are separate. Rust
uses observation-derived forks rather than exact clones to start simulations.
Java player copies can include hidden-state sampling. A missing entry means that
API was not executed, not that it costs zero. Search does not initialize games.

| Java/Rust time | Game | Players | Operation | Rust µs | Java µs |
|---|---|---|---|---|---|
| 1.89 | Tic-Tac-Toe | 2 | Actions | 0.1205 | 0.2275 |
| 74.11 | Tic-Tac-Toe | 2 | Exact copy probe | 0.0749 | 5.5488 |
| 23.98 | Tic-Tac-Toe | 2 | Copy + apply probe | 0.2712 | 6.5042 |
| 4.24 | Tic-Tac-Toe | 2 | In-place apply | 0.1323 | 0.5614 |
| — | Tic-Tac-Toe | 2 | Simulation fork | 0.0864 | — |
| — | Tic-Tac-Toe | 2 | Actual exact-copy calls | — | 5.1643 |
| 2.32 | Connect 4 | 2 | Actions | 0.1351 | 0.3131 |
| 76.37 | Connect 4 | 2 | Exact copy probe | 0.1180 | 9.0125 |
| 12.27 | Connect 4 | 2 | Copy + apply probe | 0.9372 | 11.4968 |
| 3.43 | Connect 4 | 2 | In-place apply | 0.5672 | 1.9473 |
| — | Connect 4 | 2 | Simulation fork | 0.1039 | — |
| — | Connect 4 | 2 | Actual exact-copy calls | — | 8.1124 |
| 7.46 | Can't Stop | 2 | Actions | 0.1128 | 0.8421 |
| 30.01 | Can't Stop | 2 | Exact copy probe | 0.3257 | 9.7750 |
| 17.16 | Can't Stop | 2 | Copy + apply probe | 0.6093 | 10.4583 |
| 1.49 | Can't Stop | 2 | In-place apply | 0.1974 | 0.2933 |
| — | Can't Stop | 2 | Simulation fork | 0.1125 | — |
| — | Can't Stop | 2 | Actual exact-copy calls | — | 9.3355 |
| 5.37 | Can't Stop | 3 | Actions | 0.1607 | 0.8623 |
| 22.61 | Can't Stop | 3 | Exact copy probe | 0.4594 | 10.3885 |
| 13.12 | Can't Stop | 3 | Copy + apply probe | 0.8455 | 11.0935 |
| 1.05 | Can't Stop | 3 | In-place apply | 0.2861 | 0.2995 |
| — | Can't Stop | 3 | Simulation fork | 0.1403 | — |
| — | Can't Stop | 3 | Actual exact-copy calls | — | 10.0937 |
| 6.33 | Can't Stop | 4 | Actions | 0.2056 | 1.3005 |
| 27.02 | Can't Stop | 4 | Exact copy probe | 0.5676 | 15.3387 |
| 15.38 | Can't Stop | 4 | Copy + apply probe | 1.0675 | 16.4154 |
| 0.99 | Can't Stop | 4 | In-place apply | 0.4007 | 0.3975 |
| — | Can't Stop | 4 | Simulation fork | 0.1690 | — |
| — | Can't Stop | 4 | Actual exact-copy calls | — | 14.8088 |
| 4.75 | Love Letter | 2 | Actions | 0.2774 | 1.3169 |
| 6.70 | Love Letter | 2 | Exact copy probe | 3.4706 | 23.2679 |
| 19.49 | Love Letter | 2 | Observation/player copy | 1.2205 | 23.7918 |
| 4.80 | Love Letter | 2 | Copy + apply probe | 5.3965 | 25.9141 |
| 2.24 | Love Letter | 2 | In-place apply | 0.8099 | 1.8175 |
| — | Love Letter | 2 | Simulation fork | 17.8591 | — |
| — | Love Letter | 2 | Actual exact-copy calls | — | 32.3958 |
| 4.45 | Love Letter | 3 | Actions | 0.2954 | 1.3133 |
| 5.17 | Love Letter | 3 | Exact copy probe | 3.6845 | 19.0467 |
| 15.04 | Love Letter | 3 | Observation/player copy | 1.2960 | 19.4978 |
| 3.88 | Love Letter | 3 | Copy + apply probe | 5.5773 | 21.6381 |
| 1.98 | Love Letter | 3 | In-place apply | 0.6917 | 1.3714 |
| — | Love Letter | 3 | Simulation fork | 21.9430 | — |
| — | Love Letter | 3 | Actual exact-copy calls | — | 26.6577 |
| 3.63 | Love Letter | 4 | Actions | 0.4816 | 1.7471 |
| 3.91 | Love Letter | 4 | Exact copy probe | 5.6407 | 22.0289 |
| 13.78 | Love Letter | 4 | Observation/player copy | 1.6797 | 23.1437 |
| 2.91 | Love Letter | 4 | Copy + apply probe | 8.4683 | 24.6759 |
| 1.47 | Love Letter | 4 | In-place apply | 0.9948 | 1.4589 |
| — | Love Letter | 4 | Simulation fork | 33.0049 | — |
| — | Love Letter | 4 | Actual exact-copy calls | — | 17.8857 |

## Instrumented complete-game execution

These times include API counters/timers and shadow probes, agent setup and real
play. Probe duration is recorded separately; it is not subtracted to invent an
uninstrumented result. A search decision can contain many simulated actions.
Game length and search work differ between engines even with aligned settings.

Game means pool all included games: total game duration divided by the
included game count (250 normally, 248 for the setting with two exclusions).
The 100/100/50 batches are weighted by their game counts. Per-call API means
use timed-call counts instead. [Pooled totals and denominators](profile-summary.csv)
make the computation explicit.

| Java/Rust time | Game | Players | Rust s/game | Java s/game | Rust actions/game | Java actions/game |
|---|---|---|---|---|---|---|
| 4.75 | Tic-Tac-Toe | 2 | 0.134 | 0.634 | 9.0 | 9.0 |
| 2.90 | Connect 4 | 2 | 2.292 | 6.658 | 31.4 | 32.0 |
| 4.91 | Can't Stop | 2 | 4.303 | 21.128 | 129.5 | 157.4 |
| 3.41 | Can't Stop | 3 | 8.416 | 28.682 | 179.3 | 208.4 |
| 3.65 | Can't Stop | 4 | 13.375 | 48.848 | 214.3 | 241.3 |
| 2.14 | Love Letter | 2 | 7.720 | 16.485 | 39.7 | 37.9 |
| 1.91 | Love Letter | 3 | 19.765 | 37.846 | 88.3 | 92.4 |
| 1.51 | Love Letter | 4 | 36.170 | 54.754 | 103.2 | 104.6 |

## Process memory while playing

Each value is the median across processes. Peak means the largest sampled value
during a process's measured games, with 50 ms polling. Shorter-lived peaks can be
missed. Baselines follow warmup. Private committed memory is storage committed
to the process; resident working set is currently resident RAM, including shared
pages. Existing available heap space can accommodate objects without increasing
process memory. These values include runtimes, search trees, allocators and
instrumentation; they are not state sizes.

| Java/Rust peak | Game | Players | Metric | Rust baseline MiB | Rust peak MiB | Java baseline MiB | Java peak MiB |
|---|---|---|---|---|---|---|---|
| 158.14 | Tic-Tac-Toe | 2 | Private committed | 1.70 | 2.73 | 360.73 | 431.17 |
| 59.86 | Tic-Tac-Toe | 2 | Resident working set | 4.87 | 6.32 | 296.95 | 378.10 |
| 183.07 | Connect 4 | 2 | Private committed | 1.11 | 2.92 | 372.16 | 534.92 |
| 72.38 | Connect 4 | 2 | Resident working set | 4.76 | 6.69 | 319.16 | 484.06 |
| 186.80 | Can't Stop | 2 | Private committed | 1.52 | 3.16 | 436.36 | 589.60 |
| 77.58 | Can't Stop | 2 | Resident working set | 4.63 | 7.03 | 382.60 | 545.45 |
| 181.20 | Can't Stop | 3 | Private committed | 1.05 | 3.53 | 462.46 | 639.86 |
| 80.50 | Can't Stop | 3 | Resident working set | 4.72 | 7.42 | 410.13 | 597.46 |
| 187.38 | Can't Stop | 4 | Private committed | 1.43 | 3.61 | 439.99 | 677.04 |
| 82.02 | Can't Stop | 4 | Resident working set | 5.05 | 7.48 | 386.60 | 613.86 |
| 160.98 | Love Letter | 2 | Private committed | 1.89 | 3.71 | 362.23 | 597.38 |
| 72.02 | Love Letter | 2 | Resident working set | 5.40 | 7.63 | 310.60 | 549.71 |
| 171.50 | Love Letter | 3 | Private committed | 1.93 | 3.98 | 370.16 | 682.66 |
| 76.53 | Love Letter | 3 | Resident working set | 5.58 | 7.93 | 317.32 | 606.60 |
| 158.45 | Love Letter | 4 | Private committed | 2.30 | 4.31 | 385.21 | 683.31 |
| 74.13 | Love Letter | 4 | Resident working set | 5.81 | 8.18 | 331.98 | 606.38 |

## Ratios by player count

[Speed and memory plots for Can't Stop and Love Letter](plots/ratios/README.md)
show Java/Rust ratios against the number of players, with equality at 1.

## Data and limits

[Settings](settings.json), [API summaries](api-summary.csv), [game/memory summaries](game-summary.csv)
and per-shard JSON files retain all measured games. Counts distinguish actual
API calls from shadow-copy samples. The original prepared-state study remains
a separate workload. Tiny per-call timings include timer overhead; process
timer controls and probe fractions are in the game summary. Rules-version-2
Rust Can't Stop resolves busts within Roll, while Java retains its forced bust
action and its other documented rule/knowledge differences. No language-wide
speed or memory multiplier follows from this experiment.
