# OSLA benchmark results

Measured 2026-10-16T17:15:57.920451+00:00.

Windows-11-10.0.26300-SP0; Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; 20 logical CPUs.
Python 3.14.8, NumPy 2.5.4, rustc 1.99.0 (b940084d7 2026-09-28).

See [the method and interpretation](README.md) and [raw samples](results.json).

## Full-game averages

Batch limit 1,024: all candidates for one move fit in one batch. Times include setup.

| Board | Rust-only original ms | Rust-only batched ms | Python starts, Rust runs/scores ms | Python loop, Rust scoring ms | Python loop, Python scoring ms | Manual Rust scoring/owned execution |
|---|---:|---:|---:|---:|---:|---:|
| Tic-Tac-Toe 3x3 | 0.0263 | 0.1625 | 0.468 | 1.455 | 3.376 | 3.1x |
| Tic-Tac-Toe 5x5 | 1.0022 | 1.2603 | 1.742 | 4.442 | 31.025 | 2.6x |
| Connect 4 7x6, target 4 | 0.7425 | 0.6355 | 1.061 | 2.408 | 37.327 | 2.3x |
| Connect 4 4x3, target 3 | 0.0399 | 0.0806 | 0.318 | 0.818 | 1.986 | 2.6x |

## Isolated heuristic scoring

Microseconds per candidate. Rust states and candidate arrays are prepared before timing.
The new Rust-owned execution route uses the Rust-only batched scorer; its complete-game cost is measured above.

| Board | Rust-only original us | Rust-only batched us | Python calls Rust us | Python scores us | Python/Rust scoring-call ratio |
|---|---:|---:|---:|---:|---:|
| Tic-Tac-Toe 3x3 | 0.832 | 0.206 | 3.96 | 43.51 | 11.0x |
| Tic-Tac-Toe 5x5 | 1.894 | 0.790 | 2.30 | 76.44 | 33.2x |
| Connect 4 7x6, target 4 | 5.907 | 1.941 | 4.28 | 354.25 | 82.8x |
| Connect 4 4x3, target 3 | 1.163 | 0.287 | 4.12 | 57.49 | 14.0x |

## Batch limits: complete-game means

| Board | Route | Limit 1 ms | Limit 3 ms | Limit 1,024 ms |
|---|---|---:|---:|---:|
| Tic-Tac-Toe 3x3 | Rust-only batched | 0.176 | 0.164 | 0.163 |
| Tic-Tac-Toe 3x3 | Python starts, Rust runs/scores | 0.541 | 0.483 | 0.468 |
| Tic-Tac-Toe 3x3 | Python loop, Rust scoring | 4.401 | 2.255 | 1.455 |
| Tic-Tac-Toe 3x3 | Python loop, Python scoring | 5.850 | 3.978 | 3.376 |
| Tic-Tac-Toe 5x5 | Rust-only batched | 1.503 | 1.355 | 1.260 |
| Tic-Tac-Toe 5x5 | Python starts, Rust runs/scores | 2.483 | 1.986 | 1.742 |
| Tic-Tac-Toe 5x5 | Python loop, Rust scoring | 27.970 | 11.410 | 4.442 |
| Tic-Tac-Toe 5x5 | Python loop, Python scoring | 53.082 | 36.572 | 31.025 |
| Connect 4 7x6, target 4 | Rust-only batched | 0.698 | 0.702 | 0.636 |
| Connect 4 7x6, target 4 | Python starts, Rust runs/scores | 1.197 | 1.045 | 1.061 |
| Connect 4 7x6, target 4 | Python loop, Rust scoring | 8.231 | 4.550 | 2.408 |
| Connect 4 7x6, target 4 | Python loop, Python scoring | 43.949 | 37.751 | 37.327 |
| Connect 4 4x3, target 3 | Rust-only batched | 0.090 | 0.083 | 0.081 |
| Connect 4 4x3, target 3 | Python starts, Rust runs/scores | 0.341 | 0.325 | 0.318 |
| Connect 4 4x3, target 3 | Python loop, Rust scoring | 1.839 | 1.151 | 0.818 |
| Connect 4 4x3, target 3 | Python loop, Python scoring | 2.778 | 2.281 | 1.986 |

## Sample spread

Each range spans seven sample averages, each covering repeated sets of 16 games. Batch limit 1,024.

| Board | Route | Median ms | Sample min-max ms |
|---|---|---:|---:|
| Tic-Tac-Toe 3x3 | Rust-only original | 0.0169 | 0.0168-0.0825 |
| Tic-Tac-Toe 3x3 | Rust-only batched | 0.1613 | 0.1511-0.1849 |
| Tic-Tac-Toe 3x3 | Python starts, Rust runs/scores | 0.4646 | 0.4341-0.4973 |
| Tic-Tac-Toe 3x3 | Python loop, Rust scoring | 1.4445 | 1.3056-1.6258 |
| Tic-Tac-Toe 3x3 | Python loop, Python scoring | 3.3187 | 3.1482-3.6534 |
| Tic-Tac-Toe 5x5 | Rust-only original | 0.9997 | 0.8708-1.0961 |
| Tic-Tac-Toe 5x5 | Rust-only batched | 1.2878 | 1.1102-1.3502 |
| Tic-Tac-Toe 5x5 | Python starts, Rust runs/scores | 1.6987 | 1.6141-1.8810 |
| Tic-Tac-Toe 5x5 | Python loop, Rust scoring | 4.5207 | 3.6106-4.8962 |
| Tic-Tac-Toe 5x5 | Python loop, Python scoring | 31.4017 | 28.6818-33.1549 |
| Connect 4 7x6, target 4 | Rust-only original | 0.7505 | 0.6612-0.8073 |
| Connect 4 7x6, target 4 | Rust-only batched | 0.6441 | 0.5460-0.6749 |
| Connect 4 7x6, target 4 | Python starts, Rust runs/scores | 1.0671 | 0.8907-1.1583 |
| Connect 4 7x6, target 4 | Python loop, Rust scoring | 2.4269 | 2.0053-2.7786 |
| Connect 4 7x6, target 4 | Python loop, Python scoring | 37.8666 | 34.1727-39.5415 |
| Connect 4 4x3, target 3 | Rust-only original | 0.0418 | 0.0303-0.0472 |
| Connect 4 4x3, target 3 | Rust-only batched | 0.0810 | 0.0766-0.0844 |
| Connect 4 4x3, target 3 | Python starts, Rust runs/scores | 0.3220 | 0.3018-0.3290 |
| Connect 4 4x3, target 3 | Python loop, Rust scoring | 0.8167 | 0.8107-0.8285 |
| Connect 4 4x3, target 3 | Python loop, Python scoring | 1.9955 | 1.9414-2.0244 |

## Work per game

| Board | Mean moves | Mean candidates | Manual Python requests: limits 1 / 3 / 1,024 | Owned advance calls/game |
|---|---:|---:|---|---:|
| Tic-Tac-Toe 3x3 | 9.00 | 45.00 | 45.00 / 18.00 / 9.00 | 1 |
| Tic-Tac-Toe 5x5 | 25.00 | 325.00 | 325.00 / 117.00 / 25.00 | 1 |
| Connect 4 7x6, target 4 | 12.88 | 87.69 | 87.69 / 37.44 / 12.88 | 1 |
| Connect 4 4x3, target 3 | 5.00 | 20.00 | 20.00 / 10.00 / 5.00 | 1 |

## Separate Python call breakdown

Instrumented pass, mean milliseconds per game, batch limit 1,024. Includes timer overhead.

| Board | Python calls | Prepare | Score | Finish | Commit | Scoring share of these calls |
|---|---|---:|---:|---:|---:|---:|
| Tic-Tac-Toe 3x3 | Rust scorer | 0.479 | 0.248 | 0.346 | 0.090 | 21.3% |
| Tic-Tac-Toe 3x3 | Python scorer | 0.521 | 1.910 | 0.452 | 0.106 | 63.9% |
| Tic-Tac-Toe 5x5 | Rust scorer | 1.740 | 0.766 | 0.840 | 0.232 | 21.4% |
| Tic-Tac-Toe 5x5 | Python scorer | 2.157 | 24.581 | 1.657 | 0.314 | 85.6% |
| Connect 4 7x6, target 4 | Rust scorer | 0.792 | 0.446 | 0.412 | 0.128 | 25.1% |
| Connect 4 7x6, target 4 | Python scorer | 1.132 | 28.978 | 1.017 | 0.166 | 92.6% |
| Connect 4 4x3, target 3 | Rust scorer | 0.225 | 0.108 | 0.170 | 0.046 | 19.6% |
| Connect 4 4x3, target 3 | Python scorer | 0.284 | 1.314 | 0.233 | 0.053 | 69.7% |
