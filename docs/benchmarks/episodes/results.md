# Individual and bulk learner episodes: CPU measurements

Recorded 17-10-2026 00:00 (Europe/Paris). Raw samples: [results.json](results.json).

Times are mean milliseconds per complete workload, including environment construction,
seeded reset, learner action selection, native opponent play, output handling, final
facts/checkpoint extraction and close. Each workload completes the listed number of
independent games. A ratio above 1 means bulk took less time in this run.

| Game | Rust opponent | Environments | Individual ms | Bulk ms | Individual/bulk | Step calls individual/bulk |
|---|---|---:|---:|---:|---:|---:|
| tictactoe | random | 1 | 0.128 | 0.272 | 0.47 | 5/5 |
| tictactoe | random | 2 | 1.327 | 1.804 | 0.74 | 7/5 |
| tictactoe | random | 4 | 2.573 | 2.713 | 0.95 | 14/5 |
| tictactoe | random | 8 | 5.181 | 5.243 | 0.99 | 28/5 |
| tictactoe | random | 16 | 10.025 | 9.453 | 1.06 | 53/5 |
| tictactoe | random | 32 | 18.258 | 17.485 | 1.04 | 106/5 |
| tictactoe | random | 64 | 37.819 | 34.182 | 1.11 | 209/5 |
| tictactoe | random | 128 | 76.383 | 64.029 | 1.19 | 413/5 |
| tictactoe | osla_python | 1 | 0.758 | 1.129 | 0.67 | 3/3 |
| tictactoe | osla_python | 2 | 1.453 | 1.803 | 0.81 | 5/3 |
| tictactoe | osla_python | 4 | 3.057 | 3.368 | 0.91 | 10/3 |
| tictactoe | osla_python | 8 | 4.814 | 4.694 | 1.03 | 20/3 |
| tictactoe | osla_python | 16 | 11.106 | 9.731 | 1.14 | 40/3 |
| tictactoe | osla_python | 32 | 20.178 | 16.709 | 1.21 | 80/3 |
| tictactoe | osla_python | 64 | 43.949 | 36.677 | 1.20 | 160/3 |
| tictactoe | osla_python | 128 | 89.214 | 71.856 | 1.24 | 320/3 |
| connect4 | random | 1 | 1.000 | 1.584 | 0.63 | 7/7 |
| connect4 | random | 2 | 1.634 | 2.341 | 0.70 | 12/7 |
| connect4 | random | 4 | 3.693 | 4.395 | 0.84 | 27/11 |
| connect4 | random | 8 | 5.683 | 6.419 | 0.89 | 45/11 |
| connect4 | random | 16 | 11.447 | 11.377 | 1.01 | 89/11 |
| connect4 | random | 32 | 24.552 | 22.602 | 1.09 | 177/12 |
| connect4 | random | 64 | 50.051 | 39.375 | 1.27 | 343/12 |
| connect4 | random | 128 | 104.895 | 87.564 | 1.20 | 701/13 |
| connect4 | osla_python | 1 | 1.145 | 1.564 | 0.73 | 5/5 |
| connect4 | osla_python | 2 | 2.306 | 2.787 | 0.83 | 11/6 |
| connect4 | osla_python | 4 | 4.325 | 4.598 | 0.94 | 20/6 |
| connect4 | osla_python | 8 | 8.097 | 7.623 | 1.06 | 36/6 |
| connect4 | osla_python | 16 | 15.346 | 13.068 | 1.17 | 75/6 |
| connect4 | osla_python | 32 | 31.987 | 27.829 | 1.15 | 154/7 |
| connect4 | osla_python | 64 | 63.201 | 50.718 | 1.25 | 304/7 |
| connect4 | osla_python | 128 | 125.183 | 105.114 | 1.19 | 596/7 |

Seven warmed samples per route normally accumulate at least 200 ms each; raw
settings/samples record the exact run. Route order alternates. Exact per-transition
boards, results, counters and opponent RNG checkpoints are compared before timing;
every timed repetition also checks final facts outside its timed interval.

Step calls count collector step entry points, excluding reset/creation/close and
internal implementation calls. Bulk executes members sequentially in Rust, without
threads, GPU inference or automatic reset. Scalar-info construction and Python seed
coordination remain per member. Stacking copies numeric payloads; no zero-copy or
universal speedup is claimed. Peak memory and component timings are unmeasured.

Learners alternate seats, always choose the first legal ID, and use reset seeds
700+member_index. Payload is cells. Each individual/bulk pair uses the same settings
and seed pairs; the raw records include total learner steps and underlying moves.
