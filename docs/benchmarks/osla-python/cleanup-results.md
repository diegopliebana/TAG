# External-evaluation cleanup: targeted CPU measurements

Measured 16-10-2026 21:05 (Europe/Paris). [Raw before/after samples](cleanup-results.json).

These measurements check the effect of the final external-evaluation refactoring on four
cheap-scoring workloads. Each workload runs 16 or 64 independent default games,
using a constant NumPy evaluator, cells input and the same controller seeds as
the scaling experiment. Both routes validate final facts/checkpoints against
individual execution at every repetition. Seven warmed samples per route each
accumulate at least 200 ms of execution. The table shows grouped execution only;
raw samples also retain the individual route as a reference.

Each time covers the entire set of games, including setup, play, extracting final
facts/checkpoints and cleanup. The time ratio divides before time by after time.
Values above 1 mean the measured refactored workload was faster; values below 1
mean it took longer.

| Game | Games in workload | Before cleanup (ms) | After cleanup (ms) | Time ratio: before / after |
|---|---:|---:|---:|---:|
| tictactoe | 16 | 2.374 | 2.636 | 0.90 |
| tictactoe | 64 | 39.197 | 33.119 | 1.18 |
| connect4 | 16 | 27.695 | 24.532 | 1.13 |
| connect4 | 64 | 97.212 | 85.038 | 1.14 |

The result is mixed: three workloads became faster in this run, and the smallest
Tic-Tac-Toe workload became slower. CPU/interpreter variation and different
coordination/metadata costs affect these small workloads. This pass therefore
makes no claim of a universal performance improvement. Its concrete changes
remove repeated format construction, accumulated-score copies and duplicate
bookkeeping while making the implementation easier to follow.

Retaining format descriptions adds persistent metadata storage per game; it
avoids reconstructing the same line indices on every grouped size check. Peak
memory was not measured. The intermediate-score change helps multi-chunk
decisions; these particular default-board workloads fit within one chunk, so
the table does not isolate that change's performance effect.

The earlier [scaling report](scaling-results.md) and five-route measurements are
preserved snapshots from before this cleanup. Run
`.venv/Scripts/python.exe tools/benchmarks/osla/grouped.py --scaling` to obtain
fresh measurements of the current implementation. The same `measure()` helper
produced these saved before/after samples.
