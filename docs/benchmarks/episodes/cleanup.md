# Episode conversion cleanup: selected CPU workloads

Recorded 17-10-2026 22:44 (Europe/Paris). Raw [before](cleanup-before.json) and [after](cleanup-after.json) samples.

The selected cleanup builds metadata directly for each batch member. Previously
conversion constructed an empty payload dictionary and empty NumPy mask/action-ID
arrays, then removed them from that dictionary. Individual and batched returns
now share the player-metadata conversion helper. Final observations, counters,
rewards, flags, RNG and failure behavior retain their existing contract.

These release samples use the existing whole-workload runner: default boards,
Random opponents, learner seats alternating by member, cells payload, reset seed
700+member index and first legal actions. Exact transition/RNG checks run before
each case. Each route retains seven warmed samples of at least 200 ms; route
order alternates within each phase. Setup/reset, play, output handling and cleanup
are timed. Final-fact comparisons run outside timing.

| Game | Members | Route | Before ms | After ms | Before/after |
|---|---:|---|---:|---:|---:|
| tictactoe | 1 | individual | 0.152 | 0.144 | 1.05 |
| tictactoe | 1 | bulk | 0.375 | 0.319 | 1.17 |
| tictactoe | 128 | individual | 78.792 | 78.998 | 1.00 |
| tictactoe | 128 | bulk | 65.462 | 67.978 | 0.96 |
| connect4 | 1 | individual | 1.069 | 0.989 | 1.08 |
| connect4 | 1 | bulk | 1.729 | 1.534 | 1.13 |
| connect4 | 128 | individual | 109.377 | 96.428 | 1.13 |
| connect4 | 128 | bulk | 95.626 | 83.786 | 1.14 |

Results are mixed. At 128 members, Tic-Tac-Toe bulk time rose from about 65.5 to
68.0 ms, while Connect 4 bulk time fell from about 95.6 to 83.8 ms. Individual
Connect 4 improved by a similar proportion. These sequential before/after phases
also reflect machine variation: other load, CPU frequency and thermal state were
uncontrolled. Builds and regression suites ran outside the timed phases.

The retained change simplifies conversion and removes the temporary objects
described above. Further performance tuning should start with component profiling.
The [original collection](README.md) remains the dated 32-case baseline.
Q-table allocation/load cleanup and transition collection are outside these timings.
Peak memory remains unmeasured.

## Reproduce a phase

Prepare the corresponding source version and release extension, then run from the
repository root. The phase label identifies the build being measured.

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/episodes/cleanup.py --phase after-cleanup --output .scratch/m6-cleanup-repeat.json
```

The runner reuses [measure.py](../../../tools/benchmarks/episodes/measure.py).
Raw samples record interpreter/package versions, source base commit and phase.
The two recorded phases used uncommitted work on base 8c279ed.
