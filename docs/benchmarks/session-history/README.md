# Session history and simulation measurements

## Reading the measurements

A workload is the specific operation being measured, such as copying a state or
replaying a complete game. A harness is the program that repeats and times it.
The median is the middle measured value after sorting; it describes these samples,
not a guaranteed time for a future machine. Cache capacity is how many rebuilt
states are kept to avoid repeated work. A checkpoint interval says how many moves
separate eligible cached positions. Read the units and included/excluded work
before comparing numbers. Raw samples, commands and historical results below are
unchanged. See [the glossary](../../reference/framework/glossary.md) for engine terms.


Measured 11-10-2026 23:05 (Europe/Paris), Windows x64, Rust 1.99.0, release build.
This report records session cleanup measurements. Current APIs are described in
[recordings and branches](../../reference/persistence/recordings.md).

## Reproduce

```powershell
cargo run --offline --release --example history_profile
```

The [harness](../../../examples/history_profile.rs) creates deterministic
first-available matches, records accepted moves and reconstructs every position
in forward or reverse order. Each traversal starts with a fresh empty cache;
reverse traversal first reconstructs the last position. Compare capacities of
4 and 64 with checkpoint interval 16. Recording setup is outside traversal timing.
Every returned state is passed through black_box. No browser, I/O, animations or
Java runtime is included in the timings.

It also times 100,000 independent initial-state copies, 100 unrecorded complete
first-available games, and 100 initial OSLA decisions. Each OSLA decision evaluates
all offered candidates; one persistent seeded agent is used for that measurement.
OSLA input observation/action construction is outside that decision timing.
Game execution includes offered observations/actions and rules. These workloads
have different units and should not be treated as a breakdown of one another.

Three post-build runs are saved as [sample 1](sample-1.txt), [sample 2](sample-2.txt)
and [sample 3](sample-3.txt). The table shows median milliseconds. It is a local
exploratory measurement, with visible timing variation and no claim of production
latency or improvement over the previous engine implementation.

| Workload | Tic-Tac-Toe 32x32 | Connect 4 32x32, target 32 |
|---|---:|---:|
| Accepted moves per match | 993 | 993 |
| Forward traversal, capacity 4 | 73.075 | 95.637 |
| Backward traversal, capacity 4 | 123.437 | 181.652 |
| Backward traversal, capacity 64 | 72.585 | 97.336 |
| 100,000 initial-state copies | 5.310 | 22.348 |
| 100 complete first-available games | 1071.213 | 1232.891 |
| 100 initial OSLA decisions | 2060.576 | 91.059 |

Standard configurations are also measured in the samples. The two large OSLA
workloads differ: Tic-Tac-Toe offers 1024 candidates and Connect 4 offers 32.
Their timings cannot establish relative framework efficiency.

## Conclusions and limits

Larger cache capacity reduces repeated backward reconstruction in these long
matches. Capacity 64 can retain all 62 interval checkpoints of a 993-move match;
capacity 4 repeatedly discards earlier checkpoints. The extra state copies cost
memory. Keep the portable default at 4 and let callers select a limit appropriate
to game state size. Browser byte limits still need adapter implementation and
measurement; a state-count limit alone cannot establish a fixed memory budget.

Controller restoration now copies one final snapshot per seat. A counted-Clone
test verifies that operation count independently of noisy timing measurements.
No before/after speedup claim is made for this cleanup or OSLA allocation removal.

State-copy, full-game and OSLA timings identify useful follow-up workloads, but do
not isolate the cost of win scans, draw scans, evaluations or offered-action
construction. Keep game-rule algorithms and state representations unchanged until
focused profiling can attribute costs and tests can preserve winning-line ordering.
The temporary agent is still restored again during AI commitment; changing that
requires a separate cost measurement and consideration of proposal ownership.
