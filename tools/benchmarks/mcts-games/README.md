# Measure complete MCTS games and encountered APIs

These probes play complete games with native MCTS in every seat. Rust uses a
release build; Java uses public TAG classes in a disposable instrumented build.
Rules and adapters in the public reference are never edited or built in place.

## Setup and run

Use the Rust/JDK/Python setup and disposable Java build described in
[research tools](../research/README.md). No PyTorch or neural model is involved.
Compile the measurement probes from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/mcts-games/build.py
```

`run.py` requires explicit choices for `--aligned`, `--seed-copies` and
`--skip-forced`; each accepts `true` or `false`. For example, this is a small
smoke test using aligned search settings and seeded Java copy RNGs:

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/mcts-games/run.py --output .scratch/mcts-games-smoke --games 1 --shards 1 --iterations 32 --warmup 1 --aligned true --seed-copies true --skip-forced true
```

For a full run, choose a new output directory, `--games 500 --shards 5
--iterations 5000 --warmup 6`, and the explicitly selected Boolean options.
The eight profiles are Tic-Tac-Toe/Connect 4 with two players, and Can't Stop/
Love Letter with 2–4. There are 500 games per engine/profile, split across five
fresh processes of 100 games. Engines run sequentially, with order alternating
between shards; no games from different engines run concurrently.

Completed shards are reusable only with identical measurement settings; collection
counts can change through the explicit resize procedure below. Interrupted shard
progress remains a diagnostic log; it is not counted as a complete shard.
Use `--engine`, `--game` or `--players` for restricted validation/pilots.
`--seed 42` is the default; `auto` resolves OS randomness once and records it.
Rollout length is fixed at 10, with random rollout choices and game-native values.

To reduce an existing collection without rerunning its completed 100-game batches,
stop its collector after the current batch commits, then resume with `--games 250
--shards 3 --shard-sizes 100,100,50 --resize` and the same measurement arguments.
`--resize` permits collection-count changes only; it preserves the initial settings
as `original-settings.json` and leaves existing result identities unchanged.
Seeds for later batches follow cumulative batch sizes, preserving the first 250
game seeds. The report states the actual batch sizes and included sample counts.
The `resize_after_batch.py` controller performs this transition for the active
recorded run; its target process and batch are explicit in the source/status file.
It uses an owned directory at the next stderr-log path to stop the old collector
before another process launches, then removes that gate before resuming. This
requires no process-tree enumeration or expanded permissions.

## Search and rules

`--aligned true` uses UCB/UCT with exploration constant sqrt(2), final selection
by visits and a maximum tree depth of 1,000 in both engines. `false` preserves
the implementations' other defaults. Tree reuse is disabled; agents have
independent persistent RNGs across their decisions. Java's Love Letter search
uses `Information_Set`; the other Java games use `Open_Loop`. Rust uses its
existing observation-derived simulation forks. Their algorithms, heuristics,
knowledge and stochastic trajectories are not claimed equivalent.

`--skip-forced true` executes the sole offered action without search in both
runners. `false` runs the configured iterations even in those cases; only the
disposable Java MCTS wrapper's early return is disabled. Every searched decision
must complete the configured iterations. Reports distinguish accepted actions
and searched decisions. Warmup uses the same budget at positions with choices.

Configure both frameworks for 3×3 Tic-Tac-Toe, 7×6 Connect 4, Can't Stop heights
3/5/7/9/11/13/11/9/7/5/3 with three markers/three columns to win, and classic Love
Letter with winning-token targets 6/5/4. This aligns configuration, not all rules:
for example, public TAG still offers a forced bust action and can pass at the
start of a Can't Stop turn. Rust resolves busts inside Roll. Identical seed
numbers do not produce identical games across implementations.

`--seed-copies true` explicitly initializes Java's otherwise automatically seeded
simulation-copy RNGs in the disposable build, independently of the game RNG.
The seed derivation and code are recorded; `false` retains Java's initialization.
Game and agent seeds derive from the recorded master and game/seat indices.

## API measurements

Final per-game means pool totals across all included games: divide by 250 for a
complete setting, or by its actual included count after exclusions. Do not average
the 100/100/50 batch means equally. API latencies use total sampled time divided
by the number of timed calls; process-memory peaks remain explicitly labelled
process statistics. `profile-summary.csv` records game totals and denominators.

Counters separate real execution from search. API-boundary calls are counted;
internal rule helpers are included in their enclosing call, not counted again.
Real-execution calls are timed throughout each game. Search call latency is
sampled for the first 16 calls per game/category, then every 1,024th call by
default. `--stride` changes that search interval; zero disables
timing/probes for instrumentation controls. Timings include a two-clock overhead;
an empty timer measurement is recorded, not automatically subtracted.

Categories are initialization, available actions, exact copy, observer/player
copy and copy+apply. Initialization occurs only for actual games; search does
not initialize a fresh game at every simulation. Exact copy and copy+apply are
also probed on encountered pre-transition states because the native algorithms
do not necessarily execute those compound operations themselves. Those probes
are separately counted samples, not invented actual API calls. `actual_copy`
separately records explicit exact-copy API calls; copies inside a fork or
observation remain included in those enclosing operations. Rust reconstructs
simulation worlds from observations; their cost is additionally recorded as
`fork`, independently of an exact copy. Isolated in-place `apply` is also reported.

Shadow probes apply the same action to an independent full copy, never the live
or searched state. Java temporarily substitutes a probe-only copy RNG and restores
the source's generator; probe construction does not advance the benchmark's
actual copy-RNG seed sequence. Probe recursion is excluded from counters. Rust
uses independent exact clones. Outputs escape optimization; destruction/GC is
outside the per-call copy+apply timer. Instrumented complete-game times include
probe/counter overhead and are labelled accordingly; probe durations are retained.

## Memory

The parent samples the same Windows process counters for both engines every
50 ms: private committed bytes and resident working set. The baseline is after
untimed warmup, before the game group. Each game's sampled peak is retained, as
are OS process-lifetime high-water counters (which can include warmup).
Processes wait after emitting each result so the final sample is taken before
they exit. Java runs with `-Xms64m -Xmx512m`; a pre-baseline GC discards warmup
objects. Normal GC remains enabled during games. Rust has no tracing collector.
These are instrumented process footprints, not isolated state sizes or exact
instantaneous per-game peaks. A 50 ms sampler can miss short-lived peaks.

## Files

`settings.json` owns the shared settings. One JSON file per engine/profile/shard
contains per-game work, API samples, baseline and process-memory peaks. Partial
progress and stderr logs are separate. Preserve pilot/control outputs outside
the final result directory. Do not merge incomplete shards into results.

Before acknowledging the process's ready message, the runner atomically saves
`*.baseline.json` with its identity, command, warmup timing and memory baseline.
Each completed game is flushed to disk before allowing the next game to start.
Run the same collection command again after an interruption: the runner retains
complete records and starts a warmed process at the next unrecorded game seed.
Only an unfinished final log record is trimmed; gaps, duplicates and changed
settings cause an explicit error. Game IDs remain continuous across processes.
Each restarted process has its own saved baseline and its own memory peaks;
`process-memory-summary.json` lists these segments and the memory table takes
the median across actual processes. Per-game means still pool included games.
Partial records created before baseline checkpointing require archiving and
recollection because their process baseline cannot be reconstructed.

An optional `exclusions.json` lists explicitly approved interrupted games by
engine, game, player count, shard, zero-based game ID and seed, with a reason.
The reporter preserves raw records, validates each exclusion against its seed,
and removes the whole game from API/game/memory statistics. It reports collected,
included and excluded counts separately. It does not replace games or automatically
discard slow games. A setting may therefore have fewer included games than collected.
