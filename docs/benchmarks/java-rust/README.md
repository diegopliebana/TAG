# Java TAG / Rust engine speed comparison

## Reading the measurements

A workload is the specific operation being measured, such as copying a state or
replaying a complete game. A harness is the program that repeats and times it.
The median is the middle measured value after sorting; it describes these samples,
not a guaranteed time for a future machine. Cache capacity is how many rebuilt
states are kept to avoid repeated work. A checkpoint interval says how many moves
separate eligible cached positions. Read the units and included/excluded work
before comparing numbers. Raw samples, commands and historical results below are
unchanged. See [the glossary](../../reference/framework/glossary.md) for engine terms.


Recorded 11-10-2026 00:22 (Europe/Paris). User authorised comparing rules and
copying. The two engines replay the same recorded legal games, with
matched board outcomes and available-action counts.

## Results

Values are medians of 15 batch averages: five batches in each of three fresh
processes per engine. Each batch repeats the 32-game corpus 500 times.
Times below are microseconds per call, except replay, which is per complete game.
The ratio is Java time divided by Rust time.

| Game | Operation | Rust microseconds | Java microseconds | Java/Rust time |
|---|---|---:|---:|---:|
| Tic-Tac-Toe 3x3 | Initialize | 0.045 | 1.925 | 43.0x |
| Tic-Tac-Toe 3x3 | Available actions | 0.087 | 0.146 | 1.7x |
| Tic-Tac-Toe 3x3 | Copy state | 0.029 | 3.615 | 123.8x |
| Tic-Tac-Toe 3x3 | Apply move | 0.080 | 0.812 | 10.1x |
| Tic-Tac-Toe 3x3 | Complete replay | 1.089 | 7.171 | 6.6x |
| Connect 4 7x6/target 4 | Initialize | 0.050 | 3.803 | 76.3x |
| Connect 4 7x6/target 4 | Available actions | 0.096 | 0.360 | 3.8x |
| Connect 4 7x6/target 4 | Copy state | 0.043 | 8.010 | 184.9x |
| Connect 4 7x6/target 4 | Apply move | 0.744 | 3.498 | 4.7x |
| Connect 4 7x6/target 4 | Complete replay | 16.341 | 44.427 | 2.7x |

Full medians, minimum/maximum batch values and operation counts are in
[summary.tsv](summary.tsv). Every sample and warmup count is preserved in
rust-fork-1.tsv through rust-fork-3.tsv and java-fork-1.tsv through java-fork-3.tsv.
No samples were discarded. These results describe the current engine APIs and
these workloads. Framework features contribute to the costs as described below.

## Matched workload and validation

[benchmark-corpus.tsv](../../../tools/java-comparison/benchmark-corpus.tsv)
contains 32 terminal Tic-Tac-Toe histories selected evenly across the recorded
terminal boards, plus all 32 default Connect 4 histories in its recorded corpus.
The histories were derived from the previously validated fixtures; their final
board and player-0 result are included. Both harnesses validate those fields
before timing. Every pre-move prefix is ongoing.

| Corpus | Complete games | Pre-move prefixes/transitions | Total available actions over prefixes |
|---|---:|---:|---:|
| Tic-Tac-Toe | 32 | 221 | 1,318 |
| Connect 4 | 32 | 648 | 4,381 |

All six processes agree on these counts. The summary script checks all 300 sample
rows, matching counts, sample counts and operations per batch before summarising.
All 78 normal Rust tests, formatting, offline Clippy with warnings denied and
all-target WebAssembly compilation passed after adding the harness.

Corpus SHA-256:
`ef3fb224b58a2fcd856dfb7a9e2654a79ec9ad521876f18aadf260bda4b9ddfd`.

## Timing procedure

Each operation receives at least one second of warmup and at least 200 corpus
passes. Timed batches run sequentially. Process order alternates: Rust/Java,
Java/Rust, Rust/Java. Corpus parsing, prefix preparation and result validation
are outside the measured intervals. Printing is outside the measured intervals.
Parameters are prepared once and reused for initialization in both engines.

- Initialize: construct a state and run public setup/init_state.
- Available actions: generate the action list for each prepared ongoing prefix.
- Copy: call public full-state copy() or Rust state.clone() for each prefix.
- Apply move: create fresh prefix copies before starting the timer, then apply
  each prefix's preselected recorded action exactly once.
- Replay: initialize a state, generate available actions and locate/apply each
  recorded move until the complete history has been played.

Rust uses std::hint::black_box to retain results and Java uses a volatile object
sink. Their overhead is included and differs between runtimes. Rust drops
temporary objects during measured execution; Java allocation and garbage
collection follow JVM lifecycle behaviour. Transition preparation copies are
outside its timer, but their allocations can contribute to later Java collections.
Timer boundaries occur once per corpus pass. Java is run with a 256 MiB initial
heap and 512 MiB maximum heap, using other default JVM settings.

These are small manual benchmarks. Batch averages and ranges show timing
variation; broader hardware and workload coverage would be needed for general
claims. Warmup is an explicit measured procedure; compiler/GC stability has not
been established through a dedicated profiler or a JMH harness.

## Implementation differences

Both engines use their normal public APIs. Java TAG's next() records action
history and human-readable action text. Its copy() includes parameters, component
bookkeeping, player timers, histories and branch RNG creation. Grid storage uses
framework components. Java's competitionMode remains false, the public default.

Rust currently owns compact grid values, outcomes and progress counters, with
checked actions. These deterministic game states have a smaller set of framework
fields. The large copy/init ratios include this difference in stored information
and framework machinery. Future history, chance and experiment capabilities may
change Rust's costs. These measurements establish a baseline for that evolution.

An OSLA speed comparison would additionally require matched evaluators, planning
inputs and tie policy. This task measures engine rules and full-state copies.
The earlier OSLA measurements use different workloads and include search work.

## Versions and provenance

- Rust engine checkout: 6b8d1b1a217fd6816836c550130c0c82bccd5052, with the new
  benchmark tooling added to the working tree. Engine rule implementations unchanged.
- rustc 1.99.0 (b940084d7), LLVM 23.1.1, x86_64-pc-windows-msvc; Cargo default
  release profile. Cargo.lock retains the previously selected dependencies.
- Java 21.0.12+7-LTS-205, Java HotSpot 64-Bit Server VM.
- Existing disposable public TAG build, baseline commit
  d65ad79f04992e238663c37c812dc2c5f6ce7ff2. Earlier provenance is in
  [source notices](../../source-notices.md) and the recorded fixtures.
  The build was reused; this task did not independently verify its original Git
  commit or rebuild the framework. Corpus validation passed against these classes.
- javac emitted the benchmark class and exited successfully, while again printing
  AccessDeniedException when closing the Netty Linux/aarch64 archive in the Maven
  cache. All benchmark processes subsequently completed successfully.
- CPU/OS-detail queries were unavailable under the existing environment permissions.

## Reproduce

From the project root, with the existing disposable Java build/classpath present:

```powershell
./tools/java-comparison/run-benchmarks.ps1 -Passes 500 -Forks 3
python tools/java-comparison/summarize-benchmarks.py .scratch/java-comparison/benchmark-results --forks 3
```

The [runner script](../../../tools/java-comparison/run-benchmarks.ps1) builds only
the Rust example and Java benchmark class, then runs sequential processes and
saves new results under .scratch. It confines output to the project workspace.
The [Rust harness](../../../examples/java_comparison_measure.rs) and
[Java harness](../../../tools/java-comparison/BenchmarkRules.java) contain the
matching workloads. The public reference checkout remains read-only.
