# Research benchmark tools

Plot exports share `figure_layout.py`, which places deduplicated legends in a
reserved band above the plotting area, with no visible figure titles.

These tools measure the implemented game APIs, agent families, adapters and
memory use. Start with the [current measurement report](../../../docs/benchmarks/research/README.md)
and [workload definitions](workloads.json). This is a Windows CPU study, with
release Rust and a warmed Java 21 JVM. It measures execution cost, not playing strength.

## Requirements

Complete the [tool setup](../../../docs/guides/setup.md). Use Rust, JDK 21, Maven,
Python, the release Python adapter, NumPy, CPU PyTorch and Matplotlib. If plotting
support is missing, install it in your own terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

The recorded package versions are in the report's environment manifest. These
drivers use Windows process-memory APIs. Native browser-adapter measurements run
as ordinary host programs; they include no Web Worker transport or rendering.

Build Java only in a disposable writable copy at `.scratch/java-comparison`.
Reuse an existing copy/classpath if available. For a fresh setup, first copy your
public TAG checkout there; do not build in a read-only reference. With the public
checkout available at the relative path below, and the destination absent:

```powershell
if (Test-Path .scratch/java-comparison) { throw 'Reuse or choose a different disposable copy.' }
New-Item -ItemType Directory -Force .scratch | Out-Null
Copy-Item -LiteralPath ../references/tag-public -Destination .scratch/java-comparison -Recurse
Push-Location .scratch/java-comparison
mvn -DskipTests compile
mvn dependency:build-classpath '-Dmdep.outputFile=target/comparison-classpath.txt'
Pop-Location
```

Replace the source path with your public checkout if needed. Initial dependency
downloads require network access in your terminal. Stop on build failures; do not
substitute estimates for missing results. The measured source hashes identify the
reference used here; another public revision is a different benchmark condition.

## Build the probes

From the repository root, in PowerShell:

```powershell
New-Item -ItemType Directory -Force .scratch/research-bench | Out-Null
cargo build --release --features mcts-json,replay --example research_measure
Copy-Item target/release/examples/research_measure.exe .scratch/research-bench/rust-timing.exe
cargo rustc --release --features mcts-json,replay --example research_measure -- --cfg benchmark_alloc
Copy-Item target/release/examples/research_measure.exe .scratch/research-bench/rust-memory.exe
cargo build --release --example research_features
Copy-Item target/release/examples/research_features.exe .scratch/research-bench/minimal-features.exe
cargo build --release --features mcts-json,replay,inference --example research_features
Copy-Item target/release/examples/research_features.exe .scratch/research-bench/optional-features.exe
cargo build --release -p tag-browser --example research_adapter
cargo build --release --features mcts-json,replay,inference --example match_play --example tournament
$javaRoot = Join-Path $PWD '.scratch/java-comparison'
$javaDependencies = (Get-Content "$javaRoot/target/comparison-classpath.txt" -Raw).Trim()
$javaClasspath = "$javaRoot/target/classes;$javaDependencies"
javac -proc:none -implicit:none -sourcepath "$javaRoot/empty-source" -cp $javaClasspath -d "$javaRoot/target/classes" tools/java-comparison/BenchmarkResearch.java tools/java-comparison/BenchmarkTreeMemory.java
```

`--release` enables optimization; it is essential for interpreting these as
performance measurements. `--cfg benchmark_alloc` selects a counter-equipped
allocator in this probe only. Its timings are not used as release timing results.
The `--` separator passes compiler options through Cargo to rustc. `-p` selects
the browser package; `--features` selects optional engine capabilities.
[Command syntax](../../../docs/guides/setup.md#understanding-the-commands) explains
these options. Add `--offline` to Cargo commands only when all dependencies are
already cached and network access should be disabled.

The Java compiler's `-d` directs generated classes into the disposable build;
`-cp` supplies compiled TAG classes and dependencies. `-sourcepath` prevents an
implicit rebuild of unrelated sources; `-proc:none` disables annotation processors
that the measurement classes do not use. Check compilation and the smoke runs:
the recorded environment emitted an archive-close access diagnostic despite
returning exit code 0 and producing working classes.

## Run a complete collection

Use a new output directory; existing primary sample files are preserved:

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/research/run.py --phase timing --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/continue.py --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/extras.py --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/extras.py --episodes 1 --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/extras.py --episodes 2 --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/extras.py --episodes 3 --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/tree_memory.py --output .scratch/research-bench/new-results
.\.venv\Scripts\python.exe tools/benchmarks/research/summarize.py .scratch/research-bench/new-results
```

Run phases sequentially so one workload does not compete with another for the
CPU. `continue.py` expects the complete three-process, seed-42 timing collection;
it runs allocation, retained-state, Python/model, native-adapter, optional-feature
and recorded-corpus phases. A completed phase is reused on continuation; a partial
phase requires inspection and a fresh output directory. The additional commands
collect CLI/episode/search-memory detail, then regenerate all plots.

Models are small, deterministic **untrained fixtures**, generated under
`.scratch/research-bench/models`. Full-vector MCTS models and discounted scalar
OSLA models have different contracts. No existing trained model is required.
Matplotlib's cache also stays under `.scratch/research-bench`.

## Driver options

| Program | Arguments and defaults |
|---|---|
| `run.py` | Required `--phase timing`, `allocations` or `retention`; `--output docs/benchmarks/research`; `--forks 3` (1-10); `--seed 42` or `auto`; optional `--engine rust` / `java` and `--operation` for a restricted run. `auto` resolves one OS-random nonnegative Java-long seed and records it. The published full collection uses the fixed fixture seed 42. |
| `continue.py` | `--output docs/benchmarks/research`; expects the complete fixed seed-42 timing set and the prebuilt probes. |
| `python_measure.py` | Required integer `--fork`; `--output docs/benchmarks/research`; `--models .scratch/research-bench/models`; optional `--game tictactoe`, `connect4`, `cantstop`, `loveletter` and `--players 2`, `3`, `4` restrict the profiles. One invocation is one fresh-process repetition. |
| `extras.py` | `--output docs/benchmarks/research`; omission of `--episodes` measures CLI schedules; `--episodes N` records learner collection for fresh-process repetition N. |
| `tree_memory.py` | `--output docs/benchmarks/research`; three fresh-process repetitions at 32 and 128 iterations for all eight game/player profiles. |
| `summarize.py` | Optional positional directory, default `docs/benchmarks/research`; validates sample coverage and writes summaries plus SVG/PDF/PNG plots. |

The low-level Rust/Java rule probe takes eight positional arguments:

```text
game players operation count seed iterations warm_ms sample_ms
```

`game` is one of the four IDs above; placement games require two players,
stochastic games support 2-4. `count` is a minimum operation count per timed batch,
or the retained-state count for `retain_states`. `seed` is a resolved integer;
`iterations` is the MCTS budget. Warmup and minimum sample duration are milliseconds.
Timing executes blocks of 32 operations, so a batch can exceed its minimum count.
There are five timed batches. The supported operations and their included work
are defined in `workloads.json`; Java has no Rust-only encoding/persistence modes.
For example:

```powershell
.\.scratch\research-bench\rust-timing.exe cantstop 3 mcts 1 42 32 150 80
```

`research_features` takes `game players seed` and waits at two protocol checkpoints
for the parent sampler. `research_adapter` takes portable model-file paths; each
model specifies its game and player count. `BenchmarkTreeMemory` takes
`game players iterations`, using fixture seed 42. These are orchestration probes,
not interactive game commands.

## Validate the analysis

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools/benchmarks/research -p test_summary.py -v
.\.venv\Scripts\python.exe tools/benchmarks/research/validate.py
```

The analysis tests check normalization, independent-process aggregation and
rejection of duplicate process IDs. Validation checks workload coverage, actual
iteration counts, paired outcome checksums and the standalone plot exports.

## Complete remeasurement and workflow diagnostics

The [current data index](../../../docs/benchmarks/research/data.md) lists the
complete sequential collection commands. Use a fresh output directory; do not
overwrite the earlier samples. Rebuild all Rust probes/CLI binaries and the Java
harnesses before collection, and verify the release Python adapter matches source.

`workflows.py --output DIRECTORY --phase timing` records three process repetitions
of prepared-state/component probes and complete-game controls. `--phase allocations`
uses the separately instrumented Rust binary and one allocation repetition.
The fixture seed is 42, profiles are the same eight game/player combinations and
prefixes are 0, 8 and 128 additional actions, capped before game completion.
The optional `@N` suffix on a low-level operation selects that prepared prefix.

| Additional operation | Included work |
|---|---|
| `game_direct` | Complete games with direct uniform choice and no observation. Diagnostic only; forced choices consume no selection draw, matching the actual agents. |
| `game_aligned`, `game_aligned_direct` | Two-player Love Letter with both winning-token targets set to six, with/without observations. Only Java's configurable target differs from its default. |
| `validate` | Enumerate the state-derived legal list and test membership of the prepared action. |
| `apply` | Apply a checked action to prepared owned inputs; cloning those inputs is outside each timed block. |
| `apply_control` | Opening clock/loop control over 32,000 calls (1,000 blocks), without a transition. Never subtracted. |
| `history_copy` | Love Letter retained event/private containers in Rust, action/text lists in Java; contents differ. |

The ordinary `game` mode uses actions from state, observation/player-copy creation
and the actual Random agent in **both** engines. Each measured block covers the
same 32 controller seeds, 42-73, within that implementation. `game_work` records
measured games/actions/logical turns/rounds/observation calls, excluding warmup.
The original game harness used asymmetric paths and a different choice stream.

After collecting all phases:

```powershell
python tools/benchmarks/research/workflow_summary.py docs/benchmarks/research
python tools/benchmarks/research/plot_workflows.py docs/benchmarks/research
python tools/benchmarks/research/report_remeasurement.py docs/benchmarks/research
python tools/benchmarks/research/validate.py docs/benchmarks/research
python tools/benchmarks/research/validate_workflows.py docs/benchmarks/research
python -m unittest discover -s tools/benchmarks/research -p 'test_*.py' -v
```

Each analysis command takes one required report-directory argument; plot output
is under its `plots/workflows/`. Summaries provide JSON and CSV. The five workflow
figures supplement the twelve main figure categories. The validator checks
the complete matrix, normalized counts, fixed CLI identities, source hashes and
all 15 supplemental exports. See the methodology for control and GC limits.
