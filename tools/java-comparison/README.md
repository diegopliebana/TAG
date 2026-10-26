# Java/Rust comparison tools

The comparison tools replay known action sequences in Java and Rust. A corpus
is the collection of those sequences; a fixture is saved comparison data. Each
comparison checks the boards, actions and results for the specified sequences.
Additional sequences and tests are needed to check other behavior.


The [four-game research tools](../benchmarks/research/README.md) and
[current speed/memory report](../../docs/benchmarks/research/README.md) extend
these fixture/corpus checks to stochastic games, agents and adapters.

## Speed comparison

Completed 11-10-2026 00:22 (Europe/Paris): matched rules/copy workloads now have
three-process results for each engine. See the
[measurement report](../../docs/benchmarks/java-rust/README.md) for timings,
workload definitions, implementation differences and reproduction commands.

Reference: d65ad79f04992e238663c37c812dc2c5f6ce7ff2. User reported a clean
public checkout and successful Java build on 2026-10-09. Original reference
remains read-only. Build/run only its disposable .scratch/java-comparison copy.

ExportTicTacToe.java calls public TAG APIs to export each of the 5,478 reachable
boards plus a legal recorded move sequence, full-state signature and both
player-copy signatures. Each signature includes ordered actions, status/results,
scores, decision, historical player, move/turn/round counters and first player.
This compares one legal replay per board, not every possible history or invalid
action. Rust's separate exhaustive test covers every reachable transition.

Normal Rust tests use the preserved tests/fixtures/tictactoe-java.tsv; Java is
needed only to regenerate results. Comparison failures name the differing field
and show coordinate moves. To regenerate, run this in your own PowerShell using your Maven
configuration. Maven requires access to its configured dependency cache; a
restricted environment may not be able to regenerate this file.

```powershell
Push-Location .scratch/java-comparison
mvn dependency:build-classpath '-Dmdep.outputFile=target/comparison-classpath.txt'
Pop-Location
```

Once the dependency list exists, compile/run from the project root:

```powershell
# Run from the repository root.
$comparisonRoot = Join-Path $PWD '.scratch/java-comparison'
$comparisonDeps = (Get-Content "$comparisonRoot/target/comparison-classpath.txt" -Raw).Trim()
$comparisonClasspath = "$comparisonRoot/target/classes;$comparisonDeps"
javac -cp $comparisonClasspath -d "$comparisonRoot/target/classes" tools/java-comparison/ExportTicTacToe.java
java -cp $comparisonClasspath ExportTicTacToe "$comparisonRoot/target/tictactoe-java.tsv"
$env:JAVA_TAG_FIXTURE = "$comparisonRoot/target/tictactoe-java.tsv"
cargo test --offline --test integration games::tictactoe::java_comparison -- --nocapture
Remove-Item Env:JAVA_TAG_FIXTURE
```

Check each command succeeds before proceeding. Regenerated files remain ignored
in .scratch until reviewed; the accepted fixture is preserved in tests/fixtures.
No additional Rust dependencies are needed. JAVA_TAG_FIXTURE overrides the saved
fixture only for this run; without it, the normal test uses the preserved results.

Known semantic choices: Rust rejects invalid moves without penalties, and offers
no moves to a nonacting observer. Java player copies retain current-player action
enumeration; compare visible facts separately. No attempt is made to reproduce
Java's inconsistent isGameOver helper; completion uses isNotTerminal.

Validated 2026-10-09: Java exported all 5,478 positions and the explicit Rust test
passed for every recorded replay. Full-state signatures matched, and both Java
player-copy signatures matched the same visible facts. Invalid moves and every
possible history are outside this comparison. On 2026-10-10 the accepted TSV was
preserved in tests/fixtures/tictactoe-java.tsv and enabled in normal Rust tests.

Compilation emitted the exporter but also printed a javac AccessDeniedException
when closing a Netty dependency archive in the user's Maven cache. No permission
workaround was attempted. Exporter execution and Rust comparison exited successfully;
the compiler diagnostic is retained as an environment limitation.

## Connect 4 regeneration

Added 10-10-2026 19:52 (Europe/Paris). The existing disposable build/classpath is
also used by ExportConnect4.java. It records 128 deterministic complete games and
both public copies; see [fixture schema/coverage](../../tests/fixtures/connect4-java.md).
From the project root, after setting the comparison variables as above:

```powershell
javac -cp $comparisonClasspath -d "$comparisonRoot/target/classes" tools/java-comparison/ExportConnect4.java
java -cp $comparisonClasspath ExportConnect4 "$comparisonRoot/target/connect4-java.tsv"
$env:JAVA_CONNECT4_FIXTURE = "$comparisonRoot/target/connect4-java.tsv"
cargo test --offline --test integration games::connect4::java_comparison
Remove-Item Env:JAVA_CONNECT4_FIXTURE
```

Check each command succeeds. Generated results stay in .scratch until reviewed.
The accepted fixture is embedded in normal tests; the override is optional.
Compilation again printed the dependency-archive access diagnostic described
above; class generation, export and comparison succeeded. No workaround used.
