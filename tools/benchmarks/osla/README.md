# OSLA comparison tools

This directory contains the five-route comparison machinery. User-facing play
examples are in `adapters/python/examples/`; reusable Rust/Python interfaces stay
in the engine and adapter packages.

Run from the repository root using the configured Python environment:

```powershell
$env:PYO3_PYTHON = (Resolve-Path '.venv/Scripts/python.exe').Path
.\.venv\Scripts\python.exe tools/benchmarks/osla/run.py --output docs/benchmarks/osla-python/results.json
```

Build the release extension first when adapter or engine code changes. The
[benchmark report](../../../docs/benchmarks/osla-python/README.md) gives that
command, timing definitions, limitations and interpretation.

| File | Responsibility |
|---|---|
| run.py | `main()` runs the native benchmark; `benchmark_case()` checks scores/traces and times the Python routes. `measure()` and `sample()` handle warm-up and timing. |
| workloads.py | `play()` executes a manual Python-coordinated game; `play_owned()` uses one Rust-owned run call. `verify_owned()` checks the original traces and whole-call state/RNG parity outside timing. |
| native.rs | Measures both Rust-only routes and builds the candidate corpus. `play_with_agents()` supplies the common game loop for both native agents. Cargo builds this as the explicitly registered `benchmark_osla` example target. |
| report.py | `write_tables()` turns saved JSON values into Markdown tables. It performs no game execution or timing. |

The measured Python comparison heuristic lives in
[open_lines.py](../../../adapters/python/examples/open_lines.py). It is an
educational implementation of the Rust formula. The ordinary
[play example](../../../adapters/python/examples/play_osla_python.py) selects
Rust scoring by default and contains no timing or report code.

`grouped.py` separately compares 16 individual games with Rust-coordinated grouped
requests. It uses the real Python heuristic and a constant NumPy scorer, checks
final facts/checkpoints and records seven warmed samples of at least 200 ms per
route. Run with `.venv/Scripts/python.exe tools/benchmarks/osla/grouped.py`.
Results are in `docs/benchmarks/osla-python/grouped-results.json`; the accompanying
[report](../../../docs/benchmarks/osla-python/grouped-results.md) explains limits.

For scaling, add `--scaling` to measure 1, 2, 4, 8, 16, 32, 64 and 128 games,
then run `scaling_report.py` to render the saved JSON without rerunning games.
The [scaling report](../../../docs/benchmarks/osla-python/scaling-results.md)
records per-game times, speed factors, call counts and methodology. The original
16-game artifacts are preserved when using `--scaling`.
