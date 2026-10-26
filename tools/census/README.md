# File, line and test census

Run from the repository root with Python 3.14 or later (standard library only):

```powershell
python tools/census/measure.py --output docs/benchmarks/framework-size/data --private-output .scratch/census/working
python -m unittest discover -s tools/census -v
```

Both output arguments are required directory paths. The first receives a
public-scope file manifest and grouped JSON/CSV totals. The second receives the
whole working inventory. Neither command publishes or changes runtime code.
Version 1.0 counts the union of Git-tracked and nonignored untracked files.
Deleted files, symlinks and junctions are skipped. Run after saving all intended
source/documentation changes; regenerate when those inputs change.

Each file has one category and language. The public projection omits development
handover/review records and repository agent instructions. Git ignore rules exclude
builds, environments, dependencies, generated bindings and disposable reference
copies. The counter additionally excludes `target`, `node_modules`, `.scratch`,
`__pycache__`, `.git`, directories beginning `.venv`, and known generated binding
paths even if ignore rules change. Census data/plot outputs exclude themselves from their input inventory;
their product inventory is reported separately. Tracked training outputs remain
visible as data, rather than being counted as framework implementation.

Lines are physical lines, with an unterminated final line counted. Blank means
whitespace only. Nonblank includes comments: it is **not executable code LOC**.
No code/comment lexer is used. Binary files have bytes and file counts, with null
line counts. HTML scripts/styles and Markdown fences stay in their containing
file's language; they are not counted again as extra files or source lines.
Tests embedded in implementation files remain in those files' line totals.

The [report](../../docs/benchmarks/framework-size/README.md) explains test
identities, execution contexts, generated deliverables and limits. Test execution
is separate from this read-only file counter.

## Test census

Run native discovery and execution from the repository root:

```powershell
cargo test --workspace --all-features --all-targets -- --list
cargo test --workspace --all-features --all-targets
cargo test --workspace --all-features --doc
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
cargo test --manifest-path adapters/python/Cargo.toml
python -m unittest discover -s adapters/python/tests -v
node --test --test-reporter=tap web/tests/*.test.js
python -m unittest discover -s tools/census -v
python -m unittest discover -s tools/benchmarks/research -p 'test_*.py' -v
```

Use the configured environment's Python executable if `python` resolves elsewhere.
`--all-targets` includes example test binaries; `--doc` runs documentation examples
separately. `--all-features` enables optional runtime capabilities. These are
development-profile correctness checks, not release performance measurements.
See [setup](../../docs/guides/setup.md) for Cargo syntax and tool installation.

The report counts discovered Python methods as its unit; `subTest` iterations and
internal parameter loops remain inside their method. Node counts named tests.
Rust identities include their crate, except explicitly shared match/tournament
modules, whose repeat compilations are merged. Assertions and benchmark iterations
are never tests.

`test_inventory.py --logs DIRECTORY --output FILE` aggregates UTF-8 captured logs
named `rust-all-list.log`, `rust-all-run.log`, `rust-doctests.log`, `python-rust.log`,
`python.log`, `node.log`, `census-unit.log` and `statistics-unit.log`.
`python-discovered.json` must contain IDs from unittest discovery. Generate it with:

```powershell
python tools/census/test_inventory.py --discover-python adapters/python/tests --output .scratch/census/python-discovered.json
```

`--discover-python` selects a unittest directory and writes sorted method IDs to
`--output`, without executing them. The collector checks discovery/execution
counts and rejects incomplete successful-run logs. Keep captured logs alongside
the measurement so failures remain inspectable.

Operational suites are separate: build release `match_play` and `tournament`
examples with `mcts-json,replay`, then run `python tools/matches/check-cli.py` and
`python tools/matches/check-tournament.py`. Build browser bindings using the
[browser manual](../../docs/guides/running/browser.md), then run
`node tools/browser/test-browser.mjs` for real-browser checks. A browser startup
failure is unavailable execution, not passed page tests.
