# Data and reproduction

This directory contains the current default benchmark results, collected from
the updated source.

| File | Contents |
|---|---|
| `timing.jsonl` | Primary three-process rule/agent/complete-game timing repetitions |
| `allocations.jsonl` | Separate allocation-instrumented repetitions |
| `retention.jsonl` | Initialized and loaded memory at 1/128/2,048 retained states |
| `python-*.jsonl` | Python interfaces, CPU evaluation and neural search |
| `adapter-*.jsonl` | Native browser-adapter protocol, local policies and history/branches |
| `features.json` | Minimal/optional feature initialized memory and binary sizes |
| `matched/` | Fresh validated placement-game corpus measurements |
| `cli.json`, `episodes-*.json` | Complete CLI schedules and learner collection |
| `tree-memory.json` | Native transient and Java retained search-tree memory |
| `workflows-timing.jsonl`, `workflows-allocations.jsonl` | Additional component/prepared-state and Random diagnostic probes |
| `game-summary.json` | Actual game/action/turn denominators and normalized timings |
| `component-summary.json` | Prepared-position timing medians, ranges and fixture metadata |
| `summary.json`, `summary.csv` | Primary process-level aggregates |
| `environment.json`, `source-hashes.json`, `java-class-hashes.json` | Versions, settings, binary and source/class identities |

Run commands from the repository root. Complete tool setup and builds using the
[benchmark tools](../../../tools/benchmarks/research/README.md), then select a fresh
output directory. Run collection phases sequentially, without competing workloads.

```powershell
python tools/benchmarks/research/run.py --phase timing --output docs/benchmarks/research
python tools/benchmarks/research/continue.py --output docs/benchmarks/research
python tools/benchmarks/research/extras.py --output docs/benchmarks/research
python tools/benchmarks/research/extras.py --episodes 1 --output docs/benchmarks/research
python tools/benchmarks/research/extras.py --episodes 2 --output docs/benchmarks/research
python tools/benchmarks/research/extras.py --episodes 3 --output docs/benchmarks/research
python tools/benchmarks/research/tree_memory.py --output docs/benchmarks/research
python tools/benchmarks/research/workflows.py --phase timing --output docs/benchmarks/research
python tools/benchmarks/research/workflows.py --phase allocations --output docs/benchmarks/research
python tools/benchmarks/research/summarize.py docs/benchmarks/research
python tools/benchmarks/research/workflow_summary.py docs/benchmarks/research
python tools/benchmarks/research/plot_workflows.py docs/benchmarks/research
python tools/benchmarks/research/report_remeasurement.py docs/benchmarks/research
python tools/benchmarks/research/validate.py docs/benchmarks/research
python tools/benchmarks/research/validate_workflows.py docs/benchmarks/research
```

Use the configured environment's Python executable. `workflows.py` requires
`--output`; `--phase` defaults to `timing`, with three independent process repetitions.
`allocations` uses one separate repetition. Seed 42 and the fixed profile/position
matrix define this diagnostic fixture. The low-level operation suffix `@N` prepares
up to N additional actions before timing, stopping at the last nonterminal state.
These are benchmark controls, not ordinary match-runner arguments.

Read [methodology](methodology.md) before interpreting timing ratios or memory.

`analysis-source-hashes.json` identifies aggregation, plotting and validation
sources separately from the frozen collection sources. `checksums.json` covers
accepted report artifacts, excluding itself. Reproduction also requires the
recorded model fixtures and the exact runtime versions in `environment.json`.
