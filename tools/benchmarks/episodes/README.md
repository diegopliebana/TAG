# Episode CPU comparison tools

These tools compare complete individual and custom bulk learner episodes using the
same release extension, board settings, alternating learner seats and explicit
seed pairs. They require the already selected NumPy/Gymnasium environment.

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
.\.venv\Scripts\python.exe tools/benchmarks/episodes/measure.py
```

The default runs seven warmed samples of at least 200 ms per route for 1,2,4,8,16,
32,64,128 environments on both games with random and native-scored OSLAPython.
Route order alternates. Trace collection runs outside timing; final fact extraction
is timed consistently and comparisons happen afterward. The time ratio describes
these workloads on this machine, not guaranteed scaling or GPU throughput.

Raw results are written to docs/benchmarks/episodes/results.json. Regenerate only
the Markdown table with --render. The [report](../../../docs/benchmarks/episodes/results.md)
owns results and limitations; ordinary examples do not import benchmark modules.

For the episode-conversion cleanup comparison, cleanup.py records the 1/128-member
Random cases for the currently installed release build. Specify --phase and
--output explicitly. [The cleanup report](../../../docs/benchmarks/episodes/cleanup.md)
contains the method, saved samples and reproduction command.
