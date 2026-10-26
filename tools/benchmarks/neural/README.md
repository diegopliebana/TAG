# Neural pipeline CPU measurements

From the repository root, with the existing release Python adapter and neural
dependencies installed:

```powershell
.\.venv\Scripts\python.exe tools/benchmarks/neural/measure.py
cargo build --offline --release --example java_comparison_measure
target/release/examples/java_comparison_measure.exe 500
.\.venv\Scripts\python.exe tools/learning/stage5-proof.py
```

The Python tool writes `target/m7-s5/measurements.json` (the tool's retained output path). Defaults are five samples
per operation, one untimed warm-up call and at least .08 seconds per sample.
Use --samples and --seconds to change sampling. Avoid concurrent builds, tests or
other experiments while collecting timings. Record the full environment and
spread rather than treating a median as a guaranteed latency.

Components cover observation export, neural batch packing, owned input copies,
prepared-tensor forward execution, output/reply copies, checked frozen inference,
snapshot construction and loss/backward/Adam on a fixed teacher group. Component
batches repeat a cached two-move position at sizes 1,16,64. CPU tensors share the
newly copied local NumPy storage; they do not share Rust game buffers.

Full workloads cover neural/heuristic complete games, 16-episode policy collection
and fresh 32-episode training with two updates. The fixed-target optimizer timing
updates the same model repeatedly and is a local kernel/workflow measurement,
not continued reinforcement learning. The full-training workload recreates a
session for each repetition. No artifact I/O is included in those timed workloads.
The integration proof separately exercises CLI artifact saving/loading/resumption.

The native example provides rules-only release measurements on the existing
verified replay corpus. A transition uses a prepared mutable successor, with
cloning outside its timed section. This differs from the Python
clone/observe/apply workload, whose label includes those operations and bridge
costs. Do not subtract their timings to estimate isolated FFI overhead. The
native example warms each mode for at least one second and 200 passes, then
records five samples of the selected pass count.

The generated JSON records environment, workload settings and raw samples.
Retain those alongside any reported numbers and interpretation. No GPU/device
transfer timings or Java speed comparison are claimed by this CPU-only study.

The optional review-diagnostics.py tool recalculates the earlier Tic-Tac-Toe
diagnostics using native OSLA's near-best tolerance. It requires the original and
broader-training artifacts under `target`; these local training artifacts are not
bundled with the measurement tool.
