# Engine/browser cleanup measurements

## Reading the measurements

A workload is the specific operation being measured, such as copying a state or
replaying a complete game. A harness is the program that repeats and times it.
The median is the middle measured value after sorting; it describes these samples,
not a guaranteed time for a future machine. Cache capacity is how many rebuilt
states are kept to avoid repeated work. A checkpoint interval says how many moves
separate eligible cached positions. Read the units and included/excluded work
before comparing numbers. Raw samples, commands and historical results below are
unchanged. See [the glossary](../../reference/framework/glossary.md) for engine terms.


Measured 13-10-2026 22:42 (Europe/Paris). These native release-mode workloads
measure complete browser JSON view requests and nested branch switching. They
exclude browser DOM work and Wasm/browser runtime overhead. The compared source changes affect view construction and branch restoration;
the samples below describe their measured scope.

## Workloads and results

Each workload ran three samples in one process. Views warm up with 100 requests,
then time 2,000 views of an empty two-human Tic-Tac-Toe board. Timing includes JSON
decoding, observation/display construction, scores/results and JSON encoding.
Branch switching uses a 32x32 board with a 900-move root, a child forked at 400 and
extended to 600, and a nested fork at 550 extended to 600. Twenty round trips select
the root and nested tips; every restored tip is compared to its original snapshot.

| Workload | Before median | After median |
|---|---:|---:|
| 16x16 views, 2,000 requests | 129.705 ms | 149.343 ms |
| 64x64 views, 2,000 requests | 2,460.304 ms | 2,155.380 ms |
| Nested branches, 20 round trips | 200.622 ms | 135.387 ms |

Nested switching took about 33% less time in these samples. The transition-count
regression test independently proves that a nested branch's inherited prefix is
applied once during selection. Browser view results are mixed and sample variance
is substantial, so these data do not establish a general view speed improvement.
The view changes remove one simulation-state clone and seed-only agent checkpoint
captures, while request handling now performs additional envelope decoding.
Allocation counts were not measured. These are local microbenchmarks, with ordinary
Windows scheduling and desktop activity; neither multi-game throughput nor future
complex-game costs are established.

Raw samples: [before](before.txt), [after](after.txt). Baseline samples were captured
before production edits. Final after samples ran after builds/tests completed.
An intermediate run overlapped browser validation and is excluded.

## Reproduce

```powershell
cargo run --offline --release -p tag-browser --example cleanup_profile
```

The [workload source](../../../adapters/browser/examples/cleanup_profile.rs) uses
black_box, explicit repetitions and state equality checks. Reproduce the baseline
against the pre-cleanup source plus this same harness. A single after run cannot
reconstruct the old implementation's timing.
