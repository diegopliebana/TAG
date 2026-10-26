# Individual and bulk learner episodes

Measured 17-10-2026 00:00 (Europe/Paris), using the release Python extension on
Windows x64, CPython 3.14.8, NumPy 2.5.4, Gymnasium 1.4.0 and Rust 1.99.0.
Environment metadata is preserved with the [raw samples](results.json).
The [generated table](results.md) owns every workload time and call count.

The measured working tree used the bulk-episode implementation on base commit
78d831dd71a305b72d65926326d3437b6c91eeb8; it was not committed for measurement.
No builds or regression suites ran alongside these timings. Other machine load,
CPU frequency and thermal conditions were not isolated or recorded.

## What was compared

Both routes create the same GameEnv instances, alternate learner seats and use
reset seed 700+member_index. Learners choose the first legal action ID. Payload is
cells; opponents are Random or native-scored OSLAPython. Native policy settings
remain at their defaults. The count varies through 1,2,4,8,16,32,64,128.

The individual route completes one independent game at a time using GameEnv.step.
The bulk route retains those independent games in GameEnvBatch and advances all
active learner decisions in each call. Finished members become inactive; their
final outputs are recorded when returned. There is no automatic reset, worker pool,
GPU inference or simultaneous processing of game rules.

Before timing, every transition's cells, results, learner/underlying move counters
and exact opponent RNG checkpoint must match. Every timed repetition checks final
facts/checkpoints outside its timed interval. Trace capture is excluded from timing.

Times include environment construction, descriptor/space creation, seeded reset,
learner action selection, native execution, numeric stacking/conversion, Python
info/result handling, final fact extraction and close. Each of seven warmed samples
accumulates at least 200 ms of measured workloads, then reports its mean. The table
averages those seven means. Route order alternates. Raw samples retain repetition
counts, so variation can be inspected rather than hidden behind a single ratio.

Step-call counts are collector entry points, not counts of all native methods or
Python operations. Creation/specification, two-phase reset and close have their own
costs, included in timing. Each row still receives Python seed/info handling.
The two routes advance the same total learner actions and underlying moves.

## Interpretation

Bulk execution reduces step entry points substantially. At 128 environments the
calls fall from 413 to 5 for Random Tic-Tac-Toe and from 701 to 13 for Random
Connect 4. For native-scored OSLAPython the corresponding counts are 320 to 3 and
596 to 7. These are independent episodes, not multiple candidates within one tree.

Whole-workload timing gains are smaller than call-count reductions. At 128
environments the individual/bulk time ratios are approximately 1.19 and 1.24 for
Tic-Tac-Toe's Random and OSLAPython workloads, and 1.20 and 1.19 for Connect 4.
Individual execution is faster for the measured one/two-environment cases.
Several middle-size cases are close. Retain these slower and near-equal cases;
the data does not establish a universal speed improvement.

The count-one workload has learner seat 0 only; higher counts include both seats
and different seed sets/trajectory lengths. Compare each individual/bulk pair
within its count. These tables are not an isolated breakdown of fixed bridge
costs or a promise of linear scaling. FirstAvailable/original OSLA and configured
boards are correctness-tested but are not timed in this collection.

Stacking copies numeric payloads; Python still formats per-member metadata. Fixed
encoding descriptions are retained in each adapter episode for compatibility and
size checks. That retains persistent metadata storage and avoids rebuilding rules
or descriptions during bulk checks. Peak memory, CPU component costs and GPU/model
throughput were not measured. The numeric byte cap is not a process-memory cap.

## Reproduce

See [the tool README](../../../tools/benchmarks/episodes/README.md) for build and
measurement commands. The default seven samples/200 ms recreate the method; fresh
times may differ. Running measure.py --render regenerates the saved Markdown
without executing games. [the guide](../../reference/python/batched-episodes.md)
owns public bulk API semantics.

The subsequent [conversion-cleanup comparison](cleanup.md) retains separate
before/after samples for selected 1/128-member Random workloads.
