# Measurement method

This collection reruns the complete four-game speed/memory study against the
updated Rust implementation and unmodified public Java TAG. Results compare implementations
and execution paths; they do not isolate programming languages or playing strength.

## Coverage and repetition

There are eight game/player profiles: two-player Tic-Tac-Toe and Connect 4, and
2-4-player Can't Stop and Love Letter. Release Rust and the warmed Java 21 JVM run
in separate processes, with alternating engine order across three repetitions.
Each timing process records five batches. Summary values are the median of three
process medians; ranges show those three observed medians, not a confidence interval.
No batch is treated as an independent process and no latency outliers are removed.

The primary suite retains initialization, actions, observations, full copies,
copy-plus-transition, scores/results, forks, First Available, Random, OSLA, MCTS,
complete games, encoding and persistence. The complete rerun also includes
allocation counters, retained states, optional features, tree memory, Python/model
batches, the native browser-adapter protocol, histories/branches, CLI schedules,
learner episodes and the validated placement-game replay corpus.

Java warms for at least 1,000 ms; primary Rust timing warms for 150 ms. Five
measured batches each run for at least 80 ms and a minimum of one block of 32 calls.
Allocation probes run separately from release latency probes. Java uses 64-512 MiB
heap bounds for generic probes; the matched corpus retains its original 256-512 MiB
configuration and its own validated histories/warmup. The environment manifest
pins binaries, source/class hashes, package versions, processor and power plan.

## Complete Random games and actual work

The normal `game` workflow now does the same sequence in both engines:

1. Create a game and seeded Random controller.
2. Generate available actions from the real state.
3. Construct the acting player's observation (Rust) or player copy (Java).
4. Ask the actual Random controller for a choice, then apply it to the real state.

Every timed block covers the same 32 agent seeds, 42 through 73, in a cycle. The
real game setup seed is 42. A complete block covers every member regardless of
where warmup left the cycle. The diagnostic `game_direct` uses the same choice
stream but bypasses observation creation and the ordinary agent call. Forced
single choices consume no selection RNG draw, matching each actual Random agent.
Smoke checks verified identical action/turn/round counts between these two
workflows within each engine, across all eight profiles. Their trajectories are
not matched across Rust and Java.

This homogeneous kernel uses one Random controller across the acting seats; it
does not model per-seat strategic memory. The separately measured native CLI
schedules exercise independent seat controllers, rotation and complete reporting.

The per-batch `game_work` vector records **measured work only**, excluding warmup:
completed games, accepted actions, logical player turns, completed rounds and
observation constructions. Per-game lengths and elapsed time per action/turn use
those actual denominators. Setup, agent construction and final state disposal/GC
effects remain in complete-game elapsed time.

End-to-end loops measure the optimized homogeneous Random program; compiler/JIT
optimizations of unused view data remain possible. Observation counts describe
source-level calls, not a claim about physical allocation count. The standalone
component probes retain opaque/escaping outputs to measure materialized views and
copies. Instrumented allocation probes can affect optimization and have separate
timings; they are not substituted for the release latency samples.

A logical turn includes the terminal turn: one placement/card play in the three
placement/card games, and one push-your-luck turn ending in bank/bust in Can't Stop.
It is separate from accepted action count (a Can't Stop turn has several actions)
and from implementation-specific progress counters. In particular, Java resets
its turn counter at round boundaries and both board engines omit the final turn
from some framework counters; those counters are not substituted for logical turns.

Default two-player Love Letter uses six winning tokens in Rust and seven in Java.
An additional `game_aligned`/`game_aligned_direct` pair configures Java's public
`nTokensWin2` parameter to six; Rust keeps six. No game source is patched. This
aligns the threshold, not every rule, action order, RNG or knowledge convention.
Other stochastic/search comparisons likewise retain documented implementation
differences. Java/Rust timing ratios describe the measured fixtures; values above
1 mean Rust was faster, below 1 mean Rust was slower.

## Component costs and retained history

Prepared-position probes use the opening, eight additional accepted actions, and
128 additional actions or the last nonterminal position if the game ends earlier.
Actual prepared lengths are recorded for each engine. Root preparation is outside
the timer. Stochastic positions and late-game lengths differ between engines.

- `observe`: the complete owned observation/player-copy operation.
- `actions`: enumerate choices from state.
- `validate`: re-enumerate the legal list and test membership of a prepared action.
  This matches the main legality work in Love Letter and external action submission;
  it is not claimed to expose every game's internal validation instructions.
- `apply`: mutate prepared successor inputs through the public checked transition.
  Its input copies are made before each block timer; legality checks remain inside
  the transition. It includes automatic consequences and internal allocations.
- `copy`: an ordinary full-state copy, including its retained data and disposal.
- `history_copy`: Love Letter's retained public/current/last-round private containers
  in Rust, and Java's action/text history lists. These are different representations;
  no isolated equal-content language comparison is claimed.
- `save`/`load`: Rust-only sufficient-state serialization and checked restoration,
  not an assumed equivalent Java persistence API.

Prepared transition timing uses one clock pair per 32 updates. `apply_control`
reports the clock/loop control at opening positions over 32,000 calls (1,000 timed
blocks), avoiding minutes of input cloning outside an almost-empty timer.
Control values are not subtracted. Allocation/GC caused by preparation can still
affect a later timed block; these are practical component probes, not perfectly
isolated instruction costs. Component medians should not be added as an exact
decomposition of complete-game time.

Rust has no match-long provenance in simulation state. It retains current and
previous-round knowledge, so copy costs can vary within and across rounds but
must not be interpreted as growing with the whole match history. Java history
and observation semantics differ. Hypothetical worlds and agent planning RNGs
remain separate from the actual hidden state and environment RNG.

## Memory and platform limits

Requested Rust allocation bytes omit inline state storage; the counting allocator's
reallocation fallback can temporarily allocate old and new buffers. Java allocation
bytes come from the current thread. Java post-GC heap, private committed process
memory and resident working set are different quantities. Initialization/scaffolding
is measured separately; JVM baseline memory is not per-state resident RAM.
The probes retain 1, 128 and 2,048 states and search at 32/128 iterations.

In the process-memory plot, “Before retaining state copies” marks memory already
used by the initialized runtime, framework, prepared root and benchmark objects.
Bars show total memory with 2,048 copies retained. Subtract the marker's numeric
value from the bar's numeric value to obtain additional process memory; the
logarithmic axis means their visual separation is not that difference. Total
memory remains relevant when comparing complete application footprints.

The native browser-adapter probe measures Rust/JSON and local inference on the
host. It does not measure actual Wasm execution, Web Worker transport or rendering.
Neural fixtures are deterministic untrained CPU networks, not trained playing
strength or GPU benchmarks. Python/batch contracts remain those in the original
study, including separate full-vector MCTS and discounted scalar OSLA models.

The public Java reference is read-only; compilation uses its disposable copy.
Its copy API branches future randomness internally, and Love Letter additionally
has a clock-seeded redeterminization RNG. Fixed setup/agent seeds do not make Java
sampled search fully reproducible. These source behaviors are preserved and their
hashes recorded. Java revision metadata was unavailable; source/class hashes
identify the measured reference instead.
