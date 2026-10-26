# Batched learner episodes

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

GameEnvBatch advances compatible independent learner environments in one Rust call.
Use it when collecting several episodes or preparing a leading batch dimension for
learning code. Rust runs members sequentially. The [single-environment guide](gym-episodes.md) defines each member's
rules, rewards, seeds and failure lifecycle. Gymnasium 1.4.0 remains the optional
dependency.

## Run independent episodes together

```python
import numpy as np
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch

environments = [GameEnv("connect4", learner=i % 2, opponent=policy,
                        payloads=("cells", "board"))
                for i, policy in enumerate(("random", "osla_python", "osla", "first_available"))]
try:
    with GameEnvBatch(environments) as batch:
        result = batch.reset(seeds=[100, 101, 102, 103])
        if any(result.errors):
            raise RuntimeError(result.errors)
        active = result.valid.copy()
        final_info = [None] * len(environments)
        while active.any():
            actions = [int(np.flatnonzero(result.observations["action_mask"][i])[0])
                       if active[i] else None for i in range(len(environments))]
            result = batch.step(actions)
            if any(result.errors):
                raise RuntimeError(result.errors)
            for i in np.flatnonzero(result.valid):
                if result.terminated[i] or result.truncated[i]:
                    final_info[i] = result.infos[i]
                    active[i] = False
        print([info["results"].tolist() for info in final_info])
finally:
    for environment in environments:
        environment.close()
```

The collector supplies learner actions. Rust validates all selected actions first,
then applies each learner move and its native opponent reply using the shared episode runtime.
Each original GameEnv keeps its own board, policy, RNG and wrapper seed progression.
You can inspect/use originals between bulk calls. Shared mutation is locked for the
whole operation, including reset/output conversion. Using an old result after a
separate individual operation requires care: its mask describes the earlier board.

A batch retains references to supplied environments. Closing the coordinator
releases its references and leaves originals open. Close originals explicitly or
use their normal context management. Two batches may retain the same originals;
concurrent operations are rejected rather than scheduled across them.

## Compatibility and spaces

A batch accepts 1-256 distinct GameEnv objects. They must have identical complete
encoding/action descriptions and the same ordered payload names. Different
learner seats, Rust opponents, seeds, horizons and resource limits are permitted
when those descriptions match. Create separate batches for different board sizes,
winning targets, games or payload layouts. Unsupported agents still fail during
ordinary environment construction.

single_action_space and single_observation_space describe one row using the first
environment's space objects, including its action-sampling RNG. Choose actions
from each returned row's legal mask.

For B default Connect 4 environments with cells and board, observations contain:

| Key | Shape | Type and meaning |
|---|---|---|
| cells | [B,42] | float32; own=+1, opposing=-1, empty=0 for that row's learner. |
| board | [B,3,6,7] | float32; own/opposing/empty channels. |
| action_mask | [B,7] | int8; 1 for legal learner column IDs. |

Rows always retain their original member index. Rust copies independent numeric
outputs into stacked allocations while detached from Python, then creates NumPy
objects after reattaching. Python formats per-member info dictionaries and handles
wrapper seeds.

## Read the result before collecting a transition

reset and step return an EpisodeBatchResult with full-size arrays and tuples:

| Field | Meaning |
|---|---|
| observations | Dictionary of stacked payloads and legal mask. |
| rewards | float64 [B]; learner rewards for valid rows. |
| terminated / truncated | bool [B]; ending flags for valid rows. |
| valid | bool [B]; this row has a new usable return. A reset return is an initial observation, not a training transition. |
| infos | B entries: member info dictionary for a valid row, otherwise None. |
| errors | B entries: None or a member-specific runtime/output error string. |
| statuses | B strings: new, ready, ended, failed or closed. |

Inactive and failed rows have zero numeric arrays, reward/flags false or zero,
valid=False and info=None. These zeros are placeholders, never an automatic reset
or a new transition. Statuses describe live or successfully replaced episodes,
including inactive members; flags describe only valid returns. Record a terminal
transition when it appears: later inactive rows do not repeat its final observation.
Earlier returned arrays/dictionaries remain independent of subsequent calls.
A row slice is a NumPy view into that one returned batch allocation; copy it if
independent mutation within the same result is needed.

An error after the learner move marks that environment failed and requires reset.
Its incomplete transition must not enter training data. Other valid members retain
their successful transitions. If constructing the entire step result fails, no
result is delivered and all otherwise completed selected members require reset.
An ordinary invalid action or wrong lifecycle rejects the whole input before any
member advances.

## Active members and explicit partial reset

step(actions) accepts exactly B entries. None means inactive. Alternatively,
step(actions, active=boolean_mask) uses the supplied B-entry Boolean mask;
inactive action entries are ignored. Every active entry must be a nonnegative
integer ID legal in that member's current learner decision. Failed/new/ended/
closed selected members reject the whole call. Inactive members can stay in those
states. An empty active selection returns only invalid placeholder rows.

reset(indices=None, seeds=None) selects every member by default. Pass distinct
indices to replace only selected members, and an optional seed list in that exact
selection order. None entries continue those members' ordinary seed generators.
For example, reset(indices=[3,1], seeds=[500,None]) reseeds member 3 and continues
member 1's seed progression. An empty index selection has no state/seed side effects.

All indices/seeds and selected lifecycle/generation checks are validated before
live replacement. Fresh games, openings and outputs are prepared independently.
A later failed reset, including seed staging or output construction, preserves
that member's previous board, lifecycle and seed progression, while successful
members replace theirs. The wrapper commits only
rows whose complete output was constructed successfully. Owner/generation checks
reject foreign, stale and consumed native reset candidates. No automatic reset is
performed after terminal, truncated or failed episodes.

## Limits, validation and measurement

max_bytes defaults to 64 MiB. Before allocating the output, a conservative memory estimate counts all B rows,
including placeholders: requested float32 elements, action masks and an allowance
for numeric metadata/output flags. The constructor and each bulk call reject an
excessive estimate before advancing members. The limit concerns final numeric
outputs, not all memory. Live games, candidate resets, immutable descriptions,
search copies, temporary Rust rows, Python dictionaries and earlier retained
outputs require additional memory; peak process memory is unmeasured.

[CPU results](../../benchmarks/episodes/results.md)
compare release-mode individual and bulk workloads for 1-128 environments, with
exact transition/RNG validation. Small batches can be slower: stacking and
coordination have costs even though Python makes fewer bulk step calls.

Run the checks and example from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -v
.\.venv\Scripts\python.exe adapters/python/examples/play_episode_batch.py
```

Implementation entry points are [gym_batch.py](../../../adapters/python/python/tag_rust/episodes/batch.py),
[episode_batch.rs](../../../adapters/python/src/episode_batch.rs) and
[episode_arrays.rs](../../../adapters/python/src/episode_arrays.rs). These reuse
the GameEnv/episode runtime and stay separate from pending-evaluator grouping.

For complete transition records and an update/save/reload workflow, follow
[Collect transitions and learn a table](../learning/tabular-learning.md).

The API supports optional stacked intermediate observations with a separate validity mask,
bulk `step_teacher(active=...)` and per-selected-member teacher reset options.
The [neural play guide](../learning/neural-play.md#capture-observations-between-learner-and-opponent-actions)
owns the new fields, mixed capture configurations and limits. The reusable
collector can now call one batch action policy or native teacher step per round;
the earlier per-member callback remains available.
