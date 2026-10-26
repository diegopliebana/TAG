# Play with a neural model and collect complete episodes

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

This workflow connects [CPU inference](neural-inference.md) to direct learner actions and
OSLAPython successor values. It also collects native OSLA demonstrations and
observations immediately after learner actions. The examples use untrained
networks to demonstrate execution and data semantics. Training and saved neural
models are implemented in the [training workflow](neural-training.md).

Use the existing [Python environment](../python/setup.md), optional PyTorch setup
in the inference guide and optional Gymnasium setup in the [episode guide](../python/gym-episodes.md).
The verified environment has CPython 3.14.8, NumPy 2.5.4, Gymnasium 1.4.0 and
PyTorch 2.14.0+cpu on Windows x64. Rebuild the adapter after Rust changes.

## Keep the model parameters fixed during evaluation

```python
import torch
import tag_rust as tag
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.inputs import ModelContract
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.play import NeuralValueEvaluator

torch.manual_seed(42)
with tag.create("tictactoe") as game:
    contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
model = PolicyValueNetwork(contract)
snapshot = FrozenModel(model)
evaluator = NeuralValueEvaluator(snapshot)
```

`FrozenModel` copies model state once. Updating the source `model` afterwards
leaves this snapshot's predictions unchanged. Its identity includes the declared
model implementation/settings, complete model input/output specification and ordered tensor state,
including persistent buffers. This binds evaluation requests to specific weights.
Create a new snapshot and evaluator after a training update.

The reference fully connected architecture identifies itself automatically.
Custom CPU `torch.nn.Module` implementations require an explicit implementation
identity and JSON settings describing their forward behavior. They must keep
tensor state in `state_dict()` and behave as stateless evaluators in evaluation
mode. Extra state objects, nonpersistent buffers, changed tensor state during copy
and shared source storage/modules reject. Python code identity and purity remain
caller declarations; arbitrary Python behavior cannot be established from weights.
The snapshot keeps its model private and returns independent prediction arrays.
Concurrent use of the same snapshot raises an error; callers serialize its use.

Digest version 1 requires a little-endian CPU. It hashes canonical UTF-8 JSON
(sorted keys, compact separators, ASCII escaping), then sorted state entries.
Each entry contributes a JSON name/dtype/shape header followed by contiguous tensor
bytes. Every frame is prefixed by its eight-byte unsigned little-endian length.
The fixed zero-network fixture in [the tests](../../../adapters/python/tests/test_neural_play.py)
checks the exact identity. This identifies model state; it is not a cross-platform
or cross-version numerical reproducibility guarantee or a saved checkpoint format.

## Run neural-value OSLA

```python
with tag.create("tictactoe") as game:
    game.set_controller(0, evaluator.controller(seed=7))
    game.set_controller(1, tag.OSLAPython.rust_heuristic(seed=8))
    stop = game.advance()
    while stop.reason == "evaluation":
        values = evaluator.evaluate(stop.request)
        stop = game.resume(stop.request, values)
    assert stop.reason == "finished"
```

Rust creates candidate successor positions and retains their original actor as
observer. Python checks the request's encoding/evaluator identity and result/
player arrays. Terminal rows use declared win/draw/loss rewards; only ongoing
rows enter neural inference. Results are scattered back into original row order
as an independently owned finite float64 vector. Rust validates the reply and
retains its existing near-best selection tolerance of 0.001 and agent RNG.

The network sees observer/next-player context for the successor. An opponent-turn
row has no policy operation; the original observer's value remains applicable.
A declared player-vector output is indexed for that observer without sign changes.
The default evaluator requires the observer-return reward units, gamma 0.99 and learner-step
clock. Other meanings need explicit compatible terminal rewards/units/clock in
the evaluator constructor. Missing player outputs and incompatible conventions
reject. All-terminal requests execute no network.

The evaluator returns state values. Rust validates the reply before applying the
selected action. A valid older snapshot
can still compute values from retained arrays after reset, but Rust rejects their
stale request on submission. A different snapshot/evaluator rejects the request's
identity before scoring. Retry/cancel semantics follow the
[existing evaluator guide](adding-evaluator.md).

## Group requests from independent games

```python
from contextlib import ExitStack
from collections import deque

with ExitStack() as stack:
    games = [stack.enter_context(tag.create("tictactoe")) for _ in range(2)]
    games[1].apply(4, position=games[1].position)
    for index, game in enumerate(games):
        for seat in (0, 1):
            game.set_controller(seat, evaluator.controller(seed=10 + 2*index + seat))
    batch = stack.enter_context(tag.GameBatch(games))
    pending = deque(batch.advance().groups)
    while pending:
        group = pending.popleft()
        result = batch.resume(group, evaluator.evaluate(group))
        for stop in result.stops:
            if stop.error is not None:
                raise RuntimeError(stop.error)
        pending.extend(result.groups)
    assert all(game.next_player is None for game in games)
```

Grouped rows retain their own observer IDs. Compatibility grouping and Rust's
whole-reply validation remain in [GameBatch](../python/grouped-evaluation.md). The shared
row packer creates no ordinary Python Game-observation object per candidate.
Tests use absolute CPU tolerance 1e-6 for ordinary single/grouped value agreement.
Exact move/RNG parity uses controlled zero-model values because batched floating
point execution can differ slightly near an action-selection boundary.

## Capture observations between learner and opponent actions

```python
from tag_rust.episodes.gym import GameEnv

with GameEnv("connect4", teacher="osla", payloads=("cells",),
             include_intermediate=True) as env:
    observation, info = env.reset(seed=7, options={"teacher_seed":17})
    assert info["intermediate"] is None
    observation, reward, terminated, truncated, info = env.step_teacher()
    selected_action = info["action"]
    intermediate = info["intermediate"]
    assert intermediate["info"]["observer"] == info["observer"]
    assert not intermediate["observation"]["action_mask"].any()
```

`step_teacher()` chooses with an independent original Rust OSLA agent, then runs
the ordinary learner step. `step(action)` remains available. Teacher settings are
fixed at construction; a configured teacher requires an explicit uint64 reset
option `teacher_seed`. Diagnostics include that seed and `teacher_rng` as
`(version, initial_seed, stream, word_position)`. The native opponent retains its
separate RNG. Its existing two-word PCG64 reset derivation is unchanged.

`include_intermediate=True` encodes the learner's permitted view immediately after
its action and before opponent play. Single returns keep the ordinary Gym tuple
and observation space; `info["intermediate"]` contains `observation` payloads/mask
and public `info` identity, observer, next player, progress and results.
A learner move that ends the game still has a terminal intermediate view.
No extra discount factor is introduced; Training uses the corresponding learner-step
return for nonterminal intermediate values.

Bulk calls use `GameEnvBatch.step_teacher(active=...)` or ordinary `step`.
`reset(options=...)` accepts one reset-options dictionary per selected member.
Intermediate data uses dedicated result fields:

| Field | Meaning |
|---|---|
| `intermediate_observations` | Full-member stacked payloads and action masks. |
| `intermediate_infos` | Public metadata per valid intermediate row; otherwise None. |
| `intermediate_valid` | True only for a successfully completed enabled step. |

All three are None when capture is disabled throughout the batch. Mixed enabled/
disabled members are supported. Reset, inactive, disabled and failed rows are
invalid placeholders. Check this validity mask separately from ordinary `valid`.
Changing capture on/off preserves accepted actions, final data, rewards, counters
and native RNG progression.

Additional numeric arrays are included in the conservative output-size checks. An
entirely disabled batch creates no intermediate arrays. Packing/stacking and
snapshot construction copy data; peak memory and throughput are not measured here.
Runtime/output failure retains accepted moves, requires reset and exposes no
usable intermediate training row. Reset replacement still commits only after
complete output construction; no per-step rollback snapshot was introduced.

## Collect completed episode groups

```python
from tag_rust.learning.collection import collect_episode_group

teacher = collect_episode_group("tictactoe", episode_count=3, master_seed=42,
                               phase="teacher", payloads=("cells",))
actor = collect_episode_group("tictactoe", episode_count=3, master_seed=42,
                             first_episode_id=teacher.next_episode_id,
                             snapshot=snapshot)
assert [episode.seat for episode in teacher.episodes] == [0, 1, 0]
assert [episode.seat for episode in actor.episodes] == [1, 0, 1]
```

The teacher default opponent is OSLA; actor-critic/evaluation default to Random.
All four registered opponents can be selected. Set `learner=0` or `learner=1` for
a fixed seat, otherwise the global logical episode ID alternates seats. A group
contains 1-256 compatible independent episodes, each collected through natural
termination or the optional `max_episode_steps` horizon. Return `next_episode_id`
to the next training group, including across teacher/actor phase boundaries.
Evaluation uses `phase="evaluation"` and its own episode counter.

Each `EpisodeRecord` retains logical ID, seat, phase, behavior identity/settings,
reset/teacher/policy seeds and ordered `CollectedStep` records. A step wraps the
existing owned `Transition` and, for neural collection, frozen value/log-probability
diagnostics. The transition holds before/after observations and metadata, action,
reward, separate termination/truncation flags and optional intermediate data.
Opponent settings and RNG diagnostics remain in transition info. Terminal and
truncated final observations are retained after environment close. Partial failure
raises `GroupCollectionError` with successful partial episode records and the
underlying diagnostics; it does not return an update-ready group.

The reusable `collect_episodes` still accepts its per-member action callback.
Its new `choose_batch(members, observations, infos)` path receives active rows
together. `NeuralPolicy` evaluates those rows once, then samples through separate
per-episode PCG64 streams. `teacher=True` instead uses native bulk selection/steps.
Ended members are excluded from later decisions. Each callback receives independent
inputs; invalid returned action counts reject before that round's Rust execution.

`episode_seed` implements the version-1 compact ASCII JSON/SHA-256 derivation.
Reset, teacher and policy-sampling purposes use separate streams. Explicit teacher
seeds add no draw to the existing environment/opponent seed generator. Model
initialization and evaluation do not advance training episode streams. Saved
continuation and checkpoint files are implemented in the [training workflow](neural-training.md).

## Run the complete examples and checks

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_collect.py --game tictactoe --episodes 5 --batch-size 3
.\.venv\Scripts\python.exe adapters/python/examples/neural_collect.py --game connect4 --episodes 5 --batch-size 3
.\.venv\Scripts\python.exe adapters/python/examples/neural_collect.py --game tictactoe --mode policy --episodes 5 --learner 1 --opponent osla --max-steps 2
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -p 'test_neural*.py' -v
```

The default `--mode all` runs exactly N teacher, N direct-policy and N neural-value
OSLA episodes. Teacher/policy collection shares its continuing training ID counter;
OSLA evaluation has its own counter. JSON reports combined and per-seat counts,
opponents, seeds and model identity. The OSLA example supports native-scored
OSLAPython as its opponent; other registered opponents are available in teacher/
policy modes. No weights are updated, loaded or saved. The examples demonstrate collection and evaluation, not training or playing strength.

The general model contracts support explicit player perspectives. Playable
learner-step episodes support Tic-Tac-Toe and Connect 4; arbitrary-player value selection also has labelled fixture tests.
New hidden-information games require their own observation/sampling guarantees.
Candidate-policy models can supply value-only inference. Native candidate-action
executors, graph/image adapters and GPU execution are not implemented.
[Neural MCTS](neural-mcts.md) uses its separate policy/value request contract.

For weight updates, model files and checkpoint continuation, use the implemented
[training workflow](neural-training.md). Implementations are now in topic folders;
the [package map](../../../adapters/python/python/tag_rust/README.md) explains
the paths and retained compatibility imports.
