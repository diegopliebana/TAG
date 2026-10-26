# Collect learner episodes

Use an episode environment when a learner supplies one action and receives its
next observation, reward and ending flags after native opponent play. This is a
different time step from applying one basic game action. Complete [Python setup](../setup.md),
including Gymnasium. Check the [capability matrix](../../reference/framework/overview.md#supported-workflows)
for registered learner environments; basic game/search access does not imply
episode support.

## Run one environment

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/play_episode.py --seed 42
```

The example prints the master seed and the learner's final reward, game results
and learner-step count. Omit the seed or use `auto` for a fresh stream. A script
can choose the first legal action rather than sampling:

```python
import numpy as np
from tag_rust.episodes.gym import GameEnv

env = GameEnv("connect4", learner=1, opponent="random", payloads=("board",))
try:
    observation, info = env.reset(seed=42)
    while True:
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(reward, info["results"].tolist(), info["learner_steps"])
            break
finally:
    env.close()
```

`action_mask` marks fixed action IDs currently offered to the learner. A step can
include the learner move and an opponent reply. `terminated` means the rules
ended the game. `truncated` means an external horizon ended collection; it does
not create a game winner. Reset before stepping again after either flag. Retain
the final observation when calculating learning targets.

## Configure the environment

| Setting | Meaning |
|---|---|
| `learner` | Seat 0 or 1; choose seats explicitly when creating environments |
| `opponent` | `first_available`, `random`, `osla`, `osla_python` where registered |
| `opponent_settings` | Validated configuration for that opponent; see reference |
| `payloads` | Requested numeric observation representations |
| `max_episode_steps` | Positive learner-step horizon or None to disable it |
| `include_intermediate` | Capture the state after the learner action and before opponent play |
| `teacher="osla"` | Enable native teacher selection through `step_teacher()` |

Initialize an explicit seed with `reset(seed=42)`. Gym `reset(seed=None)` uses OS
entropy on first use and subsequently continues its seed stream. This differs
from basic `Game.reset()`, which repeats the original setup. Native teachers,
opponents and learner sampling retain separate randomness; preserve their
recorded seeds/checkpoints for continuation.

Teacher selection follows the ordinary step path. `step_teacher()` reports its
chosen action; intermediate views retain the learner's perspective. Only valid
nonterminal intermediate rows should receive a bootstrapped value target.

## Run environments in bulk

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/play_episode_batch.py --seed 42
```

This fixed four-member example prints each learner's reward. `GameEnvBatch`
advances compatible independent members through one Rust call. Its result has a
leading member dimension and a `valid` mask: read outputs only for valid rows.
After a member ends, send None for its action or exclude it from the active mask.
An ending row is returned once; later inactive rows do not repeat its final data.

Bulk reset accepts selected indices and seeds in that order. Validate `errors`
for every member, retain successful outputs and reset failed members as required.
Bulk execution is sequential native work, not a promise of parallel simulation.
The [complete batch example](../../reference/python/batched-episodes.md#run-independent-episodes-together)
shows masking, final-info retention and cleanup.

## Collect trajectories for learning

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_collect.py --game connect4 --mode all --episodes 3 --batch-size 2 --seed 42
```

`teacher` collects native OSLA demonstrations; `policy` samples an untrained
frozen policy; `osla` uses frozen successor values. `all` runs all three. The
OSLA mode supports its native-scored OSLAPython opponent; teacher/policy modes
accept the other registered opponents. This command collects data without
updating weights. [All flags and defaults](../../reference/python/command-line.md#neural_collectpy)
include fixed seats, step horizons and opponent selection.

Owned transitions preserve observations before/after the step, action, reward,
termination/truncation flags and metadata after reset/closure. Episode groups
add logical IDs, seeds and optional neural diagnostics. Carry `next_episode_id`
into subsequent groups to maintain alternating seats. Use
[Q-learning](tabular-learning.md) or [neural training](neural-training.md) for
actual parameter updates. Whole-game [MCTS self-play](mcts-selfplay.md) uses a
separate collection/value contract.

## Reference

[Gym API, rewards and failures](../../reference/python/gym-episodes.md),
[bulk lifecycle and masks](../../reference/python/batched-episodes.md),
[owned transitions and Q-learning](../../reference/learning/tabular-learning.md),
[neural collection](../../reference/learning/neural-play.md).
