# Single-learner Gymnasium episodes

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

The single-environment interface supports for Tic-Tac-Toe and Connect 4.
Rust owns the rules, chosen opponent and episode bookkeeping. Python supplies the
learner's actions and receives independent NumPy arrays. [Batched episodes](batched-episodes.md)
are also implemented. Gymnasium is optional: import this module explicitly.

## Setup

The selected optional dependency is Gymnasium 1.4.0. Install it manually in normal
PowerShell if missing, then rebuild the extension after Rust changes:

```powershell
.\.venv\Scripts\python.exe -m pip install "gymnasium==1.4.0"
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
```

The package also declares a gym optional extra for installation from a built
wheel. Ordinary tag_rust imports do not load Gymnasium. Native/browser builds
continue to use the existing workspaces. See [setup](setup.md) for dependency and platform limits.

## Run a complete episode

This example chooses among currently legal actions at random. It collects
transitions without training a neural network.

```python
import numpy as np
from tag_rust.episodes.gym import GameEnv

env = GameEnv("connect4", learner=1, opponent="random", payloads=("board",))
try:
    observation, info = env.reset(seed=42)
    env.action_space.seed(123)
    trajectory = []
    while True:
        action = env.action_space.sample(mask=observation["action_mask"])
        next_observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append((observation, int(action), reward,
                           next_observation, terminated, truncated))
        observation = next_observation
        if terminated or truncated:
            break
    print(reward, info["results"], len(trajectory))
finally:
    env.close()
```

A step applies one learner action and then the opponent reply, unless the game
already ended. If seat 1 learns, reset runs seat 0's opening move first. Every
returned board is from the learner's viewpoint. Cells encode own pieces as +1,
opposing pieces as -1 and empty cells as 0. Board channels are own/opposing/empty
in channel/row/column order. All payloads are float32; the legal mask is int8.

For Tic-Tac-Toe use GameEnv("tictactoe", {"grid_size": 4}, ...). For Connect 4,
parameters are width, height and winning_length. Use learner=0 or learner=1.
Payloads can be cells, board or both; names, shapes, types and numeric bounds
come from Rust descriptions. Settings remain fixed for an environment's lifetime.
The [array guide](../learning/observation-arrays.md) explains the coordinates and indices.

## Choose a Rust opponent

| Identifier | Policy | Accepted opponent_settings |
|---|---|---|
| first_available | Chooses the first offered action. | Empty dictionary. |
| random | Uniformly samples offered actions with its own Rust RNG. | Empty dictionary. |
| osla | Original native OSLA, evaluating copied game states. | Empty dictionary. |
| osla_python | Separate batched OSLA agent, scoring candidates entirely in Rust. This is the default. | max_rows (default 1024), max_candidates (default 4096). |

For example, opponent="osla_python", opponent_settings={"max_rows": 3} limits
candidate rows per batch. max_candidates must cover the configuration's full
action space so predictable limit failures are rejected before episode play.
Unknown policies/settings fail explicitly. These agents preserve their existing
native selection and tie/RNG rules. A new Rust policy needs Rust factory/settings
registration and validation. External Python-evaluated opponents remain outside
this synchronous interface; use [Game/GameBatch](rust-owned-execution.md) for them.

Do not put seeds or live checkpoints in opponent_settings. reset(seed=...) derives
fresh episode/opponent seeds. This keeps one experiment seed contract across
opponents without silently accepting a second source of randomness.

## Rewards, ending flags and limits

Reward is 0 while ongoing, +1 for a learner win, -1 for a learner loss and 0 for a
draw. These rewards describe that learner, even when the opponent moves last.
Info's results array is ordered by absolute seat ID: ongoing=0, win=1, draw=2,
loss=3. Result labels themselves are not numeric rewards.

An optional positive max_episode_steps counts learner actions and their complete
opponent replies. At the horizon, an ongoing game returns truncated=True and its
learner-ready observation/legal mask. A rules ending returns terminated=True and
an empty mask, taking precedence over the horizon. The caller's discount applies
once per learner step; returned rewards are undiscounted.

Reset explicitly after either flag. Further step calls are rejected. Never sample
an all-zero terminal mask: Gymnasium's Discrete mask fallback can return an ID even
when no legal move exists. Keep the final observation/reward before starting a new
episode. These games reject learner seat 1 with target one because reset would
finish before the learner could act. Seat 0 can use that configuration.

max_cells defaults to 4096 and max_bytes to 64 MiB of numeric output/index data.
These bounds do not cover total process memory, game/search copies or Python
object overhead. Rust checks fixed layouts and candidate limits before play.

## Failures and ownership

Reset builds a fresh game and opponent, runs any opening, then constructs the
return value before replacing the old episode. A failed reset preserves the old
state, lifecycle and seed progression. It does not copy the old board.

Step checks types, legal IDs, lifecycle and counters before execution. Invalid
input leaves the episode unchanged and usable. Once execution starts, an unexpected
rule/agent/encoding/output error can retain accepted moves or agent RNG changes.
The environment becomes failed and returns an exception rather than a transition.
Do not put that failed operation into training data; reset successfully before
another step. Earlier completed transitions remain valid. status reports new,
ready, ended, failed or closed.

There is no extra rollback snapshot on every step. Ordinary observation and search
copies still occur. Unrecoverable process/allocator failures have no recovery
guarantee. Concurrent mutating calls on one environment are rejected; use separate
environments for independent work. Returned arrays keep independent storage after
moves, reset and close. close is idempotent; reset/step after close fail.

## Seeds and diagnostic information

An explicit reset seed restarts NumPy PCG64 and sets episode_index to zero.
reset(seed=None) continues its progression; the first unseeded reset initializes
it from entropy. Derivation version 1 draws two raw u64 words per successful
reset in environment/opponent order. The environment word is reserved metadata
for these deterministic games; their rules consume no chance stream. Step does
not draw wrapper seeds. Seed action_space separately for reproducible sampling.

Info records encoding identity, observer, next_player, progress and results;
learner_steps, underlying_moves and moves_this_call; opponent/settings;
opponent_rng (RNG version, initial seed, stream, word position); and environment_seed,
opponent_seed, seed_derivation_version, seed_generator, numpy_version and
episode_index. Each call returns separate dictionaries/arrays. Seeds alone are
not checkpoints for a partially played game. Matching traces require recorded
versions, settings, learner actions and execution ordering.

## Checks and implementation path

The API supports optional `include_intermediate=True` capture and `teacher="osla"`
learner-side selection. `step_teacher()` uses the ordinary learner step path;
teacher resets require explicit `options={"teacher_seed": seed}`. The
[neural play guide](../learning/neural-play.md#capture-observations-between-learner-and-opponent-actions)
owns these APIs, intermediate info fields, independent RNGs and checked failure
behavior. Ordinary Gym tuples and observation spaces remain as described here.

Run the Python suite after installing Gymnasium and rebuilding the extension:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -v
```

The official checker samples unmasked actions and can select illegal moves on
occupied boards. Tests use passive checks with legal inputs and additional
mask-aware full-episode, seed, ownership and failure tests. An unmodified checker
smoke test covers a 1 by 1, learner-seat-0 terminal episode. That smoke test does
not prove all configurations. A horizon of one also needs its own tests because
the checker assumes its determinism smoke step does not truncate.

- [gym.py](../../../adapters/python/python/tag_rust/episodes/gym.py) maps Rust descriptions
  to spaces, owns wrapper seed progression and formats standard Gym returns.
- [episode.rs](../../../adapters/python/src/episode.rs) owns typed Rust games,
  registered opponents, preflight and episode execution.
- [episode_binding.rs](../../../adapters/python/src/episode_binding.rs) stages
  reset candidates, rejects foreign/stale commits and tracks failed lifecycle.
- [tests](../../../adapters/python/tests/test_gym.py) cover native bindings and
  optional Gymnasium checks. Missing Gymnasium is reported as explicit skips.

## Constructor and reset argument reference

| Argument | Default | Accepted meaning |
|---|---|---|
| `game`, `parameters` | Required game ID; parameters None | Registered learner game and validated integer rule parameters |
| `learner` | 0 | Seat 0 or 1 |
| `opponent` | `osla_python` | Registered opponent identifier from the table above |
| `opponent_settings` | None (empty object) | Only that opponent's documented fields |
| `payloads` | `("board",)` | Non-string sequence of advertised observation names |
| `max_episode_steps` | None | Positive learner-step cap or disabled |
| `max_cells`, `max_bytes` | 4,096; 64 MiB | Positive configuration/array resource limits |
| `include_intermediate` | False | Boolean capture of post-learner/pre-opponent data |
| `teacher` | None | None or `"osla"` |

`reset(seed=None, options=None)` accepts only `teacher_seed` in options. A configured
teacher requires an explicit u64 `options={"teacher_seed": 7}` on every reset;
without a teacher, that option is absent. `step(action)` takes one integer fixed
action ID. `step_teacher()` requires the configured teacher. `close()` releases
the environment; results/owned transitions already returned retain their storage.
The wrapper does not accept a rendering mode.
