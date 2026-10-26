# Train, save and evaluate a neural player

For operating instructions, use the [short manual](../../guides/learning/neural-training.md). This page contains
the detailed reference.

This workflow trains the network introduced in [neural inference](neural-inference.md),
using the training trajectories collected by [neural play](neural-play.md). Rust executes
the rules, teachers and opponents. Python/PyTorch calculates training targets,
losses and weight updates. The executable examples support default Tic-Tac-Toe
and Connect 4; the network/input descriptions retain their general tabletop
interfaces. New games still need their own observation encoders and episode
execution support.

## Run from the repository root

Use the existing environment with PyTorch and Gymnasium installed. All commands
below are PowerShell commands from the repository root:

```powershell
# Run from the repository root.
Test-Path .\.venv\Scripts\python.exe
```

The path check should print `True`. A command-not-found error for this interpreter
means that the current directory or environment path needs correcting. Installation
commands belong in the [inference setup guide](neural-inference.md).

Train with the recommended CPU settings:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --game tictactoe --output target/models/tictactoe.json --checkpoint target/models/tictactoe.pt
```

For Connect 4, use `--game connect4` and different output paths. Defaults are 256
teacher episodes followed by 1024 actor-critic episodes, groups of 16, an OS-random master seed,
two 64-unit tanh layers, Adam at learning rate 0.001, and one CPU computation
thread. One complete group produces one update. The final group of each phase
may be smaller. An episode is one game from the learner's viewpoint.

Each line reports the phase, episode IDs, losses, model identities and per-seat
outcomes. A negative actor loss is possible because its advantage can be negative;
loss magnitude alone does not measure playing strength. For a quick execution
check use `--teacher-episodes 5 --actor-episodes 5 --batch-size 3`.

Teacher/opponent settings are independent. The learner's teacher is original
Rust OSLA. Its warm-up opponent defaults to `osla`; the actor-critic opponent
defaults to `random`. `--teacher-opponent` and `--opponent` select any registered
episode opponent. `--learner 0` or `--learner 1` requests a fixed seat; otherwise
logical episode IDs alternate seats across groups and across the phase boundary.

## How the network learns

The policy output predicts probabilities of choosing each legal action. During
imitation learning from the teacher, its target is the action selected by Rust
OSLA. Masked cross-entropy increases the probability of that demonstrated action while keeping
illegal actions outside the distribution.

The value output predicts the observing player's discounted future reward. A win
gives +1, a loss -1, and a draw 0. Discounting by 0.99 once per learner step gives
earlier decisions some preference for reaching a good outcome sooner. For a
three-step win with rewards `[0,0,1]`, targets are `[0.9801,0.99,1]`. At an external
step limit, the frozen model estimates the remaining return from the final
observation. A one-step truncated segment with continuation value 0.8 has target
`0 + 0.99 * 0.8 = 0.792`. Rule termination always uses its actual final reward.

Both the observation before an action and the nonterminal observation immediately
after that action receive the same return target. The latter is the kind of
position OSLA scores, with the original observer retained while the opponent may
be next to act. There is no extra discount within that learner step. Terminal
intermediate observations are omitted from value training because the evaluator
already handles their exact results.

During actor-critic training, a copy of the policy with fixed parameters samples learner actions for a complete
group of episodes. The advantage is the observed return minus the frozen value prediction:
an action with a better outcome than expected receives a positive advantage.
The update increases its log probability, weighted by `0.99**step_index`.
Episode sums are averaged over episodes. Value squared errors average decisions
and nonterminal intermediate rows separately. Entropy measures how spread out
the legal action probabilities are and encourages exploration.

The total actor-critic loss is policy loss plus `0.5 * decision_value_loss` plus
`0.5 * intermediate_value_loss` minus `0.01 * entropy`. Teacher warm-up uses
demonstration cross-entropy in place of the actor loss and zero entropy weight.
An empty intermediate set contributes zero. Returns and advantages are fixed
numeric targets; gradients change network predictions, not those targets.
The implementation in [training.py](../../../adapters/python/python/tag_rust/learning/training.py)
defines these losses and applies updates to each collected group.

## Save, stop and resume

Training creates two files:

| File | Contents and purpose |
|---|---|
| `.json` evaluation model | Versioned network implementation, architecture, complete observation, action and value specification, finite float32 weights, snapshot digest, and training settings and model history. Load to play or evaluate. |
| `.pt` training checkpoint | Evaluation model plus Adam tensors/settings, phase progress, episode/update counters, history, private PyTorch RNG state and CPU/software conditions. Load to continue training. |

Files are replaced after a complete temporary file has been written. The CLI
updates the checkpoint after every successful group and exports the evaluation
model when the requested run ends. Stop deliberately after two groups:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --teacher-episodes 5 --actor-episodes 5 --batch-size 3 --max-groups 2 --output target/models/partial.json --checkpoint target/models/partial.pt
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --resume target/models/partial.pt --output target/models/continued.json --checkpoint target/models/continued.pt
```

Resume uses the saved training settings. It restores model and optimizer state
and starts fresh environments at the next logical episode ID. Checkpoints contain
no live game or pending evaluation. A failed group prevents further updates/saves
from that session; reload the last successful checkpoint. The existing checkpoint
remains available if writing its replacement fails.

Exact continuation is tested within matching recorded CPU/software conditions.
The loader rejects a different runtime. Model initialization and each update use
a temporary PyTorch RNG scope; per-episode policy sampling and Rust seeds derive
from separate explicit streams. Evaluation leaves the training schedule and RNG
state unchanged. Training selects the process-wide PyTorch thread count and
enables deterministic algorithms; use separate processes for experiments requiring
different runtime settings. The reference model uses no random training layers.

Checkpoint loading uses `torch.load(..., weights_only=True, map_location="cpu")`
and reconstructs the declared reference model. Shape/type/finite checks and the
snapshot digest reject incompatible evaluation weights. The loaders impose a
64 MiB file limit, which does not bound every temporary Python allocation.
See PyTorch's [saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
for the roles of model and optimizer state.

## Play the saved network

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_neural.py --model target/models/tictactoe.json --mode policy --opponent random --episodes 100
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_neural.py --model target/models/tictactoe.json --mode osla --episodes 100
```

`policy` samples legal actions directly from the network against any registered
episode opponent. `osla` uses Rust candidate generation and action selection with
the loaded network's successor values, against native-scored `osla_python`.
This OSLA evaluation currently supports that opponent only. Both commands play
exactly the requested count, alternate seats independently of training, and
report combined/per-seat counts and rates. Use multiple evaluation seeds and
opponents when assessing strength. Small run results establish that the workflow executes. Training against Random does not establish optimal play or value
accuracy under repeated OSLA continuation.

## Inspect the code and extend an experiment

Implementations now live in topic folders under `tag_rust`; the
[package map](../../../adapters/python/python/tag_rust/README.md) explains them.
Historical imports remain usable through compatibility modules.

| Module | Main responsibilities |
|---|---|
| [training.py](../../../adapters/python/python/tag_rust/learning/training.py) | `TrainingSettings`, `TrainingSession`, `discounted_returns`, `make_targets`, and `loss_terms`. |
| [artifacts.py](../../../adapters/python/python/tag_rust/learning/artifacts.py) | Portable evaluation JSON and resumable checkpoints, reconstruction and compatibility checks. |
| [evaluation.py](../../../adapters/python/python/tag_rust/learning/evaluation.py) | Exact-N policy/OSLA evaluation with independent seeds and per-seat outcomes. |
| [tabular.py](../../../adapters/python/python/tag_rust/learning/tabular.py) | Existing `QTable` workflow, preserving its versioned JSON format. |

`TrainingSettings` also exposes hidden widths and all loss coefficients for Python
experiments. Current training uses observer scalar returns and fixed action IDs;
other value/policy forms remain supported by the inference boundary and need
their own training algorithm. CPU training for the default two games is the
implemented workflow. No GPU training or new game is added here.

## Browser execution

The selected [browser learned-player extension](../browser/learned-controllers.md) reads the
portable reference model JSON and evaluates it in optional Rust/Wasm code inside
the worker. The Q-table retains its existing JSON and greedy action-value
meaning. Use the seat model picker for evaluation JSON; Python training checkpoints
remain Python-only. Native and actual-Wasm tests check model parity. Real graphical checks
separately exercise UI/file-picker behaviour. The [integration checklist](neural-integration.md)
covers new-game registration, compatibility and buffer ownership; the [benchmark collection](../../benchmarks/README.md) records existing measured
workloads and their scope.

For explicit or automatic seeds and recorded continuation, see the
[seed manual](../experiments/reproducibility.md). `TrainingSettings.master_seed=None`
is resolved once by TrainingSession, not each training group or checkpoint load.

## Complete argument mapping

The [CLI reference](../python/command-line.md#neural_trainpy) lists every training
flag, effective fresh-run default and required output. It is derived from the
parser and `TrainingSettings`. Python uses the same settings with these name changes:

| CLI flag | `TrainingSettings` field |
|---|---|
| `--batch-size` | `group_size` (1-256 independent episodes per update) |
| `--seed` | `master_seed` (u64 or None for automatic setup) |
| `--opponent` | `actor_opponent` |
| `--max-steps` | `max_episode_steps` (positive or None) |

Other training-setting flags replace hyphens with underscores. `--resume`,
`--max-groups`, `--output` and `--checkpoint` control the runner, not the settings
dataclass. A resumed session retains saved settings; use only the resume,
additional-group and destination flags. Required model/checkpoint paths must be
distinct. The collector supports only registered learner environments; discovering
a game ID does not enable this training algorithm for it.
