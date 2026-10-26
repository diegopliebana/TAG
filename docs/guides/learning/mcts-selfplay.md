# Train a Neural MCTS model

Self-play uses MCTS for every player, then trains a network from search visit
counts and final game outcomes. The next cycle uses the updated network.
Complete [Python setup](../setup.md), including PyTorch.

[Rust/Python evaluation diagram](../../reference/learning/neural-mcts.md#advance-evaluate-and-resume) shows the search boundary; [artifact diagram](../../reference/persistence/README.md#choose-the-file-for-the-task) distinguishes models from checkpoints.

## Train and save

### Check the complete workflow quickly

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/quickcheck.json --max-cycles 1
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --resume examples/selfplay/results/quickcheck-training.pt --max-cycles 1
```

This [small configuration](../../../examples/selfplay/quickcheck.json) runs two
cycles of two complete games, four search iterations per decision and one update
per cycle. The first command saves a completed-cycle checkpoint; the second
continues it. Inspect `quickcheck-results.json` beside the model/checkpoint for
cycle counts, losses and identities. Outputs are replaced on a fresh run, so use
a copied configuration with different output paths for another experiment.
This checks operation, not playing strength.


### Train one cycle

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/connect4.json --max-cycles 1
```

For Tic-Tac-Toe, use `examples/selfplay/tictactoe.json`. These two placement configurations
request ten cycles. One cycle generates 16 games, performs 64 training updates
and evaluates 20 games against native MCTS. Omit `--max-cycles 1` to run all cycles.

### Train Can't Stop with 2-4 players

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/cantstop.json --max-cycles 1
```

This configuration uses the default rules, three players, three games per cycle,
eight MCTS iterations per decision and eight training updates. It requests two
cycles. All seats use the same frozen network during collection; each decision
records the actual acting seat. Reports include game seeds, per-seat outcomes and
combined player appearances. Opponent evaluation is disabled in this short training
example; [model evaluation](evaluate-mcts-models.md) shows how to enable it.

Change `players` to 2 or 4 in a copy of the configuration for those player counts.
Keep `model.value_output: "players"`: Can't Stop learns one terminal value per
absolute seat, using win +1, loss -1, draw 0 and gamma=1. The selected observation
arrays and 80 action IDs come from Rust. A model is specific to its player count
and rules. Optional rule changes go in `parameters.rules`.

Set top-level `seed` to an integer or null for OS randomness; the report/checkpoint
records the resolved master seed. The workflow derives and records a separate
dice seed for every game. Omit `parameters.seed` because game seeds come from that
master seed. The CLI prints the resolved master before collection, so a failing
automatic run can also be reproduced. Seeds do not enter observations or model inputs.

The example saves `cantstop-model.json`, `cantstop-training.pt` and
`cantstop-results.json` under `examples/selfplay/results/`. Continue with:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --resume examples/selfplay/results/cantstop-training.pt --max-cycles 1
```

A checkpoint resumes after a completed cycle. If a game reaches `limits.max_moves`
or a collection limit, the collection group fails without adding partial-game
targets. A failed cycle cannot replace the last successful checkpoint. Resume
from that checkpoint; mid-game/search resume is not implemented. Completing these
short runs demonstrates the workflow, not improved playing strength.

### Train Love Letter

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/loveletter.json --max-cycles 1
```

The supplied configuration uses three players, default classic rules, three
complete games per cycle, eight MCTS iterations per decision and eight updates.
It requests two cycles. Change `players` to 2 or 4 in a copy; each model is specific
to its player count and rules. Starting seats rotate by logical game ID. Set
`parameters.first_player` for a fixed starter. Set top-level `seed` to an integer
or null for OS randomness; omit `parameters.seed`.

Keep `model.value_output: "players"`. The short example disables opponent evaluation. The model
uses one acting player's permitted observation and learns a whole-game value for
every seat: each winner +1, each loser -1. Round elimination is nonterminal.
Search inputs exclude `last_private_history`; the example selects supported
arrays. Incomplete collection groups fail without adding targets or updates.

This example saves to the repository's `results/` directory. Resume with:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --resume results/loveletter-training.pt --max-cycles 1
```

Use `results/loveletter-model.json` with the
[Python MCTS demonstration](../agents/rust-mcts-from-python.md). Checkpoints, reports
and exported-model provenance are labelled privileged research data: they contain
private experiences or seeds that reproduce deals. Model inputs remain a single
player's permitted view. The [game reference](../../reference/games/loveletter.md#complete-game-self-play-and-training)
explains replay validation and reproducibility. Use the shared
[model evaluation guide](../games/loveletter.md#train-and-evaluate)
to test an exported model or enable evaluation after training cycles. Load compatible exports through the [browser model picker](../running/browser.md).
These small runs establish operation, not playing strength.

## Configure the experiment

| JSON setting | Controls |
|---|---|
| `game`, `parameters` | Game and rules; the model must match them |
| `cycles`, `games_per_cycle`, `parallel_games` | Total cycles and generated games |
| `search_config_file` or `search` | MCTS budgets, policies and evaluators |
| `model.hidden_sizes`, `model.value_output` | Network layers; `observer` scalar or `players` vector |
| `data.mode` | Default `replay_buffer`, or only the `newest_group` |
| `data.capacity`, `data.max_bytes` | Retained training-data bounds |
| `optimization` | Updates per cycle, batch size, learning rate and loss weights |
| `exploration` | Independent visit sampling and root noise |
| `evaluation` | Separate opponent games; default alternating seats |
| `seed`, `output` | Reproducibility and output destinations |

## Outputs

Paths in the configuration resolve from its directory. The Connect 4 example saves:

| File | Use |
|---|---|
| `examples/selfplay/results/connect4-model.json` | Load into a Neural MCTS agent |
| `examples/selfplay/results/connect4-training.pt` | Resume training |
| `examples/selfplay/results/connect4-results.json` | Inspect settings, losses, evaluation outcomes and model identities |

Saving replaces these destinations. Copy the configuration and change `output`
paths to keep a separate experiment. A fresh run is not a resume.

## Resume

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --resume examples/selfplay/results/connect4-training.pt --max-cycles 1
```

Resume uses settings and destinations in the checkpoint; omit `--config`.
It continues at a completed-cycle boundary and stops at the configured cycle count.

## Evaluate and play

For Tic-Tac-Toe and Connect 4, in the [browser](../running/browser.md), select the matching game, **Neural MCTS**, and
load/select the `-model.json`. Use the same export in
[single matches or tournaments](../agents/models.md). Do not load the `.pt` for play.

For Can't Stop, load the exported model into Python MCTS:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/cantstop_mcts.py --players 3 --searches 3 --seed 42 --model examples/selfplay/results/cantstop-model.json
```

Use the matching player count/rules. See [Run Rust MCTS from Python](../agents/rust-mcts-from-python.md)
for applying recommendations. To play against the exported model in the
[browser](../running/browser.md), select Can't Stop with the same player count
and rules, choose **Neural MCTS** for an opponent, and load/select its `-model.json`.
Use [model evaluation](evaluate-mcts-models.md) for opponent lineups and reports.

## Reference

[Complete CLI flags and defaults](../../reference/python/command-line.md#selfplay_trainpy).

[every self-play parameter and checkpoint rule](../../reference/learning/mcts-selfplay.md),
[MCTS search settings](../agents/mcts.md).
