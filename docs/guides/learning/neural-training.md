# Train a Neural OSLA model

This workflow first learns from Rust OSLA examples, then improves a policy/value
network through actor-critic learning. Complete [Python setup](../setup.md),
including PyTorch and Gymnasium. For Neural MCTS, use [self-play](mcts-selfplay.md) instead.

## Train and save

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --game connect4 --output target/models/connect4-osla.json --checkpoint target/models/connect4-osla.pt
```

Use `--game tictactoe` with different output paths for Tic-Tac-Toe. Defaults use
256 teacher episodes, 1,024 actor-critic episodes and two 64-unit hidden layers.
For a quick check, add `--teacher-episodes 5 --actor-episodes 5 --batch-size 3`.

## Configure the experiment

| Option | Purpose |
|---|---|
| `--teacher-opponent` | Opponent during OSLA-example collection; default `osla` |
| `--opponent` | Opponent during actor-critic learning; default `random` |
| `--teacher-episodes`, `--actor-episodes` | Games in each phase |
| `--batch-size` | Complete episodes per training update |
| `--learning-rate`, `--hidden-sizes` | Optimization/model settings |
| `--seed`, `--learner` | Experiment seed; omit learner to alternate seats |
| `--max-groups` | Stop after a bounded number of groups and save |

## Outputs

The `.json` contains evaluation weights. The `.pt` contains resumable training
state. Both selected destinations are replaced when saving; use separate names
for separate experiments. Use `--help` for the complete option list.

## Resume

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --resume target/models/connect4-osla.pt --output target/models/connect4-osla.json --checkpoint target/models/connect4-osla.pt
```

Resume restores the saved schedule/settings. It does not extend a completed
schedule automatically; set the desired episode counts when starting the run.

## Evaluate and play

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_neural.py --model target/models/connect4-osla.json --mode osla --episodes 100
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_neural.py --model target/models/connect4-osla.json --mode policy --opponent random --episodes 100
```

In the [browser](../running/browser.md), select the matching default game and **Neural OSLA**,
then load/select the `.json`. This model uses discounted observer values; it is
not the gamma=1 model required by the native Neural MCTS match/tournament agent.

## Reference
 [training parameters, losses and checkpoint compatibility](../../reference/learning/neural-training.md).

[Every CLI flag and default](../../reference/python/command-line.md#neural_trainpy).
