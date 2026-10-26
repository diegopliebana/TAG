# Train a Q-table

This small example learns values for Tic-Tac-Toe actions from played games.
It supports the default 3×3 game. Complete [Python setup](../setup.md) first.

## Train and save

```powershell
New-Item -ItemType Directory -Force target/models | Out-Null
.\.venv\Scripts\python.exe adapters/python/examples/tabular_q.py --output target/models/tictactoe-q.json
```

The command evaluates the initial table, trains, saves JSON, reloads it and
evaluates the reloaded table. Defaults train 1,024 episodes across both seats.
For a short workflow check, add `--rounds 4 --eval-rounds 2 --members 4`.

## Configure the experiment

| Option | Purpose |
|---|---|
| `--opponent random` | Training opponent; also `first_available`, `osla`, `osla_python` |
| `--rounds`, `--members` | Training groups and independent games per group |
| `--alpha`, `--gamma`, `--epsilon` | Learning rate, discount and exploration |
| `--seed` | Reproducible experiment seed |
| `--learner 0` or `1` | Fixed seat; omit to alternate seats |
| `--output` | Saved Q-table path; choose a fresh experiment name |

Use `--help` for all options.

## Outputs

`--output` writes the learned Q-table as JSON; the example reloads it to check
saved evaluation behaviour. Keep a separate path for each experiment. This is a
small state/action-value table, not a neural network or optimizer checkpoint.

## Resume

The CLI does not save a complete resumable training session. The saved table can
be evaluated or loaded through the [QTable API](../../reference/learning/tabular-learning.md)
for further updates; it does not retain the entire experiment schedule/history.

## Evaluate and play

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_tabular_q.py --model target/models/tictactoe-q.json --games 1000 --opponent osla --seed 90001
```

In the [browser](../running/browser.md), choose default Tic-Tac-Toe, select **Q-learning**
for a seat and load/select `target/models/tictactoe-q.json`.
The general native match/tournament CLI does not currently register Q-table agents.

## Reference
 [Q-learning, transitions, file format and seat scheduling](../../reference/learning/tabular-learning.md).

[Every CLI flag and default](../../reference/python/command-line.md#tabular_qpy).
