# Tic-Tac-Toe

## Run

From the repository root, run two games of MCTS against Random:

```powershell
cargo run --release --features mcts-json --example mcts -- examples/mcts/basic.json tictactoe 2
```

The arguments select the MCTS settings file, game ID (`tictactoe`) and game count.
MCTS alternates seats. The program prints resolved seeds, wins/draws/losses per
seat and combined, and search-work totals. Edit the settings file to change the
search budget or fix its seed; `null` selects and prints an automatic seed.
The Random opponent also uses OS randomness by default; append its integer seed
as a fourth argument to reproduce it. See [seed options](../running/README.md#reproducibility).
For configurable opponents, use the [match manual](../running/matches.md).

## Rules

Two players take turns marking empty cells. A complete row, column or main
diagonal wins; a full board without a winner is a draw. The default grid is 3 by 3.
Read the [Exploratorium's Tic-Tac-Toe rules](https://annex.exploratorium.edu/brain_explorer/tictactoe.html),
an educational reference for this traditional game.

## Configure the game

Use this as the `game` object in a [match](../running/matches.md) or
[tournament game entry](../running/tournaments.md):

```json
{"name": "tictactoe", "parameters": {"grid_size": 3}}
```

There are always two players. `grid_size` is a positive integer; changing it to N
creates an N by N board and requires an entire N-cell line to win. Allocation
limits also apply. This generalized size is a framework option, beyond the
standard rules. Seat 0 starts; experiment defaults rotate agent seats.

## Play in the browser

Follow [browser setup and shared controls](../running/browser.md), then select
**Tic-Tac-Toe** and choose a controller for each seat. Select an outlined empty cell to place a mark.
The board highlights the winning line. Use the shared history, fork and save
controls to review or continue play.

## Use from Python

After [Python setup](../setup.md), create the game with
`tag_rust.create("tictactoe")`. Use [game operations](../running/game-operations-from-python.md)
for observation, action IDs, copies and reset, or [run Rust MCTS from Python](../agents/rust-mcts-from-python.md)
for search decisions. The supplied neural-search demonstration is:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_mcts.py --game tictactoe --searches 4 --seed 42
```

It uses an untrained model unless `--model` supplies compatible exported weights.

## Train and evaluate

[Q-learning](../learning/tabular-learning.md) supports the default 3x3 game.
[Teacher/actor-critic training](../learning/neural-training.md) and
[MCTS self-play](../learning/mcts-selfplay.md) support the default game; use
`examples/selfplay/tictactoe.json` for self-play. Learner-step Gym episodes are
also available. Models must match the board configuration, action mapping and
value convention. [Load models](../agents/models.md) for browser or native neural
matches, and [evaluate frozen MCTS models](../learning/evaluate-mcts-models.md)
without changing their weights.

## Reference

For native Rust, use `TicTacToeForwardModel`, `TicTacToeParameters` and `PlaceMark`
from [games::tictactoe](../../../src/games/tictactoe/mod.rs).
`TicTacToeEvaluator` estimates promising open lines; terminal outcomes override
its estimates. The [Random example](../../../examples/random.rs) shows the generic runner.

The [game reference](../../reference/games/tictactoe.md) defines parameters,
state/action semantics and observation arrays.
