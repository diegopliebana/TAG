# Connect 4

## Run

From the repository root, run two games of MCTS against Random:

```powershell
cargo run --release --features mcts-json --example mcts -- examples/mcts/basic.json connect4 2
```

The arguments select the MCTS settings file, game ID (`connect4`) and game count.
MCTS alternates seats. The program prints resolved seeds, wins/draws/losses per
seat and combined, and search-work totals. Edit the settings file to change the
search budget or fix its seed; `null` selects and prints an automatic seed.
The Random opponent also uses OS randomness by default; append its integer seed
as a fourth argument to reproduce it. See [seed options](../running/README.md#reproducibility).
Use [match configuration](../running/matches.md) to select other opponents;
[the supplied matchup](../../../examples/matches/mcts-vs-osla.json) uses MCTS versus OSLA.

## Rules

Two players alternate dropping a disc into an available column. It falls to the
lowest empty cell. Four consecutive discs horizontally, vertically or diagonally
win; a full board without a winner is a draw. The standard board has seven
columns and six rows. Hasbro's
[official instructions page](https://instructions.hasbro.com/en-us/instruction/Connect-4-Game)
includes the downloadable rulebook. The framework implements the classic dropping
game; other branded Connect 4 variants have separate rules.

## Configure the game

Use this as the `game` object in a [match](../running/matches.md) or
[tournament game entry](../running/tournaments.md):

```json
{"name": "connect4", "parameters": {"width": 7, "height": 6, "winning_length": 4}}
```

There are always two players. Width, height and winning length must be positive;
winning length cannot exceed the larger dimension. Allocation limits also apply.
Other dimensions/targets are framework options beyond the standard game. Seat 0
starts; experiment defaults rotate agent seats.

## Play in the browser

Follow [browser setup and shared controls](../running/browser.md), then select
**Connect 4** and choose a controller for each seat. Select an available column to drop a disc; gravity is applied by Rust.
The board highlights the winning line. Use the shared history, fork and save
controls to review or continue play.

## Use from Python

After [Python setup](../setup.md), create the game with
`tag_rust.create("connect4")`. Use [game operations](../running/game-operations-from-python.md)
for observation, action IDs, copies and reset, or [run Rust MCTS from Python](../agents/rust-mcts-from-python.md)
for search decisions. The supplied neural-search demonstration is:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_mcts.py --game connect4 --searches 4 --seed 42
```

It uses an untrained model unless `--model` supplies compatible exported weights.

## Train and evaluate

[Teacher/actor-critic training](../learning/neural-training.md) and
[MCTS self-play](../learning/mcts-selfplay.md) support the default game; use
`examples/selfplay/connect4.json` for self-play. Learner-step Gym episodes are
also available. Models must match the board configuration, action mapping and
value convention. [Load models](../agents/models.md) for browser or native neural
matches, and [evaluate frozen MCTS models](../learning/evaluate-mcts-models.md)
without changing their weights.

## Reference

For native Rust, use `Connect4ForwardModel`, `Connect4Parameters` and `DropPiece`
from [games::connect4](../../../src/games/connect4/mod.rs).
`Connect4Evaluator` estimates promising open lines; terminal outcomes override
its estimates. The [Random example](../../../examples/random.rs) shows the generic runner.

The [game reference](../../reference/games/connect4.md) defines parameters,
state/action semantics and observation arrays.
