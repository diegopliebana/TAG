# Turn a player's observation into learning inputs

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

This is the starting point for the learning and Python guides. The Rust engine
describes and produces numeric inputs for Tic-Tac-Toe, Connect 4, Can't Stop and
Love Letter.
The compiled Python adapter can create these games, return NumPy arrays, copy
states and apply moves. For supported placement-game controllers, OSLAPython
prepares candidate batches for external
evaluation, then selects and commits a move in Rust. `advance()`/`resume()` can also
keep the game loop in Rust, returning when external input is needed. Start with
[Rust-owned execution](../python/rust-owned-execution.md) to run agents from Python.
[Single-learner Gymnasium episodes](../python/gym-episodes.md) provide a
learner-step environment for Tic-Tac-Toe and Connect 4.
[Bulk episodes](../python/batched-episodes.md) advance compatible independent environments
through one Rust call with explicit reset selection and validity masks.
The pages here explain supported behaviour and its current limits.

Start with [the project introduction](../../start-here.md) for game, agent and
observation terminology. The [Rust reading guide](../framework/rust-for-java-cpp.md) explains
traits, associated types, references and error handling.

## What the learning code does

Game rules work with types such as `TicTacToeState` and `PlaceMark`. Learning
algorithms use numeric inputs, such as a vector describing which cells are
occupied. An **encoding** specifies how information from a player's observation
becomes those numbers.

There are two steps. First, `learning_spec` (or `learning_spec_for_players` for
a configurable player count) describes the names, dimensions and
meanings of the available arrays. Then, `encode` fills requested arrays for one
observation. Both operations belong to the `LearningGame` trait implemented by
the existing game model. The game defines these meanings once in Rust, so a
generic Python adapter can discover them.

The description is a `LearningSpec`. The values for one observation are an
`EncodedObservation`. The API calls each named observation representation a **payload**. The placement
games offer `cells` and `board`. Can't Stop supplies named track/dice/context
arrays for 2-4 players; its [reference](../games/cantstop.md#python-game-operations-and-learning-arrays)
owns their exact layout. Python supports its [neural search](neural-mcts.md),
[self-play training](mcts-selfplay.md) and [frozen-model evaluation](model-evaluation.md).

## Model inputs and support for the Python heuristic

The encoding also provides information for testing Python evaluation. These two kinds
of data have different purposes:

| Data | Purpose | When it changes |
|---|---|---|
| Observation arrays (`cells`, `board`) | Describe the current position from a player's viewpoint. These are the inputs intended for learning models. | Values change with the position or observer. |
| Heuristic line descriptions (`LearningSpec.heuristic`) | Tell the hand-written Python evaluator which cells form each line it should score. | Remain fixed for a given board configuration and description version. |

The line descriptions support the comparison between native OSLA and
OSLAPython using the same hand-written scoring formula. They are stored alongside
the array descriptions for convenience. They are not supplied as model inputs in
the current design. Progress, results and action masks have separately documented
roles in describing the situation, constructing training targets and choosing legal
actions; each learning algorithm must specify which of these it uses.

## A first example

On a 3 by 3 Tic-Tac-Toe board, suppose Player 0 places X in the center. From that
player's viewpoint, the `cells` observation vector contains nine floating-point values:

```text
[0, 0, 0, 0, 1, 0, 0, 0, 0]
```

Each entry describes a board cell: +1 is an own piece, -1 an opposing piece, and
0 an empty cell. The center has index 4. An observation for Player 1 gives -1 at
that index because X belongs to the other player.

The same board can also be requested as `board`, with shape `[3, 3, 3]`: three
channels, three rows and three columns. The channels separately mark own pieces,
opposing pieces and empty cells. This representation contains 27 values.

For a standard Connect 4 board, `cells` has 42 values and `board` has 126 values
with shape `[3, 6, 7]`. The meanings of own, opposing and empty stay the same.
The [array walkthrough](observation-arrays.md) shows the actual layouts for both
games and explains the information returned alongside the arrays.

## Choose how far to go

For a first Python run, follow [setup](../python/setup.md),
[game operations](../python/game-api.md), then [Rust-owned execution](../python/rust-owned-execution.md).
Continue to grouped evaluation when running several games, or to adding an
evaluator when supplying your own values. To implement a game's learning inputs,
start with the LearningGame walkthrough and finish with the adding-support checklist.
The topic table lets you look up each part in more detail:

| Page | What you will learn |
|---|---|
| [Batched episodes](../python/batched-episodes.md) | Advance independent learner environments together, preserve final outputs, explicitly reset selected members and interpret validity/errors. |
| [Gymnasium episodes](../python/gym-episodes.md) | Run reset/step with one learner seat and a selectable Rust opponent; understand masks, rewards, endings and failed episodes. |
| [Grouped evaluation](../python/grouped-evaluation.md) | Combine compatible candidates from independent games for one Python evaluation call. |
| [Rust-owned execution from Python](../python/rust-owned-execution.md) | Configure agents in Python, run native decisions in Rust and resume after external evaluations. |
| [The LearningGame trait](learning-game.md) | What its interface operations provide and how the placement games implement them. Includes Rust calls with explained results. |
| [Observation arrays](observation-arrays.md) | What every number means, how cells become vectors and channels, how the observer changes the values, and what progress/results contain. |
| [Action IDs and legal choices](action-ids.md) | How a move becomes a fixed ID, how masks describe available choices, and how to decode a selection. |
| [Descriptions, versions and limits](data-descriptions.md) | How to read LearningSpec and TensorSpec, check compatibility, choose requests and handle errors. Also introduces the image/graph descriptions. |
| [Support for the Python heuristic evaluator](heuristic-lines.md) | How fixed cell groups support hand-written scoring for the OSLA comparison, separately from learning-model inputs. |
| [Add learning support to a game](adding-support.md) | Which responsibilities belong to the game, which helpers are reusable, and what to test. |
| [Build the Python connection](../python/setup.md) | Set up the verified Windows/Python environment, compile the extension and run checks. |
| [Use the Python connection](../python/game-api.md) | Create games, obtain NumPy arrays, apply moves safely, copy states, reset and close games. |
| [Add an evaluator](adding-evaluator.md) | Define values and player viewpoints, implement a Python function, use grouped requests and version its settings. |
| [Collect transitions and learn a table](tabular-learning.md) | Collect complete episodes, update Tic-Tac-Toe action values, save and reload the table. |
| [Neural inference](neural-inference.md) | Build an optional PyTorch model from the observation and action specifications and obtain checked CPU policy/value predictions. |
| [Neural play and collection](neural-play.md) | Use frozen model snapshots, direct batched policy play, neural-value OSLA and native teacher/intermediate trajectory collection. |
| [OSLAPython batches](../python/external-osla.md) | Attach the new agent, inspect candidates, return values, cancel work and commit one move with reproducible RNG progress. |

## Follow the code

| File | Responsibility |
|---|---|
| [learning/mod.rs](../../../src/learning/mod.rs) | Defines the LearningGame trait. |
| [learning/types.rs](../../../src/learning/types.rs) | Defines descriptions, returned arrays, IDs, limits and errors. |
| [learning/placement.rs](../../../src/learning/placement.rs) | Creates the arrays shared by the two current piece-placement games. |
| [Tic-Tac-Toe learning implementation](../../../src/games/tictactoe/learning.rs) | Supplies square-board dimensions, X/O ownership and cell actions. |
| [Connect 4 learning implementation](../../../src/games/connect4/learning.rs) | Supplies rectangular dimensions, piece ownership and column actions. |

Run the existing example and learning tests from the repository root:

```powershell
cargo run --offline --example learning
cargo test --offline --test integration core::learning
```

The [example](../../../examples/learning.rs) prints the array shapes, action counts
and heuristic line counts for both games. The [tests](../../../tests/core/learning.rs)
check numeric values, action mappings, player viewpoints, independent storage and
error handling.

## Where Python fits

The [implemented adapter](../python/game-api.md) lets Python request these arrays
and call Rust game operations. OSLAPython prepares several candidate positions in
Rust, accepts their values from Python and selects a move in Rust. A group evaluated
together is a **batch**. The hand-written Python heuristic can evaluate those batches.
[Rust-owned execution](../python/rust-owned-execution.md) and
[independent-game grouping](../python/grouped-evaluation.md) reduce repeated
Python/Rust crossings. [Gym episodes](../python/gym-episodes.md) and
[bulk episodes](../python/batched-episodes.md) support learner-step collection.
[Tabular learning](tabular-learning.md) and [neural training](neural-training.md)
provide training, saving and reload. [Native and neural MCTS](neural-mcts.md) run with simulation in
Rust and optional batched model evaluation through Python.
[MCTS self-play training](mcts-selfplay.md) generates visit-policy and terminal-value
targets, trains the next model and saves completed-cycle checkpoints.
[Local inference](local-mcts.md) uses those exports for browser MCTS and general
neural matches/tournaments without a Python process during play. Browser MCTS
supports all four games; native local neural experiment registration supports
Tic-Tac-Toe and Connect 4.

Use the [learning index](README.md) to choose an encoding, inference or training
reference, and the [training guides](../../guides/learning/README.md) for commands.

Python implementations are grouped by topic; see the
[package map](../../../adapters/python/python/tag_rust/README.md).
