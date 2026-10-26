# The LearningGame trait

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md)

`LearningGame` gives callers a common way to discover a game's numeric inputs,
encode an observation, and translate between actions and numeric IDs. A generic
caller can perform these operations for any implementing game using the same method
names. Each game supplies the meanings appropriate to its rules.

## How it fits beside the rules

The declaration begins with `pub trait LearningGame: ForwardModel`. This requires
every type implementing `LearningGame` to also implement `ForwardModel`, the
interface for running the game. `LearningGame` uses that model's associated
`Parameters`, `Observation` and `Action` types.

For example, `TicTacToeForwardModel` implements both traits. Its learning methods
accept `TicTacToeParameters`, read `TicTacToeObservation` and convert `PlaceMark`
actions. `Connect4ForwardModel` uses `Connect4Parameters`, `Connect4Observation`
and `DropPiece` instead.

Observations carry the information a particular player is allowed to know.
Tic-Tac-Toe and Connect 4 have public boards. Love Letter encodes only public
facts and the observer's own private knowledge; other players' hidden cards are
not available as model input. This requirement applies to all numeric features as well as images.

For multiplayer descriptions, call `learning_spec_for_players(parameters, players,
limits)`. The explicit player count determines array/value shapes and identity.
The default method preserves existing fixed-player implementations; Can't Stop
and Love Letter provide player-count-aware descriptions.

## Describe a configuration with learning_spec

`learning_spec(parameters, limits)` returns a `LearningSpec` describing the arrays
and actions for the supplied game settings. Here is a complete setup for a 3 by 3
Tic-Tac-Toe board. The small examples on this page use fixed valid inputs and
`unwrap()` to stop on an unexpected error.

```rust
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{LearningGame, LearningLimits};

let model = ttt::TicTacToeForwardModel;
let parameters = ttt::TicTacToeParameters {
    grid_size: 3,
    ..Default::default()
};
let limits = LearningLimits::default();
let spec = model.learning_spec(&parameters, limits).unwrap();
assert_eq!(spec.actions.size, 9);
```

`spec` advertises a `cells` vector with shape `[9]` and a `board` array with shape
`[3, 3, 3]`. It also describes four progress counters, two result labels and nine
action IDs. Its identity records the game, board dimensions, target length,
player count and relevant versions. The [description guide](data-descriptions.md)
explains these fields individually.

The same spec also carries fixed line groups in `heuristic` to support the
hand-written Python evaluator. This is supporting information for the OSLA
comparison. The arrays advertised in `payloads` are the observation inputs
intended for learning models. See [the distinction](overview.md#model-inputs-and-support-for-the-python-heuristic).

In [Tic-Tac-Toe's implementation](../../../src/games/tictactoe/learning.rs),
`learning_spec` takes width, height and winning target from `parameters.grid_size`.
It asks `placement_spec` to describe cell-based actions, then calls `init_state`
to check the parameters against the game's normal setup rules.

Connect 4 takes three separate settings: width, height and winning length:

```rust
let c4_model = c4::Connect4ForwardModel;
let c4_parameters = c4::Connect4Parameters::default();
let c4_spec = c4_model.learning_spec(&c4_parameters, limits).unwrap();
assert_eq!(c4_spec.identity.width, 7);
assert_eq!(c4_spec.identity.height, 6);
assert_eq!(c4_spec.identity.target, 4);
assert_eq!(c4_spec.actions.size, 7);
```

The default board has 42 cells, so `cells` has shape `[42]`. Its three-channel
`board` has shape `[3, 6, 7]`. It has seven action IDs because a move chooses a
column. A 4 by 3 board with a target of three would instead describe 12 cell
values, a `[3, 3, 4]` board and four action IDs.

## Encode one player's observation

`encode(observation, payloads, limits)` creates the named arrays for one
observation. Continue the Tic-Tac-Toe example by placing X in the center:

```rust
let mut state = model.init_state(&parameters, 2).unwrap();
model.next(&mut state, PlayerId::new(0), &ttt::PlaceMark { x: 1, y: 1 }).unwrap();
let observation = model.observe(&state, PlayerId::new(0)).unwrap();
let encoded = model.encode(&observation, &["cells", "board"], limits).unwrap();

assert_eq!(encoded.tensors[0].shape, vec![9]);
assert_eq!(encoded.tensors[0].values[4], 1.0);
assert_eq!(encoded.tensors[1].shape, vec![3, 3, 3]);
assert_eq!(encoded.progress, [1, 1, 1, 0]);
assert!(encoded.offered_action_ids.is_empty());
```

The request order determines the order in `encoded.tensors`: `cells` first, then
`board`. The first has nine values, including +1 at the center's index 4. The
second has 27 values: nine for each channel. Progress records one move, one
completed turn, one turn in the current round and zero completed rounds.

This observation is for Player 0 after Player 0 has moved. Player 1 acts next,
so `offered_action_ids` is empty and the nine-entry mask is entirely false.
An observation for Player 1 would encode X as -1 and offer the eight empty cells.
The board and legal choices therefore depend on different facts: whose viewpoint
we encode, and whose turn it is.

Both games supply a small cell-reading function to the shared encoder. The
Tic-Tac-Toe implementation maps `Cell::X` to Player 0 and `Cell::O` to Player 1.
Connect 4 reads the player ID already stored in `Cell::Piece(player)`.
The shared code compares that ID with the observer to choose +1/-1 or the correct
channel. It reads cells through the observation and creates separate output storage.

In each implementation, that cell reader is a **closure**, a function defined
inside `encode` that can read the surrounding observation. The closure receives
a flat cell index and returns either `None` for empty or `Some(PlayerId)` for
the owner. `PlacementView` carries this reader alongside dimensions, observer,
next decision, counters, results and offered IDs. `encode_placement` uses those
fields to produce the output. For the center X above, the reader returns
`Some(PlayerId::new(0))` at index 4; comparing it with observer 0 produces +1.

For example, Connect 4 can encode Player 0's first piece in column 2:

```rust
let mut c4_state = c4_model.init_state(&c4_parameters, 2).unwrap();
c4_model.next(&mut c4_state, PlayerId::new(0), &c4::DropPiece { column: 2 }).unwrap();
let c4_observation = c4_model.observe(&c4_state, PlayerId::new(1)).unwrap();
let c4_encoded = c4_model.encode(&c4_observation, &["cells", "board"], limits).unwrap();

assert_eq!(c4_encoded.tensors[0].values.len(), 42);
assert_eq!(c4_encoded.tensors[0].values[17], -1.0);
assert_eq!(c4_encoded.tensors[1].values.len(), 126);
assert_eq!(c4_encoded.offered_action_ids, vec![0, 1, 2, 3, 4, 5, 6]);
```

Gravity places the piece at `(2, 5)`. Its vector index is `2 * 6 + 5 = 17`.
Player 1 is observing, so the piece is opposing and its value is -1. All seven
columns have room and it is Player 1's turn, so all seven IDs are offered.

## Translate actions with action_id and action_from_id

`action_id` assigns a concrete action its numeric ID. `action_from_id` constructs
the action belonging to an ID. Using the observations above:

```rust
let cell_action = ttt::PlaceMark { x: 2, y: 1 };
assert_eq!(model.action_id(&observation, &cell_action).unwrap(), 7);
let decoded_cell = model.action_from_id(&observation, 7).unwrap();
assert_eq!((decoded_cell.x, decoded_cell.y), (2, 1));

let column_action = c4::DropPiece { column: 2 };
assert_eq!(c4_model.action_id(&c4_observation, &column_action).unwrap(), 2);
assert_eq!(c4_model.action_from_id(&c4_observation, 2).unwrap().column, 2);
```

Tic-Tac-Toe uses `x * height + y`, so `(2, 1)` gives `2 * 3 + 1 = 7`.
Connect 4 uses the column number directly. These IDs keep their meaning as the
board fills. The methods use the observation's dimensions to check that a cell,
column or ID belongs to this game configuration.

To play a decoded action, a caller must also check that it is currently offered
and apply it through `ForwardModel::next`, which enforces the rules. The
[action guide](action-ids.md) walks through occupied cells, full columns and the
difference between an action ID and its position in the available-action list.

## Where the shared work happens

Each game's `learning.rs` supplies dimensions, ownership, progress, outcomes and
the IDs of its offered actions. [placement.rs](../../../src/learning/placement.rs)
builds the descriptions and fills the arrays using that information. This shared
implementation keeps the meanings of `cells` and `board` consistent across games.

| Step inside the implementation | Tic-Tac-Toe | Connect 4 |
|---|---|---|
| Describe dimensions and winning target | Uses grid_size for width, height and target. | Uses width, height and winning_length separately. |
| Describe possible actions | One ID per cell, with CellXMajor mapping. | One ID per column, with Column mapping. |
| Read cell ownership | Translates Empty, X and O into no owner, Player 0 or Player 1. | Reads Empty or the PlayerId carried by Piece. |
| Read current choices | Calls offered_actions on the observation, then action_id on each PlaceMark. | Calls offered_actions on the observation, then action_id on each DropPiece. |
| Fill requested arrays | Passes the resulting PlacementView to encode_placement. | Passes the resulting PlacementView to the same encoder. |

The same division applies when converting IDs back into actions. Shared
`ActionSpec` helpers calculate coordinates; the Tic-Tac-Toe implementation builds
`PlaceMark { x, y }`, while Connect 4 builds `DropPiece { column: x }`.

Read [observation arrays](observation-arrays.md) next to inspect the numbers and
their layouts in detail. For another game, use the [implementation checklist](adding-support.md).
