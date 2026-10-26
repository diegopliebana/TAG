# Tic-Tac-Toe rules, state and encoding

Use the [game manual](../../guides/games/tictactoe.md) for commands and rules sources.
This page records the game-specific meanings of configuration, actions and
observation arrays. Shared API usage and algorithms are explained in the linked
references rather than repeated here.

## Configuration and state

`games::tictactoe` exports `TicTacToeForwardModel`, `TicTacToeParameters`,
`TicTacToeState`, `TicTacToeObservation`, `PlaceMark` and `TicTacToeEvaluator`.

| Parameter | Default | Requirement |
|---|---|---|
| Player count | 2 | Exactly 2 |
| `grid_size` | 3 | Positive N; board dimensions and winning-line length are N |
| Native `limits` | `BoardLimits::default()` | Grid allocation policy; checked multiplication and cell limit |

The state retains the board, current player, progress counters, outcome and winning
cells. Coordinates start at zero: x increases rightward and y downward. Seat 0
places X first; seat 1 places O. A move fills one empty cell. A full row, column or
corner-to-corner diagonal wins; otherwise a full board draws, or the other player
acts next. Invalid actions preserve state.

All rule-relevant facts are public. Observations and cloned states own independent
copies. `PublicInformationModel` reconstructs a simulation from those facts;
`SimulationModel` uses that deterministic reconstruction without chance draws.
For save formats and validation, see [game persistence](../persistence/recordings.md).

## Actions and observation arrays

`PlaceMark { x, y }` has the fixed action ID `x*N+y`, from 0 to N*N-1. For N=3,
(2,1) has ID 7. Empty cells are offered column by column, with increasing y within
each column. Occupied cells retain their IDs but have false legal-action mask
entries. Nonacting and terminal observations offer no actions.

Both default observation arrays are float32:

| Array | Shape | Meaning |
|---|---|---|
| `cells` | [N*N] | Index x*N+y; own mark +1, opponent -1, empty 0 |
| `board` | [3,N,N] | Channels own/opponent/empty; one-hot occupancy at [channel,y,x] |

Encoding identity records `tictactoe`, rule/encoding versions 1, two players and
width=height=target=N. Models must match that identity. For worked array examples
and common progress/results metadata, see [observation arrays](../learning/observation-arrays.md);
for mask/ID conversion, see [action IDs](../learning/action-ids.md).

`TicTacToeEvaluator` estimates open full-length lines in [-0.9,0.9]; terminal
values are win +1, loss -1, draw 0. See [heuristic lines](../learning/heuristic-lines.md)
for the shared representation. Neural search declares its value perspective
through the [model contract](../learning/neural-mcts.md#player-values-and-model-compatibility).

Source: [rules](../../../src/games/tictactoe/forward_model.rs),
[encoding](../../../src/games/tictactoe/learning.rs),
[state evaluation](../../../src/games/tictactoe/evaluation.rs).
