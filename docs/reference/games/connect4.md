# Connect 4 rules, state and encoding

Use the [game manual](../../guides/games/connect4.md) for commands and rules sources.
This page records the game-specific meanings of configuration, actions and
observation arrays. Shared API usage and algorithms are explained in the linked
references rather than repeated here.

## Configuration and state

`games::connect4` exports `Connect4ForwardModel`, `Connect4Parameters`,
`Connect4State`, `Connect4Observation`, `DropPiece` and `Connect4Evaluator`.

| Parameter | Default | Requirement |
|---|---|---|
| Player count | 2 | Exactly 2 |
| `width` | 7 | Positive number of columns |
| `height` | 6 | Positive number of rows |
| `winning_length` | 4 | Positive; at most max(width,height) |
| Native `limits` | `BoardLimits::default()` | Grid allocation policy; checked multiplication and cell limit |

The state retains the board, winning length, current player, progress counters,
outcome and winning cells. Coordinates start at zero: x increases rightward and
y downward. Seat 0 starts. A valid move drops the actor's piece to the largest
empty y in the selected column. A winning move terminates immediately; otherwise
a full board draws, or the other player acts next. Invalid actions preserve state.

All rule-relevant facts are public. Observations and cloned states own independent
copies. `PublicInformationModel` reconstructs a simulation from those facts;
`SimulationModel` uses that deterministic reconstruction without chance draws.
For save formats and validation, see [game persistence](../persistence/recordings.md).

## Actions and observation arrays

`DropPiece { column }` has the fixed action ID `column`, from 0 to width-1.
Actions are offered left to right for nonfull columns. Full columns retain their
IDs but have a false legal-action mask entry. Nonacting and terminal observations
offer no actions.

Both default observation arrays are float32:

| Array | Shape | Meaning |
|---|---|---|
| `cells` | [width*height] | Index x*height+y; own piece +1, opponent -1, empty 0 |
| `board` | [3,height,width] | Channels own/opponent/empty; one-hot occupancy at [channel,y,x] |

Encoding identity records `connect4`, rule/encoding versions 1, two players and
the configured dimensions/winning length. Models must match that identity.
For worked array examples and common progress/results metadata, see
[observation arrays](../learning/observation-arrays.md); for mask/ID conversion,
see [action IDs](../learning/action-ids.md).

`Connect4Evaluator` estimates open winning-length lines in [-0.9,0.9]; terminal
values are win +1, loss -1, draw 0. Gravity can prevent an empty cell in a promising
line being playable immediately. See [heuristic lines](../learning/heuristic-lines.md)
for the shared representation. Neural search declares its value perspective
through the [model contract](../learning/neural-mcts.md#player-values-and-model-compatibility).

Source: [rules](../../../src/games/connect4/forward_model.rs),
[encoding](../../../src/games/connect4/learning.rs),
[state evaluation](../../../src/games/connect4/evaluation.rs).
