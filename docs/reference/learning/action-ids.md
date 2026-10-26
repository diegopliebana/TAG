# Action IDs and legal choices

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md) · [LearningGame calls](learning-game.md)

A network that produces one output per possible action needs each output to keep
the same meaning as the game progresses. An **action ID** identifies that move
within a particular game configuration. `LearningSpec.actions` describes how many
IDs exist and how to interpret them.

## Tic-Tac-Toe identifies cells

For a 3 by 3 board, there are nine IDs, using `id = x * height + y`:

```text
       x=0 x=1 x=2
y=0     0   3   6
y=1     1   4   7
y=2     2   5   8
```

`PlaceMark { x: 2, y: 1 }` therefore has ID 7. Decoding ID 7 divides by the
height: `x = 7 / 3 = 2` and `y = 7 % 3 = 1`, where `/` is integer division and
`%` gives the remainder. The mapping is named `ActionMapping::CellXMajor`.

Suppose X occupies `(1, 2)` and O occupies `(2, 0)`, as in the
[array example](observation-arrays.md). Player 0 acts next. The occupied cells
have IDs 5 and 6, so the action information is:

```text
action_mask:        [true, true, true, true, true, false, false, true, true]
offered_action_ids: [0, 1, 2, 3, 4, 7, 8]
```

The mask has nine entries because every cell retains its ID. The offered list
has seven entries because only seven cells are empty. If an agent returns list
index 5, it selects action ID 7, the cell `(2, 1)`.

The number of IDs depends on the board dimensions. A 5 by 5 game has 25 IDs and
a mask of length 25. Within that configuration, each ID keeps its cell meaning.

## Connect 4 identifies columns

Connect 4 actions specify a column. The rules determine where the piece lands.
The mapping `ActionMapping::Column` uses the column number directly, so
`DropPiece { column: 2 }` has ID 2. A default seven-column board has seven IDs,
regardless of its current piece count.

While all columns have room, the mask and offered IDs are:

```text
action_mask:        [true, true, true, true, true, true, true]
offered_action_ids: [0, 1, 2, 3, 4, 5, 6]
```

Once column 0 fills, they become:

```text
action_mask:        [false, true, true, true, true, true, true]
offered_action_ids: [1, 2, 3, 4, 5, 6]
```

The first offered action now has list index 0 and action ID 1. Network output 1
still represents column 1. An algorithm selecting among current list entries can
use `offered_action_ids` to translate its selected index back to the fixed ID.

The mask describes choices offered to the encoded observer. It is all false if
another player acts next or if the game has ended. The observation still contains
the board; the mask identifies who can act on it now.

## Decode, then apply through the rules

For richer action spaces, `ActionMapping::Indexed` declares a fixed set whose
meaning is defined by the game's `LearningGame` methods. `Candidates` declares
a variable list (`size=0` in the static description); each decision supplies
features and an explicit mapping back to concrete actions. Placement coordinate
helpers reject these mappings. The [neural candidate guide](neural-inference.md#score-supplied-action-candidates)
describes implemented Python batching/scoring and the remaining native executor
integration required for a new candidate-action game.

`action_id` and `action_from_id` check the coordinates or ID against the configured
board. On a 3 by 3 board, decoding ID 9 returns an error because valid IDs run
from 0 to 8. Decoding an occupied cell's ID constructs its `PlaceMark`; applying
that action through `ForwardModel::next` returns the game's occupied-cell error.
Connect 4 similarly decodes an in-range column even when it is full.

A caller choosing a network output should follow these steps:

1. Read `action_mask` from the observation used to produce the network inputs.
2. Select among IDs whose mask entry is true.
3. Call `action_from_id` with that observation and the selected ID.
4. Apply the resulting action through the game's normal decision/session interface,
   which checks the player, current position and game rules.

A request can become outdated while external code is working. The
[Python connection](../python/game-api.md) requires the position token from the
observation when applying a move; it rejects old and cross-game tokens. The next
agent stage will add pending-evaluation request checks. The numeric encoder
provides the mapping and the observation's choices.

## Lower-level coordinate helpers

`ActionSpec::id(identity, x, y)` and `coordinates(identity, id)` implement the
shared conversion. For column actions, their coordinate convention uses `y=0`;
this value identifies the column for conversion. Connect 4's forward model
calculates the actual landing row when applying the move.

Use the game-level `action_id` and `action_from_id` methods when working with
`PlaceMark` or `DropPiece`. They supply the appropriate dimensions and construct
the concrete action for you. See the [LearningGame examples](learning-game.md#translate-actions-with-action_id-and-action_from_id).
