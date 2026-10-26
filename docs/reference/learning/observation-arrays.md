# Read the arrays for one observation

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md) · [LearningGame calls](learning-game.md)

`EncodedObservation` holds the requested numeric arrays and information about
the player, game progress, results and actions. Each requested array is a
`Float32Tensor`, containing a `name`, a `shape` and a flat `values` vector of
32-bit floating-point numbers. Shape tells us how to interpret that vector as a
multidimensional array.

## From a Tic-Tac-Toe board to cells

Use this legal two-move position: Player 0 placed X at `(1, 2)`, then Player 1
placed O at `(2, 0)`. Coordinates start at zero; x increases rightward and y
downward. Player 0 observes and acts next.

```text
       x=0 x=1 x=2
y=0     .   .   O
y=1     .   .   .
y=2     .   X   .
```

![The board becomes nine cell values, read down each column. X at (1,2) has index 5 and value 1; O at (2,0) has index 6 and value -1.](../../images/learning-board-vector.svg)

For `cells`, read down column 0, then column 1, then column 2. This is **x-major
order**, with `index = x * height + y`. Here `height` is 3:

```text
coordinates: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2)
index:         0     1     2     3     4     5     6     7     8
value:         0     0     0     0     0    +1    -1     0     0
```

The shape is `[9]` because there are nine cells. Each cell contributes one number:
+1 for an own piece, -1 for an opposing piece, or 0 for an empty cell.

## From the same board to three channels

Requesting `board` gives shape `[3, 3, 3]`, meaning `[channel, row, column]`.
A **channel** is a grid answering one question about each cell. The channels,
in order, mark own pieces, opposing pieces and empty cells:

```text
own (channel 0)       opposing (channel 1)   empty (channel 2)
0 0 0                 0 0 1                  1 1 0
0 0 0                 0 0 0                  1 1 1
0 1 0                 0 0 0                  1 0 1
```

![Three grids for the same board mark own, opposing and empty cells. The array axes are channel, row and column.](../../images/learning-board-tensor.svg)

Each channel contains nine numbers, giving `3 * 3 * 3 = 27` values in total.
At every cell exactly one channel is 1 and the other two are 0. Thus an empty
board has zeros in the first two channels and ones throughout the empty channel.

The flat `values` vector stores each channel row by row, then proceeds to the next
channel. For width W and height H, the flat index for `[channel, y, x]` is:

```text
channel * (W * H) + y * W + x
```

For X at `(1, 2)`, the own-channel value is at `0 * 9 + 2 * 3 + 1 = 7`.
The `cells` value for X is at index 5. Building `board` therefore requires both
creating the three channels and arranging each in row-by-row order.

## Connect 4 uses the same meanings on a larger board

On the standard seven-column, six-row board, let Player 0 drop a piece into column
2. It lands at `(2, 5)`, the bottom cell. Player 1 observes and acts next.

The `cells` shape is `[42]`: seven columns times six rows. Its value at index
`2 * 6 + 5 = 17` is -1, since the piece belongs to Player 0. The other 41 values
are 0. Grouping the vector by column makes this visible:

```text
column 0: [0, 0, 0, 0, 0,  0]
column 1: [0, 0, 0, 0, 0,  0]
column 2: [0, 0, 0, 0, 0, -1]
column 3: [0, 0, 0, 0, 0,  0]
column 4: [0, 0, 0, 0, 0,  0]
column 5: [0, 0, 0, 0, 0,  0]
column 6: [0, 0, 0, 0, 0,  0]
```

`board` has shape `[3, 6, 7]` and 126 values. The opposing channel is 1 at
`[1, 5, 2]`, flat index `1 * 42 + 5 * 7 + 2 = 79`. The empty channel is 0 at
that cell and 1 at the other 41 cells. The own channel is entirely zero.

Observing the same position as Player 0 changes the piece's `cells` value to +1
and puts the 1 in the own channel. The empty channel stays the same.

## Information returned with every request

`encode` always returns the following fields, including when called with an
empty list of requested observation arrays `&[]`. In that case `tensors` is empty.

| Field | Contents |
|---|---|
| `identity` | The game, versions, dimensions, target length and player count used to interpret these values. |
| `observer` | The PlayerId whose viewpoint determines own and opposing pieces. |
| `next_player` | `Some(PlayerId)` when a player acts next; `None` when play has ended. |
| `progress` | Four int64 counters in the order explained below. |
| `results` | One ResultTag per player in absolute seat order: Player 0, then Player 1. |
| `action_mask` | One Boolean for every representable action ID; true means the action is offered to this observer now. |
| `offered_action_ids` | IDs for the offered actions in the game's offered-action order. |

These fields are often called **metadata** because they describe the situation
and how to interpret or use the arrays.

`progress` contains `[moves, completed turns, turns in the current round, completed
rounds]`. After the two nonterminal Tic-Tac-Toe moves above it is `[2, 2, 2, 0]`.
Tic-Tac-Toe keeps the round count at zero. After the first Connect 4 move it is
`[1, 1, 1, 0]`; after both players have made a nonterminal move it becomes
`[2, 2, 0, 1]`. Connect 4 completes a round after that second turn.

A terminal move increments the move count while retaining the completed-turn
counters from before that move. The rules determine when turns and rounds end;
the encoder copies those counters into the agreed order.

`results` contains enum labels with numeric codes: `Ongoing=0`, `Win=1`, `Draw=2`,
`Loss=3`. The ongoing examples have `[Ongoing, Ongoing]`. If Player 0 wins, results
become `[Win, Loss]` for either observer. A learning algorithm separately defines
the reward associated with each outcome.

The two-move Tic-Tac-Toe example has a nine-entry mask and seven offered IDs for
Player 0. Player 1 can still observe this board, but receives an all-false mask
and an empty offered list until it is their turn. At completion, both players'
masks are all false. See [action IDs](action-ids.md) for worked masks.

## Keeping the data independent

The returned arrays own their storage. Editing `Float32Tensor.values` changes
only that encoded array. Keeping an earlier encoded observation lets you inspect
those earlier numbers after the game advances. The encoder reads its input
observation through a reference and leaves it unchanged.

Read [descriptions and versions](data-descriptions.md) to see how callers discover
these layouts, or [LearningGame](learning-game.md) for Rust calls producing them.
