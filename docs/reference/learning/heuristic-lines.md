# Support for the Python heuristic evaluator

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md)

The current OSLA evaluators estimate how promising a board is by examining groups
of consecutive cells. A **window** is a group long enough to win: three cells on
a 3 by 3 Tic-Tac-Toe board, or four on default Connect 4. `OpenLinesSpec` lists
those groups so the Python evaluator can use the same game-specific lines.

These descriptions support a hand-written scoring function. The function reads
the current `cells` values and uses the fixed groups to calculate its estimate.
The observation arrays are the inputs intended for learning models; the line
descriptions serve the separate OSLA/Python comparison. The current design does
not pass these descriptions to a neural network.

For example, the top-row group `[0, 3, 6]` stays the same throughout a 3 by 3 game.
The values read from `cells[0]`, `cells[3]` and `cells[6]` change as players place
pieces. This separates board structure used by the heuristic from the current
position represented by the observation arrays.

## Read the exported groups

`LearningSpec.heuristic` contains an `OpenLinesSpec` with `version`,
`window_length` and `indices`. The indices use the same x-major order as the
`cells` vector. They are stored consecutively; each `window_length` entries
describe one group.

For 3 by 3 Tic-Tac-Toe, `window_length` is 3 and there are eight groups: three
rows, three columns and two diagonals. Thus `indices` has `8 * 3 = 24` entries.
The first three groups in the current traversal are:

```text
[0, 3, 6]   top row:           (0,0), (1,0), (2,0)
[0, 1, 2]   left column:       (0,0), (0,1), (0,2)
[0, 4, 8]   downward diagonal: (0,0), (1,1), (2,2)
```

Reading these indices from an encoded `cells` vector gives the three piece values
in each line. The [shared traversal](../../../src/core/lines.rs) visits starting
cells column by column and tries right, down, diagonally down-right, then
diagonally up-right, retaining windows that fit inside the board.

## Connect 4 has overlapping windows

On the seven-column, six-row board with target four, there are 24 horizontal,
21 vertical and 24 diagonal windows, giving 69 in total. `window_length` is 4
and `indices` has `69 * 4 = 276` entries.

For example, the first four positions in the bottom row have x-major indices
`[5, 11, 17, 23]`. The next horizontal window uses `[11, 17, 23, 29]`. Windows
overlap because several different sets of four consecutive cells could win.

Changing board dimensions or target length changes these groups. When the target
is one, each cell forms one group and appears once in `indices`.

## Calculate an estimate from cells and groups

For an ongoing position, the current native evaluators assign a contribution to
each group. A group containing only own pieces and empty cells contributes
`(own_count / window_length)^2`. One containing only opposing pieces and empty
cells contributes the negative equivalent. Mixed groups and entirely empty
groups contribute zero. Add all contributions, divide by the number of groups,
then multiply by 0.9.

For a 3 by 3 board with just an own X at the center, four of the eight groups
contain that piece. Each contributes `(1 / 3)^2`; the remaining four contribute
zero. The estimate is `0.9 * 4 * (1 / 3)^2 / 8 = 0.05`.

Terminal win/loss/draw is checked first and produces +1/-1/0 for the chosen
perspective. Encoding includes results as well as cells, so an evaluator can
handle terminal cases before applying the ongoing-position formula. Its viewpoint
must agree with the observer used for own/opposing cell values.

`OpenLinesSpec` supplies the cell groups for that calculation. The current native
evaluators implement the formula in Rust, and
[the Python comparison heuristic](../../../adapters/python/examples/open_lines.py)
implements it using the exported groups. The [learning tests](../../../tests/core/learning.rs) calculate estimates
from exported data and compare them with the native evaluators. The Python
comparison also preserves the contribution order when checking exact float64
agreement, because floating-point summation order can affect rounding.

See [external OSLA](../python/external-osla.md) for batched Python evaluation and [adding support](adding-support.md) for testing another game.
