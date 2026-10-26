# Saved-state and replay examples

## Tournament replay fixtures

`tournament-tictactoe.tag-replay.json` and `tournament-connect4.tag-replay.json`
were exported by the native tournament CLI on 22-10-2026. The agent pool, in order,
was MCTS (inline config: schema_version 1, budgets.iterations 5, rollout_length 1;
all other settings default) and Random. The exhaustive tournament used master
seed 42, seat permutations and one repetition. Game entries, in order, were
default Tic-Tac-Toe and Connect 4 with width 5, height 4, winning_length 3.
These files are the first seat allocation of each game entry (match indices 0
and 2). They use human controllers to preserve the selected actions independently
of the original algorithms. The browser file tests import both and navigate every
recorded position. The tournament tests separately verify newly generated logs
and reconstructed final outcomes, including MCTS games.

## Original session examples

These are example files, not hand-edited rule definitions. A state file saves one position; a replay saves recorded moves and agent progress so earlier positions can be visited and play continued.


Generated 13-10-2026 17:12 (Europe/Paris) by the engine's real export functions.
Each file was loaded and validated before being written.

| Game | Mid-game state | Finished state | Branched replay |
|---|---|---|---|
| Tic-Tac-Toe | [tictactoe.tag-state.json](tictactoe.tag-state.json) | [tictactoe-finished.tag-state.json](tictactoe-finished.tag-state.json) | [tictactoe.tag-replay.json](tictactoe.tag-replay.json) |
| Connect 4 | [connect4.tag-state.json](connect4.tag-state.json) | [connect4-finished.tag-state.json](connect4-finished.tag-state.json) | [connect4.tag-replay.json](connect4.tag-replay.json) |

Both mid-game states contain three accepted moves, with player 1 next to act.
They can be loaded with new controllers using the unchanged saved parameters.
Finished states show player 1 winning: Tic-Tac-Toe after six moves and Connect 4
after eight. The Connect 4 file includes its ordered vertical winning cells.

Each replay retains an original OSLA/OSLA branch (seeds 7 and 8), then forks after
move 2. In branch 1, player 0 is Human and player 1 preserves OSLA's progress.
The generator supplies the first offered action for each human decision.
The original branches have nine moves; child branches finish after six/eight
absolute moves. Selected branch is 1, displayed cursor is 2 and AI delay is 250 ms.
That saved display position precedes the child's recorded continuation.

Cells are x-major: coordinate (x,y) uses x * height + y. Occupants are 0/1;
null means empty. Counters are moves, completed turns, turns within the round,
and completed rounds. Board layout and API details belong in
[portable formats](../../docs/reference/persistence/formats.md).

Regenerate these exact files from the project root with:

```powershell
cargo run --offline --features replay --example save_files
```

[Generator source](../save_files.rs) creates both games, exports their state/history,
checks the imports and writes these six JSON files. Fixture regression checks live
in [persistence tests](../../tests/core/sessions/persistence.rs).
