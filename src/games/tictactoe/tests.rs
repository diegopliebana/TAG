//! Snapshot ownership and independence checks.
use super::*;
use crate::core::{ForwardModel, PlayerId};

#[test]
fn clone_has_independent_board_and_progress() {
    // Use real transitions to test snapshots.
    let model = TicTacToeForwardModel;
    let mut source = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    model
        .next(&mut source, PlayerId::new(0), &PlaceMark { x: 0, y: 0 })
        .unwrap();

    // Two explicit snapshots are made:
    //   original: a baseline copy that this test will leave unchanged;
    //   branch:   a mutable simulation-like copy that the test will alter.
    let original = source.clone(); // clone borrows source; original owns an independent snapshot.
    let mut branch = source.clone(); // Explicit clone; branch owns its mutable snapshot.

    // assert_eq! is a testing macro. It checks == and reports diagnostics if unequal.
    assert_eq!(branch, source);
    model
        .next(&mut branch, PlayerId::new(1), &PlaceMark { x: 0, y: 1 })
        .unwrap();
    assert_eq!(branch.move_count(), 2);
    assert_eq!(branch.completed_turn_count(), 2);
    assert_eq!(branch.current_player(), PlayerId::new(0));
    // This verifies that all branch edits left the original source state unchanged.
    assert_eq!(source, original);
    assert_eq!(source.cell(0, 1), Ok(Cell::Empty));
    assert_eq!(branch.cell(0, 1), Ok(Cell::O));

    // The final edit tests independence in the OTHER direction: editing source
    // leaves the branch snapshot intact.
    model
        .next(&mut source, PlayerId::new(1), &PlaceMark { x: 0, y: 2 })
        .unwrap();
    assert_eq!(source.cell(0, 2), Ok(Cell::O));
    assert_eq!(branch.cell(0, 2), Ok(Cell::Empty));
}
