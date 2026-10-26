//! Internal ownership checks; public initialization tests are in tests/games/connect4/initialization.rs.
use super::*;
use crate::core::ForwardModel;
use crate::core::PlayerId;

#[test]
fn snapshots_own_board_and_winning_coordinates_independently() {
    let model = Connect4ForwardModel;
    let mut source = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    let mut branch = source.clone(); // Explicit independent snapshot.
    // Actual moves alter the branch's board, counters and winning coordinates.
    for column in [0, 1, 0, 1, 0, 1, 0] {
        let actor = branch.current_player();
        model
            .next(&mut branch, actor, &DropPiece { column })
            .unwrap();
    }
    assert_eq!(source.cell(0, 5), Ok(Cell::Empty));
    assert!(source.winning_cells().is_empty());
    assert_eq!(source.move_count(), 0);
    assert_eq!(branch.cell(0, 5), Ok(Cell::Piece(PlayerId::new(0))));
    assert_eq!(branch.winning_cells(), &[(0, 5), (0, 4), (0, 3), (0, 2)]);
    model
        .next(&mut source, PlayerId::new(0), &DropPiece { column: 2 })
        .unwrap();
    assert_eq!(branch.cell(2, 5), Ok(Cell::Empty));
}

#[test]
fn ordered_scan_covers_both_seats_all_directions_and_longer_runs() {
    let model = Connect4ForwardModel;
    for seat in 0..2 {
        for cells in [
            vec![(2, 5), (2, 4), (2, 3), (2, 2), (2, 1)],
            vec![(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)],
            vec![(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)],
            vec![(4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
            vec![(2, 5), (3, 4), (4, 3), (5, 2)],
            vec![(6, 4), (5, 3), (4, 2), (3, 1)],
        ] {
            // Deliberately construct line fixtures to isolate scanning from gravity rules.
            let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
            let actor = PlayerId::new(seat);
            for &(x, y) in &cells {
                *state.board.get_mut(x, y).unwrap() = Cell::Piece(actor);
            }
            assert_eq!(
                super::forward_model::winning_line(&state, actor),
                Some(cells[..4].to_vec())
            );
        }
    }
}

#[test]
fn simultaneous_lines_select_columns_before_rows_and_diagonal_start_order() {
    let model = Connect4ForwardModel;
    let actor = PlayerId::new(0);
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    for (x, y) in [(0, 5), (0, 4), (0, 3), (0, 2), (1, 5), (2, 5), (3, 5)] {
        *state.board.get_mut(x, y).unwrap() = Cell::Piece(actor);
    }
    assert_eq!(
        super::forward_model::winning_line(&state, actor),
        Some(vec![(0, 5), (0, 4), (0, 3), (0, 2)])
    );
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    for (x, y) in [
        (0, 4),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 5),
        (2, 4),
        (1, 3),
        (0, 2),
    ] {
        *state.board.get_mut(x, y).unwrap() = Cell::Piece(actor);
    }
    // The left-edge rising-right diagonal is visited before bottom-start rising-left diagonals.
    assert_eq!(
        super::forward_model::winning_line(&state, actor),
        Some(vec![(0, 4), (1, 3), (2, 2), (3, 1)])
    );
}

#[test]
fn observation_clones_own_their_board_and_winning_coordinates() {
    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    for column in [0, 1, 0, 1, 0, 1, 0] {
        let actor = state.current_player();
        model
            .next(&mut state, actor, &DropPiece { column })
            .unwrap();
    }
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    let mut copy = view.clone(); // Explicit owned copy of the observation's state and observer ID.
    // Private test edits isolate storage ownership; agents receive read-only queries.
    *copy.state.board.get_mut(0, 5).unwrap() = Cell::Empty;
    copy.state.winning_cells.clear();
    assert_eq!(view.cell(0, 5), Ok(Cell::Piece(PlayerId::new(0))));
    assert_eq!(view.winning_cells(), &[(0, 5), (0, 4), (0, 3), (0, 2)]);
    assert_eq!(state.cell(0, 5), Ok(Cell::Piece(PlayerId::new(0))));
    assert_eq!(state.winning_cells(), view.winning_cells());
}
