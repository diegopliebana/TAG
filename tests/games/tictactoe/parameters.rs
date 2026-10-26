use tag_rust::core::grid::{BoardLimits, GridError};
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId};
use tag_rust::games::tictactoe::{
    Cell, PlaceMark, TicTacToeError, TicTacToeForwardModel, TicTacToeParameters,
};

#[test]
fn configurable_square_sizes_support_real_moves_and_wins() {
    let model = TicTacToeForwardModel;
    for grid_size in [2, 4, 7] {
        let mut state = model
            .init_state(
                &TicTacToeParameters {
                    grid_size,
                    ..Default::default()
                },
                2,
            )
            .unwrap();
        assert_eq!(state.grid_size(), grid_size);
        assert_eq!(state.board().len(), grid_size * grid_size);
        assert_eq!(
            model
                .available_actions(&state, PlayerId::new(0))
                .unwrap()
                .len(),
            grid_size * grid_size
        );
        for x in 0..grid_size {
            model
                .next(&mut state, PlayerId::new(0), &PlaceMark { x, y: 0 })
                .unwrap();
            if x + 1 < grid_size {
                model
                    .next(&mut state, PlayerId::new(1), &PlaceMark { x, y: 1 })
                    .unwrap();
            }
        }
        assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
        assert_eq!(model.next_player(&state), Decision::Finished);
        assert_eq!(
            state.winning_cells(),
            (0..grid_size).map(|x| (x, 0)).collect::<Vec<_>>()
        );
        assert_eq!(state.move_count(), 2 * grid_size - 1);
        assert_eq!(state.completed_turn_count(), 2 * grid_size - 2);
        assert_eq!(state.round_count(), 0);
        assert_eq!(
            model.observe(&state, PlayerId::new(1)).unwrap().board(),
            state.board()
        );
    }
}

#[test]
fn one_cell_game_wins_on_first_move_without_completing_a_turn() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(
            &TicTacToeParameters {
                grid_size: 1,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    model
        .next(&mut state, PlayerId::new(0), &PlaceMark { x: 0, y: 0 })
        .unwrap();
    assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
    assert_eq!(state.winning_cells(), vec![(0, 0)]);
    assert!(view.winning_cells().is_empty());
    assert_eq!(state.move_count(), 1);
    assert_eq!(state.completed_turn_count(), 0);
    assert_eq!(view.board(), &[Cell::Empty]);
}

#[test]
fn tictactoe_cell_limits_are_configurable_and_checked_before_allocation() {
    let model = TicTacToeForwardModel;
    assert_eq!(
        model.init_state(
            &TicTacToeParameters {
                grid_size: 1001,
                ..Default::default()
            },
            2
        ),
        Err(TicTacToeError::Grid(GridError::CellLimitExceeded {
            requested: 1_002_001,
            limit: 1_000_000
        }))
    );
    assert_eq!(
        model.init_state(
            &TicTacToeParameters {
                grid_size: 4,
                limits: BoardLimits { max_cells: 15 }
            },
            2
        ),
        Err(TicTacToeError::Grid(GridError::CellLimitExceeded {
            requested: 16,
            limit: 15
        }))
    );
    assert!(
        model
            .init_state(
                &TicTacToeParameters {
                    grid_size: 4,
                    limits: BoardLimits { max_cells: 16 }
                },
                2
            )
            .is_ok()
    );
    assert_eq!(
        model.init_state(
            &TicTacToeParameters {
                limits: BoardLimits { max_cells: 0 },
                ..Default::default()
            },
            2
        ),
        Err(TicTacToeError::Grid(GridError::InvalidCellLimit))
    );
}
