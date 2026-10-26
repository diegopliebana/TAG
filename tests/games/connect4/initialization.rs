use tag_rust::core::ForwardModel;
use tag_rust::core::grid::{BoardLimits, GridError};
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, PlayerId, PlayerResult};
use tag_rust::games::connect4::{Cell, Connect4Error, Connect4ForwardModel, Connect4Parameters};

#[test]
fn default_connect4_initialization_is_empty_and_player_zero_acts() {
    let model = Connect4ForwardModel;
    let state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    assert_eq!(
        (state.width(), state.height(), state.winning_length()),
        (7, 6, 4)
    );
    assert_eq!(state.board(), &[Cell::Empty; 42]);
    assert_eq!(
        model.next_player(&state),
        Decision::Player(PlayerId::new(0))
    );
    assert_eq!(model.player_count(&state), 2);
    assert_eq!(state.current_player(), PlayerId::new(0));
    assert_eq!(state.first_player(), PlayerId::new(0));
    assert_eq!(state.status(), SingleWinnerOutcome::Ongoing);
    assert_eq!(state.move_count(), 0);
    assert_eq!(state.completed_turn_count(), 0);
    assert_eq!(state.turn_count_in_round(), 0);
    assert_eq!(state.round_count(), 0);
    assert!(state.winning_cells().is_empty());
    for x in 0..7 {
        for y in 0..6 {
            assert_eq!(state.cell(x, y), Ok(Cell::Empty));
        }
    }
    for player in [PlayerId::new(0), PlayerId::new(1)] {
        assert_eq!(
            model.player_result(&state, player),
            Ok(PlayerResult::Ongoing)
        );
        assert_eq!(model.player_score(&state, player), Ok(0.0));
    }
    let original = state.clone();
    for player in [PlayerId::new(2), PlayerId::new(usize::MAX)] {
        assert_eq!(
            model.player_result(&state, player),
            Err(Connect4Error::InvalidPlayer { player })
        );
        assert_eq!(
            model.player_score(&state, player),
            Err(Connect4Error::InvalidPlayer { player })
        );
    }
    assert!(state.cell(7, 0).is_err());
    assert!(state.cell(0, 6).is_err());
    assert!(state.cell(usize::MAX, usize::MAX).is_err());
    assert_eq!(state, original);
}

#[test]
fn experimental_parameters_support_varied_board_shapes_and_targets() {
    let model = Connect4ForwardModel;
    for (width, height, winning_length) in [(1, 1, 1), (1, 9, 7), (13, 2, 12), (11, 9, 8)] {
        let state = model
            .init_state(
                &Connect4Parameters {
                    width,
                    height,
                    winning_length,
                    ..Default::default()
                },
                2,
            )
            .unwrap();
        assert_eq!(
            (state.width(), state.height(), state.winning_length()),
            (width, height, winning_length)
        );
        assert_eq!(state.board().len(), width * height);
    }
}

#[test]
fn invalid_configurations_fail_before_allocating_the_board() {
    let model = Connect4ForwardModel;
    for player_count in [0, 1, 3, usize::MAX] {
        assert_eq!(
            model.init_state(&Connect4Parameters::default(), player_count),
            Err(Connect4Error::UnsupportedPlayerCount {
                requested: player_count
            })
        );
    }
    assert_eq!(
        model.init_state(
            &Connect4Parameters {
                winning_length: 0,
                ..Default::default()
            },
            2
        ),
        Err(Connect4Error::InvalidWinningLength { requested: 0 })
    );
    assert_eq!(
        model.init_state(
            &Connect4Parameters {
                width: 5,
                height: 4,
                winning_length: 6,
                ..Default::default()
            },
            2
        ),
        Err(Connect4Error::ImpossibleWinningLength {
            requested: 6,
            longest_line: 5
        })
    );
    assert_eq!(
        model.init_state(
            &Connect4Parameters {
                width: 0,
                ..Default::default()
            },
            2
        ),
        Err(Connect4Error::Grid(GridError::InvalidDimensions {
            width: 0,
            height: 6
        }))
    );
    assert_eq!(
        model.init_state(
            &Connect4Parameters {
                width: usize::MAX,
                height: 2,
                ..Default::default()
            },
            2
        ),
        Err(Connect4Error::Grid(GridError::SizeOverflow))
    );
    assert_eq!(
        model.init_state(
            &Connect4Parameters {
                limits: BoardLimits { max_cells: 41 },
                ..Default::default()
            },
            2
        ),
        Err(Connect4Error::Grid(GridError::CellLimitExceeded {
            requested: 42,
            limit: 41
        }))
    );
    assert!(
        model
            .init_state(
                &Connect4Parameters {
                    limits: BoardLimits { max_cells: 42 },
                    ..Default::default()
                },
                2
            )
            .is_ok()
    );
}
