use tag_rust::core::grid::{BoardLimits, Grid, GridError};
use tag_rust::core::outcomes::{SingleWinnerOutcome, win_draw_loss_score};
use tag_rust::core::players::validate_player;
use tag_rust::core::progress::TurnCounters;
use tag_rust::core::{Decision, PlayerId, PlayerResult};

#[test]
fn grid_supports_rectangles_noncopy_cells_ordering_and_independent_snapshots() {
    let mut grid = Grid::filled(2, 3, String::from("empty"), BoardLimits::default()).unwrap();
    assert_eq!(
        grid.coordinates().collect::<Vec<_>>(),
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    );
    *grid.get_mut(1, 2).unwrap() = "piece".to_owned();
    assert_eq!(grid.cells()[5], "piece");
    let mut branch = grid.clone(); // Explicit deep copy of owned Strings and storage.
    branch.get_mut(1, 2).unwrap().push_str(" changed");
    assert_eq!(grid.get(1, 2).unwrap(), "piece");
    grid.get_mut(0, 0).unwrap().push_str(" source");
    assert_eq!(branch.get(0, 0).unwrap(), "empty");
    assert_eq!(grid.width(), 2);
    assert_eq!(grid.height(), 3);
    let original = grid.clone();
    for (x, y) in [(2, 0), (0, 3), (usize::MAX, usize::MAX)] {
        assert_eq!(
            grid.get_mut(x, y),
            Err(GridError::OutOfBounds {
                x,
                y,
                width: 2,
                height: 3
            })
        );
        assert_eq!(grid, original);
    }
}

#[test]
fn grid_reports_limits_overflow_and_unrepresentable_allocation() {
    assert_eq!(BoardLimits::default().max_cells, 1_000_000);
    for (width, height) in [(0, 1), (1, 0), (0, 0)] {
        assert_eq!(
            Grid::filled(width, height, 0u8, BoardLimits::default()),
            Err(GridError::InvalidDimensions { width, height })
        );
    }
    assert_eq!(
        Grid::filled(1, 1, 0u8, BoardLimits { max_cells: 0 }),
        Err(GridError::InvalidCellLimit)
    );
    assert_eq!(
        Grid::filled(usize::MAX, 2, 0u8, BoardLimits::default()),
        Err(GridError::SizeOverflow)
    );
    assert_eq!(
        Grid::filled(3, 4, 0u8, BoardLimits { max_cells: 11 }),
        Err(GridError::CellLimitExceeded {
            requested: 12,
            limit: 11
        })
    );
    assert!(Grid::filled(3, 4, 0u8, BoardLimits { max_cells: 12 }).is_ok());
    // This requested u64 buffer exceeds the representable allocation size.
    assert_eq!(
        Grid::filled(
            usize::MAX,
            1,
            0u64,
            BoardLimits {
                max_cells: usize::MAX
            }
        ),
        Err(GridError::AllocationFailed)
    );
}

#[test]
fn counters_do_not_impose_turn_or_round_policy() {
    let mut no_rounds = TurnCounters::default();
    let mut paired_turns = TurnCounters::default();
    for index in 0..4 {
        no_rounds.record_move();
        no_rounds.complete_turn();
        paired_turns.record_move();
        paired_turns.complete_turn();
        if index % 2 == 1 {
            paired_turns.complete_round();
        }
    }
    assert_eq!(no_rounds.round_count(), 0);
    assert_eq!(no_rounds.turn_count_in_round(), 4);
    assert_eq!(paired_turns.round_count(), 2);
    assert_eq!(paired_turns.turn_count_in_round(), 0);
    assert_eq!(paired_turns.completed_turn_count(), 4);
    paired_turns.record_move(); // Record the terminal move while retaining completed-turn counters.
    assert_eq!(paired_turns.move_count(), 5);
    assert_eq!(paired_turns.completed_turn_count(), 4);
    assert_eq!(paired_turns.round_count(), 2);
}

#[test]
fn optional_single_winner_outcomes_validate_seats_without_assuming_two_players() {
    let winner = PlayerId::new(2);
    let outcome = SingleWinnerOutcome::Won(winner);
    for index in 0..4 {
        let expected = if index == 2 {
            PlayerResult::Win
        } else {
            PlayerResult::Loss
        };
        assert_eq!(outcome.player_result(PlayerId::new(index), 4), Ok(expected));
    }
    assert_eq!(outcome.next_player(winner), Decision::Finished);
    assert_eq!(
        SingleWinnerOutcome::Ongoing.next_player(winner),
        Decision::Player(winner)
    );
    assert!(outcome.player_result(PlayerId::new(4), 4).is_err());
    assert!(outcome.player_result(PlayerId::new(0), 2).is_err()); // Invalid stored winner also checked.
    assert!(validate_player(PlayerId::new(usize::MAX), 4).is_err());
    assert!(validate_player(PlayerId::new(0), 0).is_err());
    assert_eq!(win_draw_loss_score(PlayerResult::Win), 1.0);
    assert_eq!(win_draw_loss_score(PlayerResult::Draw), 0.0);
    assert_eq!(win_draw_loss_score(PlayerResult::Loss), -1.0);
    assert_eq!(
        SingleWinnerOutcome::Draw.player_result(winner, 4),
        Ok(PlayerResult::Draw)
    );
}

#[test]
fn ordered_line_helper_resets_at_gaps_and_returns_first_target_cells() {
    use tag_rust::core::lines::{first_matching_run, ray};
    let cells = [true, true, false, true, true, true, true];
    assert_eq!(
        first_matching_run(ray(7, 1, (0, 0), (1, 0)), 3, |(x, _)| {
            // Match occupied cells; the gap must break the first short sequence.
            cells[x]
        }),
        Some(vec![(3, 0), (4, 0), (5, 0)])
    );
    assert_eq!(
        ray(1, 4, (0, 3), (0, -1)).collect::<Vec<_>>(),
        vec![(0, 3), (0, 2), (0, 1), (0, 0)]
    );
    assert_eq!(
        ray(4, 3, (3, 2), (-1, -1)).collect::<Vec<_>>(),
        vec![(3, 2), (2, 1), (1, 0)]
    );
    assert_eq!(ray(1, 1, (0, 0), (0, 0)).count(), 0);
    assert_eq!(ray(1, 1, (1, 0), (1, 0)).count(), 0);
    assert_eq!(
        first_matching_run(ray(1, 1, (0, 0), (1, 0)), 0, |_| {
            // A zero target returns None.
            true
        }),
        None
    );
}
