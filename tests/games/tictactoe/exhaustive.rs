//! Explore valid play using a separate row-based board and outcome calculation.

use std::collections::HashSet;
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::tictactoe::{
    Cell, PlaceMark, TicTacToeError, TicTacToeForwardModel, TicTacToeParameters, TicTacToeState,
};

// The test board uses rows containing columns, unlike the engine's flat x-major board.
// X is +1, O is -1, empty is 0. A line wins when its sum is +3 or -3.
type Board = [[i8; 3]; 3];

fn outcome(board: &Board) -> SingleWinnerOutcome {
    // Shared test-board borrow; returns a copied expected status.
    for mark in [1, -1] {
        let target = 3 * mark;
        let mut won = false;
        for (i, row) in board.iter().enumerate() {
            let row_sum = row[0] + row[1] + row[2];
            let column_sum = board[0][i] + board[1][i] + board[2][i];
            won |= row_sum == target || column_sum == target;
        }
        won |= board[0][0] + board[1][1] + board[2][2] == target;
        won |= board[0][2] + board[1][1] + board[2][0] == target;
        if won {
            return SingleWinnerOutcome::Won(PlayerId::new(if mark == 1 { 0 } else { 1 }));
        }
    }
    for row in board {
        for cell in row {
            if *cell == 0 {
                return SingleWinnerOutcome::Ongoing;
            }
        }
    }
    SingleWinnerOutcome::Draw
}

fn board_key(board: &Board) -> usize {
    // Shared board borrow; copied base-three key, used only for test deduplication.
    let mut key = 0;
    for row in board {
        for cell in row {
            key = key * 3 + (*cell + 1) as usize;
        }
    }
    key
}

#[derive(Debug, Default)] // No Clone/Copy or PartialEq/Eq: statistics are mutated and checked field by field.
struct Coverage {
    positions: HashSet<usize>,
    transitions: usize,
    x_wins: usize,
    o_wins: usize,
    draws: usize,
}

fn explore(
    model: &TicTacToeForwardModel, // Shared rule implementation borrow.
    state: &TicTacToeState,        // Shared engine state borrow; children own explicit clones.
    board: &Board,                 // Shared independent expected board.
    moves: usize,                  // Copied expected move count.
    coverage: &mut Coverage,       // Exclusive statistics borrow, shared across traversal.
) {
    let expected_status = outcome(board);
    let finished = expected_status != SingleWinnerOutcome::Ongoing;
    let actor = PlayerId::new(if finished { (moves - 1) % 2 } else { moves % 2 });
    let decision = if finished {
        Decision::Finished
    } else {
        Decision::Player(actor)
    };
    let completed_turns = if finished { moves - 1 } else { moves };
    let original = state.clone(); // Independent baseline to detect unintended mutation.
    assert_eq!(state.status(), expected_status, "test board: {board:?}");
    let winning = state.winning_cells();
    if let SingleWinnerOutcome::Won(winner) = expected_status {
        assert_eq!(winning.len(), 3);
        let expected_mark = if winner.index() == 0 { 1 } else { -1 };
        for &(x, y) in &winning {
            assert_eq!(board[y][x], expected_mark);
        }
        let dx = winning[1].0 as isize - winning[0].0 as isize;
        let dy = winning[1].1 as isize - winning[0].1 as isize;
        assert!(dx.abs() <= 1 && dy.abs() <= 1 && (dx != 0 || dy != 0));
        assert_eq!(winning[2].0 as isize - winning[1].0 as isize, dx);
        assert_eq!(winning[2].1 as isize - winning[1].1 as isize, dy);
    } else {
        assert!(winning.is_empty());
    }
    assert_eq!(state.move_count(), moves);
    assert_eq!(state.completed_turn_count(), completed_turns);
    assert_eq!(state.current_player(), actor);
    assert_eq!(state.round_count(), 0);
    assert_eq!(state.first_player(), PlayerId::new(0));
    assert_eq!(model.player_count(state), 2);
    assert_eq!(model.next_player(state), decision);

    let mut expected_actions = Vec::new();
    for x in 0..3 {
        for (y, row) in board.iter().enumerate() {
            let expected_cell = match row[x] {
                1 => Cell::X,
                -1 => Cell::O,
                _ => Cell::Empty,
            };
            assert_eq!(state.cell(x, y), Ok(expected_cell));
            assert_eq!(state.board()[x * 3 + y], expected_cell);
            if !finished && row[x] == 0 {
                expected_actions.push(PlaceMark { x, y });
            }
        }
    }
    let views = [
        model.observe(state, PlayerId::new(0)).unwrap(),
        model.observe(state, PlayerId::new(1)).unwrap(),
    ]; // Owned observations.
    let saved_views = views.clone(); // Explicit independent copies for branch-independence checks.
    for (player_index, view) in views.iter().enumerate() {
        let player = PlayerId::new(player_index);
        // view: shared borrow; the observation array retains ownership.
        let expected_result = match expected_status {
            SingleWinnerOutcome::Ongoing => PlayerResult::Ongoing,
            SingleWinnerOutcome::Draw => PlayerResult::Draw,
            SingleWinnerOutcome::Won(winner) if winner == player => PlayerResult::Win,
            SingleWinnerOutcome::Won(_) => PlayerResult::Loss,
        };
        let expected_score = match expected_result {
            PlayerResult::Win => 1.0,
            PlayerResult::Loss => -1.0,
            _ => 0.0,
        };
        assert_eq!(model.player_result(state, player), Ok(expected_result));
        assert_eq!(model.player_score(state, player), Ok(expected_score));
        assert_eq!(view.observer(), player);
        assert_eq!(view.board(), state.board());
        assert_eq!(view.status(), expected_status);
        assert_eq!(view.winning_cells(), winning);
        assert_eq!(view.next_player(), decision);
        assert_eq!(view.current_player(), actor);
        assert_eq!(view.move_count(), moves);
        assert_eq!(view.completed_turn_count(), completed_turns);
        assert_eq!(view.round_count(), 0);
        assert_eq!(view.first_player(), PlayerId::new(0));
        for seat in [PlayerId::new(0), PlayerId::new(1)] {
            assert_eq!(view.player_result(seat), model.player_result(state, seat));
            assert_eq!(view.player_score(seat), model.player_score(state, seat));
        }
        if !finished && player == actor {
            assert_eq!(model.offered_actions(view), expected_actions);
            assert_eq!(
                model.available_actions(state, player),
                Ok(expected_actions.clone())
            );
        } else {
            assert!(model.offered_actions(view).is_empty());
            let expected = if finished {
                Ok(vec![])
            } else {
                Err(TicTacToeError::WrongPlayer {
                    expected: actor,
                    actual: player,
                })
            };
            assert_eq!(model.available_actions(state, player), expected);
        }
    }

    let mut rejected = state.clone(); // Owned branch used only for invalid moves.
    for player in [PlayerId::new(2), PlayerId::new(usize::MAX)] {
        let error = TicTacToeError::InvalidPlayer { player };
        assert_eq!(model.observe(state, player), Err(error.clone()));
        assert_eq!(model.available_actions(state, player), Err(error.clone()));
        assert_eq!(
            model.next(&mut rejected, player, &PlaceMark { x: 0, y: 0 }),
            Err(error)
        );
        assert_eq!(rejected, original);
    }
    if finished {
        for player in [PlayerId::new(0), PlayerId::new(1)] {
            assert_eq!(
                model.next(&mut rejected, player, &PlaceMark { x: 0, y: 0 }),
                Err(TicTacToeError::MoveAfterGameFinished)
            );
            assert_eq!(rejected, original);
        }
    } else {
        let wrong = PlayerId::new(1 - actor.index());
        assert_eq!(
            model.next(&mut rejected, wrong, &PlaceMark { x: 0, y: 0 }),
            Err(TicTacToeError::WrongPlayer {
                expected: actor,
                actual: wrong
            })
        );
        for (x, y) in [(3, 0), (0, 3), (usize::MAX, usize::MAX)] {
            assert_eq!(
                model.next(&mut rejected, actor, &PlaceMark { x, y }),
                Err(TicTacToeError::OutOfBounds { x, y })
            );
            assert_eq!(rejected, original);
        }
        for (y, row) in board.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if *cell != 0 {
                    assert_eq!(
                        model.next(&mut rejected, actor, &PlaceMark { x, y }),
                        Err(TicTacToeError::OccupiedCell { x, y })
                    );
                    assert_eq!(rejected, original);
                }
            }
        }
    }
    assert_eq!(rejected, original);
    assert_eq!(*state, original);

    // Check every arriving transition before deduplicating; expand each board only once.
    // In this game the board determines move count, historical actor and all other progress.
    if !coverage.positions.insert(board_key(board)) {
        return;
    }
    match expected_status {
        SingleWinnerOutcome::Won(winner) if winner.index() == 0 => coverage.x_wins += 1,
        SingleWinnerOutcome::Won(_) => coverage.o_wins += 1,
        SingleWinnerOutcome::Draw => coverage.draws += 1,
        SingleWinnerOutcome::Ongoing => {}
    }
    for action in expected_actions {
        let mut child = state.clone(); // Independent simulation branch; source stays untouched.
        let mut child_board = *board; // Copy the small test array; it contains only Copy integers.
        child_board[action.y][action.x] = if actor.index() == 0 { 1 } else { -1 };
        model.next(&mut child, actor, &action).unwrap();
        coverage.transitions += 1;
        assert_eq!(*state, original);
        assert_eq!(views, saved_views); // Every branch leaves earlier observations unchanged.
        explore(model, &child, &child_board, moves + 1, coverage);
    }
}

#[test]
fn every_reachable_position_and_transition_matches_independent_rules() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let mut coverage = Coverage::default();
    explore(&model, &state, &[[0; 3]; 3], 0, &mut coverage);
    // Exact counts guard against accidentally skipping branches or continuing after completion.
    assert_eq!(coverage.positions.len(), 5_478);
    assert_eq!(coverage.transitions, 16_167);
    assert_eq!(coverage.x_wins, 626);
    assert_eq!(coverage.o_wins, 316);
    assert_eq!(coverage.draws, 16);
    println!(
        "Checked {} positions and {} transitions; {} X wins, {} O wins, {} draws",
        coverage.positions.len(),
        coverage.transitions,
        coverage.x_wins,
        coverage.o_wins,
        coverage.draws
    );
}
