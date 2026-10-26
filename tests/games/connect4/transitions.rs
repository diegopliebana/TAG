//! Independent small-board rules and focused transition checks.
use std::collections::HashSet;
use tag_rust::core::ForwardModel;
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, PlayerId};
use tag_rust::games::connect4::{
    Cell, Connect4Error, Connect4ForwardModel, Connect4Parameters, Connect4State, DropPiece,
};

fn play(model: &Connect4ForwardModel, state: &mut Connect4State, column: usize) {
    // Borrow rules/state; copy the current actor and borrow a temporary action.
    model
        .next(state, state.current_player(), &DropPiece { column })
        .unwrap();
}

#[test]
fn gravity_actions_counters_terminal_timing_and_snapshots() {
    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    let snapshot = state.clone();
    for (i, column) in [0, 1, 0, 1, 0, 1, 0].into_iter().enumerate() {
        play(&model, &mut state, column);
        assert_eq!(state.move_count(), i + 1);
        let turns = (i + 1).min(6);
        assert_eq!(state.completed_turn_count(), turns);
        assert_eq!(state.turn_count_in_round(), turns % 2);
        assert_eq!(state.round_count(), turns / 2);
    }
    assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
    assert_eq!(state.winning_cells(), &[(0, 5), (0, 4), (0, 3), (0, 2)]);
    assert_eq!(model.next_player(&state), Decision::Finished);
    assert!(
        model
            .available_actions(&state, PlayerId::new(1))
            .unwrap()
            .is_empty()
    );
    assert_eq!(snapshot.move_count(), 0);
    assert_eq!(snapshot.cell(0, 5), Ok(Cell::Empty));
    let frozen = state.clone();
    assert_eq!(
        model.next(&mut state, PlayerId::new(0), &DropPiece { column: 0 }),
        Err(Connect4Error::MoveAfterGameFinished)
    );
    assert_eq!(state, frozen);
}

#[test]
fn rejected_moves_are_atomic_and_full_columns_disappear() {
    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    for (actor, column, error) in [
        (
            2,
            usize::MAX,
            Connect4Error::InvalidPlayer {
                player: PlayerId::new(2),
            },
        ),
        (
            1,
            usize::MAX,
            Connect4Error::WrongPlayer {
                expected: PlayerId::new(0),
                actual: PlayerId::new(1),
            },
        ),
        (
            0,
            usize::MAX,
            Connect4Error::InvalidColumn {
                column: usize::MAX,
                width: 7,
            },
        ),
    ] {
        let before = state.clone();
        assert_eq!(
            model.next(&mut state, PlayerId::new(actor), &DropPiece { column }),
            Err(error)
        );
        assert_eq!(state, before);
    }
    for _ in 0..6 {
        play(&model, &mut state, 0);
    }
    let before = state.clone();
    assert_eq!(
        model.next(&mut state, PlayerId::new(0), &DropPiece { column: 0 }),
        Err(Connect4Error::FullColumn { column: 0 })
    );
    assert_eq!(state, before);
    assert_eq!(
        model.available_actions(&state, PlayerId::new(0)).unwrap(),
        (1..7)
            .map(|column| {
                // Each remaining nonfull column offers one drop action, left to right.
                DropPiece { column }
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn target_one_wins_immediately_in_rectangles_and_single_cell_boards() {
    let model = Connect4ForwardModel;
    for (width, height) in [(1, 1), (9, 1), (1, 9), (7, 6)] {
        let mut state = model
            .init_state(
                &Connect4Parameters {
                    width,
                    height,
                    winning_length: 1,
                    ..Default::default()
                },
                2,
            )
            .unwrap();
        play(&model, &mut state, width - 1);
        assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
        assert_eq!(state.winning_cells(), &[(width - 1, height - 1)]);
        assert_eq!(state.completed_turn_count(), 0);
        assert_eq!(state.round_count(), 0);
    }
}

// Independent row-major array: examine each occupied cell in four directions.
fn oracle(board: &[i8; 9]) -> SingleWinnerOutcome {
    for y in 0..3_i32 {
        for x in 0..3_i32 {
            let mark = board[(y * 3 + x) as usize];
            if mark == 0 {
                continue;
            }
            for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                let mut won = true;
                for step in 1..3 {
                    let (xx, yy) = (x + dx * step, y + dy * step);
                    if !(0..3).contains(&xx)
                        || !(0..3).contains(&yy)
                        || board[(yy * 3 + xx) as usize] != mark
                    {
                        won = false;
                        break;
                    }
                }
                if won {
                    return SingleWinnerOutcome::Won(PlayerId::new((mark - 1) as usize));
                }
            }
        }
    }
    if board.contains(&0) {
        SingleWinnerOutcome::Ongoing
    } else {
        SingleWinnerOutcome::Draw
    }
}

fn visit(
    model: &Connect4ForwardModel,
    state: Connect4State,
    board: [i8; 9],
    seen: &mut HashSet<[i8; 9]>,
    coverage: &mut [usize; 3],
) {
    if !seen.insert(board) {
        return;
    }
    assert_eq!(state.status(), oracle(&board));
    for y in 0..3 {
        for x in 0..3 {
            let expected = match board[y * 3 + x] {
                0 => Cell::Empty,
                mark => Cell::Piece(PlayerId::new((mark - 1) as usize)),
            };
            assert_eq!(state.cell(x, y), Ok(expected));
        }
    }
    if state.status() != SingleWinnerOutcome::Ongoing {
        match state.status() {
            SingleWinnerOutcome::Won(p) => coverage[p.index()] += 1,
            SingleWinnerOutcome::Draw => coverage[2] += 1,
            _ => unreachable!(),
        }
        return;
    }
    let actor = state.current_player();
    let mut expected_actions = Vec::new();
    for x in 0..3 {
        // Independently locate the lowest empty row, using row-major oracle storage.
        let mut landing = None;
        for y in (0..3).rev() {
            if board[y * 3 + x] == 0 {
                landing = Some(y);
                break;
            }
        }
        if let Some(y) = landing {
            expected_actions.push(DropPiece { column: x });
            let mut next_board = board;
            next_board[y * 3 + x] = (actor.index() + 1) as i8;
            let mut branch = state.clone();
            play(model, &mut branch, x);
            let terminal = oracle(&next_board) != SingleWinnerOutcome::Ongoing;
            assert_eq!(
                branch.completed_turn_count(),
                state.completed_turn_count() + usize::from(!terminal)
            );
            if terminal {
                assert_eq!(branch.round_count(), state.round_count());
                assert_eq!(branch.turn_count_in_round(), state.turn_count_in_round());
                assert_eq!(branch.current_player(), actor);
            }
            if let SingleWinnerOutcome::Won(winner) = branch.status() {
                assert_eq!(branch.winning_cells().len(), 3);
                for &(xx, yy) in branch.winning_cells() {
                    assert_eq!(next_board[yy * 3 + xx], (winner.index() + 1) as i8);
                }
            }
            visit(model, branch, next_board, seen, coverage);
        }
    }
    assert_eq!(
        model.available_actions(&state, actor).unwrap(),
        expected_actions
    );
}

#[test]
fn all_reachable_three_by_three_positions_match_independent_gravity_and_outcomes() {
    let model = Connect4ForwardModel;
    let state = model
        .init_state(
            &Connect4Parameters {
                width: 3,
                height: 3,
                winning_length: 3,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    let mut seen = HashSet::new();
    let mut coverage = [0; 3];
    visit(&model, state, [0; 9], &mut seen, &mut coverage);
    assert!(seen.len() > 100);
    assert!(coverage.iter().all(|&count| {
        // The corpus must include wins for both players and full-board draws.
        count > 0
    }));
}
