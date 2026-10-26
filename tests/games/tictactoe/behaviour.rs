use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::tictactoe::{
    Cell, PlaceMark, TicTacToeError, TicTacToeForwardModel, TicTacToeParameters, TicTacToeState,
};

#[test]
fn observations_preserve_both_player_views_and_remain_independent() {
    let model = TicTacToeForwardModel;
    let mut state = play(&[(1, 1), (0, 2)]);
    let original = state.clone(); // Explicit independent state snapshot.
    for observer in [PlayerId::new(0), PlayerId::new(1)] {
        let view = model.observe(&state, observer).unwrap(); // Owned view.
        let saved = view.clone(); // Explicit independent observation snapshot.
        assert_eq!(view.observer(), observer);
        assert_eq!(view.board(), state.board());
        assert_eq!(view.cell(1, 1), Ok(Cell::X));
        assert_eq!(view.cell(0, 2), Ok(Cell::O));
        assert_eq!(view.next_player(), model.next_player(&state));
        assert_eq!(view.current_player(), state.current_player());
        assert_eq!(view.move_count(), state.move_count());
        assert_eq!(view.completed_turn_count(), state.completed_turn_count());
        assert_eq!(view.round_count(), state.round_count());
        assert_eq!(view.first_player(), state.first_player());
        assert_eq!(view.status(), state.status());
        for player in [PlayerId::new(0), PlayerId::new(1)] {
            assert_eq!(
                view.player_result(player),
                model.player_result(&state, player)
            );
            assert_eq!(
                view.player_score(player),
                model.player_score(&state, player)
            );
        }
        if observer == state.current_player() {
            assert_eq!(
                model.offered_actions(&view),
                model.available_actions(&state, observer).unwrap()
            );
        } else {
            assert!(model.offered_actions(&view).is_empty());
        }
        assert_eq!(state, original); // The game retains its original state.
        let mut branch = state.clone();
        model
            .next(&mut branch, PlayerId::new(0), &PlaceMark { x: 0, y: 0 })
            .unwrap();
        assert_eq!(view, saved); // The earlier view retains its snapshot.
        assert_eq!(view.cell(0, 0), Ok(Cell::Empty));
    }
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    model
        .next(&mut state, PlayerId::new(0), &PlaceMark { x: 0, y: 0 })
        .unwrap();
    assert_eq!(view.cell(0, 0), Ok(Cell::Empty)); // The original game can also advance independently.
    assert_eq!(view.move_count(), 2);
    assert_eq!(state.move_count(), 3);
}

#[test]
fn terminal_observations_show_results_and_offer_no_moves() {
    let model = TicTacToeForwardModel;
    for state in [
        play(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]),
        play(&[
            (0, 0),
            (1, 0),
            (2, 0),
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 2),
            (0, 2),
            (2, 2),
        ]),
    ] {
        for observer in [PlayerId::new(0), PlayerId::new(1)] {
            let view = model.observe(&state, observer).unwrap();
            assert_eq!(view.board(), state.board());
            assert_eq!(view.next_player(), Decision::Finished);
            assert_eq!(view.status(), state.status());
            assert_eq!(view.current_player(), state.current_player());
            assert_eq!(view.move_count(), state.move_count());
            assert_eq!(view.completed_turn_count(), state.completed_turn_count());
            assert!(model.offered_actions(&view).is_empty());
            for player in [PlayerId::new(0), PlayerId::new(1)] {
                assert_eq!(
                    view.player_result(player),
                    model.player_result(&state, player)
                );
                assert_eq!(
                    view.player_score(player),
                    model.player_score(&state, player)
                );
            }
        }
    }
}

#[test]
fn invalid_observers_and_observation_queries_return_errors() {
    let model = TicTacToeForwardModel;
    let state = play(&[(0, 0)]);
    let original = state.clone();
    let view = model.observe(&state, PlayerId::new(1)).unwrap();
    let saved = view.clone();
    for player in [PlayerId::new(2), PlayerId::new(usize::MAX)] {
        let error = TicTacToeError::InvalidPlayer { player };
        assert_eq!(model.observe(&state, player), Err(error.clone()));
        assert_eq!(view.player_result(player), Err(error.clone()));
        assert_eq!(view.player_score(player), Err(error));
    }
    for (x, y) in [(3, 0), (0, 3), (usize::MAX, usize::MAX)] {
        assert_eq!(view.cell(x, y), Err(TicTacToeError::OutOfBounds { x, y }));
    }
    assert_eq!(state, original);
    assert_eq!(view, saved);
}

#[test]
fn initial_state_has_empty_board_player_zero_and_ongoing_results() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    assert_eq!(model.player_count(&state), 2);
    assert_eq!(
        model.next_player(&state),
        Decision::Player(PlayerId::new(0))
    );
    assert_eq!(state.current_player(), PlayerId::new(0));
    assert_eq!(state.first_player(), PlayerId::new(0));
    assert_eq!(state.move_count(), 0);
    assert_eq!(state.completed_turn_count(), 0);
    assert_eq!(state.round_count(), 0);
    assert_eq!(state.status(), SingleWinnerOutcome::Ongoing);
    assert_eq!(state.board(), &[Cell::Empty; 9]);
    for x in 0..3 {
        for y in 0..3 {
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
}

fn play(
    moves: &[(usize, usize)], // Shared borrow of a recorded coordinate sequence.
) -> TicTacToeState {
    // Returns an owned state reached through valid moves.
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for &(x, y) in moves {
        let Decision::Player(actor) = model.next_player(&state) else {
            panic!("sequence attempted a move after completion");
        };
        model.next(&mut state, actor, &PlaceMark { x, y }).unwrap();
    }
    state
}

#[test]
fn actions_are_complete_ordered_and_querying_does_not_change_state() {
    let model = TicTacToeForwardModel;
    let initial = play(&[]);
    let expected = vec![
        PlaceMark { x: 0, y: 0 },
        PlaceMark { x: 0, y: 1 },
        PlaceMark { x: 0, y: 2 },
        PlaceMark { x: 1, y: 0 },
        PlaceMark { x: 1, y: 1 },
        PlaceMark { x: 1, y: 2 },
        PlaceMark { x: 2, y: 0 },
        PlaceMark { x: 2, y: 1 },
        PlaceMark { x: 2, y: 2 },
    ];
    assert_eq!(
        model.available_actions(&initial, PlayerId::new(0)),
        Ok(expected.clone())
    );
    let state = play(&[(1, 1), (0, 2)]);
    let original = state.clone();
    let remaining: Vec<_> = expected
        .into_iter()
        .filter(|action| {
            // Keep moves to cells other than the two already occupied cells.
            (action.x, action.y) != (1, 1) && (action.x, action.y) != (0, 2)
        })
        .collect();
    assert_eq!(
        model.available_actions(&state, PlayerId::new(0)),
        Ok(remaining)
    );
    assert_eq!(state, original);
    assert_eq!(state.cell(1, 1), Ok(Cell::X));
    assert_eq!(state.cell(0, 2), Ok(Cell::O));
    assert_eq!(state.move_count(), 2);
    assert_eq!(state.completed_turn_count(), 2);
    assert_eq!(state.current_player(), PlayerId::new(0));
    assert_eq!(model.player_score(&state, PlayerId::new(0)), Ok(0.0));
    assert_eq!(model.player_score(&state, PlayerId::new(1)), Ok(0.0));
}

#[test]
fn every_winning_line_is_detected_for_either_player() {
    // Explicit coordinates describe each winning line.
    let lines = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(2, 0), (1, 1), (0, 2)],
    ];
    let model = TicTacToeForwardModel;
    for target in lines {
        let others: Vec<_> = (0..3)
            .flat_map(|x| {
                // Visit all three cells in this board column.
                (0..3).map(move |y| {
                    // Make the coordinates of the cell at this column and row.
                    (x, y)
                })
            })
            .filter(|cell| {
                // Keep only cells outside the winning line being tested.
                !target.contains(cell)
            })
            .collect();
        // Pick three opponent cells that keep play ongoing until player 1's last move.
        let fillers = (0..others.len())
            .flat_map(|a| {
                // Try every choice of two later cells to join this first opponent cell.
                let others = &others; // Shared borrow captured by iterator.
                (a + 1..others.len()).flat_map(move |b| {
                    // Try every later third cell to join these first two opponent cells.
                    (b + 1..others.len()).map(move |c| {
                        // Make a candidate set of three distinct cells for the opponent.
                        [others[a], others[b], others[c]]
                    })
                })
            })
            .find(|cells| {
                // Accept these opponent cells when player 0's position remains ongoing.
                !lines.iter().any(|line| {
                    // Check whether this winning line is covered by the opponent cells.
                    line.iter().all(|cell| {
                        // Check whether this line's cell is among the opponent cells.
                        cells.contains(cell)
                    })
                })
            })
            .unwrap();
        for winner_index in 0..2 {
            let mut moves = Vec::new();
            for i in 0..3 {
                if winner_index == 0 {
                    moves.push(target[i]);
                    if i < 2 {
                        moves.push(fillers[i]);
                    }
                } else {
                    moves.push(fillers[i]);
                    moves.push(target[i]);
                }
            }
            let mut state = play(&moves[..moves.len() - 1]);
            assert_eq!(state.status(), SingleWinnerOutcome::Ongoing);
            let winner = PlayerId::new(winner_index);
            let &(x, y) = moves.last().unwrap();
            model.next(&mut state, winner, &PlaceMark { x, y }).unwrap();
            assert_eq!(
                state.status(),
                SingleWinnerOutcome::Won(winner),
                "{target:?}"
            );
            assert_eq!(state.winning_cells(), target);
            for observer in [winner, PlayerId::new(1 - winner_index)] {
                assert_eq!(
                    model.observe(&state, observer).unwrap().winning_cells(),
                    target
                );
            }
            assert_eq!(model.next_player(&state), Decision::Finished);
            assert_eq!(state.current_player(), winner);
            assert_eq!(state.move_count(), moves.len());
            assert_eq!(state.completed_turn_count(), moves.len() - 1);
            assert_eq!(model.player_result(&state, winner), Ok(PlayerResult::Win));
            assert_eq!(model.player_score(&state, winner), Ok(1.0));
            let loser = PlayerId::new(1 - winner_index);
            assert_eq!(model.player_result(&state, loser), Ok(PlayerResult::Loss));
            assert_eq!(model.player_score(&state, loser), Ok(-1.0));
            for player in [winner, loser] {
                assert_eq!(model.available_actions(&state, player), Ok(vec![]));
            }
        }
    }
}

#[test]
fn full_board_draw_keeps_final_actor_and_stops_play() {
    let state = play(&[
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 2),
        (0, 2),
        (2, 2),
    ]);
    let model = TicTacToeForwardModel;
    assert_eq!(state.status(), SingleWinnerOutcome::Draw);
    assert_eq!(model.next_player(&state), Decision::Finished);
    assert_eq!(state.current_player(), PlayerId::new(0));
    assert_eq!(state.move_count(), 9);
    assert_eq!(state.completed_turn_count(), 8);
    for player in [PlayerId::new(0), PlayerId::new(1)] {
        assert_eq!(model.player_result(&state, player), Ok(PlayerResult::Draw));
        assert_eq!(model.player_score(&state, player), Ok(0.0));
        assert_eq!(model.available_actions(&state, player), Ok(vec![]));
    }
}

#[test]
fn winning_on_ninth_move_takes_precedence_over_full_board_draw() {
    let state = play(&[
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    ]);
    assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
    assert_eq!(state.move_count(), 9);
    assert_eq!(state.completed_turn_count(), 8);
}

#[test]
fn invalid_moves_and_queries_are_rejected_without_mutation() {
    let model = TicTacToeForwardModel;
    let mut state = play(&[(0, 0)]);
    let original = state.clone();
    let cases = [
        (
            PlayerId::new(2),
            PlaceMark { x: 1, y: 0 },
            TicTacToeError::InvalidPlayer {
                player: PlayerId::new(2),
            },
        ),
        (
            PlayerId::new(usize::MAX),
            PlaceMark { x: 1, y: 0 },
            TicTacToeError::InvalidPlayer {
                player: PlayerId::new(usize::MAX),
            },
        ),
        (
            PlayerId::new(0),
            PlaceMark { x: 1, y: 0 },
            TicTacToeError::WrongPlayer {
                expected: PlayerId::new(1),
                actual: PlayerId::new(0),
            },
        ),
        (
            PlayerId::new(1),
            PlaceMark { x: 0, y: 0 },
            TicTacToeError::OccupiedCell { x: 0, y: 0 },
        ),
        (
            PlayerId::new(1),
            PlaceMark { x: 3, y: 0 },
            TicTacToeError::OutOfBounds { x: 3, y: 0 },
        ),
        (
            PlayerId::new(1),
            PlaceMark { x: 0, y: 3 },
            TicTacToeError::OutOfBounds { x: 0, y: 3 },
        ),
        (
            PlayerId::new(1),
            PlaceMark {
                x: usize::MAX,
                y: usize::MAX,
            },
            TicTacToeError::OutOfBounds {
                x: usize::MAX,
                y: usize::MAX,
            },
        ),
    ];
    for (actor, action, error) in cases {
        assert_eq!(model.next(&mut state, actor, &action), Err(error));
        assert_eq!(state, original);
    }
    for actor in [PlayerId::new(2), PlayerId::new(usize::MAX)] {
        assert_eq!(
            model.available_actions(&state, actor),
            Err(TicTacToeError::InvalidPlayer { player: actor })
        );
    }
    assert_eq!(
        model.available_actions(&state, PlayerId::new(0)),
        Err(TicTacToeError::WrongPlayer {
            expected: PlayerId::new(1),
            actual: PlayerId::new(0)
        })
    );
    assert_eq!(state, original);
}

#[test]
fn completed_games_reject_moves_without_mutation() {
    let model = TicTacToeForwardModel;
    for moves in [
        &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)][..],
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 2),
            (0, 2),
            (2, 2),
        ][..],
    ] {
        let mut state = play(moves);
        let original = state.clone();
        for actor in [PlayerId::new(0), PlayerId::new(1)] {
            assert_eq!(
                model.next(&mut state, actor, &PlaceMark { x: 2, y: 2 }),
                Err(TicTacToeError::MoveAfterGameFinished)
            );
            assert_eq!(state, original);
        }
        assert_eq!(
            model.available_actions(&state, PlayerId::new(2)),
            Err(TicTacToeError::InvalidPlayer {
                player: PlayerId::new(2)
            })
        );
    }
}

#[test]
fn unsupported_configuration_is_rejected() {
    let model = TicTacToeForwardModel;
    for count in [0, 1, 3, usize::MAX] {
        assert_eq!(
            model.init_state(&TicTacToeParameters::default(), count),
            Err(TicTacToeError::UnsupportedPlayerCount { requested: count })
        );
    }
    use tag_rust::core::grid::GridError;
    assert_eq!(
        model.init_state(
            &TicTacToeParameters {
                grid_size: 0,
                ..Default::default()
            },
            2
        ),
        Err(TicTacToeError::Grid(GridError::InvalidDimensions {
            width: 0,
            height: 0
        }))
    );
    assert_eq!(
        model.init_state(
            &TicTacToeParameters {
                grid_size: usize::MAX,
                ..Default::default()
            },
            2
        ),
        Err(TicTacToeError::Grid(GridError::SizeOverflow))
    );
}

#[test]
fn invalid_queries_return_errors_without_changing_state() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let original = state.clone(); // Shared borrow for clone; returned snapshot is independently owned.
    for player in [PlayerId::new(2), PlayerId::new(usize::MAX)] {
        assert_eq!(
            model.player_result(&state, player),
            Err(TicTacToeError::InvalidPlayer { player })
        );
        assert_eq!(
            model.player_score(&state, player),
            Err(TicTacToeError::InvalidPlayer { player })
        );
    }
    for (x, y) in [(3, 0), (0, 3), (usize::MAX, 0), (0, usize::MAX)] {
        assert_eq!(state.cell(x, y), Err(TicTacToeError::OutOfBounds { x, y }));
    }
    assert_eq!(state, original);
}

#[test]
fn simultaneous_diagonal_wins_select_the_first_line_and_return_owned_coordinates() {
    let state = play(&[
        (0, 0),
        (1, 0),
        (2, 2),
        (1, 2),
        (2, 0),
        (0, 1),
        (0, 2),
        (2, 1),
        (1, 1),
    ]);
    let expected = vec![(0, 0), (1, 1), (2, 2)];
    let mut coordinates = state.winning_cells();
    assert_eq!(coordinates, expected);
    coordinates.clear();
    assert_eq!(state.winning_cells(), expected);
}
