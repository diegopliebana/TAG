//! Tic-Tac-Toe position estimates checked against known values and independent arithmetic.
use crate::support::evaluation::{close, oracle};
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::tictactoe::{
    Cell as TicTacToeCell, PlaceMark, TicTacToeEvaluator, TicTacToeForwardModel,
    TicTacToeParameters,
};

#[test]
fn tictactoe_known_positions_use_explicit_perspective_and_leave_state_unchanged() {
    let model = TicTacToeForwardModel;
    let mut evaluator = TicTacToeEvaluator;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    close(evaluator.evaluate(&state, PlayerId::new(0)).unwrap(), 0.0);
    model
        .next(&mut state, PlayerId::new(0), &PlaceMark { x: 1, y: 1 })
        .unwrap();
    let before = state.clone(); // Independent snapshot for mutation detection.
    // The center belongs to four of eight lines, each contributing (1/3)^2.
    close(evaluator.evaluate(&state, PlayerId::new(0)).unwrap(), 0.05);
    close(evaluator.evaluate(&state, PlayerId::new(1)).unwrap(), -0.05);
    assert_eq!(state, before);
    assert!(evaluator.evaluate(&state, PlayerId::new(2)).is_err());
}

#[test]
fn configured_tictactoe_prefixes_match_independent_formula_and_terminal_values() {
    let model = TicTacToeForwardModel;
    let mut evaluator = TicTacToeEvaluator;
    for size in [1, 2, 3, 4, 7] {
        let mut state = model
            .init_state(
                &TicTacToeParameters {
                    grid_size: size,
                    ..Default::default()
                },
                2,
            )
            .unwrap();
        loop {
            let cells = state
                .board()
                .iter()
                .map(|cell| {
                    // Translate each mark into a seat for the independent oracle.
                    match cell {
                        TicTacToeCell::Empty => None,
                        TicTacToeCell::X => Some(PlayerId::new(0)),
                        TicTacToeCell::O => Some(PlayerId::new(1)),
                    }
                })
                .collect::<Vec<_>>();
            for player in [PlayerId::new(0), PlayerId::new(1)] {
                let expected = match state.player_result(player).unwrap() {
                    PlayerResult::Ongoing => oracle(size, size, size, &cells, player),
                    PlayerResult::Win => 1.0,
                    PlayerResult::Loss => -1.0,
                    PlayerResult::Draw => 0.0,
                };
                close(evaluator.evaluate(&state, player).unwrap(), expected);
                assert!(expected.is_finite() && (-1.0..=1.0).contains(&expected));
            }
            let Decision::Player(actor) = state.next_player() else {
                break;
            };
            let action = model.available_actions(&state, actor).unwrap().remove(0);
            model.next(&mut state, actor, &action).unwrap();
        }
        assert!(
            evaluator
                .evaluate(&state, PlayerId::new(usize::MAX))
                .is_err()
        );
    }
}

#[test]
fn tictactoe_draw_is_zero_for_both_players() {
    let model = TicTacToeForwardModel;
    let mut evaluator = TicTacToeEvaluator;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for (x, y) in [
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 2),
        (0, 2),
        (2, 2),
    ] {
        let actor = state.current_player();
        model.next(&mut state, actor, &PlaceMark { x, y }).unwrap();
    }
    for player in [PlayerId::new(0), PlayerId::new(1)] {
        assert_eq!(state.player_result(player).unwrap(), PlayerResult::Draw);
        assert_eq!(evaluator.evaluate(&state, player).unwrap(), 0.0);
    }
}
