//! Connect 4 position estimates checked against known values and independent arithmetic.
use crate::support::evaluation::{close, oracle};
use tag_rust::core::evaluation::{EVALUATION_TIE_TOLERANCE, Evaluator};
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{
    Cell as Connect4Cell, Connect4Evaluator, Connect4ForwardModel, Connect4Parameters, DropPiece,
};

#[test]
fn connect4_known_position_counts_diagonal_potential_despite_gravity() {
    let model = Connect4ForwardModel;
    let mut evaluator = Connect4Evaluator;
    let parameters = Connect4Parameters {
        width: 4,
        height: 3,
        winning_length: 3,
        ..Default::default()
    };
    let mut state = model.init_state(&parameters, 2).unwrap();
    close(evaluator.evaluate(&state, PlayerId::new(0)).unwrap(), 0.0);
    model
        .next(&mut state, PlayerId::new(0), &DropPiece { column: 0 })
        .unwrap();
    let before = state.clone();
    // The bottom corner belongs to three of fourteen windows. The diagonal is
    // potential whose empty cells require supporting pieces below them.
    close(
        evaluator.evaluate(&state, PlayerId::new(0)).unwrap(),
        0.9 / 42.0,
    );
    close(
        evaluator.evaluate(&state, PlayerId::new(1)).unwrap(),
        -0.9 / 42.0,
    );
    assert_eq!(state, before);
    assert!(evaluator.evaluate(&state, PlayerId::new(2)).is_err());
}

#[test]
fn configured_connect4_prefixes_match_independent_formula_and_terminal_values() {
    let model = Connect4ForwardModel;
    let mut evaluator = Connect4Evaluator;
    let mut saw_win = false;
    let mut saw_draw = false;
    for (width, height, target) in [
        (1, 1, 1),
        (1, 6, 4),
        (6, 1, 4),
        (3, 3, 3),
        (7, 6, 4),
        (9, 4, 5),
    ] {
        let mut state = model
            .init_state(
                &Connect4Parameters {
                    width,
                    height,
                    winning_length: target,
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
                    // Translate each piece into a seat for the independent oracle.
                    match cell {
                        Connect4Cell::Empty => None,
                        Connect4Cell::Piece(seat) => Some(*seat),
                    }
                })
                .collect::<Vec<_>>();
            for player in [PlayerId::new(0), PlayerId::new(1)] {
                let expected = match state.player_result(player).unwrap() {
                    PlayerResult::Ongoing => {
                        let estimate = oracle(width, height, target, &cells, player);
                        assert!(estimate.abs() <= 0.9);
                        assert!(1.0 - estimate > EVALUATION_TIE_TOLERANCE);
                        estimate
                    }
                    PlayerResult::Win => {
                        saw_win = true;
                        1.0
                    }
                    PlayerResult::Loss => -1.0,
                    PlayerResult::Draw => {
                        saw_draw = true;
                        0.0
                    }
                };
                close(evaluator.evaluate(&state, player).unwrap(), expected);
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
    assert!(saw_win && saw_draw);
}
