//! The same execution path serves both games through the shared interface.
use std::fmt::Debug;
use tag_rust::agents::FirstAvailableAgent;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{Cell, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::runner::{RunnerError, run_game};

// This test helper uses the shared interface.
fn complete<F: ForwardModel>(model: &F, parameters: &F::Parameters) -> F::State
where
    F::Error: Debug,
{
    // Borrow model/configuration; return the owned state for game-specific assertions.
    let mut state = model.init_state(parameters, 2).unwrap();
    let mut agents = (0..model.player_count(&state))
        .map(|seat| {
            // Each seat receives a separately constructed controller with an explicit seed.
            FirstAvailableAgent::new(seat as u64 + 100)
        })
        .collect::<Vec<_>>();
    run_game(model, &mut state, &mut agents).unwrap();
    assert_eq!(model.next_player(&state), Decision::Finished);
    for seat in 0..model.player_count(&state) {
        let player = PlayerId::new(seat);
        let view = model.observe(&state, player).unwrap();
        assert!(model.offered_actions(&view).is_empty());
        assert!(model.available_actions(&state, player).unwrap().is_empty());
        assert_ne!(
            model.player_result(&state, player).unwrap(),
            PlayerResult::Ongoing
        );
    }
    state
}

#[test]
fn unchanged_runner_and_baseline_complete_both_default_games() {
    let ttt = complete(&TicTacToeForwardModel, &TicTacToeParameters::default());
    assert_eq!(ttt.move_count(), 7);
    assert_eq!(ttt.player_result(PlayerId::new(0)), Ok(PlayerResult::Win));
    let model = Connect4ForwardModel;
    let mut c4 = complete(&model, &Connect4Parameters::default());
    // The first-action agent fills three columns, then wins across four bottom cells.
    assert_eq!(c4.move_count(), 19);
    assert_eq!(c4.completed_turn_count(), 18);
    assert_eq!(c4.round_count(), 9);
    assert_eq!(c4.turn_count_in_round(), 0);
    assert_eq!(c4.winning_cells(), &[(0, 5), (1, 5), (2, 5), (3, 5)]);
    assert_eq!(c4.player_result(PlayerId::new(0)), Ok(PlayerResult::Win));
    let before = c4.clone();
    // A finished game returns immediately, preserving the final state.
    run_game(&model, &mut c4, &mut [] as &mut [FirstAvailableAgent]).unwrap();
    assert_eq!(c4, before);
}

#[test]
fn same_execution_handles_configured_wins_and_draws() {
    for (width, height, target, moves, result) in [
        (1, 1, 1, 1, PlayerResult::Win),
        (1, 6, 4, 6, PlayerResult::Draw),
        (5, 2, 4, 7, PlayerResult::Win),
        (9, 4, 5, 17, PlayerResult::Win),
    ] {
        let parameters = Connect4Parameters {
            width,
            height,
            winning_length: target,
            ..Default::default()
        };
        let state = complete(&Connect4ForwardModel, &parameters);
        assert_eq!(state.move_count(), moves);
        assert_eq!(state.player_result(PlayerId::new(0)), Ok(result));
    }
    for size in [1, 2, 4] {
        let state = complete(
            &TicTacToeForwardModel,
            &TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
        );
        assert!(state.move_count() <= size * size);
    }
}

#[test]
fn connect4_execution_can_resume_after_a_missing_controller() {
    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    assert_eq!(
        run_game(&model, &mut state, &mut [FirstAvailableAgent::new(100)]),
        Err(RunnerError::MissingController {
            player: PlayerId::new(1)
        })
    );
    assert_eq!(state.move_count(), 1);
    assert_eq!(state.cell(0, 5), Ok(Cell::Piece(PlayerId::new(0))));
    assert_eq!(state.current_player(), PlayerId::new(1));
    run_game(
        &model,
        &mut state,
        &mut [FirstAvailableAgent::new(100), FirstAvailableAgent::new(200)],
    )
    .unwrap();
    assert_eq!(state.move_count(), 19);
    assert_eq!(state.player_result(PlayerId::new(0)), Ok(PlayerResult::Win));
}
