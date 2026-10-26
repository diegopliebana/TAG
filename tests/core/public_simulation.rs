//! Permitted reconstruction copies the observation snapshot.
use std::fmt::Debug;
use tag_rust::core::{Decision, PlayerId, PublicInformationModel};
use tag_rust::games::connect4::{Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters};

fn check<F: PublicInformationModel>(model: &F, parameters: &F::Parameters)
where
    F::State: Clone + Debug + PartialEq,
    F::Error: Debug,
{
    // Borrow rules/configuration and run shared snapshot checks.
    let mut live = model.init_state(parameters, 2).unwrap();
    let initial = live.clone();
    let observer = PlayerId::new(0);
    let view = model.observe(&live, observer).unwrap();
    let frozen_view = view;
    let mut branch = model.simulation_state(&frozen_view).unwrap();
    assert_eq!(branch, initial);
    let action = model
        .available_actions(&branch, observer)
        .unwrap()
        .remove(0);
    model.next(&mut branch, observer, &action).unwrap();
    assert_ne!(branch, initial);
    assert_eq!(live, initial);
    assert_eq!(model.simulation_state(&frozen_view).unwrap(), initial);
    // Earlier views and reconstructed copies retain their snapshots after a move.
    model.next(&mut live, observer, &action).unwrap();
    assert_eq!(model.simulation_state(&frozen_view).unwrap(), initial);
    assert_eq!(live, branch);
    // Both seats can reconstruct all public facts, including terminal state.
    while let Decision::Player(actor) = model.next_player(&live) {
        let action = model.available_actions(&live, actor).unwrap().remove(0);
        model.next(&mut live, actor, &action).unwrap();
    }
    for seat in 0..2 {
        let terminal_view = model.observe(&live, PlayerId::new(seat)).unwrap();
        let reconstructed = model.simulation_state(&terminal_view).unwrap();
        assert_eq!(reconstructed, live);
        assert_eq!(model.next_player(&reconstructed), Decision::Finished);
    }
}

#[test]
fn tictactoe_reconstructs_independent_public_states() {
    check(&TicTacToeForwardModel, &TicTacToeParameters::default());
    check(
        &TicTacToeForwardModel,
        &TicTacToeParameters {
            grid_size: 4,
            ..Default::default()
        },
    );
}

#[test]
fn connect4_reconstructs_independent_public_states() {
    check(&Connect4ForwardModel, &Connect4Parameters::default());
    check(
        &Connect4ForwardModel,
        &Connect4Parameters {
            width: 3,
            height: 2,
            winning_length: 3,
            ..Default::default()
        },
    );
}
