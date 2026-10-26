//! Generate and validate educational save files for both supported games.
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;
use std::path::Path;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::replay::{
    ReplayGame, ReplayLimits, export_replay, export_state, import_replay, import_state,
};
use tag_rust::session::{BranchId, BranchSession, BuiltinAgent, Controller, Session};

/// Borrow a branch collection and complete its active continuation with explicit human inputs.
/// Human seats select their first offered move; AI seats use their configured policy/RNG.
fn finish<F, E>(game: &mut BranchSession<F, BuiltinAgent<E>>)
where
    F: PublicInformationModel + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: Debug,
    E::RestoreError: Debug,
{
    while let Decision::Player(actor) = game.active().session().next_player() {
        let version = game.active().session().position_version();
        if matches!(
            game.active().session().controller(actor),
            Some(Controller::Human)
        ) {
            let view = game.active().session().decision_view(actor).unwrap();
            game.submit_human(version, actor, view.actions.into_iter().next().unwrap())
                .unwrap();
        } else {
            game.advance_ai(version).unwrap();
        }
    }
}
/// Borrow output directory and consume rules/parameters/evaluator to generate one game's examples.
/// Exports use the engine's codec; import checks finish before files are written.
fn write_examples<F, E>(directory: &Path, model: F, parameters: F::Parameters, evaluator: E)
where
    F: ReplayGame<SavedEvaluator = E> + PublicInformationModel,
    F::State: Clone + PartialEq + Debug,
    F::Action: Clone + PartialEq + Serialize + DeserializeOwned,
    F::Parameters: Clone + Serialize + DeserializeOwned,
    F::Error: Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()> + Copy,
    E::Error: Debug,
    E::RestoreError: Debug,
{
    let mut game = BranchSession::new(
        Session::create(
            model.clone(),
            &parameters,
            2,
            vec![
                Controller::osla(evaluator, 7),
                Controller::osla(evaluator, 8),
            ],
        )
        .unwrap(),
    );
    for _ in 0..3 {
        game.advance_ai(game.active().session().position_version())
            .unwrap();
    }
    let middle = export_state(&model, &parameters, game.active().session().state()).unwrap();
    let (_, restored) = import_state(&middle, &model, ReplayLimits::default()).unwrap();
    assert_eq!(&restored, game.active().session().state());
    finish(&mut game);
    let original_length = game.info(BranchId(0)).unwrap().move_count;
    let child = game
        .fork(BranchId(0), 2, vec![Some(Controller::Human), None])
        .unwrap();
    finish(&mut game);
    let finished = export_state(&model, &parameters, game.active().session().state()).unwrap();
    let (_, restored) = import_state(&finished, &model, ReplayLimits::default()).unwrap();
    assert_eq!(&restored, game.active().session().state());
    let replay = export_replay(&game, &parameters, 2, 250).unwrap();
    let loaded = import_replay::<F, E>(&replay, model, ReplayLimits::default()).unwrap();
    assert_eq!(
        loaded.session.active().session().state(),
        game.active().session().state()
    );
    for (suffix, text) in [
        (".tag-state.json", middle),
        ("-finished.tag-state.json", finished),
        (".tag-replay.json", replay),
    ] {
        std::fs::write(directory.join(format!("{}{suffix}", F::GAME_ID)), text).unwrap();
    }
    println!(
        "{}: root {original_length} moves, branch {} {} moves; saved state/replay files validated.",
        F::GAME_ID,
        child.0,
        game.info(child).unwrap().move_count
    );
}
fn main() {
    let directory = Path::new("examples/saves");
    std::fs::create_dir_all(directory).unwrap();
    write_examples(
        directory,
        TicTacToeForwardModel,
        TicTacToeParameters::default(),
        TicTacToeEvaluator,
    );
    write_examples(
        directory,
        Connect4ForwardModel,
        Connect4Parameters::default(),
        Connect4Evaluator,
    );
}
