//! Native save/load demonstration; optional output paths are handled by this caller.
use tag_rust::core::{Decision, PlayerId};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::replay::{ReplayLimits, export_replay, export_state, import_replay, import_state};
use tag_rust::session::{BranchId, BranchSession, Controller, Session};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parameters = TicTacToeParameters::default();
    let mut game = BranchSession::new(Session::create(
        TicTacToeForwardModel,
        &parameters,
        2,
        vec![
            Controller::osla(TicTacToeEvaluator, 7),
            Controller::osla(TicTacToeEvaluator, 8),
        ],
    )?);
    for version in 0..2 {
        game.advance_ai(version)?;
    }
    let branch = game.fork(BranchId(0), 1, vec![Some(Controller::Human), None])?;
    while let Decision::Player(actor) = game.active().session().next_player() {
        let version = game.active().session().position_version();
        if matches!(
            game.active().session().controller(actor),
            Some(Controller::Human)
        ) {
            let action = game
                .active()
                .session()
                .decision_view(actor)?
                .actions
                .into_iter()
                .next()
                .unwrap();
            game.submit_human(version, actor, action)?;
        } else {
            game.advance_ai(version)?;
        }
    }
    let replay = export_replay(&game, &parameters, game.info(branch)?.move_count, 250)?;
    let loaded = import_replay::<_, TicTacToeEvaluator>(
        &replay,
        TicTacToeForwardModel,
        ReplayLimits::default(),
    )?;
    assert_eq!(
        loaded.session.active().session().state(),
        game.active().session().state()
    );
    let state = export_state(
        &TicTacToeForwardModel,
        &parameters,
        game.active().session().state(),
    )?;
    let (_, restored) = import_state(&state, &TicTacToeForwardModel, ReplayLimits::default())?;
    assert_eq!(&restored, game.active().session().state());
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 2 {
        std::fs::write(&args[0], &replay)?;
        std::fs::write(&args[1], &state)?;
    } else if !args.is_empty() {
        return Err("supply both replay and state output paths".into());
    }
    println!(
        "Validated {} branches; selected branch {} has {} moves.",
        loaded.session.branch_count(),
        branch.0,
        loaded.session.info(branch)?.move_count
    );
    println!(
        "Replay: {} bytes; standalone state: {} bytes.",
        replay.len(),
        state.len()
    );
    println!(
        "Player 0 result: {:?}",
        restored.player_result(PlayerId::new(0))?
    );
    Ok(())
}
