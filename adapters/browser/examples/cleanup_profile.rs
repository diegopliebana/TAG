//! Release-mode workloads for browser views and nested branch selection.
use std::{hint::black_box, time::Instant};
use tag_browser::BrowserEngine;
use tag_rust::core::PlayerId;
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::session::{BranchId, BranchSession, BuiltinAgent, Controller, Session};

/// Measure complete JSON view requests for a configured public board.
///
/// # Arguments
/// `size` is the copied board side; `requests` is the copied number of views.
///
/// # Returns
/// Unit after printing elapsed milliseconds. Each response is retained by black_box.
fn views(size: usize, requests: usize) {
    let mut engine = BrowserEngine::new();
    let setup = format!(
        r#"{{"request_id":"1","generation":"0","command":{{"type":"create","game":"tictactoe","parameters":{{"grid_size":{size},"limits":{{"max_cells":1000000}}}}}}}}"#
    );
    let reply: serde_json::Value = serde_json::from_str(&engine.handle(&setup)).unwrap();
    assert!(reply["error"].is_null(), "{reply}");
    let request = r#"{"request_id":"2","generation":"1","command":{"type":"view"}}"#;
    for _ in 0..100 {
        black_box(engine.handle(request));
    }
    let start = Instant::now();
    for _ in 0..requests {
        black_box(engine.handle(request));
    }
    println!(
        "views {size}x{size}, {requests} requests: {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

type Match = BranchSession<TicTacToeForwardModel, BuiltinAgent<TicTacToeEvaluator>>;

/// Advance a shared-borrow-free live branch with the first offered human action.
///
/// # Arguments
/// `game` is accessed through an exclusive mutable reference; `moves` is the copied number of accepted moves.
///
/// # Returns
/// Unit after extending the active branch. This workload stays below any winning line.
fn advance(game: &mut Match, moves: usize) {
    for _ in 0..moves {
        let session = game.active().session();
        let actor = PlayerId::new(session.state().current_player().index());
        let input = session.decision_view(actor).unwrap();
        game.submit_human(input.position_version, actor, input.actions[0])
            .unwrap();
    }
}

/// Measure selecting long nested continuations with their exact recorded state.
///
/// # Arguments
/// `cycles` is the copied number of round trips between the root and nested branch.
///
/// # Returns
/// Unit after printing time and validating both tips against independent snapshots.
fn branches(cycles: usize) {
    let mut game = Match::new(
        Session::create(
            TicTacToeForwardModel,
            &TicTacToeParameters {
                grid_size: 32,
                ..Default::default()
            },
            2,
            vec![Controller::Human, Controller::Human],
        )
        .unwrap(),
    );
    advance(&mut game, 900);
    let root_tip = game.active().session().state().clone();
    let child = game.fork(BranchId(0), 400, vec![None, None]).unwrap();
    advance(&mut game, 200);
    let nested = game.fork(child, 550, vec![None, None]).unwrap();
    advance(&mut game, 50);
    let nested_tip = game.active().session().state().clone();
    let start = Instant::now();
    for _ in 0..cycles {
        game.select(BranchId(0)).unwrap();
        assert_eq!(game.active().session().state(), &root_tip);
        game.select(nested).unwrap();
        assert_eq!(game.active().session().state(), &nested_tip);
    }
    println!(
        "branches 900/600 moves, {cycles} round trips: {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

fn main() {
    for _ in 0..3 {
        views(16, 2000);
        views(64, 2000);
        branches(20);
    }
}
