//! Headless Random versus Random using the same trait/state/action path as OSLA.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::RandomAgent;
use tag_rust::core::ForwardModel;
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::runner::run_game;
// This example uses fixed, valid setup data. unwrap() stops on an unexpected
// error; application code should propagate or report Result errors instead.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() > 2 {
        return Err("usage: random [player-0 seed|auto] [player-1 seed|auto]".into());
    }
    let seeds = [
        seeds::resolve_seed(arguments.first().map(String::as_str))?,
        seeds::resolve_seed(arguments.get(1).map(String::as_str))?,
    ];
    println!("Random seeds: {seeds:?}");
    // A unit struct is a value with no fields: the rules need no private progress.
    let model = ttt::TicTacToeForwardModel;
    let mut state = model
        .init_state(&ttt::TicTacToeParameters::default(), 2)
        .unwrap();
    // The runner borrows rules read-only and state/controllers exclusively.
    // Each agent owns a separate RNG; borrowing does not copy or restart it.
    run_game(
        &model,
        &mut state,
        &mut [RandomAgent::new(seeds[0]), RandomAgent::new(seeds[1])],
    )
    .unwrap();
    println!(
        "Tic-Tac-Toe: {:?}, {} moves",
        state.status(),
        state.move_count()
    );
    let model = c4::Connect4ForwardModel;
    let mut state = model
        .init_state(&c4::Connect4Parameters::default(), 2)
        .unwrap();
    run_game(
        &model,
        &mut state,
        &mut [RandomAgent::new(seeds[0]), RandomAgent::new(seeds[1])],
    )
    .unwrap();
    println!(
        "Connect 4: {:?}, {} moves",
        state.status(),
        state.move_count()
    );
    Ok(())
}
