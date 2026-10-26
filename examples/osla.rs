//! Run optionally seeded one-step agents in both games through the shared runner.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::OslaAgent;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{PlayerId, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::runner::run_game;

/// Run a match and print each player's result using shared framework interfaces.
///
/// # Arguments
/// Rules, parameters and name are borrowed; evaluator construction is consumed.
///
/// # Returns
/// An owned completed state, allowing game-specific reporting by the caller.
///
/// # Errors
/// Initialization, execution or result-query failures are returned to the caller.
fn play<F, E>(
    name: &str,
    model: &F,
    parameters: &F::Parameters,
    seeds: [u64; 2],
    evaluator: impl Fn() -> E,
) -> Result<F::State, Box<dyn std::error::Error>>
where
    F: PublicInformationModel,
    F::State: Clone,
    F::Error: std::error::Error + 'static,
    E: Evaluator<F::State>,
    E::Error: std::error::Error + 'static,
{
    let mut state = model.init_state(parameters, 2)?;
    let mut agents = [
        OslaAgent::new(evaluator(), seeds[0]),
        OslaAgent::new(evaluator(), seeds[1]),
    ];
    run_game(model, &mut state, &mut agents)?;
    println!("{name}: OSLA seeds {seeds:?}");
    for seat in 0..model.player_count(&state) {
        println!(
            "  Player {seat}: {:?}",
            model.player_result(&state, PlayerId::new(seat))?
        );
    }
    Ok(state)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() > 2 {
        return Err("usage: osla [player-0 seed|auto] [player-1 seed|auto]".into());
    }
    let seeds = [
        seeds::resolve_seed(arguments.first().map(String::as_str))?,
        seeds::resolve_seed(arguments.get(1).map(String::as_str))?,
    ];
    println!("OSLA seeds: {seeds:?}");
    let ttt = play(
        "Tic-Tac-Toe (3x3)",
        &TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        seeds,
        || TicTacToeEvaluator,
    )?;
    println!("  Moves: {}", ttt.move_count());
    let c4 = play(
        "Connect 4 (7x6, target 4)",
        &Connect4ForwardModel,
        &Connect4Parameters::default(),
        seeds,
        || Connect4Evaluator,
    )?;
    println!("  Moves: {}", c4.move_count());
    Ok(())
}
