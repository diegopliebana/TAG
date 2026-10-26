//! Run batched lookahead headlessly with the open-line heuristic entirely in Rust.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::{
    EvaluationLimits, EvaluatorIdentity, OpenLinesBatchEvaluator, OslaPythonAgent, OslaPythonConfig,
};
use tag_rust::core::{PlayerId, PublicInformationModel};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{LearningGame, LearningLimits};
use tag_rust::runner::run_game;

fn play<F>(
    name: &str,
    model: F,
    parameters: F::Parameters,
    seeds: [u64; 2],
) -> Result<(), Box<dyn std::error::Error>>
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::error::Error + 'static,
{
    let settings = OslaPythonConfig {
        evaluator: EvaluatorIdentity {
            name: "rust-open-lines".into(),
            version: 1,
            configuration: "{}".into(),
        },
        payloads: vec!["cells".into()],
        limits: EvaluationLimits {
            max_rows: 3,
            ..Default::default()
        },
    };
    // The game supplies its ordered winning-line groups once. Both independent
    // controllers score the resulting candidate arrays directly in Rust.
    let backend = OpenLinesBatchEvaluator::new(
        &model.learning_spec(&parameters, LearningLimits::default())?,
    )?;
    let mut agents = [
        OslaPythonAgent::new(backend.clone(), settings.clone(), seeds[0])?,
        OslaPythonAgent::new(backend, settings, seeds[1])?,
    ];
    let mut state = model.init_state(&parameters, 2)?;
    run_game(&model, &mut state, &mut agents)?;
    println!(
        "{name}: {:?}, {:?}",
        model.player_result(&state, PlayerId::new(0))?,
        model.player_result(&state, PlayerId::new(1))?
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() > 2 {
        return Err("usage: osla_python [player-0 seed|auto] [player-1 seed|auto]".into());
    }
    let seeds = [
        seeds::resolve_seed(arguments.first().map(String::as_str))?,
        seeds::resolve_seed(arguments.get(1).map(String::as_str))?,
    ];
    println!("OSLAPython seeds: {seeds:?}");
    play(
        "Tic-Tac-Toe",
        ttt::TicTacToeForwardModel,
        Default::default(),
        seeds,
    )?;
    play(
        "Connect 4",
        c4::Connect4ForwardModel,
        Default::default(),
        seeds,
    )
}
