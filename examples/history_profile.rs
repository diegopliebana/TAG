//! Native release-mode measurements of reconstruction and game simulation costs.
use std::hint::black_box;
use std::time::Instant;
use tag_rust::agents::{FirstAvailableAgent, OslaAgent};
use tag_rust::core::PublicInformationModel;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::recording::{ControllerSnapshot, GameHistory, HistoryCachePolicy};
use tag_rust::runner::{run_game, run_game_recorded};

fn profile<F, E>(name: &str, model: F, parameters: F::Parameters, evaluator: E)
where
    F: PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Copy,
    E::Error: std::fmt::Debug,
{
    let mut agents = [FirstAvailableAgent::new(0), FirstAvailableAgent::new(1)];
    let mut state = model.init_state(&parameters, 2).unwrap();
    let root = state.clone();
    let seats = agents
        .iter()
        .map(|agent| ControllerSnapshot::Ai(agent.snapshot()))
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    run_game_recorded(&model, &mut state, &mut agents, |record| {
        records.push(record)
    })
    .unwrap();
    println!("{name}: {} accepted moves", records.len());
    for (reverse, max_states) in [(false, 4), (true, 4), (true, 64)] {
        let mut history = GameHistory::with_cache_policy(
            root.clone(),
            seats.clone(),
            HistoryCachePolicy {
                max_states,
                ..Default::default()
            },
        );
        for record in records.iter().cloned() {
            history.record(record);
        }
        let start = Instant::now();
        for offset in 0..=records.len() {
            let position = if reverse {
                records.len() - offset
            } else {
                offset
            };
            black_box(history.position_at(&model, position).unwrap());
        }
        println!(
            "  {} navigation (cache limit {max_states}): {:.3} ms",
            if reverse { "backward" } else { "forward" },
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let start = Instant::now();
    for _ in 0..100_000 {
        black_box(root.clone());
    }
    println!(
        "  100000 independent state copies: {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let start = Instant::now();
    let mut moves = 0;
    for _ in 0..100 {
        let mut state = root.clone();
        let mut agents = [FirstAvailableAgent::new(0), FirstAvailableAgent::new(1)];
        run_game(&model, &mut state, &mut agents).unwrap();
        black_box(state);
        moves += records.len();
    }
    println!(
        "  100 first-available games ({moves} moves): {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    // One decision exercises observation reconstruction, candidate copies, rules and evaluation.
    let actor = tag_rust::core::PlayerId::new(0);
    let view = model.observe(&root, actor).unwrap();
    let actions = model.available_actions(&root, actor).unwrap();
    let start = Instant::now();
    let mut agent = OslaAgent::new(evaluator, 0);
    use tag_rust::agents::Agent;
    for _ in 0..100 {
        black_box(agent.choose(&model, actor, &view, &actions).unwrap());
    }
    println!(
        "  100 OSLA initial decisions ({} candidates each): {:.3} ms",
        actions.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );
}
fn main() {
    profile(
        "Tic-Tac-Toe 3",
        TicTacToeForwardModel,
        TicTacToeParameters::default(),
        TicTacToeEvaluator,
    );
    profile(
        "Tic-Tac-Toe 32",
        TicTacToeForwardModel,
        TicTacToeParameters {
            grid_size: 32,
            ..Default::default()
        },
        TicTacToeEvaluator,
    );
    profile(
        "Connect 4 standard",
        Connect4ForwardModel,
        Connect4Parameters::default(),
        Connect4Evaluator,
    );
    profile(
        "Connect 4 32x32 target32",
        Connect4ForwardModel,
        Connect4Parameters {
            width: 32,
            height: 32,
            winning_length: 32,
            ..Default::default()
        },
        Connect4Evaluator,
    );
}
