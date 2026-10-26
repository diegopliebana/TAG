//! Run configured native MCTS against Random, alternating the MCTS seat.
//! For configurable opponents and game rules, use the match_play example.
#[path = "support/seeds.rs"]
mod seeds;
use std::error::Error;
use tag_rust::agents::{Agent, RandomAgent};
use tag_rust::core::{Decision, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::connect4::{Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::search::mcts::{EvaluatorRegistry, MctsAgent, Settings};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.is_empty() || arguments.len() > 4 {
        return Err(
            "usage: mcts <config.json> [game ID] [positive game count] [opponent seed|auto]".into(),
        );
    }
    let settings = Settings::from_json(&std::fs::read_to_string(&arguments[0])?)?;
    let count = arguments
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    if count == 0 {
        return Err("game count must be positive".into());
    }
    let opponent_seed = seeds::resolve_seed(arguments.get(3).map(String::as_str))?;
    match arguments.get(1).map(String::as_str).unwrap_or("tictactoe") {
        "tictactoe" => play(
            TicTacToeForwardModel,
            TicTacToeParameters::default(),
            EvaluatorRegistry::<TicTacToeForwardModel>::for_game(),
            settings,
            count,
            opponent_seed,
        ),
        "connect4" => play(
            Connect4ForwardModel,
            Connect4Parameters::default(),
            EvaluatorRegistry::<Connect4ForwardModel>::for_game(),
            settings,
            count,
            opponent_seed,
        ),
        name => Err(format!("unknown game '{name}' in the standalone MCTS example; use match_play for registered configurable matches").into()),
    }
}

fn play<F>(
    model: F,
    parameters: F::Parameters,
    registry: EvaluatorRegistry<F>,
    settings: Settings,
    games: usize,
    opponent_seed: u64,
) -> Result<(), Box<dyn Error>>
where
    F: PublicInformationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: Error + 'static,
{
    let mut agent = MctsAgent::new(settings, &registry)?;
    // Report the resolved seed so automatic runs can be reproduced explicitly.
    let mut opponent = RandomAgent::new(opponent_seed);
    println!(
        "MCTS seed: {}; Random seed: {opponent_seed}",
        agent.rng().initial_seed()
    );
    println!("Effective MCTS settings:\n{}", agent.settings().to_json()?);
    let mut totals = [[0usize; 3]; 2];
    let mut transitions = 0u64;
    let mut iterations = 0u64;
    for game in 0..games {
        let learner = PlayerId::new(game % 2);
        let mut state = model.init_state(&parameters, 2)?;
        while let Decision::Player(actor) = model.next_player(&state) {
            let observation = model.observe(&state, actor)?;
            let actions = model.available_actions(&state, actor).unwrap();
            let index = if actor == learner {
                let index = agent.choose(&model, actor, &observation, &actions)?;
                let report = agent
                    .last_report()
                    .expect("successful choose retains its report");
                transitions = transitions
                    .checked_add(report.counters.forward_model_calls)
                    .ok_or("transition total overflow")?;
                iterations = iterations
                    .checked_add(report.counters.completed_iterations)
                    .ok_or("iteration total overflow")?;
                index
            } else {
                opponent.choose(&model, actor, &observation, &actions)?
            };
            model.next(&mut state, actor, &actions[index])?;
        }
        let result = match model.player_result(&state, learner)? {
            PlayerResult::Win => 0,
            PlayerResult::Draw => 1,
            PlayerResult::Loss => 2,
            PlayerResult::Ongoing => return Err("finished game has ongoing learner result".into()),
        };
        totals[learner.index()][result] += 1;
    }
    for (seat, counts) in totals.iter().enumerate() {
        println!(
            "MCTS seat {seat}: wins={} draws={} losses={}",
            counts[0], counts[1], counts[2]
        );
    }
    println!(
        "Combined: wins={} draws={} losses={}; completed_iterations={iterations}; simulated_actions={transitions}",
        totals[0][0] + totals[1][0],
        totals[0][1] + totals[1][1],
        totals[0][2] + totals[1][2]
    );
    Ok(())
}
