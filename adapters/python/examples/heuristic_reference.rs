//! Record native OSLA scores and seeded moves for the Python heuristic tests.
use serde_json::{Value, json};
use tag_rust::agents::{Agent, OslaAgent};
use tag_rust::core::{Decision, PublicInformationModel, evaluation::Evaluator};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::LearningGame;

fn record<F, E>(model: F, parameters: F::Parameters, evaluator: E) -> Vec<Value>
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Clone,
    E::Error: std::fmt::Debug,
{
    let mut state = model.init_state(&parameters, 2).unwrap();
    let mut agents = [
        OslaAgent::new(evaluator.clone(), 7),
        OslaAgent::new(evaluator.clone(), 8),
    ];
    let mut scoring = evaluator;
    let mut records = Vec::new();
    while let Decision::Player(actor) = model.next_player(&state) {
        let observation = model.observe(&state, actor).unwrap();
        let actions = model.available_actions(&state, actor).unwrap();
        let mut values = Vec::new();
        let mut ids = Vec::new();
        for action in &actions {
            let mut candidate = state.clone();
            model.next(&mut candidate, actor, action).unwrap();
            values.push(scoring.evaluate(&candidate, actor).unwrap());
            ids.push(model.action_id(&observation, action).unwrap());
        }
        let chosen = agents[actor.index()]
            .choose(&model, actor, &observation, &actions)
            .unwrap();
        records.push(json!({"actor": actor.index(), "ids": ids, "values": values,
            "chosen": ids[chosen]}));
        model.next(&mut state, actor, &actions[chosen]).unwrap();
    }
    records
}

fn main() {
    let mut output = Vec::new();
    for size in [1, 3, 5] {
        let parameters = ttt::TicTacToeParameters {
            grid_size: size,
            ..Default::default()
        };
        output.push(json!({"game":"tictactoe", "parameters":{"grid_size":size},
            "moves":record(ttt::TicTacToeForwardModel, parameters, ttt::TicTacToeEvaluator)}));
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3), (3, 2, 1)] {
        let parameters = c4::Connect4Parameters {
            width,
            height,
            winning_length,
            ..Default::default()
        };
        output.push(
            json!({"game":"connect4", "parameters":{"width":width,"height":height,
            "winning_length":winning_length},
            "moves":record(c4::Connect4ForwardModel, parameters, c4::Connect4Evaluator)}),
        );
    }
    println!("{}", serde_json::to_string(&output).unwrap());
}
