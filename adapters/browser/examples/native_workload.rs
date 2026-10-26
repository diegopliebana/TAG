//! Record portable native workloads for comparison with the real browser/worker.
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tag_rust::agents::{FirstAvailableAgent, OslaAgent, RandomAgent};
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, PublicInformationModel};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::recording::{ControllerSnapshot, GameHistory};
use tag_rust::replay::{BoardStateData, ReplayGame, export_replay};
use tag_rust::runner::run_game_recorded;
use tag_rust::session::{BranchSession, BuiltinAgent};
// Browser-only winning emphasis is derived; persisted Tic-Tac-Toe files stay unchanged.
trait PresentationState {
    fn winning_cells(&self) -> Vec<(usize, usize)>;
}
impl PresentationState for ttt::TicTacToeState {
    fn winning_cells(&self) -> Vec<(usize, usize)> {
        self.winning_cells()
    }
}
impl PresentationState for c4::Connect4State {
    fn winning_cells(&self) -> Vec<(usize, usize)> {
        self.winning_cells().to_vec()
    }
}
fn record<F, E>(
    name: &str,
    model: F,
    parameters: F::Parameters,
    evaluator: E,
    policies: [&str; 2],
) -> Value
where
    F: ReplayGame<SavedState = BoardStateData, SavedEvaluator = E> + PublicInformationModel,
    F::State: Clone + PartialEq + PresentationState,
    F::Action: Clone + PartialEq + Serialize + DeserializeOwned,
    F::Parameters: Clone + Serialize,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()> + Copy,
    E::Error: std::fmt::Debug,
    E::RestoreError: std::fmt::Debug,
{
    let root = model.init_state(&parameters, 2).unwrap();
    let mut state = root.clone();
    let mut agents = policies
        .iter()
        .enumerate()
        .map(|(seat, policy)| {
            let seed = if seat == 0 { 7 } else { 8 };
            match *policy {
                "random" => BuiltinAgent::Random(RandomAgent::new(seed)),
                "osla" => BuiltinAgent::Osla(OslaAgent::new(evaluator, seed)),
                "first_available" => BuiltinAgent::FirstAvailable(FirstAvailableAgent::new(seed)),
                _ => panic!("unknown workload policy"),
            }
        })
        .collect::<Vec<_>>();
    let controllers = json!([{"kind":policies[0],"seed":"7"},{"kind":policies[1],"seed":"8"}]);
    let mut history = GameHistory::new(
        root.clone(),
        agents
            .iter()
            .map(|a| ControllerSnapshot::Ai(a.snapshot()))
            .collect(),
    );
    let mut records = Vec::new();
    run_game_recorded(&model, &mut state, &mut agents, |record| {
        records.push(record)
    })
    .unwrap();
    let mut replay = root.clone();
    let mut decisions = Vec::new();
    for record in records {
        history.record(record.clone());
        let actions = model.available_actions(&replay, record.actor).unwrap();
        model
            .next(&mut replay, record.actor, &record.action)
            .unwrap();
        let mut displayed = model.state_data(&replay);
        displayed.winning_cells = replay.winning_cells();
        decisions.push(json!({"actor":record.actor.index(),"actions":actions,"action":record.action,"next_player":match model.next_player(&replay){Decision::Player(p)=>Some(p.index()),Decision::Finished=>None},"board":displayed}));
    }
    assert_eq!(model.state_data(&replay), model.state_data(&state));
    let session = BranchSession::from_history(model.clone(), history).unwrap();
    let file = export_replay(&session, &parameters, 0, 0).unwrap();
    json!({"name":name,"game":F::GAME_ID,"parameters":parameters,"controllers":controllers,"initial":model.state_data(&root),"decisions":decisions,"replay":file})
}
fn main() {
    let cases = vec![
        record(
            "tictactoe-standard",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters::default(),
            ttt::TicTacToeEvaluator,
            ["first_available", "first_available"],
        ),
        record(
            "tictactoe-mixed",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters::default(),
            ttt::TicTacToeEvaluator,
            ["first_available", "osla"],
        ),
        record(
            "tictactoe-configured",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters {
                grid_size: 5,
                ..Default::default()
            },
            ttt::TicTacToeEvaluator,
            ["first_available", "first_available"],
        ),
        record(
            "connect4-standard",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters::default(),
            c4::Connect4Evaluator,
            ["first_available", "first_available"],
        ),
        record(
            "connect4-mixed",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters::default(),
            c4::Connect4Evaluator,
            ["first_available", "osla"],
        ),
        record(
            "connect4-configured",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters {
                width: 4,
                height: 3,
                winning_length: 3,
                ..Default::default()
            },
            c4::Connect4Evaluator,
            ["first_available", "first_available"],
        ),
        record(
            "tictactoe-random",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters::default(),
            ttt::TicTacToeEvaluator,
            ["random", "random"],
        ),
        record(
            "connect4-random",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters::default(),
            c4::Connect4Evaluator,
            ["random", "random"],
        ),
        record(
            "tictactoe-random-osla",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters::default(),
            ttt::TicTacToeEvaluator,
            ["random", "osla"],
        ),
        record(
            "connect4-random-osla",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters::default(),
            c4::Connect4Evaluator,
            ["random", "osla"],
        ),
        record(
            "tictactoe-random-configured",
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters {
                grid_size: 5,
                ..Default::default()
            },
            ttt::TicTacToeEvaluator,
            ["random", "first_available"],
        ),
        record(
            "connect4-random-configured",
            c4::Connect4ForwardModel,
            c4::Connect4Parameters {
                width: 4,
                height: 3,
                winning_length: 3,
                ..Default::default()
            },
            c4::Connect4Evaluator,
            ["random", "first_available"],
        ),
    ];
    let path = std::path::Path::new(".scratch/browser-proof");
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(
        path.join("workload.json"),
        serde_json::to_string_pretty(&cases).unwrap(),
    )
    .unwrap();
    println!("Recorded {} native comparison workloads.", cases.len());
}
