//! Fixed-work native searches for comparison with Python's suspended searches.
use serde_json::{Value, json};
use tag_rust::{
    core::{Decision, ForwardModel},
    games::cantstop::*,
    learning::LearningGame,
    search::mcts::{
        EvaluatorRegistry, MctsAgent, Settings, SimulationContext, StateEvaluator, StateValues,
    },
};

struct Constant;
impl StateEvaluator<CantStopForwardModel> for Constant {
    fn evaluate(
        &mut self,
        state: &CantStopState,
        context: &mut SimulationContext<'_, CantStopForwardModel>,
    ) -> tag_rust::search::mcts::Result<StateValues> {
        Ok(StateValues::Full(
            (0..context.model().player_count(state))
                .map(|seat| (seat + 1) as f64 / 8.0)
                .collect(),
        ))
    }
}

fn main() {
    let model = CantStopForwardModel;
    let mut cases = Vec::<Value>::new();
    for players in 2..=4 {
        let parameters = CantStopParameters {
            seed: 42,
            rules: CantStopRules {
                markers: 1,
                column_heights: [2; 11],
                columns_to_win: 1,
            },
        };
        let mut state = model.init_state(&parameters, players).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut moves = Vec::new();
        let mut registry = EvaluatorRegistry::<CantStopForwardModel>::for_game();
        registry
            .register_state("constant", |_| Ok(Box::new(Constant)))
            .unwrap();
        while let Decision::Player(actor) = model.next_player(&state) {
            let observation = model.observe(&state, actor).unwrap();
            let actions = model.available_actions(&state, actor).unwrap();
            let phase = if actions == [CantStopAction::Bust] {
                "ForcedBust".to_owned()
            } else {
                format!("{:?}", observation.position().phase())
            };
            if seen.insert(phase.clone()) {
                for evaluator in ["game_heuristic", "constant"] {
                    let settings = Settings::from_json(
                        &json!({
                            "schema_version": 1, "seed": 91, "rollout_length": 3,
                            "budgets": {"iterations": 32, "forward_model_calls": 80},
                            "state_evaluator": {"name": evaluator}
                        })
                        .to_string(),
                    )
                    .unwrap();
                    let report = MctsAgent::new(settings.clone(), &registry)
                        .unwrap()
                        .search(&model, actor, &observation, &actions)
                        .unwrap();
                    let rows = report
                        .root_actions
                        .iter()
                        .map(|row| {
                            json!({
                                "action_id": model.action_id(&observation, &row.action).unwrap(),
                                "visits": row.visits, "mean_values": row.mean_values
                            })
                        })
                        .collect::<Vec<_>>();
                    cases.push(json!({"players": players, "parameters": parameters,
                        "moves": moves, "phase": phase, "evaluator": evaluator,
                        "settings": serde_json::from_str::<Value>(&settings.to_json().unwrap()).unwrap(),
                        "report": {"action_id": model.action_id(&observation, &actions[report.selected_index]).unwrap(),
                            "root_actions": rows, "nodes": report.nodes, "edges": report.edges,
                            "stop_reasons": report.stop_reasons.iter().map(|r|format!("{r:?}")).collect::<Vec<_>>(),
                            "counters": {"completed_iterations":report.counters.completed_iterations,
                                "forward_model_calls":report.counters.forward_model_calls,
                                "state_copies":report.counters.state_copies,
                                "state_evaluations":report.counters.state_evaluations,
                                "action_evaluations":report.counters.action_evaluations}}}));
                }
            }
            // Continue rolling instead of banking so a one-marker game reaches Bust.
            let action = actions
                .iter()
                .find(|a| **a == CantStopAction::Roll)
                .unwrap_or(&actions[0]);
            moves.push(model.action_id(&observation, action).unwrap());
            model.next(&mut state, actor, action).unwrap();
            assert!(moves.len() < 1000);
            if seen.len() == 4 {
                break;
            }
        }
        assert_eq!(seen.len(), 4);
    }
    println!("{}", serde_json::to_string(&cases).unwrap());
}
