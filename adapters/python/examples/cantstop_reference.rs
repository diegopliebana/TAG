//! Native seeded trajectories used to check Python conversion and game operations.
use serde_json::{Value, json};
use tag_rust::{
    agents::{Agent, RandomAgent},
    core::{Decision, ForwardModel, PlayerId},
    games::cantstop::*,
    learning::{LearningGame, LearningLimits},
};
const NAMES: &[&str] = &["permanent", "temporary", "owners", "dice", "context"];
fn views(model: &CantStopForwardModel, state: &CantStopState) -> Vec<Value> {
    (0..model.player_count(state)).map(|seat| {
        let observation=model.observe(state,PlayerId::new(seat)).unwrap();
        let value=model.encode(&observation,NAMES,LearningLimits::default()).unwrap();
        json!({"observer":seat,"next_player":value.next_player.map(|p|p.index()),"progress":value.progress,
            "results":value.results.iter().map(|r|*r as i8).collect::<Vec<_>>(),"action_mask":value.action_mask,"offered_action_ids":value.offered_action_ids,
            "payloads":value.tensors.iter().map(|t|(t.name.to_owned(),json!({"shape":t.shape,"values":t.values}))).collect::<serde_json::Map<_,_>>()})
    }).collect()
}
fn main() {
    let model = CantStopForwardModel;
    let mut cases = Vec::new();
    for players in 2..=4 {
        let parameters = CantStopParameters {
            seed: 42 + players as u64,
            rules: CantStopRules {
                column_heights: [3; 11],
                columns_to_win: 2,
                ..Default::default()
            },
        };
        let mut state = model.init_state(&parameters, players).unwrap();
        let mut agents = (0..players)
            .map(|seat| RandomAgent::new(100 + seat as u64))
            .collect::<Vec<_>>();
        let mut positions = vec![views(&model, &state)];
        let mut moves = Vec::new();
        while let Decision::Player(actor) = model.next_player(&state) {
            assert!(moves.len() < 10000);
            let observation = model.observe(&state, actor).unwrap();
            let actions = model.available_actions(&state, actor).unwrap();
            let selected = agents[actor.index()]
                .choose(&model, actor, &observation, &actions)
                .unwrap();
            let action = actions[selected];
            let id = model.action_id(&observation, &action).unwrap();
            moves.push(json!({"actor":actor.index(),"action_id":id}));
            model.next(&mut state, actor, &action).unwrap();
            positions.push(views(&model, &state));
        }
        cases.push(
            json!({"players":players,"parameters":parameters,"moves":moves,"positions":positions}),
        );
    }
    println!("{}", serde_json::to_string(&cases).unwrap());
}
