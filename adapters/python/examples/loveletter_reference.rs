//! Sparse native observation records for complete Python/native comparisons.
//! This is privileged test output: case seeds reproduce hidden deals.
use serde_json::{Value, json};
use tag_rust::{
    agents::{Agent, RandomAgent},
    core::{Decision, ForwardModel, PlayerId},
    games::loveletter::*,
    learning::{LearningGame, LearningLimits},
};
fn views(model: &LoveLetterForwardModel, state: &LoveLetterState) -> Vec<Value> {
    let spec = model
        .learning_spec_for_players(
            &LoveLetterParameters {
                rules: state.public().rules.clone(),
                ..Default::default()
            },
            model.player_count(state),
            LearningLimits::default(),
        )
        .unwrap();
    let names: Vec<_> = spec.payloads.iter().map(|p| p.name()).collect();
    (0..model.player_count(state)).map(|seat| {
        let o=model.observe(state,PlayerId::new(seat)).unwrap();let v=model.encode(&o,&names,LearningLimits::default()).unwrap();
        json!({"observer":seat,"next_player":v.next_player.map(|p|p.index()),"progress":v.progress,"results":v.results.iter().map(|r|*r as i8).collect::<Vec<_>>(),"action_mask":v.action_mask,"offered_action_ids":v.offered_action_ids,
            "payloads":v.tensors.iter().map(|t|(t.name.to_owned(),json!({"shape":t.shape,"nonzero":t.values.iter().enumerate().filter(|(_,v)| **v!=0.).map(|(i,v)|json!([i,v])).collect::<Vec<_>>()}))).collect::<serde_json::Map<_,_>>()})
    }).collect()
}
fn case(players: usize, seed: u64, targets: [u32; 3], retain: bool) -> (Value, bool) {
    let model = LoveLetterForwardModel;
    let parameters = LoveLetterParameters {
        seed,
        rules: LoveLetterRules {
            winning_tokens: targets,
        },
        ..Default::default()
    };
    let mut state = model.init_state(&parameters, players).unwrap();
    let mut agents: Vec<_> = (0..players)
        .map(|s| RandomAgent::new(100 + s as u64))
        .collect();
    let mut positions = vec![];
    if retain {
        positions.push(views(&model, &state));
    }
    let mut moves = vec![];
    while let Decision::Player(actor) = model.next_player(&state) {
        assert!(moves.len() < 3000);
        let o = model.observe(&state, actor).unwrap();
        let actions = model.available_actions(&state, actor).unwrap();
        let a = actions[agents[actor.index()]
            .choose(&model, actor, &o, &actions)
            .unwrap()];
        moves.push(json!({"actor":actor.index(),"action_id":model.action_id(&o,&a).unwrap()}));
        model.next(&mut state, actor, &a).unwrap();
        if retain {
            positions.push(views(&model, &state));
        }
    }
    (
        json!({"players":players,"parameters":parameters,"moves":moves,"positions":positions}),
        state.public().outcome.winners().len() > 1,
    )
}
fn main() {
    let mut cases: Vec<_> = (2..=4).map(|n| case(n, 42, [2, 2, 2], true).0).collect();
    let shared = (0..2000)
        .find(|&seed| case(4, seed, [1, 1, 1], false).1)
        .expect("shared-win fixture seed");
    cases.push(case(4, shared, [1, 1, 1], true).0);
    println!("{}", serde_json::to_string(&cases).unwrap());
}
