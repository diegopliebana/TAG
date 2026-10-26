//! Direct persistent native agents, independent of Python's opponent checkpoint adapter.
use serde_json::json;
use std::num::NonZeroU32;
use tag_rust::{
    agents::{Agent, RandomAgent, SampledOslaAgent},
    core::{Decision, ForwardModel, PlayerId},
    games::cantstop::*,
    learning::LearningGame,
    search::mcts::{EvaluatorRegistry, MctsAgent, Settings},
};
enum Controller {
    Random(Box<RandomAgent>),
    Osla(Box<SampledOslaAgent<CantStopProgress>>),
    Mcts(Box<MctsAgent<CantStopForwardModel>>),
}
impl Controller {
    fn new(kind: &str, seed: u64) -> Self {
        match kind {
            "random" => Self::Random(Box::new(RandomAgent::new(seed))),
            "osla" => Self::Osla(Box::new(SampledOslaAgent::new(
                CantStopProgress::default(),
                seed,
                NonZeroU32::new(2).unwrap(),
            ))),
            _ => {
                let settings = Settings::from_json(&json!({"schema_version":1,"seed":seed,"rollout_length":0,"budgets":{"iterations":5}}).to_string()).unwrap();
                Self::Mcts(Box::new(
                    MctsAgent::new(
                        settings,
                        &EvaluatorRegistry::<CantStopForwardModel>::for_game(),
                    )
                    .unwrap(),
                ))
            }
        }
    }
    fn choose(
        &mut self,
        model: &CantStopForwardModel,
        actor: PlayerId,
        observation: &CantStopObservation,
        actions: &[CantStopAction],
    ) -> usize {
        match self {
            Self::Random(a) => a.choose(model, actor, observation, actions).unwrap(),
            Self::Osla(a) => a.choose(model, actor, observation, actions).unwrap(),
            Self::Mcts(a) => a.choose(model, actor, observation, actions).unwrap(),
        }
    }
}
fn main() {
    let model = CantStopForwardModel;
    let mut records = Vec::new();
    for players in 2..=4 {
        for candidate in 0..players {
            for kind in ["random", "osla", "mcts"] {
                let parameters = CantStopParameters {
                    seed: 42,
                    rules: CantStopRules {
                        column_heights: [3; 11],
                        columns_to_win: 2,
                        ..Default::default()
                    },
                };
                let mut state = model.init_state(&parameters, players).unwrap();
                let mut agents = (0..players)
                    .map(|seat| Controller::new(kind, 100 + seat as u64))
                    .collect::<Vec<_>>();
                let mut actions = Vec::new();
                while let Decision::Player(actor) = model.next_player(&state) {
                    let observation = model.observe(&state, actor).unwrap();
                    let offered = model.available_actions(&state, actor).unwrap();
                    let index = if actor.index() == candidate {
                        0
                    } else {
                        agents[actor.index()].choose(&model, actor, &observation, &offered)
                    };
                    let id = model.action_id(&observation, &offered[index]).unwrap();
                    let dice = model
                        .next_recorded(&mut state, actor, &offered[index])
                        .unwrap()
                        .map(|e| e.dice);
                    actions.push(json!({"actor":actor.index(),"action_id":id,"dice":dice}));
                    assert!(actions.len() < 5000);
                }
                records.push(json!({"players":players,"candidate":candidate,"kind":kind,"parameters":parameters,"actions":actions,
                    "scores":(0..players).map(|p|model.player_score(&state,PlayerId::new(p)).unwrap()).collect::<Vec<_>>(),
                    "outcomes":(0..players).map(|p|match model.player_result(&state,PlayerId::new(p)).unwrap() {
                        tag_rust::core::PlayerResult::Win=>1, tag_rust::core::PlayerResult::Draw=>2,
                        tag_rust::core::PlayerResult::Loss=>3,tag_rust::core::PlayerResult::Ongoing=>0
                    }).collect::<Vec<_>>()}));
            }
        }
    }
    println!("{}", serde_json::to_string(&records).unwrap());
}
