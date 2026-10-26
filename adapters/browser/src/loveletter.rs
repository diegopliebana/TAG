//! Stochastic browser controllers retain both action and simulation RNG streams.
use crate::{
    games::{BrowserGame, BrowserReplay},
    learned::Registry,
    protocol::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tag_rust::{
    agents::{Agent, SampledOslaAgent, SampledOslaSnapshot},
    core::{
        PlayerId,
        checkpoint::Checkpoint,
        random::{AgentRng, AgentRngSnapshot, SimulationRng, SimulationRngSnapshot},
    },
    games::loveletter::*,
    inference::{LocalMctsModel, ModelError},
    learning::{LearningGame, LearningLimits},
    recording::ControllerSnapshot,
    replay::{PersistenceError, ReplayCodec, RngData, SeatData, counter_data},
    search::mcts::{MctsAgent, MctsSnapshot, NativeEvaluators, Settings},
    session::Controller,
};

fn error(e: impl std::fmt::Display) -> ModelError {
    ModelError(e.to_string())
}
fn bridge(e: impl std::fmt::Display) -> BridgeError {
    BridgeError::new("invalid_setup", e.to_string())
}
fn invalid(e: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Invalid(e.to_string())
}

type Game = LoveLetterForwardModel;
const GAME_ID: &str = "loveletter";
crate::stochastic::stochastic_controllers!(
    LoveLetterObservation,
    LoveLetterAction,
    LoveLetterHeuristic
);

pub(crate) fn model_parameters(
    m: &tag_rust::inference::LearnedModel,
) -> Result<(LoveLetterParameters, usize), ModelError> {
    let d = m.as_document();
    let identity = &d["contract"]["identity"];
    let pairs: Vec<(String, String)> =
        serde_json::from_value(identity["configuration"].clone()).map_err(error)?;
    let named: std::collections::BTreeMap<String, String> = pairs.into_iter().collect();
    let targets = named
        .get("winning_tokens")
        .ok_or_else(|| error("missing token targets"))?
        .split(',')
        .map(|v| v.parse::<u32>().map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    let n = identity["player_count"]
        .as_u64()
        .ok_or_else(|| error("missing player count"))? as usize;
    let expected: Vec<_> = (0..n).collect();
    let network = m
        .network()
        .ok_or_else(|| error("Love Letter requires a neural model"))?;
    if network.value_players() != Some(expected.as_slice())
        || network.payloads().contains(&"last_private_history")
    {
        return Err(error(
            "Love Letter requires absolute-seat values and excludes last_private_history",
        ));
    }
    Ok((
        LoveLetterParameters {
            rules: LoveLetterRules {
                winning_tokens: targets
                    .try_into()
                    .map_err(|_| error("expected three token targets"))?,
            },
            ..Default::default()
        },
        n,
    ))
}
impl BrowserGame for Game {
    type Agent = StochasticAgent;
    const KIND: GameKind = GameKind::Loveletter;
    const HIDDEN_INFORMATION: bool = true;
    fn dimensions(_: &LoveLetterParameters) -> (usize, usize) {
        (16, 1)
    }
    fn parse_parameters(value: Option<Value>) -> Result<LoveLetterParameters, BridgeError> {
        crate::seeds::parameters(value)
    }

    fn parameters_view(p: &LoveLetterParameters) -> Result<Value, BridgeError> {
        Ok(json!({"rules":p.rules,"seed":p.seed.to_string(),"first_player":p.first_player}))
    }
    fn validate_seats(
        p: &LoveLetterParameters,
        seats: &[SeatSetup],
        models: &Registry,
    ) -> Result<(), BridgeError> {
        if !(2..=4).contains(&seats.len()) {
            return Err(bridge("Love Letter requires 2?4 players"));
        }
        for seat in seats {
            if let SeatSetup::NeuralMcts { model_id, .. } = seat {
                let m = models
                    .get(model_id)
                    .ok_or_else(|| bridge("missing model"))?;
                let (_, n) = model_parameters(m).map_err(bridge)?;
                if n != seats.len() {
                    return Err(bridge("model player count differs from game seats"));
                }
                m.validate_spec(
                    &LoveLetterForwardModel
                        .learning_spec_for_players(p, n, LearningLimits::default())
                        .map_err(bridge)?,
                )
                .map_err(bridge)?;
            }
        }
        Ok(())
    }
    fn controller(s: SeatSetup) -> Result<Controller<StochasticAgent>, BridgeError> {
        controller(s, &Registry::new())
    }
    fn controller_with_models(
        _: &Self,
        p: &LoveLetterParameters,
        s: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<StochasticAgent>, BridgeError> {
        if let SeatSetup::NeuralMcts { model_id, .. } = &s {
            let m = models
                .get(model_id)
                .ok_or_else(|| bridge("missing model"))?;
            let (_, n) = model_parameters(m).map_err(bridge)?;
            m.validate_spec(
                &LoveLetterForwardModel
                    .learning_spec_for_players(p, n, LearningLimits::default())
                    .map_err(bridge)?,
            )
            .map_err(bridge)?;
        }
        controller(s, models)
    }
    fn settings(a: &StochasticAgent) -> SeatSetup {
        serde_json::from_value(a.saved.setup.clone()).expect("validated setup")
    }
    fn display_settings(setup: SeatSetup) -> SeatSetup {
        let mut v = serde_json::to_value(setup).expect("seat settings");
        if v.get("seed").is_some() {
            v["seed"] = json!("auto");
        }
        serde_json::from_value(v).expect("redacted settings")
    }
    fn player_values(
        o: &LoveLetterObservation,
        p: PlayerId,
    ) -> Result<(tag_rust::core::PlayerResult, f64), LoveLetterError> {
        Ok((o.public().player_result(p)?, o.public().player_score(p)?))
    }
    fn board(o: &LoveLetterObservation) -> BoardView {
        let p = o.public();
        let outcome = match p.outcome.winners() {
            [] => BoardOutcome::Ongoing,
            [one] => BoardOutcome::Won(one.index()),
            many => BoardOutcome::SharedWin(many.iter().map(|p| p.index()).collect()),
        };
        BoardView {
            cantstop: None,
            loveletter: Some(json!({
                "rules":p.rules,"round":p.round,"tokens":p.tokens,"active":p.active,"protected":p.protected,
                "reserve":p.reserve,"discards":p.discards,"draw_count":p.draw_count,"removed_count":p.removed_count,
                "hand":o.hand(),"public_events":p.events,"private_events":o.private_events(),
                "last_round":p.last_round,"last_private_events":o.last_round_private_events(),
                "observer":o.observer(),"public_only":false,
            })),
            width: 16,
            height: 1,
            cells: vec![],
            current_player: p.current_player.index(),
            counters: counter_data(p.counters),
            outcome,
            winning_cells: vec![],
        }
    }
    fn display_board(
        s: &LoveLetterState,
        o: &LoveLetterObservation,
        reveal: bool,
        public_only: bool,
    ) -> BoardView {
        let mut b = Self::board(o);
        let v = b.loveletter.as_mut().unwrap();
        if public_only {
            v["hand"] = json!([]);
            v["private_events"] = json!([]);
            v["last_private_events"] = json!([]);
            v["public_only"] = json!(true);
        }
        if reveal {
            v["privileged"] = json!({"hands":(0..s.public().player_count()).map(|p|s.privileged_hand(PlayerId::new(p)).unwrap()).collect::<Vec<_>>(),
                "removed":s.privileged_removed_card(),"draw_pile":s.privileged_draw_pile()});
        }
        b
    }
}
impl BrowserReplay for Game {
    fn codec(models: &Registry) -> impl ReplayCodec<StochasticAgent> {
        Codec(models)
    }
    fn validate_models(
        &self,
        p: &LoveLetterParameters,
        models: &Registry,
        text: &str,
    ) -> Result<(), BridgeError> {
        let file: Value = serde_json::from_str(text).map_err(bridge)?;
        let root = &file["initial_state"];
        let players = root["snapshot"]["public"]["tokens"]
            .as_array()
            .map(Vec::len)
            .or_else(|| {
                root["players"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
            })
            .filter(|n| (2..=4).contains(n))
            .ok_or_else(|| bridge("missing or invalid Love Letter root players"))?;
        for id in crate::learned::references(text)? {
            let m = models
                .get(&id)
                .ok_or_else(|| bridge("missing referenced model"))?;
            let (_, n) = model_parameters(m).map_err(bridge)?;
            if n != players {
                return Err(bridge("replay model player count differs from root"));
            }
            m.validate_spec(
                &self
                    .learning_spec_for_players(p, n, LearningLimits::default())
                    .map_err(bridge)?,
            )
            .map_err(bridge)?;
        }
        Ok(())
    }
    fn prediction(
        &self,
        s: &LoveLetterState,
        p: &LoveLetterParameters,
        m: &tag_rust::inference::LearnedModel,
        observer: usize,
    ) -> Result<Value, BridgeError> {
        crate::learned::prediction(self, s, p, m, observer)
    }
}
