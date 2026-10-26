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
    games::cantstop::*,
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
type Game = CantStopForwardModel;

pub(crate) fn model_parameters(
    m: &tag_rust::inference::LearnedModel,
) -> Result<(CantStopParameters, usize), ModelError> {
    let d = m.as_document();
    let identity = &d["contract"]["identity"];
    let pairs: Vec<(String, String)> =
        serde_json::from_value(identity["configuration"].clone()).map_err(error)?;
    let named: std::collections::BTreeMap<String, String> = pairs.into_iter().collect();
    let heights: Vec<u32> = named
        .get("column_heights")
        .ok_or_else(|| error("missing column heights"))?
        .split(',')
        .map(|s| s.parse::<u32>().map_err(error))
        .collect::<Result<_, _>>()?;
    let rules = CantStopRules {
        column_heights: heights
            .try_into()
            .map_err(|_| error("expected eleven heights"))?,
        markers: named
            .get("markers")
            .ok_or_else(|| error("missing marker count"))?
            .parse()
            .map_err(error)?,
        columns_to_win: named
            .get("columns_to_win")
            .ok_or_else(|| error("missing winning target"))?
            .parse()
            .map_err(error)?,
    };
    let n = identity["player_count"]
        .as_u64()
        .ok_or_else(|| error("missing player count"))? as usize;
    let expected: Vec<_> = (0..n).collect();
    if m.network().and_then(|network| network.value_players()) != Some(expected.as_slice()) {
        return Err(error(
            "Can't Stop requires a complete value vector in absolute player order",
        ));
    }
    Ok((CantStopParameters { rules, seed: 0 }, n))
}

const GAME_ID: &str = "cantstop";
crate::stochastic::stochastic_controllers!(CantStopObservation, CantStopAction, CantStopProgress);

impl BrowserGame for Game {
    type Agent = StochasticAgent;
    const KIND: GameKind = GameKind::Cantstop;
    fn validate_seats(
        _: &CantStopParameters,
        seats: &[SeatSetup],
        models: &Registry,
    ) -> Result<(), BridgeError> {
        for seat in seats {
            if let SeatSetup::NeuralMcts { model_id, .. } = seat {
                let m = models
                    .get(model_id)
                    .ok_or_else(|| bridge("missing model"))?;
                let (_, n) = model_parameters(m).map_err(bridge)?;
                if n != seats.len() {
                    return Err(bridge("model player count differs from game seats"));
                }
            }
        }
        Ok(())
    }
    fn dimensions(p: &CantStopParameters) -> (usize, usize) {
        (11, *p.rules.column_heights.iter().max().unwrap() as usize)
    }
    fn parse_parameters(value: Option<Value>) -> Result<CantStopParameters, BridgeError> {
        crate::seeds::parameters(value)
    }

    fn parameters_view(p: &CantStopParameters) -> Result<Value, BridgeError> {
        Ok(json!({"rules":p.rules,"seed":p.seed.to_string()}))
    }
    fn board(o: &CantStopObservation) -> BoardView {
        let p = o.position();
        let permanent: Vec<Vec<u32>> = (0..p.player_count())
            .map(|seat| {
                (2..=12)
                    .map(|c| p.permanent_position(PlayerId::new(seat), c).unwrap())
                    .collect()
            })
            .collect();
        let temporary: Vec<_> = (2..=12).map(|c| p.temporary_position(c).unwrap()).collect();
        let owners: Vec<_> = (2..=12)
            .map(|c| p.column_owner(c).unwrap().map(|s| s.index()))
            .collect();
        BoardView {
            width: 11,
            height: *p.rules().column_heights.iter().max().unwrap() as usize,
            cells: vec![],
            current_player: p.current_player().index(),
            counters: counter_data(p.counters()),
            outcome: p.status().into(),
            winning_cells: vec![],
            loveletter: None,
            cantstop: Some(
                json!({"rules":p.rules(),"phase":p.phase(),"dice":p.dice(),"permanent":permanent,"temporary":temporary,"owners":owners}),
            ),
        }
    }
    fn controller(s: SeatSetup) -> Result<Controller<StochasticAgent>, BridgeError> {
        controller(s, &Registry::new())
    }
    fn controller_with_models(
        _: &Self,
        p: &CantStopParameters,
        s: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<StochasticAgent>, BridgeError> {
        if let SeatSetup::NeuralMcts { model_id, .. } = &s {
            let m = models
                .get(model_id)
                .ok_or_else(|| bridge("missing model"))?;
            let n = m.as_document()["contract"]["identity"]["player_count"]
                .as_u64()
                .ok_or_else(|| bridge("missing player count"))? as usize;
            m.validate_spec(
                &CantStopForwardModel
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
    fn player_values(
        o: &CantStopObservation,
        p: PlayerId,
    ) -> Result<(tag_rust::core::PlayerResult, f64), CantStopError> {
        Ok((
            o.position().player_result(p)?,
            o.position().player_score(p)?,
        ))
    }
}

impl BrowserReplay for Game {
    fn codec(models: &Registry) -> impl ReplayCodec<StochasticAgent> {
        Codec(models)
    }
    fn validate_models(
        &self,
        p: &CantStopParameters,
        models: &Registry,
        text: &str,
    ) -> Result<(), BridgeError> {
        let file: Value = serde_json::from_str(text).map_err(bridge)?;
        let players = file["initial_state"]["permanent"]
            .as_array()
            .map(Vec::len)
            .ok_or_else(|| bridge("missing Can't Stop root"))?;
        for id in crate::learned::references(text)? {
            let m = models
                .get(&id)
                .ok_or_else(|| bridge("missing referenced model"))?;
            let n = m.as_document()["contract"]["identity"]["player_count"]
                .as_u64()
                .unwrap_or(0) as usize;
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
        s: &CantStopState,
        p: &CantStopParameters,
        m: &tag_rust::inference::LearnedModel,
        observer: usize,
    ) -> Result<Value, BridgeError> {
        crate::learned::prediction(self, s, p, m, observer)
    }
}
