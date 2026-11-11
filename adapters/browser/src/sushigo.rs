//! Browser-facing Sushi Go! observations and observation-derived controllers.
use crate::{games::BrowserGame, learned::Registry, protocol::*};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tag_rust::{
    agents::{Agent, SampledOslaAgent, SampledOslaSnapshot},
    core::{
        PlayerId, PlayerResult,
        checkpoint::Checkpoint,
        random::{AgentRng, AgentRngSnapshot, SimulationRng, SimulationRngSnapshot},
    },
    games::sushigo::*,
    inference::{LocalMctsModel, ModelError},
    learning::{LearningGame, LearningLimits},
    recording::ControllerSnapshot,
    replay::{PersistenceError, ReplayCodec, RngData, SeatData},
    search::mcts::{MctsAgent, MctsSnapshot, NativeEvaluators, Settings},
    session::Controller,
};

type Game = SushiGoForwardModel;
const GAME_ID: &str = "sushigo";

fn error(e: impl std::fmt::Display) -> ModelError {
    ModelError(e.to_string())
}
fn bridge(e: impl std::fmt::Display) -> BridgeError {
    BridgeError::new("invalid_setup", e.to_string())
}
fn invalid(e: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Invalid(e.to_string())
}

crate::stochastic::stochastic_controllers!(Observation, Action, SushiGoHeuristic);

pub(crate) fn codec(models: &Registry) -> Codec<'_> {
    Codec(models)
}

pub(crate) fn model_parameters(
    m: &tag_rust::inference::LearnedModel,
) -> Result<(Parameters, usize), ModelError> {
    let document = m.as_document();
    let identity = &document["contract"]["identity"];
    if identity["game"] != GAME_ID {
        return Err(error("model is not Sushi Go!"));
    }
    let count = identity["player_count"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| (2..=5).contains(n))
        .ok_or_else(|| error("model player count must be 2–5"))?;
    let values: Vec<usize> = (0..count).collect();
    let network = m
        .network()
        .ok_or_else(|| error("Sushi Go! requires a neural model"))?;
    if network.value_players() != Some(values.as_slice()) {
        return Err(error("Sushi Go! requires full absolute-seat values"));
    }
    Ok((Parameters::default(), count))
}

fn result(o: &Observation, seat: PlayerId) -> Result<PlayerResult, Error> {
    let p = seat.index();
    if p >= o.public.scores.len() {
        return Err(Error::InvalidPlayer(seat));
    }
    if !o.public.finished {
        return Ok(PlayerResult::Ongoing);
    }
    let best = *o.public.scores.iter().max().expect("players");
    let pudding = o
        .public
        .scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score == best)
        .map(|(i, _)| o.public.pudding[i])
        .max()
        .expect("winner");
    Ok(
        if o.public.scores[p] == best && o.public.pudding[p] == pudding {
            PlayerResult::Win
        } else {
            PlayerResult::Loss
        },
    )
}

impl BrowserGame for Game {
    type Agent = StochasticAgent;
    const KIND: GameKind = GameKind::Sushigo;
    const HIDDEN_INFORMATION: bool = true;
    fn dimensions(_: &Parameters) -> (usize, usize) {
        (12, 5)
    }
    fn parse_parameters(value: Option<Value>) -> Result<Parameters, BridgeError> {
        crate::seeds::parameters(value)
    }
    fn parameters_view(p: &Parameters) -> Result<Value, BridgeError> {
        Ok(json!({"seed":p.seed.to_string()}))
    }
    fn validate_seats(
        p: &Parameters,
        seats: &[SeatSetup],
        models: &Registry,
    ) -> Result<(), BridgeError> {
        if !(2..=5).contains(&seats.len()) {
            return Err(bridge("Sushi Go! requires 2–5 players"));
        }
        for setup in seats {
            if let SeatSetup::NeuralMcts { model_id, .. } = setup {
                let model = models
                    .get(model_id)
                    .ok_or_else(|| bridge("missing model"))?;
                let (_, count) = model_parameters(model).map_err(bridge)?;
                if count != seats.len() {
                    return Err(bridge("model player count differs from game seats"));
                }
                model
                    .validate_spec(
                        &SushiGoForwardModel
                            .learning_spec_for_players(p, count, LearningLimits::default())
                            .map_err(bridge)?,
                    )
                    .map_err(bridge)?;
            }
        }
        Ok(())
    }
    fn controller(setup: SeatSetup) -> Result<Controller<StochasticAgent>, BridgeError> {
        controller(setup, &Registry::new())
    }
    fn controller_with_models(
        _: &Self,
        p: &Parameters,
        setup: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<StochasticAgent>, BridgeError> {
        if let SeatSetup::NeuralMcts { model_id, .. } = &setup {
            let model = models
                .get(model_id)
                .ok_or_else(|| bridge("missing model"))?;
            let (_, count) = model_parameters(model).map_err(bridge)?;
            model
                .validate_spec(
                    &SushiGoForwardModel
                        .learning_spec_for_players(p, count, LearningLimits::default())
                        .map_err(bridge)?,
                )
                .map_err(bridge)?;
        }
        controller(setup, models)
    }
    fn settings(agent: &StochasticAgent) -> SeatSetup {
        serde_json::from_value(agent.saved.setup.clone()).expect("validated setup")
    }
    fn display_settings(setup: SeatSetup) -> SeatSetup {
        let mut value = serde_json::to_value(setup).expect("seat settings");
        if value.get("seed").is_some() {
            value["seed"] = json!("auto");
        }
        serde_json::from_value(value).expect("redacted settings")
    }
    fn player_values(o: &Observation, p: PlayerId) -> Result<(PlayerResult, f64), Error> {
        Ok((
            result(o, p)?,
            f64::from(
                *o.public
                    .scores
                    .get(p.index())
                    .ok_or(Error::InvalidPlayer(p))?,
            ),
        ))
    }
    fn board(o: &Observation) -> BoardView {
        let winners: Vec<_> = (0..o.public.scores.len())
            .filter(|&p| result(o, PlayerId::new(p)) == Ok(PlayerResult::Win))
            .collect();
        let outcome = match winners.as_slice() {
            [] => BoardOutcome::Ongoing,
            [one] => BoardOutcome::Won(*one),
            many => BoardOutcome::SharedWin(many.to_vec()),
        };
        BoardView {
            cantstop: None,
            loveletter: None,
            sushigo: Some(json!({
                "round":o.public.round,"scores":o.public.scores,"round_scores":o.public.round_scores,
                "pudding":o.public.pudding,"played":o.public.played.iter().map(|cards| cards.iter().map(|c| json!({"card":c.card,"on_wasabi":c.on_wasabi})).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "discard":o.public.discard,"draw_count":o.public.draw_count,"hand_size":o.public.hand_size,
                "hand_ids":o.public.hand_ids,"last_reveal":o.public.last_reveal,
                "hand":o.hand,"observer":o.observer.index(),"finished":o.public.finished,
                "public_only":false
            })),
            width: 12,
            height: 5,
            cells: vec![],
            current_player: o.observer.index(),
            counters: [
                o.public.round,
                o.public.hand_size,
                o.public.round_scores.len(),
                usize::from(o.public.finished),
            ],
            outcome,
            winning_cells: vec![],
        }
    }
    fn display_board(state: &State, o: &Observation, reveal: bool, public_only: bool) -> BoardView {
        let mut board = Self::board(o);
        let data = board.sushigo.as_mut().expect("Sushi Go! board");
        if reveal {
            data["opponent_hands"] = json!(
                (0..state.public.scores.len())
                    .map(|seat| {
                        (seat != o.observer.index()).then(|| {
                            state
                                .privileged_hand(PlayerId::new(seat))
                                .expect("valid seat")
                        })
                    })
                    .collect::<Vec<_>>()
            );
        }
        if public_only {
            data["hand"] = json!([]);
            data["public_only"] = json!(true);
        }
        board
    }
}
