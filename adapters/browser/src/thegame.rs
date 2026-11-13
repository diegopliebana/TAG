//! Browser presentation and observation-derived MCTS for The Game.
use crate::games::{BrowserGame, BrowserReplay};
use crate::learned::Registry;
use crate::protocol::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tag_rust::agents::{Agent, SampledOslaAgent, SampledOslaSnapshot};
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::random::{AgentRng, AgentRngSnapshot, SimulationRng, SimulationRngSnapshot};
use tag_rust::core::{ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::thegame::{
    Action, CommunicationMode, Observation, Parameters, State, TeamOutcome, TheGameForwardModel,
    TheGameHeuristic,
};
use tag_rust::inference::{LocalMctsModel, ModelError};
use tag_rust::learning::{LearningGame, LearningLimits};
use tag_rust::recording::ControllerSnapshot;
use tag_rust::replay::{PersistenceError, ReplayCodec, RngData, SeatData};
use tag_rust::search::mcts::{MctsAgent, MctsSnapshot, NativeEvaluators, Settings};
use tag_rust::session::Controller;

type Game = TheGameForwardModel;
const GAME_ID: &str = "thegame";

fn error(value: impl std::fmt::Display) -> ModelError {
    ModelError(value.to_string())
}
fn bridge(value: impl std::fmt::Display) -> BridgeError {
    BridgeError::new("invalid_setup", value.to_string())
}

fn invalid(message: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Invalid(message.to_string())
}

// The shared sampled-game controller owns both action and simulation streams.
// Its codec retains the earlier Random seat representation for old files.
crate::stochastic::stochastic_controllers!(Observation, Action, TheGameHeuristic);

pub(crate) fn model_parameters(
    model: &tag_rust::inference::LearnedModel,
) -> Result<(Parameters, usize), ModelError> {
    let document = model.as_document();
    let identity = &document["contract"]["identity"];
    if identity["game"] != GAME_ID {
        return Err(error("model is not The Game"));
    }
    let players = identity["player_count"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| (1..=5).contains(n))
        .ok_or_else(|| error("model player count must be 1–5"))?;
    let configuration: Vec<(String, String)> =
        serde_json::from_value(identity["configuration"].clone()).map_err(error)?;
    let communication = match configuration.as_slice() {
        [(key, value)] if key == "communication" && value == "limited_choices" => {
            CommunicationMode::LimitedChoices
        }
        [(key, value)] if key == "communication" && value == "silent" => CommunicationMode::Silent,
        _ => return Err(error("unsupported The Game model communication mode")),
    };
    let expected: Vec<_> = (0..players).collect();
    let network = model
        .network()
        .ok_or_else(|| error("The Game requires a neural model"))?;
    if network.value_players() != Some(expected.as_slice()) {
        return Err(error("The Game requires full absolute-seat values"));
    }
    Ok((
        Parameters {
            communication,
            ..Parameters::default()
        },
        players,
    ))
}

impl BrowserGame for Game {
    type Agent = StochasticAgent;
    const KIND: GameKind = GameKind::Thegame;
    const HIDDEN_INFORMATION: bool = true;
    fn dimensions(_: &Parameters) -> (usize, usize) {
        (4, 1)
    }
    fn parse_parameters(value: Option<Value>) -> Result<Parameters, BridgeError> {
        crate::seeds::parameters(value)
    }
    fn parameters_view(p: &Parameters) -> Result<Value, BridgeError> {
        Ok(
            json!({"seed":p.seed.to_string(),"first_player":p.first_player,"communication":p.communication}),
        )
    }
    fn validate_seats(
        parameters: &Parameters,
        seats: &[SeatSetup],
        models: &Registry,
    ) -> Result<(), BridgeError> {
        if !(1..=5).contains(&seats.len()) {
            return Err(BridgeError::new(
                "invalid_setup",
                "The Game requires 1–5 seats",
            ));
        }
        if seats.iter().any(|s| {
            !matches!(
                s,
                SeatSetup::Human
                    | SeatSetup::Random { .. }
                    | SeatSetup::Mcts { .. }
                    | SeatSetup::NeuralMcts { .. }
            )
        }) {
            return Err(BridgeError::new(
                "invalid_setup",
                "The Game supports Human, Random, MCTS and Neural MCTS controllers",
            ));
        }
        for seat in seats {
            if let SeatSetup::NeuralMcts { model_id, .. } = seat {
                let model = models
                    .get(model_id)
                    .ok_or_else(|| bridge("missing model"))?;
                let (_, players) = model_parameters(model).map_err(bridge)?;
                if players != seats.len() {
                    return Err(bridge("model player count differs from game seats"));
                }
                model
                    .validate_spec(
                        &TheGameForwardModel
                            .learning_spec_for_players(
                                parameters,
                                players,
                                LearningLimits::default(),
                            )
                            .map_err(bridge)?,
                    )
                    .map_err(bridge)?;
            }
        }
        Ok(())
    }
    fn controller(setup: SeatSetup) -> Result<Controller<StochasticAgent>, BridgeError> {
        if !matches!(
            setup,
            SeatSetup::Human
                | SeatSetup::Random { .. }
                | SeatSetup::Mcts { .. }
                | SeatSetup::NeuralMcts { .. }
        ) {
            return Err(bridge(
                "The Game supports Human, Random, MCTS and Neural MCTS controllers",
            ));
        }
        controller(setup, &Registry::new())
    }
    fn controller_with_models(
        _: &Self,
        parameters: &Parameters,
        setup: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<StochasticAgent>, BridgeError> {
        if let SeatSetup::NeuralMcts { model_id, .. } = &setup {
            let model = models
                .get(model_id)
                .ok_or_else(|| bridge("missing model"))?;
            let (_, players) = model_parameters(model).map_err(bridge)?;
            model
                .validate_spec(
                    &TheGameForwardModel
                        .learning_spec_for_players(parameters, players, LearningLimits::default())
                        .map_err(bridge)?,
                )
                .map_err(bridge)?;
        }
        controller(setup, models)
    }
    fn settings(agent: &StochasticAgent) -> SeatSetup {
        serde_json::from_value(agent.saved.setup.clone()).expect("validated setup")
    }
    fn player_values(
        o: &Observation,
        player: PlayerId,
    ) -> Result<(PlayerResult, f64), tag_rust::games::thegame::Error> {
        if player.index() >= o.public.hand_counts.len() {
            return Err(tag_rust::games::thegame::Error::InvalidPlayer(player));
        }
        let result = match o.public.outcome {
            TeamOutcome::Ongoing => PlayerResult::Ongoing,
            TeamOutcome::Won => PlayerResult::Win,
            TeamOutcome::Lost => PlayerResult::Loss,
        };
        Ok((result, o.public.placements.len() as f64))
    }
    fn observation_score(o: &Observation) -> Option<tag_rust::core::OutcomeScore> {
        Some(tag_rust::core::OutcomeScore {
            scope: tag_rust::core::ScoreScope::Team,
            values: vec![o.public.placements.len() as f64],
            unit: "cards_placed",
            maximum: Some(98.0),
        })
    }
    fn board(o: &Observation) -> BoardView {
        let p = &o.public;
        let model = TheGameForwardModel;
        let action_paths = model
            .offered_actions(o)
            .iter()
            .map(|action| model.action_path(o, action))
            .collect::<Vec<_>>();
        BoardView {
            cantstop: None,
            loveletter: None,
            sushigo: None,
            thegame: Some(json!({
                "pile_tops":p.pile_tops,"placements":p.placements,"messages":p.messages,
                "hand_counts":p.hand_counts,"draw_count":p.draw_count,
                "played_this_turn":p.played_this_turn,"turns_completed":p.turns_completed,
                "active_player":p.active_player.index(),"phase":p.phase,
                "hand":o.hand,"observer":o.observer.index(),"public_only":false,
                "action_paths":action_paths,
            })),
            width: 4,
            height: 1,
            cells: Vec::new(),
            current_player: p.active_player.index(),
            counters: [p.placements.len(), p.turns_completed, p.played_this_turn, 0],
            outcome: match p.outcome {
                TeamOutcome::Ongoing => BoardOutcome::Ongoing,
                TeamOutcome::Won => BoardOutcome::SharedWin((0..p.hand_counts.len()).collect()),
                TeamOutcome::Lost => BoardOutcome::TeamLoss,
            },
            winning_cells: Vec::new(),
        }
    }
    fn display_board(state: &State, o: &Observation, reveal: bool, public_only: bool) -> BoardView {
        let mut board = Self::board(o);
        let data = board.thegame.as_mut().expect("The Game board data");
        if public_only {
            data["hand"] = json!([]);
            data["action_paths"] = json!([]);
            data["public_only"] = json!(true);
        }
        if reveal {
            data["opponent_hands"] = json!(
                (0..state.public.hand_counts.len())
                    .map(|seat| state
                        .privileged_hand(PlayerId::new(seat))
                        .expect("valid seat"))
                    .collect::<Vec<_>>()
            );
        }
        board
    }
}

impl BrowserReplay for Game {
    fn codec(models: &Registry) -> impl ReplayCodec<StochasticAgent> {
        Codec(models)
    }
    fn validate_models(
        &self,
        parameters: &Parameters,
        models: &Registry,
        text: &str,
    ) -> Result<(), BridgeError> {
        let file: Value = serde_json::from_str(text).map_err(bridge)?;
        let root = &file["initial_state"];
        let players = root["public"]["hand_counts"]
            .as_array()
            .map(Vec::len)
            .or_else(|| {
                root["players"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
            })
            .filter(|n| (1..=5).contains(n))
            .ok_or_else(|| bridge("missing or invalid The Game root players"))?;
        for id in crate::learned::references(text)? {
            let model = models
                .get(&id)
                .ok_or_else(|| bridge("missing referenced model"))?;
            let (_, count) = model_parameters(model).map_err(bridge)?;
            if count != players {
                return Err(bridge("replay model player count differs from root"));
            }
            model
                .validate_spec(
                    &self
                        .learning_spec_for_players(parameters, players, LearningLimits::default())
                        .map_err(bridge)?,
                )
                .map_err(bridge)?;
        }
        Ok(())
    }
    fn prediction(
        &self,
        state: &State,
        parameters: &Parameters,
        model: &tag_rust::inference::LearnedModel,
        observer: usize,
    ) -> Result<Value, BridgeError> {
        crate::learned::prediction(self, state, parameters, model, observer)
    }
}
