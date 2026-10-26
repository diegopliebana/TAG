//! Convert game observations into browser display values.
//!
//! BrowserGame describes how to create controllers and display the information a
//! player is allowed to see. Rules still live in the engine, not this conversion layer.
use crate::learned::{BrowserAgent, Codec, Registry};
use crate::protocol::{BoardView, BridgeError, GameKind, SeatSetup};
use crate::runtime::integer;
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::{Debug, Display};
use tag_rust::agents::Agent;
use tag_rust::core::PublicInformationModel;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{LearningGame, LearningLimits};
use tag_rust::replay::counter_data;
use tag_rust::replay::{ReplayCodec, ReplayGame};
use tag_rust::session::{BuiltinAgent, Controller};

/// Browser presentation and controller support supplied independently of file persistence.
pub(crate) trait BrowserGame:
    Clone
    + 'static
    + ForwardModel<
        State: Clone,
        Observation: 'static,
        Action: Clone + PartialEq + Serialize + DeserializeOwned,
        Parameters: Default + DeserializeOwned,
        Error: Debug + Display + 'static,
    >
{
    type Agent: Agent<Self, Error: Debug + Display + 'static>
        + Checkpoint<Snapshot: PartialEq + 'static, RestoreError: Debug + Display + 'static>
        + 'static;
    const KIND: GameKind;
    const HIDDEN_INFORMATION: bool = false;
    fn display_board(
        state: &Self::State,
        observation: &Self::Observation,
        reveal: bool,
        public_only: bool,
    ) -> BoardView {
        let _ = (state, reveal, public_only);
        Self::board(observation)
    }
    fn display_settings(setup: SeatSetup) -> SeatSetup {
        setup
    }

    fn validate_seats(
        _: &Self::Parameters,
        _: &[SeatSetup],
        _: &Registry,
    ) -> Result<(), BridgeError> {
        Ok(())
    }
    fn parse_parameters(value: Option<serde_json::Value>) -> Result<Self::Parameters, BridgeError> {
        value
            .map(serde_json::from_value)
            .transpose()
            .map(|p| p.unwrap_or_default())
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))
    }
    fn parameters_view(parameters: &Self::Parameters) -> Result<serde_json::Value, BridgeError>
    where
        Self::Parameters: Serialize,
    {
        serde_json::to_value(parameters)
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))
    }
    /// Read the configured board dimensions for browser resource limits.
    ///
    /// # Arguments
    /// `parameters` is a shared configuration borrow.
    ///
    /// # Returns
    /// Copied (width, height) in cells.
    fn dimensions(parameters: &Self::Parameters) -> (usize, usize);
    /// Convert player-visible game facts into browser display data.
    ///
    /// # Arguments
    /// `observation` is a shared permitted-view borrow.
    ///
    /// # Returns
    /// Owned cell, outcome and progress values for the frontend.
    fn board(observation: &Self::Observation) -> BoardView;
    /// Create a controller from browser setup.
    ///
    /// # Arguments
    /// `setup` is consumed and selects a human or a supported seeded policy.
    ///
    /// # Returns
    /// An owned controller whose progress belongs to this seat.
    ///
    /// # Errors
    /// Invalid seed or unsupported controller settings.
    fn controller(setup: SeatSetup) -> Result<Controller<Self::Agent>, BridgeError>;
    /// Validate model compatibility and construct a seat from immutable loaded models.
    /// Borrowed rules/configuration/registry remain unchanged; setup is consumed.
    fn controller_with_models(
        model: &Self,
        parameters: &Self::Parameters,
        setup: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<Self::Agent>, BridgeError> {
        let _ = (model, parameters, models);
        Self::controller(setup)
    }
    /// Borrow an agent and return owned label/seed settings for display.
    fn settings(agent: &Self::Agent) -> SeatSetup;
    /// Query permitted results and scores from this observer's view.
    ///
    /// # Arguments
    /// `observation` is passed by a shared (read-only) reference; `player` is a copied score perspective.
    ///
    /// # Returns
    /// Copied player result and game-defined score.
    ///
    /// # Errors
    /// Invalid player or unavailable permitted values.
    fn player_values(
        observation: &Self::Observation,
        player: PlayerId,
    ) -> Result<(PlayerResult, f64), Self::Error>;
}
/// Build a supported built-in controller from owned setup settings.
///
/// # Arguments
/// `setup` is consumed; `E` selects the default OSLA evaluator.
///
/// # Returns
/// An owned human or independently seeded AI controller.
///
/// # Errors
/// An invalid decimal seed returns the setup error.
fn builtin_controller<E: Default>(
    setup: SeatSetup,
) -> Result<Controller<BuiltinAgent<E>>, BridgeError> {
    Ok(match setup {
        SeatSetup::SampledOsla { .. }
        | SeatSetup::QLearning { .. }
        | SeatSetup::NeuralOsla { .. }
        | SeatSetup::Mcts { .. }
        | SeatSetup::NeuralMcts { .. } => {
            return Err(BridgeError::new("invalid_setup", "select a loaded model"));
        }
        SeatSetup::Human => Controller::Human,
        SeatSetup::Random { seed } => Controller::random(integer(&seed)?),
        SeatSetup::FirstAvailable { seed } => Controller::first_available(integer(&seed)?),
        SeatSetup::Osla { seed } => Controller::osla(E::default(), integer(&seed)?),
    })
}

/// Read a built-in policy's label and initial seed from its shared RNG.
///
/// # Arguments
/// `agent` is passed by a shared (read-only) reference; `F` identifies the agent's rule contract.
///
/// # Returns
/// Owned setup data containing the exact decimal seed.
fn builtin_settings<F, E>(agent: &BuiltinAgent<E>) -> SeatSetup
where
    F: PublicInformationModel,
    F::State: Clone,
    E: Evaluator<F::State>,
{
    let seed = <BuiltinAgent<E> as Agent<F>>::rng(agent)
        .initial_seed()
        .to_string();
    match agent {
        BuiltinAgent::FirstAvailable(_) => SeatSetup::FirstAvailable { seed },
        BuiltinAgent::Random(_) => SeatSetup::Random { seed },
        BuiltinAgent::Osla(_) => SeatSetup::Osla { seed },
    }
}

impl BrowserGame for ttt::TicTacToeForwardModel {
    fn controller_with_models(
        model: &Self,
        parameters: &Self::Parameters,
        setup: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<Self::Agent>, BridgeError> {
        learned_controller(model, parameters, setup, models)
    }

    type Agent = BrowserAgent<ttt::TicTacToeEvaluator>;
    fn controller(setup: SeatSetup) -> Result<Controller<Self::Agent>, BridgeError> {
        wrap_controller::<Self, _>(setup)
    }
    fn settings(agent: &Self::Agent) -> SeatSetup {
        browser_settings::<Self, _>(agent)
    }
    fn player_values(
        o: &Self::Observation,
        player: PlayerId,
    ) -> Result<(PlayerResult, f64), Self::Error> {
        Ok((o.player_result(player)?, o.player_score(player)?))
    }

    const KIND: GameKind = GameKind::Tictactoe;
    fn dimensions(p: &Self::Parameters) -> (usize, usize) {
        (p.grid_size, p.grid_size)
    }
    fn board(o: &Self::Observation) -> BoardView {
        BoardView {
            cantstop: None,
            loveletter: None,
            width: o.grid_size(),
            height: o.grid_size(),
            cells: o
                .board()
                .iter()
                .map(|c| match c {
                    ttt::Cell::Empty => None,
                    ttt::Cell::X => Some(0),
                    ttt::Cell::O => Some(1),
                })
                .collect(),
            current_player: o.current_player().index(),
            counters: counter_data(o.counters()),
            outcome: o.status().into(),
            winning_cells: o.winning_cells(),
        }
    }
}
impl BrowserGame for c4::Connect4ForwardModel {
    fn controller_with_models(
        model: &Self,
        parameters: &Self::Parameters,
        setup: SeatSetup,
        models: &Registry,
    ) -> Result<Controller<Self::Agent>, BridgeError> {
        learned_controller(model, parameters, setup, models)
    }

    type Agent = BrowserAgent<c4::Connect4Evaluator>;
    fn controller(setup: SeatSetup) -> Result<Controller<Self::Agent>, BridgeError> {
        wrap_controller::<Self, _>(setup)
    }
    fn settings(agent: &Self::Agent) -> SeatSetup {
        browser_settings::<Self, _>(agent)
    }
    fn player_values(
        o: &Self::Observation,
        player: PlayerId,
    ) -> Result<(PlayerResult, f64), Self::Error> {
        Ok((o.player_result(player)?, o.player_score(player)?))
    }

    const KIND: GameKind = GameKind::Connect4;
    fn dimensions(p: &Self::Parameters) -> (usize, usize) {
        (p.width, p.height)
    }
    fn board(o: &Self::Observation) -> BoardView {
        BoardView {
            cantstop: None,
            loveletter: None,
            width: o.width(),
            height: o.height(),
            cells: o
                .board()
                .iter()
                .map(|c| match c {
                    c4::Cell::Empty => None,
                    c4::Cell::Piece(p) => Some(p.index()),
                })
                .collect(),
            current_player: o.current_player().index(),
            counters: counter_data(o.counters()),
            outcome: o.status().into(),
            winning_cells: o.winning_cells().to_vec(),
        }
    }
}

fn wrap_controller<F, E: Default>(
    setup: SeatSetup,
) -> Result<Controller<BrowserAgent<E>>, BridgeError>
where
    F: PublicInformationModel + LearningGame + tag_rust::search::mcts::NativeEvaluators + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: Display,
{
    if let SeatSetup::Mcts {
        seed,
        configuration,
    } = &setup
    {
        let mut settings = tag_rust::search::mcts::Settings::from_json(&configuration.to_string())
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
        settings.seed = Some(integer(seed)?);
        crate::mcts::agent::<F>(
            configuration,
            &tag_rust::core::random::AgentRng::new(integer(seed)?),
            None,
            PlayerId::new(0),
        )
        .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
        return Ok(Controller::Ai(BrowserAgent::Mcts {
            model: None,
            configuration: Box::new(settings),
            rng: tag_rust::core::random::AgentRng::new(integer(seed)?),
        }));
    }
    Ok(match builtin_controller(setup)? {
        Controller::Human => Controller::Human,
        Controller::Ai(a) => Controller::Ai(BrowserAgent::Builtin(a)),
    })
}
fn learned_controller<F, E: Default>(
    model: &F,
    parameters: &F::Parameters,
    setup: SeatSetup,
    models: &Registry,
) -> Result<Controller<BrowserAgent<E>>, BridgeError>
where
    F: LearningGame + PublicInformationModel + tag_rust::search::mcts::NativeEvaluators + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: Display,
{
    if let SeatSetup::NeuralMcts {
        seed,
        configuration,
        model_id,
        bindings,
    } = &setup
    {
        let loaded = models.get(model_id).ok_or_else(|| {
            BridgeError::new("invalid_setup", "load the selected neural MCTS model first")
        })?;
        loaded
            .validate_spec(
                &model
                    .learning_spec(parameters, LearningLimits::default())
                    .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?,
            )
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
        let binding = tag_rust::inference::LocalMctsModel::new(loaded.clone(), bindings.clone())
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
        let rng = tag_rust::core::random::AgentRng::new(integer(seed)?);
        let agent = crate::mcts::agent::<F>(configuration, &rng, Some(&binding), PlayerId::new(0))
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
        let configuration = agent.settings().clone();
        return Ok(Controller::Ai(BrowserAgent::Mcts {
            configuration: Box::new(configuration),
            rng,
            model: Some(binding),
        }));
    }
    let (id, seed, neural) = match &setup {
        SeatSetup::QLearning { model_id, seed } => (model_id, seed, false),
        SeatSetup::NeuralOsla { model_id, seed } => (model_id, seed, true),
        _ => return wrap_controller::<F, _>(setup),
    };
    let loaded = models
        .get(id)
        .ok_or_else(|| BridgeError::new("invalid_setup", "load the selected model first"))?
        .clone();
    if loaded.is_neural() != neural {
        return Err(BridgeError::new(
            "invalid_setup",
            "model type differs from controller",
        ));
    }
    if neural {
        loaded
            .network()
            .unwrap()
            .validate_osla()
            .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
    }
    loaded
        .validate_spec(
            &model
                .learning_spec(parameters, LearningLimits::default())
                .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?,
        )
        .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?;
    Ok(Controller::Ai(BrowserAgent::Learned {
        model: loaded,
        rng: tag_rust::core::random::AgentRng::new(integer(seed)?),
    }))
}
fn browser_settings<F, E>(agent: &BrowserAgent<E>) -> SeatSetup
where
    F: PublicInformationModel,
    F::State: Clone,
    E: Evaluator<F::State>,
{
    match agent {
        BrowserAgent::Mcts {
            configuration,
            rng,
            model,
        } => {
            // The seat's decimal seed string is authoritative. A JSON number in
            // the displayed search settings could lose u64 precision in JavaScript.
            let mut configuration = crate::mcts::configuration_value(configuration);
            configuration["seed"] = serde_json::Value::Null;
            if let Some(model) = model {
                SeatSetup::NeuralMcts {
                    seed: rng.initial_seed().to_string(),
                    configuration,
                    model_id: model.model().id().into(),
                    bindings: model.bindings().clone(),
                }
            } else {
                SeatSetup::Mcts {
                    seed: rng.initial_seed().to_string(),
                    configuration,
                }
            }
        }
        BrowserAgent::Builtin(a) => builtin_settings::<F, E>(a),
        BrowserAgent::Learned { model, rng } => {
            if model.is_neural() {
                SeatSetup::NeuralOsla {
                    seed: rng.initial_seed().to_string(),
                    model_id: model.id().into(),
                }
            } else {
                SeatSetup::QLearning {
                    seed: rng.initial_seed().to_string(),
                    model_id: model.id().into(),
                }
            }
        }
    }
}
pub(crate) trait BrowserReplay: BrowserGame + ReplayGame {
    fn codec(models: &Registry) -> impl ReplayCodec<Self::Agent>;
    /// Validate referenced learned controllers only when this game supports them.
    fn validate_models(
        &self,
        _parameters: &Self::Parameters,
        _models: &Registry,
        _text: &str,
    ) -> Result<(), BridgeError> {
        Ok(())
    }
    /// Query permitted numeric observations when a game supplies an encoder.
    fn prediction(
        &self,
        _state: &Self::State,
        _parameters: &Self::Parameters,
        _loaded: &tag_rust::inference::LearnedModel,
        _observer: usize,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::new(
            "unsupported",
            "this game does not supply learned-model predictions",
        ))
    }
}
impl BrowserReplay for ttt::TicTacToeForwardModel {
    fn codec(models: &Registry) -> impl ReplayCodec<Self::Agent> {
        Codec(models)
    }
    fn validate_models(
        &self,
        parameters: &Self::Parameters,
        models: &Registry,
        text: &str,
    ) -> Result<(), BridgeError> {
        crate::learned::validate_models(self, parameters, models, text)
    }
    fn prediction(
        &self,
        state: &Self::State,
        parameters: &Self::Parameters,
        loaded: &tag_rust::inference::LearnedModel,
        observer: usize,
    ) -> Result<serde_json::Value, BridgeError> {
        crate::learned::prediction(self, state, parameters, loaded, observer)
    }
}
impl BrowserReplay for c4::Connect4ForwardModel {
    fn codec(models: &Registry) -> impl ReplayCodec<Self::Agent> {
        Codec(models)
    }
    fn validate_models(
        &self,
        parameters: &Self::Parameters,
        models: &Registry,
        text: &str,
    ) -> Result<(), BridgeError> {
        crate::learned::validate_models(self, parameters, models, text)
    }
    fn prediction(
        &self,
        state: &Self::State,
        parameters: &Self::Parameters,
        loaded: &tag_rust::inference::LearnedModel,
        observer: usize,
    ) -> Result<serde_json::Value, BridgeError> {
        crate::learned::prediction(self, state, parameters, loaded, observer)
    }
}
