//! Run games using ordinary Rust data. Python conversion lives in the sibling modules.
use std::collections::BTreeMap;
use tag_rust::agents::{
    EvaluatorIdentity, OpenLinesBatchEvaluator, OslaPythonAgent, OslaPythonError,
    OslaPythonSnapshot,
};
use tag_rust::core::{PlayerId, checkpoint::Checkpoint, joint};
use tag_rust::games::{
    cantstop as cs, connect4 as c4, loveletter as ll, sushigo as sg, thegame as tg,
    tictactoe as ttt,
};
use tag_rust::learning::{
    EncodedObservation, LearningError, LearningGame, LearningLimits, LearningSpec,
};
use tag_rust::session::{EvaluationKey, EvaluationRequest, EvaluationSession, EvaluationStatus};

/// Errors the adapter can explain without retaining any Python objects.
#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Learning(LearningError),
    Agent(OslaPythonError),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::Learning(error) => error.fmt(f),
            Self::Agent(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for Error {}
impl From<LearningError> for Error {
    fn from(error: LearningError) -> Self {
        Self::Learning(error)
    }
}

impl From<OslaPythonError> for Error {
    fn from(error: OslaPythonError) -> Self {
        Self::Agent(error)
    }
}

/// Own the settings and state of one game; the model supplies its rule operations.
pub struct Game<F: LearningGame>
where
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    pub(super) session: EvaluationSession<F>,
    pub(super) native_evaluators: Vec<Option<OpenLinesBatchEvaluator>>,
    parameters: F::Parameters,
    pub(super) limits: LearningLimits,
    // Formats depend on fixed parameters, so retain the validated description.
    pub(super) description: LearningSpec,
    pub(super) joint_pending: Vec<(PlayerId, F::Action)>,
    pub(super) joint_resolutions: u64,
    pub(super) joint_submissions: u64,
}
impl<F> Game<F>
where
    F: LearningGame + Clone,
    F::Action: Clone + PartialEq,
    F::Parameters: Clone,
    F::State: Clone,
    F::Error: std::fmt::Display,
{
    fn new(
        model: F,
        parameters: F::Parameters,
        players: usize,
        limits: LearningLimits,
    ) -> Result<Self, Error> {
        let description = model.learning_spec_for_players(&parameters, players, limits)?;
        let state = model.init_state(&parameters, players).map_err(invalid)?;
        Ok(Self {
            session: EvaluationSession::from_state(model, state)?,
            native_evaluators: vec![None; players],
            parameters,
            limits,
            description,
            joint_pending: Vec::new(),
            joint_resolutions: 0,
            joint_submissions: 0,
        })
    }

    pub(super) fn next_player(&self) -> Option<PlayerId> {
        self.session
            .model()
            .decision_players(self.session.state())
            .into_iter()
            .find(|seat| !self.joint_pending.iter().any(|(chosen, _)| chosen == seat))
    }

    pub(super) fn observe(
        &self,
        observer: usize,
        names: &[&str],
    ) -> Result<EncodedObservation, Error> {
        let observation = self
            .session
            .model()
            .observe(self.session.state(), PlayerId::new(observer))
            .map_err(invalid)?;
        let mut encoded = self
            .session
            .model()
            .encode(&observation, names, self.limits)?;
        if self
            .joint_pending
            .iter()
            .any(|(seat, _)| seat.index() == observer)
        {
            encoded.action_mask.fill(false);
            encoded.offered_action_ids.clear();
        }
        Ok(encoded)
    }

    pub(super) fn apply(&mut self, id: usize, actor: Option<usize>) -> Result<(), Error> {
        let next = self
            .next_player()
            .ok_or_else(|| Error::Invalid("the game has finished".into()))?;
        let actor = actor.map(PlayerId::new).unwrap_or(next);
        if self
            .session
            .model()
            .decision_players(self.session.state())
            .len()
            > 1
        {
            if !self
                .session
                .model()
                .decision_players(self.session.state())
                .contains(&actor)
                || self.joint_pending.iter().any(|(seat, _)| *seat == actor)
            {
                return Err(invalid("seat has already chosen or is inactive"));
            }
            let observation = self
                .session
                .model()
                .observe(self.session.state(), actor)
                .map_err(invalid)?;
            let action = self.session.model().action_from_id(&observation, id)?;
            let offered = self
                .session
                .model()
                .available_actions(self.session.state(), actor)
                .map_err(invalid)?;
            if !offered.contains(&action) {
                return Err(invalid("action is not offered"));
            }
            let next_submission = self
                .joint_submissions
                .checked_add(1)
                .ok_or_else(|| invalid("joint submission counter exhausted"))?;
            let mut complete = self.joint_pending.clone();
            complete.push((actor, action));
            if complete.len()
                == self
                    .session
                    .model()
                    .decision_players(self.session.state())
                    .len()
            {
                let next_resolution = self
                    .joint_resolutions
                    .checked_add(1)
                    .ok_or_else(|| invalid("joint decision counter exhausted"))?;
                complete.sort_by_key(|(seat, _)| seat.index());
                let mut next_state = self.session.state().clone();
                joint::apply(self.session.model(), &mut next_state, &complete).map_err(invalid)?;
                self.session.reset_state(next_state)?;
                self.joint_pending.clear();
                self.joint_resolutions = next_resolution;
                self.joint_submissions = next_submission;
            } else {
                self.joint_pending = complete;
                self.joint_submissions = next_submission;
            }
            return Ok(());
        }
        let observation = self
            .session
            .model()
            .observe(self.session.state(), actor)
            .map_err(invalid)?;
        let action = self.session.model().action_from_id(&observation, id)?;
        self.session
            .apply_human(actor, action)
            .map_err(Error::Agent)
    }

    fn reset(&mut self) -> Result<(), Error> {
        let state = self
            .session
            .model()
            .init_state(
                &self.parameters,
                self.session.model().player_count(self.session.state()),
            )
            .map_err(invalid)?;
        self.session.reset_state(state)?;
        self.joint_pending.clear();
        Ok(())
    }

    fn snapshot(&self) -> Result<Self, Error> {
        if !self.joint_pending.is_empty() {
            return Err(invalid(
                "snapshot requires a complete joint decision boundary",
            ));
        }
        Ok(Self {
            session: self.session.snapshot()?,
            native_evaluators: self.native_evaluators.clone(),
            parameters: self.parameters.clone(),
            limits: self.limits,
            description: self.description.clone(),
            joint_pending: Vec::new(),
            joint_resolutions: self.joint_resolutions,
            joint_submissions: self.joint_submissions,
        })
    }
}

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::Invalid(error.to_string())
}

/// Rust registers each concrete game here. All Python-facing operations use this same dispatch.
pub enum Engine {
    TicTacToe(Game<ttt::TicTacToeForwardModel>),
    Connect4(Game<c4::Connect4ForwardModel>),
    // The stochastic session is larger; Box keeps the engine enum compact.
    CantStop(Box<Game<cs::CantStopForwardModel>>),
    LoveLetter(Box<Game<ll::LoveLetterForwardModel>>),
    SushiGo(Box<Game<sg::SushiGoForwardModel>>),
    TheGame(Box<Game<tg::TheGameForwardModel>>),
}

// A game is registered once for construction and name discovery. Its typed
// operation dispatch below still defines the adapter capabilities it implements.
type GameFactory = fn(serde_json::Value, usize, LearningLimits) -> Result<Engine, Error>;
struct GameRegistration {
    name: &'static str,
    create: GameFactory,
}
const GAME_REGISTRY: &[GameRegistration] = &[
    GameRegistration {
        name: "thegame",
        create: create_thegame,
    },
    GameRegistration {
        name: "sushigo",
        create: create_sushigo,
    },
    GameRegistration {
        name: "loveletter",
        create: create_loveletter,
    },
    GameRegistration {
        name: "cantstop",
        create: create_cantstop,
    },
    GameRegistration {
        name: "tictactoe",
        create: create_tictactoe,
    },
    GameRegistration {
        name: "connect4",
        create: create_connect4,
    },
];

pub fn registered_games() -> Vec<&'static str> {
    GAME_REGISTRY.iter().map(|game| game.name).collect()
}

fn create_tictactoe(
    parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    let mut parameters: BTreeMap<String, usize> =
        serde_json::from_value(parameters).map_err(invalid)?;
    let grid_size = parameters.remove("grid_size").unwrap_or(3);
    check_keys(&parameters)?;
    Ok(Engine::TicTacToe(Game::new(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters {
            grid_size,
            ..Default::default()
        },
        players,
        limits,
    )?))
}

fn create_connect4(
    parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    let mut parameters: BTreeMap<String, usize> =
        serde_json::from_value(parameters).map_err(invalid)?;
    let width = parameters.remove("width").unwrap_or(7);
    let height = parameters.remove("height").unwrap_or(6);
    let winning_length = parameters.remove("winning_length").unwrap_or(4);
    check_keys(&parameters)?;
    Ok(Engine::Connect4(Game::new(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters {
            width,
            height,
            winning_length,
            ..Default::default()
        },
        players,
        limits,
    )?))
}

fn create_cantstop(
    mut parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    resolve_game_seed(&mut parameters)?;
    let parameters: cs::CantStopParameters = serde_json::from_value(parameters).map_err(invalid)?;
    Ok(Engine::CantStop(Box::new(Game::new(
        cs::CantStopForwardModel,
        parameters,
        players,
        limits,
    )?)))
}

fn resolve_game_seed(parameters: &mut serde_json::Value) -> Result<(), Error> {
    let object = parameters
        .as_object_mut()
        .ok_or_else(|| invalid("parameters must be an object"))?;
    if object.get("seed").is_none_or(serde_json::Value::is_null) {
        let seed = tag_rust::core::random::automatic_seed().map_err(invalid)?;
        object.insert("seed".into(), serde_json::json!(seed));
    }
    Ok(())
}
fn create_loveletter(
    mut parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    resolve_game_seed(&mut parameters)?;
    let parameters: ll::LoveLetterParameters =
        serde_json::from_value(parameters).map_err(invalid)?;
    Ok(Engine::LoveLetter(Box::new(Game::new(
        ll::LoveLetterForwardModel,
        parameters,
        players,
        limits,
    )?)))
}
fn create_sushigo(
    mut parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    resolve_game_seed(&mut parameters)?;
    let mut fields: BTreeMap<String, u64> = serde_json::from_value(parameters).map_err(invalid)?;
    let seed = fields
        .remove("seed")
        .ok_or_else(|| invalid("missing game seed"))?;
    if let Some(key) = fields.keys().next() {
        return Err(invalid(format!("unknown parameter: {key}")));
    }
    Ok(Engine::SushiGo(Box::new(Game::new(
        sg::SushiGoForwardModel,
        sg::Parameters { seed },
        players,
        limits,
    )?)))
}
fn create_thegame(
    mut parameters: serde_json::Value,
    players: usize,
    limits: LearningLimits,
) -> Result<Engine, Error> {
    resolve_game_seed(&mut parameters)?;
    let object = parameters
        .as_object_mut()
        .ok_or_else(|| invalid("parameters must be an object"))?;
    let seed: u64 = serde_json::from_value(
        object
            .remove("seed")
            .ok_or_else(|| invalid("missing game seed"))?,
    )
    .map_err(invalid)?;
    let first_player = object
        .remove("first_player")
        .map(serde_json::from_value::<usize>)
        .transpose()
        .map_err(invalid)?
        .unwrap_or(0);
    let communication = match object.remove("communication") {
        None => tg::CommunicationMode::LimitedChoices,
        Some(serde_json::Value::String(ref value)) if value == "limited_choices" => {
            tg::CommunicationMode::LimitedChoices
        }
        Some(serde_json::Value::String(ref value)) if value == "silent" => {
            tg::CommunicationMode::Silent
        }
        _ => return Err(invalid("communication must be limited_choices or silent")),
    };
    if let Some(key) = object.keys().next() {
        return Err(invalid(format!("unknown parameter: {key}")));
    }
    Ok(Engine::TheGame(Box::new(Game::new(
        tg::TheGameForwardModel,
        tg::Parameters {
            seed,
            first_player: PlayerId::new(first_player),
            communication,
        },
        players,
        limits,
    )?)))
}
pub(crate) fn unsupported_game(game: &str, operation: &str) -> Error {
    invalid(format!(
        "{game} does not support {operation} in this Python adapter"
    ))
}

pub(crate) fn unsupported(operation: &str) -> Error {
    unsupported_game("Can't Stop", operation)
}

impl Engine {
    /// Construct through the same registry used for game discovery.
    pub fn create(
        name: &str,
        parameters: BTreeMap<String, usize>,
        players: usize,
        limits: LearningLimits,
    ) -> Result<Self, Error> {
        Self::create_value(
            name,
            serde_json::to_value(parameters).map_err(invalid)?,
            players,
            limits,
        )
    }
    pub fn create_value(
        name: &str,
        parameters: serde_json::Value,
        players: usize,
        limits: LearningLimits,
    ) -> Result<Self, Error> {
        let registration = GAME_REGISTRY
            .iter()
            .find(|game| game.name == name)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "unknown game: {name}; registered games: {}",
                    registered_games().join(", ")
                ))
            })?;
        (registration.create)(parameters, players, limits)
    }

    pub fn seed(&self) -> Option<u64> {
        match self {
            Self::CantStop(game) => Some(game.parameters.seed),
            Self::LoveLetter(game) => Some(game.parameters.seed),
            Self::SushiGo(game) => Some(game.parameters.seed),
            Self::TheGame(game) => Some(game.parameters.seed),
            _ => None,
        }
    }
    pub fn payload_names(&self) -> Vec<&'static str> {
        let description = match self {
            Self::TicTacToe(g) => &g.description,
            Self::Connect4(g) => &g.description,
            Self::CantStop(g) => &g.description,
            Self::LoveLetter(g) => &g.description,
            Self::SushiGo(g) => &g.description,
            Self::TheGame(g) => &g.description,
        };
        description.payloads.iter().map(|p| p.name()).collect()
    }

    /// Read array descriptions for the stored configuration.
    pub fn spec(&self) -> Result<LearningSpec, Error> {
        match self {
            Self::TicTacToe(game) => Ok(game.description.clone()),
            Self::Connect4(game) => Ok(game.description.clone()),
            Self::CantStop(game) => Ok(game.description.clone()),
            Self::LoveLetter(game) => Ok(game.description.clone()),
            Self::SushiGo(game) => Ok(game.description.clone()),
            Self::TheGame(game) => Ok(game.description.clone()),
        }
    }

    /// Identify the player who may act, or completion.
    pub fn next_player(&self) -> Option<usize> {
        match self {
            Self::TicTacToe(game) => game.next_player(),
            Self::Connect4(game) => game.next_player(),
            Self::CantStop(game) => game.next_player(),
            Self::LoveLetter(game) => game.next_player(),
            Self::SushiGo(game) => game.next_player(),
            Self::TheGame(game) => game.next_player(),
        }
        .map(PlayerId::index)
    }

    /// Read an observer's information and create independent requested arrays.
    pub fn observe(&self, observer: usize, names: &[&str]) -> Result<EncodedObservation, Error> {
        match self {
            Self::TicTacToe(game) => game.observe(observer, names),
            Self::Connect4(game) => game.observe(observer, names),
            Self::CantStop(game) => game.observe(observer, names),
            Self::LoveLetter(game) => game.observe(observer, names),
            Self::SushiGo(game) => game.observe(observer, names),
            Self::TheGame(game) => game.observe(observer, names),
        }
    }

    /// Apply an ID through the concrete game's validation, optionally checking an explicit actor.
    pub fn apply(&mut self, id: usize, actor: Option<usize>) -> Result<(), Error> {
        match self {
            Self::TicTacToe(game) => game.apply(id, actor),
            Self::Connect4(game) => game.apply(id, actor),
            Self::CantStop(game) => game.apply(id, actor),
            Self::LoveLetter(game) => game.apply(id, actor),
            Self::SushiGo(game) => game.apply(id, actor),
            Self::TheGame(game) => game.apply(id, actor),
        }
    }

    /// Recreate the initial state using the original settings.
    pub fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::TicTacToe(game) => game.reset(),
            Self::Connect4(game) => game.reset(),
            Self::CantStop(game) => game.reset(),
            Self::LoveLetter(game) => game.reset(),
            Self::SushiGo(game) => game.reset(),
            Self::TheGame(game) => game.reset(),
        }
    }

    /// Copy the complete state into separately owned storage.
    pub fn snapshot(&self) -> Result<Self, Error> {
        match self {
            Self::TicTacToe(game) => Ok(Self::TicTacToe(game.snapshot()?)),
            Self::Connect4(game) => Ok(Self::Connect4(game.snapshot()?)),
            Self::CantStop(game) => Ok(Self::CantStop(Box::new(game.snapshot()?))),
            Self::LoveLetter(game) => Ok(Self::LoveLetter(Box::new(game.snapshot()?))),
            Self::SushiGo(game) => Ok(Self::SushiGo(Box::new(game.snapshot()?))),
            Self::TheGame(game) => Ok(Self::TheGame(Box::new(game.snapshot()?))),
        }
    }

    pub fn set_controller(
        &mut self,
        player: usize,
        snapshot: Option<OslaPythonSnapshot>,
    ) -> Result<(), Error> {
        self.set_controller_mode(player, snapshot, "external")
    }
    pub fn set_controller_mode(
        &mut self,
        player: usize,
        snapshot: Option<OslaPythonSnapshot>,
        execution: &str,
    ) -> Result<(), Error> {
        if matches!(
            self,
            Self::CantStop(_) | Self::LoveLetter(_) | Self::SushiGo(_) | Self::TheGame(_)
        ) && snapshot.is_some()
        {
            return Err(unsupported_game(
                if matches!(self, Self::TheGame(_)) {
                    "The Game"
                } else if matches!(self, Self::LoveLetter(_)) {
                    "Love Letter"
                } else if matches!(self, Self::SushiGo(_)) {
                    "Sushi Go!"
                } else {
                    "Can't Stop"
                },
                "attached evaluation controllers",
            ));
        }
        if execution != "external" && execution != "rust_open_lines" {
            return Err(Error::Invalid("unknown controller execution mode".into()));
        }
        let backend = if execution == "rust_open_lines" {
            let snapshot = snapshot
                .as_ref()
                .ok_or_else(|| Error::Invalid("native scoring requires an AI controller".into()))?;
            if snapshot.config.evaluator.name != "rust-open-lines"
                || snapshot.config.evaluator.version != 1
                || snapshot.config.evaluator.configuration != "{}"
                || snapshot.config.payloads != ["cells"]
            {
                return Err(Error::Invalid("Rust scoring requires rust-open-lines version 1, empty settings and cells input".into()));
            }
            Some(OpenLinesBatchEvaluator::new(&self.spec()?)?)
        } else {
            None
        };
        let agent = snapshot
            .as_ref()
            .map(OslaPythonAgent::from_snapshot)
            .transpose()?;
        match self {
            Self::TicTacToe(game) => {
                game.session.set_controller(PlayerId::new(player), agent)?;
                game.native_evaluators[player] = backend;
            }
            Self::Connect4(game) => {
                game.session.set_controller(PlayerId::new(player), agent)?;
                game.native_evaluators[player] = backend;
            }
            Self::CantStop(game) => {
                game.session.set_controller(PlayerId::new(player), None)?;
            }
            Self::LoveLetter(game) => {
                game.session.set_controller(PlayerId::new(player), None)?;
            }
            Self::SushiGo(game) => {
                if !game.joint_pending.is_empty() {
                    return Err(invalid(
                        "cancel pending joint choices before changing controllers",
                    ));
                }
                game.session.set_controller(PlayerId::new(player), None)?;
            }
            Self::TheGame(game) => {
                game.session.set_controller(PlayerId::new(player), None)?;
            }
        }
        Ok(())
    }
    pub fn execution_mode(&self, player: usize) -> Result<&'static str, Error> {
        let backend = match self {
            Self::TicTacToe(game) => game.native_evaluators.get(player),
            Self::Connect4(game) => game.native_evaluators.get(player),
            Self::CantStop(game) => game.native_evaluators.get(player),
            Self::LoveLetter(game) => game.native_evaluators.get(player),
            Self::SushiGo(game) => game.native_evaluators.get(player),
            Self::TheGame(game) => game.native_evaluators.get(player),
        }
        .ok_or_else(|| Error::Invalid("unknown controller seat".into()))?;
        Ok(if backend.is_some() {
            "rust_open_lines"
        } else {
            "external"
        })
    }
    pub fn checkpoint(&self, player: usize) -> Result<Option<OslaPythonSnapshot>, Error> {
        Ok(match self {
            Self::TicTacToe(game) => game.session.checkpoint(PlayerId::new(player))?,
            Self::Connect4(game) => game.session.checkpoint(PlayerId::new(player))?,
            Self::CantStop(game) => game.session.checkpoint(PlayerId::new(player))?,
            Self::LoveLetter(game) => game.session.checkpoint(PlayerId::new(player))?,
            Self::SushiGo(game) => game.session.checkpoint(PlayerId::new(player))?,
            Self::TheGame(game) => game.session.checkpoint(PlayerId::new(player))?,
        })
    }
    pub fn prepare(&mut self) -> Result<EvaluationRequest, Error> {
        if let Some(actor) = self.next_player()
            && self.execution_mode(actor)? == "rust_open_lines"
        {
            return Err(Error::Invalid(
                "this controller uses Rust scoring; call advance()".into(),
            ));
        }
        Ok(match self {
            Self::TicTacToe(game) => game.session.prepare()?,
            Self::Connect4(game) => game.session.prepare()?,
            Self::CantStop(_) => return Err(unsupported("external evaluation")),
            Self::LoveLetter(_) => {
                return Err(unsupported_game("Love Letter", "external evaluation"));
            }
            Self::SushiGo(_) => return Err(unsupported_game("Sushi Go!", "external evaluation")),
            Self::TheGame(_) => return Err(unsupported_game("The Game", "external evaluation")),
        })
    }
    pub fn finish(
        &mut self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        ids: &[usize],
        values: &[f64],
    ) -> Result<EvaluationStatus, Error> {
        Ok(match self {
            Self::TicTacToe(game) => game.session.finish(key, evaluator, ids, values)?,
            Self::Connect4(game) => game.session.finish(key, evaluator, ids, values)?,
            Self::CantStop(_) => return Err(unsupported("external evaluation")),
            Self::LoveLetter(_) => {
                return Err(unsupported_game("Love Letter", "external evaluation"));
            }
            Self::SushiGo(_) => return Err(unsupported_game("Sushi Go!", "external evaluation")),
            Self::TheGame(_) => return Err(unsupported_game("The Game", "external evaluation")),
        })
    }
    pub fn commit(&mut self, key: &EvaluationKey) -> Result<(), Error> {
        match self {
            Self::TicTacToe(game) => game.session.commit(key)?,
            Self::Connect4(game) => game.session.commit(key)?,
            Self::CantStop(game) => game.session.commit(key)?,
            Self::LoveLetter(game) => game.session.commit(key)?,
            Self::SushiGo(game) => game.session.commit(key)?,
            Self::TheGame(game) => game.session.commit(key)?,
        }
        Ok(())
    }
    pub fn cancel(&mut self, key: &EvaluationKey) -> Result<(), Error> {
        match self {
            Self::TicTacToe(game) => game.session.cancel(key)?,
            Self::Connect4(game) => game.session.cancel(key)?,
            Self::CantStop(game) => game.session.cancel(key)?,
            Self::LoveLetter(game) => game.session.cancel(key)?,
            Self::SushiGo(game) => game.session.cancel(key)?,
            Self::TheGame(game) => game.session.cancel(key)?,
        }
        Ok(())
    }
}

fn check_keys(parameters: &BTreeMap<String, usize>) -> Result<(), Error> {
    if let Some(name) = parameters.keys().next() {
        return Err(Error::Invalid(format!("unknown parameter: {name}")));
    }
    Ok(())
}

#[cfg(test)]
mod sushigo_tests {
    use super::*;

    #[test]
    fn joint_choices_keep_observations_fixed_until_every_seat_submits() {
        let mut game = Engine::create_value(
            "sushigo",
            serde_json::json!({"seed": 12}),
            3,
            LearningLimits::default(),
        )
        .unwrap();
        assert_eq!(game.seed(), Some(12));
        assert_eq!(game.next_player(), Some(0));
        let names = ["hand", "known_hands", "last_reveal", "played"];
        let before = (0..3)
            .map(|seat| game.observe(seat, &names).unwrap())
            .collect::<Vec<_>>();
        let first = before[0].offered_action_ids[0];
        game.apply(first, Some(0)).unwrap();
        assert_eq!(game.next_player(), Some(1));
        assert!(game.snapshot().is_err());
        for (seat, expected) in before.iter().enumerate() {
            let observed = game.observe(seat, &names).unwrap();
            assert_eq!(observed.tensors, expected.tensors);
            assert_eq!(observed.progress, expected.progress);
        }
        assert!(game.observe(0, &[]).unwrap().offered_action_ids.is_empty());
        assert!(game.apply(first, Some(0)).is_err());
        for (seat, view) in before.iter().enumerate().skip(1) {
            let id = view.offered_action_ids[0];
            game.apply(id, Some(seat)).unwrap();
        }
        assert_eq!(game.next_player(), Some(0));
        let after = game.observe(0, &names).unwrap();
        assert_eq!(after.progress[1], 1);
        assert_ne!(after.tensors, before[0].tensors);
        assert_eq!(after.tensors[2].values.len(), 9);
        assert!(game.snapshot().is_ok());
        game.reset().unwrap();
        assert_eq!(game.observe(0, &names).unwrap().tensors, before[0].tensors);
    }

    #[test]
    fn submission_order_does_not_change_joint_successor() {
        let mut ordered = Engine::create_value(
            "sushigo",
            serde_json::json!({"seed": 77}),
            3,
            LearningLimits::default(),
        )
        .unwrap();
        let mut reversed = ordered.snapshot().unwrap();
        let ids = (0..3)
            .map(|seat| ordered.observe(seat, &[]).unwrap().offered_action_ids[0])
            .collect::<Vec<_>>();
        for seat in [0, 1, 2] {
            ordered.apply(ids[seat], Some(seat)).unwrap();
        }
        for seat in [2, 0, 1] {
            reversed.apply(ids[seat], Some(seat)).unwrap();
        }
        for seat in 0..3 {
            assert_eq!(
                ordered
                    .observe(seat, &["hand", "played", "last_reveal"])
                    .unwrap(),
                reversed
                    .observe(seat, &["hand", "played", "last_reveal"])
                    .unwrap()
            );
        }
    }
}
