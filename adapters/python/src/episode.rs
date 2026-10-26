//! Single-learner episodes. Rules, opponent decisions and bookkeeping run in Rust.
use crate::engine::{Engine, Game};
use std::collections::BTreeMap;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::{
    agents::{
        Agent, EvaluationLimits, EvaluatorIdentity, OpenLinesBatchEvaluator, OslaPythonAgent,
        OslaPythonConfig,
    },
    core::{PlayerId, PublicInformationModel, evaluation::Evaluator, random::AgentRngSnapshot},
    games::{connect4 as c4, tictactoe as ttt},
    learning::{EncodedObservation, LearningGame, LearningLimits, LearningSpec, ResultTag},
    session::BuiltinAgent,
};

/// Fixed episode setup. Cloning it copies configuration, never a live episode.
#[derive(Clone)]
pub struct Config {
    pub name: String,
    pub parameters: BTreeMap<String, usize>,
    pub learner: usize,
    pub opponent: String,
    pub settings: BTreeMap<String, usize>,
    pub payloads: Vec<String>,
    pub horizon: Option<u64>,
    pub limits: LearningLimits,
    /// Encode the learner's view immediately after its action, before opponent play.
    pub include_intermediate: bool,
    /// Enable an independently seeded original OSLA policy for learner actions.
    pub teacher: bool,
}

/// The current native policies retain their original choose implementations.
enum Opponent<E> {
    Builtin(Box<BuiltinAgent<E>>),
    Batch(Box<OslaPythonAgent<OpenLinesBatchEvaluator>>),
}
impl<E> Opponent<E> {
    fn rng<F>(&self) -> AgentRngSnapshot
    where
        F: LearningGame + PublicInformationModel,
        F::State: Clone,
        F::Action: Clone,
        F::Error: std::fmt::Display,
        E: Evaluator<F::State>,
    {
        match self {
            Self::Builtin(a) => <BuiltinAgent<E> as Agent<F>>::rng(a).snapshot(),
            Self::Batch(a) => {
                <OslaPythonAgent<OpenLinesBatchEvaluator> as Agent<F>>::rng(a).snapshot()
            }
        }
    }
    fn choose<F>(
        &mut self,
        model: &F,
        actor: PlayerId,
        view: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, String>
    where
        F: LearningGame + PublicInformationModel,
        F::State: Clone,
        F::Action: Clone,
        F::Error: std::fmt::Display,
        E: Evaluator<F::State>,
        E::Error: std::fmt::Display,
    {
        match self {
            Self::Builtin(a) => a
                .choose(model, actor, view, actions)
                .map_err(|e| e.to_string()),
            Self::Batch(a) => a
                .choose(model, actor, view, actions)
                .map_err(|e| e.to_string()),
        }
    }
}

pub struct EpisodeGame<F: LearningGame, E>
where
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    game: Game<F>,
    opponent: Opponent<E>,
    teacher: Option<Opponent<E>>,
    include_intermediate: bool,
    learner: usize,
    payloads: Vec<String>,
    horizon: Option<u64>,
    pub steps: u64,
    pub moves: u64,
    pub ended: bool,
}

/// Own one typed runtime; game registration remains Rust dispatch.
pub enum Runtime {
    TicTacToe(EpisodeGame<ttt::TicTacToeForwardModel, ttt::TicTacToeEvaluator>),
    Connect4(EpisodeGame<c4::Connect4ForwardModel, c4::Connect4Evaluator>),
}
/// One completed return, before conversion to independently owned Python data.
pub struct Output {
    pub observation: EncodedObservation,
    pub steps: u64,
    pub moves: u64,
    pub delta_moves: u64,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub rng: AgentRngSnapshot,
    pub teacher_rng: Option<AgentRngSnapshot>,
    pub action: Option<usize>,
    pub intermediate: Option<EncodedObservation>,
}

/// Build one registered native policy from borrowed settings/specification.
/// `evaluator` is consumed for original OSLA and `seed` initializes owned RNG.
/// Returns an independent policy, or rejects unsupported settings and limits.
fn native<E>(
    config: &Config,
    spec: &LearningSpec,
    evaluator: E,
    seed: u64,
) -> Result<Opponent<E>, String> {
    use tag_rust::agents::{FirstAvailableAgent, OslaAgent, RandomAgent};
    Ok(match config.opponent.as_str() {
        "first_available" => Opponent::Builtin(Box::new(BuiltinAgent::FirstAvailable(
            FirstAvailableAgent::new(seed),
        ))),
        "random" => Opponent::Builtin(Box::new(BuiltinAgent::Random(RandomAgent::new(seed)))),
        "osla" => Opponent::Builtin(Box::new(BuiltinAgent::Osla(OslaAgent::new(
            evaluator, seed,
        )))),
        "osla_python" => {
            let rows = config.settings.get("max_rows").copied().unwrap_or(1024);
            let candidates = config
                .settings
                .get("max_candidates")
                .copied()
                .unwrap_or(4096);
            if rows == 0 || candidates < spec.actions.size {
                return Err(
                    "max_rows must be positive and max_candidates must cover the action space"
                        .into(),
                );
            }
            let cells = spec
                .identity
                .width()
                .checked_mul(spec.identity.height())
                .ok_or("cell size overflow")?;
            let row_bytes = cells
                .checked_mul(4)
                .and_then(|v| v.checked_add(50))
                .ok_or("row size overflow")?;
            if row_bytes > config.limits.max_bytes {
                return Err("one candidate exceeds max_bytes".into());
            }
            let agent = OslaPythonAgent::new(
                OpenLinesBatchEvaluator::new(spec).map_err(|e| e.to_string())?,
                OslaPythonConfig {
                    evaluator: EvaluatorIdentity {
                        name: "rust-open-lines".into(),
                        version: 1,
                        configuration: "{}".into(),
                    },
                    payloads: vec!["cells".into()],
                    limits: EvaluationLimits {
                        learning: config.limits,
                        max_rows: rows,
                        max_candidates: candidates,
                    },
                },
                seed,
            )
            .map_err(|e| e.to_string())?;
            Opponent::Batch(Box::new(agent))
        }
        _ => return Err("unknown Rust opponent".into()),
    })
}

impl Config {
    /// Validate borrowed fixed settings and create a new owned candidate.
    /// `seed` initializes only the opponent. Failure never touches the live episode.
    pub fn create(&self, seed: u64) -> Result<Runtime, String> {
        self.create_seeded(seed, self.teacher.then_some(0))
    }
    /// Create a replacement with independent opponent and optional teacher seeds.
    ///
    /// # Arguments
    /// `seed` initializes the opponent; `teacher_seed` must be supplied exactly
    /// when teacher support is configured. No live episode is accessed.
    ///
    /// # Returns
    /// A separately owned fresh runtime; the caller runs any opening move.
    ///
    /// # Errors
    /// Invalid settings, incompatible teacher seeds, or exceeded output limits.
    pub fn create_seeded(&self, seed: u64, teacher_seed: Option<u64>) -> Result<Runtime, String> {
        if self.teacher != teacher_seed.is_some() {
            return Err("teacher_seed is required exactly when teacher is configured".into());
        }
        if self.learner > 1 || self.horizon == Some(0) || self.payloads.is_empty() {
            return Err("learner must be 0 or 1, horizon positive, and payloads nonempty".into());
        }
        for key in self.settings.keys() {
            if self.opponent != "osla_python"
                || !["max_rows", "max_candidates"].contains(&key.as_str())
            {
                return Err(format!("unsupported opponent setting: {key}"));
            }
        }
        let game = Engine::create(&self.name, self.parameters.clone(), 2, self.limits)
            .map_err(|e| e.to_string())?;
        Ok(match game {
            Engine::TicTacToe(game) => {
                let opponent = native(self, &game.description, ttt::TicTacToeEvaluator, seed)?;
                let teacher = teacher_seed
                    .map(|seed| {
                        native(
                            &self.teacher_config(),
                            &game.description,
                            ttt::TicTacToeEvaluator,
                            seed,
                        )
                    })
                    .transpose()?;
                Runtime::TicTacToe(EpisodeGame::new(game, opponent, teacher, self)?)
            }
            Engine::Connect4(game) => {
                let opponent = native(self, &game.description, c4::Connect4Evaluator, seed)?;
                let teacher = teacher_seed
                    .map(|seed| {
                        native(
                            &self.teacher_config(),
                            &game.description,
                            c4::Connect4Evaluator,
                            seed,
                        )
                    })
                    .transpose()?;
                Runtime::Connect4(EpisodeGame::new(game, opponent, teacher, self)?)
            }
            Engine::LoveLetter(_) => {
                return Err(
                    "Love Letter learner episodes are not supported by this adapter".into(),
                );
            }
            Engine::CantStop(_) => {
                return Err("Can't Stop learner episodes are not supported by this adapter".into());
            }
        })
    }
    fn teacher_config(&self) -> Self {
        let mut config = self.clone();
        config.opponent = "osla".into();
        config.settings.clear();
        config
    }
}
impl<F, E> EpisodeGame<F, E>
where
    F: LearningGame + PublicInformationModel + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Parameters: Clone,
    F::Error: std::fmt::Display,
    E: Evaluator<F::State>,
    E::Error: std::fmt::Display,
{
    fn new(
        game: Game<F>,
        opponent: Opponent<E>,
        teacher: Option<Opponent<E>>,
        config: &Config,
    ) -> Result<Self, String> {
        if config.learner == 1 && game.description.identity.target() == 1 {
            return Err("this configuration ends before learner seat 1 can act".into());
        }
        // Encoding the fixed layout now checks all payloads and output limits.
        let names = config
            .payloads
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        game.observe(0, &names).map_err(|e| e.to_string())?;
        if config.include_intermediate || config.teacher {
            check_capacity(
                &game.description,
                &config.payloads,
                if config.include_intermediate { 2 } else { 1 },
                config.limits.max_bytes,
            )?;
        }
        Ok(Self {
            game,
            opponent,
            teacher,
            include_intermediate: config.include_intermediate,
            learner: config.learner,
            payloads: config.payloads.clone(),
            horizon: config.horizon,
            steps: 0,
            moves: 0,
            ended: false,
        })
    }
    /// Advance the opponent if eligible, reading only its observation/actions.
    /// Returns after one reply or game completion; an error can retain progress.
    fn opponent_turn(&mut self) -> Result<(), String> {
        let model = self.game.session.model();
        let actor = match self.game.session.next_player() {
            tag_rust::core::Decision::Finished => return Ok(()),
            tag_rust::core::Decision::Player(p) if p.index() == self.learner => return Ok(()),
            tag_rust::core::Decision::Player(p) => p,
        };
        let view = model
            .observe(self.game.session.state(), actor)
            .map_err(|e| e.to_string())?;
        let actions = model
            .available_actions(self.game.session.state(), actor)
            .map_err(|e| e.to_string())?;
        let index = self.opponent.choose(model, actor, &view, &actions)?;
        let action = actions
            .get(index)
            .ok_or("opponent returned an invalid action index")?
            .clone();
        self.game
            .session
            .apply_human(actor, action)
            .map_err(|e| e.to_string())?;
        self.moves += 1;
        Ok(())
    }
    /// Read the current episode and proposed `id` without changing state or RNG.
    /// Reject ended episodes, overflow and IDs absent from the learner's mask.
    fn validate(&self, id: usize) -> Result<(), String> {
        if self.ended {
            return Err("episode ended; reset is required".into());
        }
        self.steps.checked_add(1).ok_or("learner step overflow")?;
        self.moves.checked_add(2).ok_or("move counter overflow")?;
        let view = self
            .game
            .observe(self.learner, &[])
            .map_err(|e| e.to_string())?;
        if view.action_mask.get(id) != Some(&true) {
            return Err("action is not currently legal".into());
        }
        Ok(())
    }
    /// Execute previously validated `id` in place and return owned learner data.
    /// Accepted actions survive later errors; the binding then requires reset.
    fn step(&mut self, id: usize) -> Result<Output, String> {
        let before = self.moves;
        self.game
            .apply(id, Some(self.learner))
            .map_err(|e| e.to_string())?;
        self.moves += 1;
        let intermediate = if self.include_intermediate {
            let names = self.payloads.iter().map(String::as_str).collect::<Vec<_>>();
            Some(
                self.game
                    .observe(self.learner, &names)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        self.opponent_turn()?;
        self.steps += 1;
        let mut out = self.output(self.moves - before)?;
        out.action = Some(id);
        out.intermediate = intermediate;
        self.ended = out.terminated || out.truncated;
        Ok(out)
    }
    /// Encode the learner view and counters; `delta` counts moves in this call.
    /// Returns separate arrays, reward and ending flags, or an encoding error.
    fn output(&self, delta: u64) -> Result<Output, String> {
        let names = self.payloads.iter().map(String::as_str).collect::<Vec<_>>();
        let observation = self
            .game
            .observe(self.learner, &names)
            .map_err(|e| e.to_string())?;
        let terminated = observation.next_player.is_none();
        let truncated = !terminated && self.horizon.is_some_and(|h| self.steps >= h);
        let reward = if terminated {
            match observation.results[self.learner] {
                ResultTag::Win => 1.,
                ResultTag::Loss => -1.,
                _ => 0.,
            }
        } else {
            0.
        };
        Ok(Output {
            observation,
            steps: self.steps,
            moves: self.moves,
            delta_moves: delta,
            reward,
            terminated,
            truncated,
            rng: self.opponent.rng::<F>(),
            teacher_rng: self.teacher.as_ref().map(|t| t.rng::<F>()),
            action: None,
            intermediate: None,
        })
    }
    fn validate_teacher(&self) -> Result<(), String> {
        if self.teacher.is_none() {
            return Err("no teacher configured".into());
        }
        let view = self
            .game
            .observe(self.learner, &[])
            .map_err(|e| e.to_string())?;
        let id = *view
            .offered_action_ids
            .first()
            .ok_or("no learner decision available")?;
        self.validate(id)
    }
    fn step_teacher(&mut self) -> Result<Output, String> {
        let model = self.game.session.model();
        let actor = PlayerId::new(self.learner);
        let view = model
            .observe(self.game.session.state(), actor)
            .map_err(|e| e.to_string())?;
        let actions = model
            .available_actions(self.game.session.state(), actor)
            .map_err(|e| e.to_string())?;
        let index = self
            .teacher
            .as_mut()
            .ok_or("no teacher configured")?
            .choose(model, actor, &view, &actions)?;
        let id = model
            .action_id(
                &view,
                actions.get(index).ok_or("invalid teacher action index")?,
            )
            .map_err(|e| e.to_string())?;
        self.step(id)
    }
}
macro_rules! dispatch {
    ($self:expr,$game:ident,$body:expr) => {
        match $self {
            Runtime::TicTacToe($game) => $body,
            Runtime::Connect4($game) => $body,
        }
    };
}
impl Runtime {
    #[cfg(test)]
    pub(super) fn inject_output_failure(&mut self) {
        dispatch!(
            self,
            game,
            game.payloads = vec!["injected-output-failure".into()]
        );
    }

    /// Read whether an episode completed or reached its learner-step horizon.
    pub fn ended(&self) -> bool {
        dispatch!(self, g, g.ended)
    }
    /// Return an independent copy of the fixed game/array descriptions.
    pub fn spec(&self) -> LearningSpec {
        dispatch!(self, g, g.game.description.clone())
    }
    /// Check borrowed `id` before mutation, returning a retryable input error.
    pub fn validate(&self, id: usize) -> Result<(), String> {
        dispatch!(self, g, g.validate(id))
    }
    /// Apply validated `id` in place; errors may leave accepted moves in state.
    pub fn step(&mut self, id: usize) -> Result<Output, String> {
        dispatch!(self, g, g.step(id))
    }
    /// Check teacher availability and lifecycle without consuming either RNG.
    pub fn validate_teacher(&self) -> Result<(), String> {
        dispatch!(self, g, g.validate_teacher())
    }
    /// Select the learner action in Rust and execute the ordinary learner step.
    /// A later failure may retain accepted actions and teacher RNG progress.
    pub fn step_teacher(&mut self) -> Result<Output, String> {
        dispatch!(self, g, g.step_teacher())
    }
    /// Run any opening opponent action and return an owned initial learner view.
    pub fn opening(&mut self) -> Result<Output, String> {
        dispatch!(self, g, {
            g.opponent_turn()?;
            g.output(g.moves)
        })
    }
}
/// Bound numeric arrays for `count` observation-sized outputs before execution.
///
/// # Arguments
/// `spec` and `names` borrow fixed descriptions; `max_bytes` is the numeric limit.
/// `count` includes every ordinary/intermediate output row, including placeholders.
///
/// # Returns
/// Unit when the conservative numeric budget fits.
///
/// # Errors
/// Returns unit or a descriptive size/overflow error; no allocation/game mutation.
pub(super) fn check_capacity(
    spec: &LearningSpec,
    names: &[String],
    count: usize,
    max_bytes: usize,
) -> Result<(), String> {
    let mut row = 256usize
        .checked_add(spec.actions.size)
        .ok_or("numeric size overflow")?;
    for name in names {
        let tensor = spec
            .payloads
            .iter()
            .find(|p| p.name() == name)
            .and_then(|p| p.tensor())
            .ok_or("payload is not a numeric tensor")?;
        let bytes = tensor
            .shape
            .iter()
            .try_fold(4usize, |n, &d| n.checked_mul(d))
            .ok_or("numeric size overflow")?;
        row = row.checked_add(bytes).ok_or("numeric size overflow")?;
    }
    if max_bytes == 0 || row.checked_mul(count).is_none_or(|n| n > max_bytes) {
        return Err("possible numeric outputs exceed max_bytes".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests;
