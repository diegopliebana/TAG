//! One-step lookahead with batches evaluated by a Rust backend or an external caller.
//! The original OslaAgent retains its existing synchronous evaluator and behavior.
mod batch;
mod open_lines;
use super::Agent;
use crate::core::{
    PlayerId, PublicInformationModel,
    checkpoint::Checkpoint,
    evaluation::EVALUATION_TIE_TOLERANCE,
    random::{AgentRng, AgentRngSnapshot},
};
use crate::learning::LearningGame;
pub use batch::{CandidateBatch, EvaluationLimits, EvaluationWork};
pub use open_lines::OpenLinesBatchEvaluator;

/// Identifies the evaluator code/version and all settings needed to reproduce its values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatorIdentity {
    pub name: String,
    pub version: u32,
    /// Opaque settings recorded by the caller; Python uses canonical JSON.
    pub configuration: String,
}

/// Persistent policy settings. Pending candidate arrays are stored in the execution session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OslaPythonConfig {
    pub evaluator: EvaluatorIdentity,
    pub payloads: Vec<String>,
    pub limits: EvaluationLimits,
}
impl OslaPythonConfig {
    /// Check owned settings before constructing or restoring an agent.
    pub fn validate(&self) -> Result<(), OslaPythonError> {
        if self.evaluator.name.is_empty()
            || self.evaluator.version == 0
            || self.limits.max_rows == 0
            || self.limits.max_candidates == 0
            || self.limits.learning.max_cells == 0
            || self.limits.learning.max_bytes == 0
            || self
                .payloads
                .iter()
                .enumerate()
                .any(|(i, name)| name.is_empty() || self.payloads[..i].contains(name))
        {
            return Err(OslaPythonError::Invalid(
                "invalid evaluator identity, payload names or limits".into(),
            ));
        }
        Ok(())
    }
}

/// Failures during preparation, external evaluation, restoration or selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OslaPythonError {
    ExternalEvaluationRequired,
    EmptyActions,
    Invalid(String),
    Simulation(String),
    ResourceLimit,
    Overflow,
    Busy,
    Stale,
    NotReady,
    ValueCount { expected: usize, actual: usize },
    NonfiniteValue { candidate: usize },
}
impl std::fmt::Display for OslaPythonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExternalEvaluationRequired => {
                f.write_str("this agent needs external batch values")
            }
            Self::EmptyActions => f.write_str("action selection requires offered actions"),
            Self::Invalid(s) | Self::Simulation(s) => f.write_str(s),
            Self::ResourceLimit => f.write_str("evaluation exceeds configured limits"),
            Self::Overflow => f.write_str("evaluation size or identifier overflow"),
            Self::Busy => f.write_str("finish or cancel the current evaluation first"),
            Self::Stale => {
                f.write_str("evaluation belongs to another session, position or completed request")
            }
            Self::NotReady => f.write_str("evaluation is not ready for this operation"),
            Self::ValueCount { expected, actual } => {
                write!(f, "expected {expected} values, received {actual}")
            }
            Self::NonfiniteValue { candidate } => {
                write!(f, "candidate {candidate} value must be finite")
            }
        }
    }
}
impl std::error::Error for OslaPythonError {}
impl From<crate::learning::LearningError> for OslaPythonError {
    fn from(error: crate::learning::LearningError) -> Self {
        match error {
            crate::learning::LearningError::ResourceLimit => Self::ResourceLimit,
            crate::learning::LearningError::Overflow => Self::Overflow,
            _ => Self::Invalid(error.to_string()),
        }
    }
}

/// A Rust backend can evaluate an entire chunk during ordinary synchronous choose().
pub trait BatchEvaluator {
    /// Rate every row of the borrowed batch for batch.observer.
    /// Return one finite value per row, in the same order. Larger values are better.
    /// A failure is returned to the caller; a stateful backend retains updates it
    /// already made, while move selection has not yet advanced the agent's RNG.
    fn evaluate(&mut self, batch: &CandidateBatch) -> Result<Vec<f64>, OslaPythonError>;
    fn needs_external_values(&self) -> bool {
        false
    }
}

/// Marks an agent whose values arrive through a prepared-evaluation session.
#[derive(Clone, Debug)]
pub struct ExternalEvaluation;
impl BatchEvaluator for ExternalEvaluation {
    fn evaluate(&mut self, _: &CandidateBatch) -> Result<Vec<f64>, OslaPythonError> {
        Err(OslaPythonError::ExternalEvaluationRequired)
    }
    fn needs_external_values(&self) -> bool {
        true
    }
}
impl Checkpoint for ExternalEvaluation {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) {}
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}

/// Complete persistent settings, backend memory and exact RNG position between moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OslaPythonSnapshot<S = ()> {
    pub policy_version: u32,
    pub config: OslaPythonConfig,
    pub rng: AgentRngSnapshot,
    pub evaluator: S,
}

/// A separate policy owning its backend, settings and RNG.
#[derive(Debug)]
pub struct OslaPythonAgent<E = ExternalEvaluation> {
    config: OslaPythonConfig,
    evaluator: E,
    rng: AgentRng,
}
impl<E> OslaPythonAgent<E> {
    /// Consume evaluator/settings and copy the seed into a separate generator.
    pub fn new(evaluator: E, config: OslaPythonConfig, seed: u64) -> Result<Self, OslaPythonError> {
        config.validate()?;
        Ok(Self {
            config,
            evaluator,
            rng: AgentRng::new(seed),
        })
    }
    pub fn config(&self) -> &OslaPythonConfig {
        &self.config
    }
    pub(crate) fn rng_ref(&self) -> &AgentRng {
        &self.rng
    }
    pub(crate) fn rng_ref_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }

    /// Validate all values before advancing this agent's RNG and return a list index.
    /// `count` is the complete candidate count; `values` is read-only and in action order.
    pub fn select(&mut self, count: usize, values: &[f64]) -> Result<usize, OslaPythonError> {
        validate_values(values, count, 0)?;
        if count == 0 {
            return Err(OslaPythonError::EmptyActions);
        }
        let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let eligible = || {
            values
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (best - v <= EVALUATION_TIE_TOLERANCE).then_some(i))
        };
        let draw = self
            .rng
            .index(eligible().count())
            .expect("finite maximum is eligible");
        Ok(eligible()
            .nth(draw)
            .expect("selected eligible index exists"))
    }
}
impl<E: Checkpoint> Checkpoint for OslaPythonAgent<E>
where
    E::RestoreError: std::fmt::Display,
{
    type Snapshot = OslaPythonSnapshot<E::Snapshot>;
    type RestoreError = OslaPythonError;
    fn snapshot(&self) -> Self::Snapshot {
        OslaPythonSnapshot {
            policy_version: 1,
            config: self.config.clone(),
            rng: self.rng.snapshot(),
            evaluator: self.evaluator.snapshot(),
        }
    }
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        if snapshot.policy_version != 1 {
            return Err(OslaPythonError::Invalid(
                "unsupported OSLAPython policy version".into(),
            ));
        }
        snapshot.config.validate()?;
        Ok(Self {
            config: snapshot.config.clone(),
            rng: AgentRng::from_snapshot(&snapshot.rng)
                .map_err(|e| OslaPythonError::Invalid(e.to_string()))?,
            evaluator: E::from_snapshot(&snapshot.evaluator)
                .map_err(|e| OslaPythonError::Invalid(e.to_string()))?,
        })
    }
}

/// Interface for agents that request external evaluation of simulated positions.
/// Preparation receives the same observation and available actions as Agent::choose.
pub trait PreparedEvaluationAgent<F: LearningGame + PublicInformationModel>: Agent<F> {
    type Work;
    /// Reconstruct a source from the acting player's allowed observation and retain its choices.
    ///
    /// # Arguments
    /// `model` supplies borrowed rule access. `actor` is the copied player choosing.
    /// `observation` and `actions` are borrowed and remain unchanged. Actions retain
    /// their supplied order. Persistent agent settings determine requested arrays/limits.
    ///
    /// # Returns
    /// Owned temporary work for a later batch request. Preparation leaves the live
    /// board and agent RNG unchanged.
    ///
    /// # Errors
    /// Invalid observer/actor data, empty actions, exceeded limits or a failed
    /// reconstruction. Concrete implementations define their error type.
    fn prepare(
        &self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<Self::Work, Self::Error>;
}

impl<F, E> PreparedEvaluationAgent<F> for OslaPythonAgent<E>
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
    E: BatchEvaluator,
{
    type Work = EvaluationWork<F>;
    fn prepare(
        &self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<Self::Work, Self::Error> {
        EvaluationWork::new(model, actor, observation, actions, &self.config)
    }
}
impl<F, E> Agent<F> for OslaPythonAgent<E>
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
    E: BatchEvaluator,
{
    type Error = OslaPythonError;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        if self.evaluator.needs_external_values() {
            return Err(OslaPythonError::ExternalEvaluationRequired);
        }
        let work = self.prepare(model, actor, observation, actions)?;
        let mut values = Vec::with_capacity(actions.len());
        while values.len() < actions.len() {
            let batch = work.batch(model, values.len())?;
            let chunk = self.evaluator.evaluate(&batch)?;
            validate_values(&chunk, batch.action_ids.len(), values.len())?;
            values.extend(chunk);
        }
        self.select(actions.len(), &values)
    }
}

pub(crate) fn validate_values(
    values: &[f64],
    expected: usize,
    offset: usize,
) -> Result<(), OslaPythonError> {
    if values.len() != expected {
        return Err(OslaPythonError::ValueCount {
            expected,
            actual: values.len(),
        });
    }
    for (i, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(OslaPythonError::NonfiniteValue {
                candidate: offset + i,
            });
        }
    }
    Ok(())
}
