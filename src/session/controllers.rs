//! Choose who supplies moves for each player seat.
//!
//! Controller is Human or Ai(agent). BuiltinAgent is an enum containing one of the
//! built-in policies, so different policies can share one Rust collection. Saved
//! controller data identifies the policy and its settings/RNG progress.
use crate::agents::{Agent, EmptyActionList, FirstAvailableAgent, OslaAgent, OslaError};
use crate::agents::{OslaRestoreError, OslaSnapshot, RandomAgent};
use crate::core::checkpoint::Checkpoint;
use crate::core::evaluation::Evaluator;
use crate::core::random::AgentRng;
use crate::core::random::{AgentRngSnapshot, RngSnapshotError};
use crate::core::{PlayerId, PublicInformationModel};
use std::fmt;

pub use crate::recording::ControllerSnapshot;

/// The built-in agent kind and saved data needed to reproduce its future choices.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: independent evaluator data; PartialEq/Eq: compare policy and progress when supported.
pub enum BuiltinAgentSnapshot<E> {
    Random(AgentRngSnapshot),
    FirstAvailable(AgentRngSnapshot),
    Osla(OslaSnapshot<E>),
}

/// An error identifying which built-in agent could not be restored.
#[derive(Debug)] // Debug: inspect owned errors.
pub enum BuiltinRestoreError<E> {
    Random(RngSnapshotError),
    FirstAvailable(RngSnapshotError),
    Osla(OslaRestoreError<E>),
}
impl<E: fmt::Display> fmt::Display for BuiltinRestoreError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random(error) | Self::FirstAvailable(error) => error.fmt(f),
            Self::Osla(error) => error.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for BuiltinRestoreError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Random(error) | Self::FirstAvailable(error) => Some(error),
            Self::Osla(error) => Some(error),
        }
    }
}

/// Who supplies a player's next move: a human or an AI agent.
#[derive(Debug)] // Debug: inspect controller configuration; each AI owns its progress.
pub enum Controller<A> {
    Human,
    Ai(A),
}

impl<A: Checkpoint> Checkpoint for Controller<A> {
    type Snapshot = ControllerSnapshot<A::Snapshot>;
    type RestoreError = A::RestoreError;
    /// Shared seat borrow; own exact human/AI controller progress.
    fn snapshot(&self) -> Self::Snapshot {
        match self {
            Self::Human => ControllerSnapshot::Human,
            Self::Ai(agent) => ControllerSnapshot::Ai(agent.snapshot()),
        }
    }
    /// Shared seat snapshot; restore a new independent controller.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        match snapshot {
            ControllerSnapshot::Human => Ok(Self::Human),
            ControllerSnapshot::Ai(agent) => Ok(Self::Ai(A::from_snapshot(agent)?)),
        }
    }
}

/// An enum that lets one controller collection contain different built-in agents.
#[derive(Debug)] // Debug: inspect the selected policy and its owned evaluator/RNG.
pub enum BuiltinAgent<E> {
    Random(RandomAgent),
    FirstAvailable(FirstAvailableAgent),
    Osla(OslaAgent<E>),
}

impl<E: Checkpoint> Checkpoint for BuiltinAgent<E> {
    type Snapshot = BuiltinAgentSnapshot<E::Snapshot>;
    type RestoreError = BuiltinRestoreError<E::RestoreError>;
    /// Shared policy borrow; own the selected policy's exact snapshot.
    fn snapshot(&self) -> Self::Snapshot {
        match self {
            Self::Random(agent) => BuiltinAgentSnapshot::Random(agent.snapshot()),
            Self::FirstAvailable(agent) => BuiltinAgentSnapshot::FirstAvailable(agent.snapshot()),
            Self::Osla(agent) => BuiltinAgentSnapshot::Osla(agent.snapshot()),
        }
    }
    /// Shared tagged checkpoint; construct an independent matching policy.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        match snapshot {
            BuiltinAgentSnapshot::Random(agent) => Ok(Self::Random(
                RandomAgent::from_snapshot(agent).map_err(BuiltinRestoreError::Random)?,
            )),
            BuiltinAgentSnapshot::FirstAvailable(agent) => Ok(Self::FirstAvailable(
                FirstAvailableAgent::from_snapshot(agent)
                    .map_err(BuiltinRestoreError::FirstAvailable)?,
            )),
            BuiltinAgentSnapshot::Osla(agent) => Ok(Self::Osla(
                OslaAgent::from_snapshot(agent).map_err(BuiltinRestoreError::Osla)?,
            )),
        }
    }
}

impl<E> Controller<BuiltinAgent<E>> {
    /// Construct a random AI seat with its own explicit seed.
    ///
    /// # Arguments
    /// `seed` is copied into the new agent's independent generator.
    ///
    /// # Returns
    /// Owned random controller, usable in mixed seat collections.
    pub fn random(seed: u64) -> Self {
        Self::Ai(BuiltinAgent::Random(RandomAgent::new(seed)))
    }
    /// Construct a first-available AI seat with its own explicit seed.
    ///
    /// # Arguments
    /// `seed` is copied into this seat's independent generator. The policy does
    /// not draw randomness, but retains the common agent RNG interface.
    ///
    /// # Returns
    /// Owned controller whose progress belongs to this seat.
    pub fn first_available(seed: u64) -> Self {
        // Copied seed; owned controller returned.
        Self::Ai(BuiltinAgent::FirstAvailable(FirstAvailableAgent::new(seed)))
    }

    /// Construct an OSLA seat with its own evaluator and explicit seed.
    ///
    /// # Arguments
    /// `evaluator` transfers ownership to the agent; `seed` is copied into its RNG.
    ///
    /// # Returns
    /// Owned controller with independent evaluator and RNG progress.
    pub fn osla(evaluator: E, seed: u64) -> Self {
        // Consumed evaluator, copied seed.
        Self::Ai(BuiltinAgent::Osla(OslaAgent::new(evaluator, seed)))
    }
}

/// Preserve the selected built-in policy's typed failure.
#[derive(Debug)] // Debug: inspect errors, which transfer to the caller.
pub enum BuiltinAgentError<G, E> {
    Random(EmptyActionList),
    FirstAvailable(EmptyActionList),
    Osla(OslaError<G, E>),
}

impl<G: fmt::Display, E: fmt::Display> fmt::Display for BuiltinAgentError<G, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error, exclusive formatter.
        match self {
            Self::Random(error) | Self::FirstAvailable(error) => error.fmt(f),
            Self::Osla(error) => error.fmt(f),
        }
    }
}
impl<G: std::error::Error + 'static, E: std::error::Error + 'static> std::error::Error
    for BuiltinAgentError<G, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Shared nested-error borrow.
        match self {
            Self::Random(error) | Self::FirstAvailable(error) => Some(error),
            Self::Osla(error) => Some(error),
        }
    }
}

impl<F, E> Agent<F> for BuiltinAgent<E>
where
    F: PublicInformationModel,
    F::State: Clone,
    E: Evaluator<F::State>,
{
    type Error = BuiltinAgentError<F::Error, E::Error>;
    fn rng(&self) -> &AgentRng {
        // Shared controller/RNG borrows.
        match self {
            Self::Random(agent) => <RandomAgent as Agent<F>>::rng(agent),
            Self::FirstAvailable(agent) => <FirstAvailableAgent as Agent<F>>::rng(agent),
            Self::Osla(agent) => <OslaAgent<E> as Agent<F>>::rng(agent),
        }
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        // Exclusive controller/RNG borrows.
        match self {
            Self::Random(agent) => <RandomAgent as Agent<F>>::rng_mut(agent),
            Self::FirstAvailable(agent) => <FirstAvailableAgent as Agent<F>>::rng_mut(agent),
            Self::Osla(agent) => <OslaAgent<E> as Agent<F>>::rng_mut(agent),
        }
    }
    /// Dispatch one decision to the selected policy.
    ///
    /// # Arguments
    /// Rules, observation containing only information that player is allowed to know and offered actions are shared borrows; actor
    /// is copied. Exclusive self access permits policy/evaluator/RNG progress.
    ///
    /// # Returns
    /// A copied offered-action index, or the selected policy's owned typed error.
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        match self {
            Self::Random(agent) => agent
                .choose(model, actor, observation, actions)
                .map_err(BuiltinAgentError::Random),
            Self::FirstAvailable(agent) => agent
                .choose(model, actor, observation, actions)
                .map_err(BuiltinAgentError::FirstAvailable),
            Self::Osla(agent) => agent
                .choose(model, actor, observation, actions)
                .map_err(BuiltinAgentError::Osla),
        }
    }
}
