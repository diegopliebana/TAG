//! One-step lookahead (OSLA): try each offered move on an independent state copy.
//!
//! The evaluator rates each resulting position for the original acting player. The
//! agent chooses randomly among values within the configured tolerance of the best.
//! Evaluation is synchronous inside the candidate loop; there is no Python batch
//! request/resume support in this agent.
use super::{Agent, EmptyActionList};
use crate::core::checkpoint::Checkpoint;
use crate::core::evaluation::{EVALUATION_TIE_TOLERANCE, Evaluator};
use crate::core::random::AgentRng;
use crate::core::random::{AgentRngSnapshot, RngSnapshotError};
use crate::core::{PlayerId, PublicInformationModel};
use std::fmt;

/// Evaluator settings/memory and RNG progress saved between accepted moves.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: independent evaluator snapshot; PartialEq/Eq: compare contents when supported.
pub struct OslaSnapshot<E> {
    pub evaluator: E,
    pub rng: AgentRngSnapshot,
}

/// Distinguish failure to restore the evaluator from failure to restore the RNG.
#[derive(Debug)] // Debug: inspect transferred restoration errors.
pub enum OslaRestoreError<E> {
    Evaluator(E),
    Rng(RngSnapshotError),
}
impl<E: fmt::Display> fmt::Display for OslaRestoreError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => error.fmt(f),
            Self::Rng(error) => error.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for OslaRestoreError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluator(error) => Some(error),
            Self::Rng(error) => Some(error),
        }
    }
}

/// A planning failure, separate from the outcome of the real game.
#[derive(Debug)] // Errors are transferred; nested errors may own data.

pub enum OslaError<G, E> {
    EmptyActions,
    Reconstruction(G),
    Transition { candidate: usize, error: G },
    Evaluation { candidate: usize, error: E },
    NonfiniteValue { candidate: usize, value: f64 },
}

impl<G: fmt::Display, E: fmt::Display> fmt::Display for OslaError<G, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error borrow; exclusive formatter borrow.
        match self {
            Self::EmptyActions => EmptyActionList.fmt(f),
            Self::Reconstruction(error) => write!(f, "simulation reconstruction failed: {error}"),
            Self::Transition { candidate, error } => {
                write!(f, "candidate {candidate} transition failed: {error}")
            }
            Self::Evaluation { candidate, error } => {
                write!(f, "candidate {candidate} evaluation failed: {error}")
            }
            Self::NonfiniteValue { candidate, value } => {
                write!(f, "candidate {candidate} evaluation is not finite: {value}")
            }
        }
    }
}

impl<G: std::error::Error + 'static, E: std::error::Error + 'static> std::error::Error
    for OslaError<G, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Return a shared nested-error borrow; the OSLA error retains ownership.
        match self {
            Self::Reconstruction(error) | Self::Transition { error, .. } => Some(error),
            Self::Evaluation { error, .. } => Some(error),
            Self::EmptyActions | Self::NonfiniteValue { .. } => None,
        }
    }
}

/// Try each offered move on a separate copy and rate the result for the player choosing now.
#[derive(Debug)] // Each controller owns its evaluator and RNG stream.

pub struct OslaAgent<E> {
    evaluator: E,
    rng: AgentRng,
}

impl<E: Checkpoint> Checkpoint for OslaAgent<E> {
    type Snapshot = OslaSnapshot<E::Snapshot>;
    type RestoreError = OslaRestoreError<E::RestoreError>;
    /// Shared agent borrow; own the evaluator snapshot and copy exact RNG progress.
    fn snapshot(&self) -> Self::Snapshot {
        OslaSnapshot {
            evaluator: self.evaluator.snapshot(),
            rng: self.rng.snapshot(),
        }
    }
    /// Shared snapshot borrow; restore independent persistent evaluator/RNG progress.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Ok(Self {
            evaluator: E::from_snapshot(&snapshot.evaluator)
                .map_err(OslaRestoreError::Evaluator)?,
            rng: AgentRng::from_snapshot(&snapshot.rng).map_err(OslaRestoreError::Rng)?,
        })
    }
}

impl<E> OslaAgent<E> {
    /// Create a controller with its own evaluator and explicitly seeded RNG.
    ///
    /// # Arguments
    /// `evaluator` is consumed; `seed` is copied.
    ///
    /// # Returns
    /// An owned agent with an independent RNG stream.
    pub fn new(evaluator: E, seed: u64) -> Self {
        Self {
            evaluator,
            rng: AgentRng::new(seed),
        }
    }
}

impl<F, E> Agent<F> for OslaAgent<E>
where
    F: PublicInformationModel,
    F::State: Clone,
    E: Evaluator<F::State>,
{
    type Error = OslaError<F::Error, E::Error>;

    fn rng(&self) -> &AgentRng {
        &self.rng
    } // Shared borrowed RNG.
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    } // Exclusive borrowed RNG.

    /// Simulate each candidate independently, then randomly select a near-best move.
    ///
    /// # Arguments
    /// Rules, observation and actions are passed by a shared (read-only) reference. `actor` is copied and
    /// remains the evaluation perspective even when the simulated turn changes.
    /// The exclusive agent borrow permits evaluator/RNG progress updates.
    ///
    /// # Returns
    /// `Ok(index)` selects an entry in the supplied nonempty list.
    /// Reconstructs one base state and clones it once per candidate; each candidate
    /// uses one transition and one evaluation.
    /// Uniform choice includes values within 0.001 of the true maximum. A unique
    /// eligible move keeps the RNG at its current position.
    ///
    /// # Errors
    /// Empty actions, typed reconstruction/transition/evaluator failures or a NaN or infinite value.
    /// Evaluator changes made before failure are retained; simulation copies are discarded.
    fn choose(
        &mut self,                    // Exclusive controller borrow.
        model: &F,                    // Shared immutable rules.
        actor: PlayerId,              // Copied original perspective.
        observation: &F::Observation, // Shared permitted player view.
        actions: &[F::Action],        // Shared candidate list.
    ) -> Result<usize, Self::Error> {
        if actions.is_empty() {
            return Err(OslaError::EmptyActions);
        }
        let base = model
            .simulation_state(observation)
            .map_err(OslaError::Reconstruction)?;
        let mut values = Vec::with_capacity(actions.len());
        let mut best = f64::NEG_INFINITY;
        for (candidate, action) in actions.iter().enumerate() {
            // Every move starts from the same position in an independently owned branch.
            let mut branch = base.clone();
            model.next(&mut branch, actor, action).map_err(|error| {
                // Identify the offered move whose simulated application failed.
                OslaError::Transition { candidate, error }
            })?;
            let value = self.evaluator.evaluate(&branch, actor).map_err(|error| {
                // Identify the successor whose evaluation failed.
                OslaError::Evaluation { candidate, error }
            })?;
            if !value.is_finite() {
                return Err(OslaError::NonfiniteValue { candidate, value });
            }
            best = best.max(value);
            values.push(value);
        }
        // Compare every value to the actual maximum. Comparing successive values
        // could wrongly chain together several small differences into a large one.
        let eligible = || {
            values.iter().enumerate().filter_map(|(candidate, &value)| {
                // Keep this move only if its estimate is close enough to the best.
                (best - value <= EVALUATION_TIE_TOLERANCE).then_some(candidate)
            })
        };
        // The finite maximum guarantees at least one eligible move.
        let selected = self
            .rng
            .index(eligible().count())
            .expect("the best move qualifies");
        Ok(eligible()
            .nth(selected)
            .expect("selected eligible move exists"))
    }
}
