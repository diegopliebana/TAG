//! A baseline agent that always chooses the first offered action.
//!
//! Useful for checking that complete games run. It retains the common agent RNG
//! interface but does not consume randomness when choosing.
use super::{Agent, EmptyActionList};
use crate::core::checkpoint::Checkpoint;
use crate::core::random::AgentRng;
use crate::core::random::{AgentRngSnapshot, RngSnapshotError};
use crate::core::{ForwardModel, PlayerId};

#[derive(Debug)] // Debug: inspect configuration.

pub struct FirstAvailableAgent {
    rng: AgentRng,
}
impl Checkpoint for FirstAvailableAgent {
    type Snapshot = AgentRngSnapshot;
    type RestoreError = RngSnapshotError;
    /// Shared controller borrow; copy its exact RNG progress into an owned snapshot.
    fn snapshot(&self) -> Self::Snapshot {
        self.rng.snapshot()
    }
    /// Shared snapshot borrow; restore an independently owned baseline controller.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Ok(Self {
            rng: AgentRng::from_snapshot(snapshot)?,
        })
    }
}
impl FirstAvailableAgent {
    /// Construct a baseline with its own explicitly seeded RNG.
    ///
    /// # Arguments
    /// `seed` is copied to initialize the owned generator.
    ///
    /// # Returns
    /// An owned controller with an independent RNG stream.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: AgentRng::new(seed),
        }
    }
}
impl<F: ForwardModel> Agent<F> for FirstAvailableAgent {
    type Error = EmptyActionList;
    fn rng(&self) -> &AgentRng {
        &self.rng
    } // Shared borrowed RNG.
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    } // Exclusive borrowed RNG.
    /// Select the first offered action.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; `actions` is a shared borrowed list.
    /// Shared rules/view and copied actor are unused by this baseline.
    ///
    /// # Returns
    /// `Ok(0)` selects the first entry in a nonempty list.
    ///
    /// # Errors
    /// An empty list returns `EmptyActionList`.
    fn choose(
        &mut self,
        _model: &F,
        _actor: PlayerId,
        _observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, EmptyActionList> {
        if actions.is_empty() {
            return Err(EmptyActionList);
        }
        Ok(0)
    }
}
