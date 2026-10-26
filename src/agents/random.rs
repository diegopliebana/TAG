//! An agent that chooses uniformly from the offered action list.
//!
//! Uniform means every list entry has the same probability. The agent receives the
//! same observation and rules as other agents, even though this policy ignores them.
use super::{Agent, EmptyActionList};
use crate::core::checkpoint::Checkpoint;
use crate::core::random::{AgentRng, AgentRngSnapshot, RngSnapshotError};
use crate::core::{ForwardModel, PlayerId};

/// Game-independent agent owning its complete random decision state.
#[derive(Debug)] // Debug: inspect the owned generator; exact copying uses Checkpoint.
pub struct RandomAgent {
    rng: AgentRng,
}
impl RandomAgent {
    /// Construct an independently seeded random controller.
    ///
    /// # Arguments
    /// `seed` is copied into this agent's owned RNG; no platform clock is read here.
    ///
    /// # Returns
    /// Owned agent with its own generator stream at its initial position.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: AgentRng::new(seed),
        }
    }
}
impl Checkpoint for RandomAgent {
    type Snapshot = AgentRngSnapshot;
    type RestoreError = RngSnapshotError;
    /// Borrow the agent and return a copied exact RNG checkpoint.
    fn snapshot(&self) -> Self::Snapshot {
        self.rng.snapshot()
    }
    /// Restore an independent agent from a borrowed checkpoint.
    ///
    /// # Arguments
    /// `snapshot` is passed by a shared (read-only) reference; restoring does not advance the source RNG.
    ///
    /// # Returns
    /// Owned agent whose next choices match the saved continuation.
    ///
    /// # Errors
    /// Unsupported RNG version or invalid generator position.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Ok(Self {
            rng: AgentRng::from_snapshot(snapshot)?,
        })
    }
}
impl<F: ForwardModel> Agent<F> for RandomAgent {
    type Error = EmptyActionList;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    /// Select uniformly from the supplied action list using this agent's RNG.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference to advance private random progress.
    /// `_model` borrows rules; `_actor` identifies the acting seat; `_observation`
    /// borrows its permitted current game state. This policy receives but does not
    /// inspect these inputs. `actions` borrows the available/offered choices.
    ///
    /// # Returns
    /// An index into `actions`, with equal probability per entry. One choice returns
    /// zero without consuming randomness, following the shared sampling contract.
    ///
    /// # Errors
    /// Empty lists return EmptyActionList without consuming randomness.
    fn choose(
        &mut self,
        _model: &F,
        _actor: PlayerId,
        _observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        self.rng.index(actions.len()).ok_or(EmptyActionList)
    }
}
