//! Result and proposal types for calculating a move before applying it.
//!
//! A prepared proposal holds the selected action and before/after agent checkpoints.
//! It is not an accepted game move until commit validates and applies it.
use super::{RecordingError, Session};
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::{ForwardModel, PlayerId};
use crate::recording::GameHistory;
use std::sync::Arc;
/// A successful value or an error identifying execution, history or agent-restoration failure.
pub type RecordingResult<T, G, A, C> = Result<T, RecordingError<G, A, C>>;

/// A calculated move ready for later validation/application, or an error explaining failure.
pub type PreparedAiResult<F, A> = RecordingResult<
    PreparedAi<<F as ForwardModel>::Action, <A as Checkpoint>::Snapshot>,
    <F as ForwardModel>::Error,
    <A as Agent<F>>::Error,
    <A as Checkpoint>::RestoreError,
>;

/// Owned live session and its recording when a wrapper is consumed.
pub struct RecordingParts<F: ForwardModel, A: Agent<F> + Checkpoint> {
    pub session: Session<F, A>,
    pub history: GameHistory<F::State, F::Action, A::Snapshot>,
}

/// A calculated move with a separate temporary agent checkpoint.
/// If the proposal is dropped instead of committed, the live game and agents stay unchanged.
pub struct PreparedAi<M, C> {
    pub(super) origin: Arc<()>,
    pub(super) position_version: u64,
    pub(super) actor: PlayerId,
    pub(super) action: M,
    pub(super) before: C,
    pub(super) after: C,
}
impl<M, C> PreparedAi<M, C> {
    /// Read the proposed action through a reference; the proposal keeps storing it.
    pub fn action(&self) -> &M {
        &self.action
    }
    /// Return the version of the game position used to calculate this proposal.
    pub fn position_version(&self) -> u64 {
        self.position_version
    }
    /// Return the player who would make this move.
    pub fn actor(&self) -> PlayerId {
        self.actor
    }
}
