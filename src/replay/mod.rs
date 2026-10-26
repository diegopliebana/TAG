//! Read and write game states and move recordings as versioned JSON files.
//!
//! Persistence means keeping data beyond the current program run. File values are
//! read into separate structures, checked, then converted into valid game state.
//! PersistableGame supports state files; ReplayGame adds the supported agents needed
//! to verify a recording and continue play with their original settings/progress.
mod codec;
mod common;
mod controllers;
mod file_validation;
mod format;
mod replay_file;
mod state_file;
mod validation;
use crate::core::ForwardModel;
pub use codec::*;
pub use controllers::{BuiltinCodec, ReplayCodec};
pub use format::*;
pub(crate) use validation::{
    placement_counters, placement_validation_work, validate_placement, validate_placement_outcome,
};

/// Optional methods for saving a game state and checking a loaded state before using it.
/// File data has a defined field layout/version. Restore methods check the game rules
/// and all state information before returning a usable State.
pub trait PersistableGame: ForwardModel {
    type SavedState: serde::Serialize + serde::de::DeserializeOwned;
    /// Checked storage-unit estimate for import resource limits (cells for current games).
    ///
    /// # Arguments
    /// `parameters` is a shared configuration borrow.
    ///
    /// # Returns
    /// `Some(count)` gives copied storage units; `None` reports checked arithmetic overflow.
    fn storage_units(parameters: &Self::Parameters) -> Option<usize>;
    /// Checked work estimate for validating standalone/root state metadata.
    ///
    /// # Arguments
    /// `parameters` and decoded `data` are shared borrows.
    ///
    /// # Returns
    /// Copied conservative state-validation work estimate, or `None` on overflow.
    fn validation_work(parameters: &Self::Parameters, data: &Self::SavedState) -> Option<usize>;
    const GAME_ID: &'static str;
    const RULES_VERSION: u32;
    /// Borrow full state and return separate data values prepared for a file or message; game invariants are preserved by restore_state.
    ///
    /// # Arguments
    /// `self` and full simulation `state` are shared borrows.
    ///
    /// # Returns
    /// Owned complete file-state data with independently owned mutable contents.
    fn state_data(&self, state: &Self::State) -> Self::SavedState;
    /// Borrow selected parameters and decoded data; return independently owned validated state.
    /// Invalid board/progress/outcome data returns a descriptive owned error.
    ///
    /// # Arguments
    /// `self`, selected `parameters` and decoded `data` are shared borrows.
    /// Parameters supply the saved ruleset; data supplies all mutable simulation facts.
    ///
    /// # Returns
    /// Independently owned state after game-specific invariant checks.
    ///
    /// # Errors
    /// Invalid configuration, references, counters, phase/outcome or other game invariants.
    fn restore_state(
        &self,
        parameters: &Self::Parameters,
        data: &Self::SavedState,
    ) -> Result<Self::State, PersistenceError>;
}

/// Extra support for verifying recordings and restoring their supported controller types.
/// PersistableGame alone is sufficient to save/load a position without a recording.
pub trait ReplayGame: PersistableGame + Clone {
    /// Whether accepted moves carry realized chance data in the replay format.
    const HAS_CHANCE: bool = false;
    /// Describe an action's realized public chance after the checked transition.
    /// `action` and `after` are read-only; returned JSON owns its data. Deterministic
    /// games return None. Import verifies this against the replayed transition.
    fn chance_event(
        &self,
        _action: &Self::Action,
        _after: &Self::State,
    ) -> Option<serde_json::Value> {
        None
    }
    /// Evaluator configuration identified by the supported replay version.
    type SavedEvaluator: crate::core::evaluation::Evaluator<Self::State>
        + crate::core::checkpoint::Checkpoint<Snapshot = ()>;
    /// Checked work estimate for one live transition or one OSLA decision.
    ///
    /// # Arguments
    /// `parameters` is passed by a shared (read-only) reference; copied `osla` selects an OSLA decision
    /// estimate when true and a single live transition when false.
    ///
    /// # Returns
    /// Copied conservative work estimate, or `None` on arithmetic overflow.
    fn verification_work(parameters: &Self::Parameters, osla: bool) -> Option<usize>;
}
