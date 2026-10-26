//! Record accepted actions and the controller progress after each move.
//!
//! These records can recreate earlier positions. They exclude temporary states
//! examined by an agent while planning.
mod history;
use crate::core::{Decision, PlayerId};
pub use history::{GameHistory, HistoricalView, HistoryCachePolicy, HistoryError};

/// A separate player observation at an earlier move, or an error explaining why rebuilding it failed.
pub type HistoricalViewResult<O, M, G> = Result<HistoricalView<O, M>, HistoryError<G>>;

/// The controller kind and saved agent progress at one recorded position.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: independent nested snapshot; PartialEq/Eq: exact identity/progress when supported.
pub enum ControllerSnapshot<C> {
    Human,
    Ai(C),
}

/// Accepted live move and its acting controller's resulting progress.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: duplicate owned move/checkpoint data; PartialEq/Eq: compare complete records when supported.
pub struct RecordedMove<M, C> {
    pub actor: PlayerId,
    pub action: M,
    pub decision: Decision,
    pub controller_after: ControllerSnapshot<C>,
}
