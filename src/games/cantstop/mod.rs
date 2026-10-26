//! Can't Stop: public dice/column facts and separately owned chance randomness.
//!
//! Rules follow the published game. Public Java TAG supplies the implementation
//! reference (games/cantstop, MIT); documented rule discrepancies are corrected.
mod evaluation;
mod forward_model;
mod learning;
#[cfg(feature = "replay")]
mod persistence;
mod state;
mod types;
#[cfg(feature = "replay")]
pub use persistence::{CantStopReplayEvaluator, CantStopSavedState};

pub use evaluation::{CantStopBaseline, CantStopProgress};
pub use forward_model::{CantStopForwardModel, RollEvent};
pub use state::{CantStopObservation, CantStopPosition, CantStopState};
pub use types::{
    Allocation, COLUMN_COUNT, CantStopAction, CantStopError, CantStopParameters, CantStopPhase,
    CantStopRules,
};

#[cfg(test)]
mod tests;
