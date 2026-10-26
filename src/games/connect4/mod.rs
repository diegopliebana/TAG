//! Configurable Connect 4 rules, owned states and independent player observations.
mod evaluation;
mod forward_model;
mod learning;
mod observation;
mod state;
mod types;

pub use evaluation::Connect4Evaluator;
pub use forward_model::Connect4ForwardModel;
pub use observation::Connect4Observation;
pub use state::Connect4State;
pub use types::{Cell, Connect4Error, Connect4Parameters, DropPiece, PLAYER_COUNT};

#[cfg(test)]
mod tests;

#[cfg(feature = "replay")]
mod persistence;
