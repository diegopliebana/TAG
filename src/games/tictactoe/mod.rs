//! Configurable square Tic-Tac-Toe rules and independently owned player views.
//!
//! This module exposes the game's public API; sibling modules divide responsibilities.
//! Internal access uses pub(super), limited to this game module and its children.

mod evaluation;
mod forward_model;
mod learning;
mod observation;
mod state;
mod types;

pub use evaluation::TicTacToeEvaluator;
pub use forward_model::TicTacToeForwardModel;
pub use observation::TicTacToeObservation;
pub use state::TicTacToeState;
pub use types::{Cell, GRID_SIZE, PLAYER_COUNT, PlaceMark, TicTacToeError, TicTacToeParameters};

#[cfg(test)]
mod tests;

#[cfg(feature = "replay")]
mod persistence;
