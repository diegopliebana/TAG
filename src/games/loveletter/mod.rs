//! Classic Love Letter from the selected Z-Man rulebook (16 cards, 2–4 players).
//!
//! Rules and exact snapshots are native capabilities. Observations own only the
//! selected player's permitted information. Planning reconstructs hypothetical
//! rounds from observations; this game does not implement public-state reconstruction.
mod evaluation;
mod events;
mod forward_model;
mod learning;
#[cfg(feature = "replay")]
mod persistence;
#[cfg(feature = "replay")]
pub use persistence::{LoveLetterSavedState, LoveLetterSnapshot};
mod simulation;
mod state;
mod types;
pub use evaluation::LoveLetterHeuristic;
pub use events::*;
pub use forward_model::LoveLetterForwardModel;
pub use simulation::DEFAULT_SAMPLING_BUDGET;
pub use state::{LoveLetterObservation, LoveLetterPublicState, LoveLetterState};
pub use types::*;

#[cfg(test)]
mod simulation_tests;
#[cfg(test)]
mod tests;
