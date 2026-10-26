//! Game rules, agents and tools used by Rust programs and the browser.
//!
//! A state stores a game position; an observation contains what one player can see.
//! Agents choose actions and forward models apply their effects. The learning module
//! turns observations into arrays. The separate adapters/python package exposes
//! these operations to Python; learning algorithms remain outside the game rules.
//! For an introduction and reading paths, see docs/start-here.md in the repository.

pub mod agents;
pub mod core;
pub mod games;
#[cfg(feature = "inference")]
pub mod inference;
pub mod learning;
pub mod recording;
pub mod runner;
pub mod search;
pub mod session;

#[cfg(feature = "replay")]
pub mod replay;
