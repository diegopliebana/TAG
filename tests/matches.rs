//! The match runner is application code; exercise it without a nested Cargo process.
#![cfg(all(feature = "mcts-json", feature = "replay"))]
#[allow(dead_code, unused_imports)]
#[path = "../examples/support/matches/mod.rs"]
mod matches;
#[allow(dead_code)]
#[path = "../examples/support/tournaments/mod.rs"]
mod tournaments;
