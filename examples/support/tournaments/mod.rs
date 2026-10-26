//! Tournament scheduling and analysis reuse the application's native match runner.
mod config;
mod execution;
mod logs;
mod ranking;
mod schedule;
use crate::matches::Result;
pub use config::TournamentConfig;
pub use execution::{TournamentReport, run};

#[cfg(test)]
mod tests;
