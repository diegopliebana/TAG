//! Platform adapter exposing shared Rust sessions to browser JavaScript.
mod cantstop;
mod engine;
mod errors;
mod games;
mod learned;
mod loveletter;
mod mcts;
mod protocol;
mod runtime;
mod seeds;
mod stochastic;
pub use engine::BrowserEngine;

#[cfg(test)]
mod tests;
