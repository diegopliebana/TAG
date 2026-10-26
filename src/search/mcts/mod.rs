//! Open-loop Monte Carlo tree search with native or resumable external evaluation.
//!
//! Every iteration reconstructs a simulation from a permitted root snapshot.
//! Completed returns contain one component per player. Tree traversal and rollouts
//! share the same transition counter, including simulations requested by heuristics.
#![doc = include_str!("../../../docs/reference/agents/mcts-api.md")]
mod budgets;
mod engine;
mod evaluation;
mod external;
#[cfg(feature = "mcts-json")]
mod json;
mod seed;
mod selection;
mod settings;
mod tree;
pub use budgets::{Clock, Counters, SimulationContext, StopReason, SystemClock};
pub use engine::{MctsAgent, MctsSnapshot, RootActionReport, SearchReport};
pub use evaluation::{
    ActionEvaluator, EvaluationFuture, EvaluatorRegistry, NativeEvaluators, PlayerEvaluator,
    StateEvaluator, StateValues, complete_values,
};
pub use external::*;
pub use settings::*;
use std::fmt;

/// Obtain one seed from operating-system randomness, using the same source as
/// automatic MCTS construction. Applications may use it for an experiment seed.
/// Unsupported platforms and operating-system failures return an error.
pub fn automatic_seed() -> Result<u64> {
    seed::system_seed()
}

/// Search failures are separate from game results. Messages preserve game/evaluator context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Game(String),
    Evaluation(String),
    ResourceLimit,
    BudgetExhausted,
    Cancelled,
    UnsupportedCheckpoint(String),
    Seed(String),
    Overflow,
    EvaluationRequired,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid MCTS input: {s}"),
            Self::Game(s) => write!(f, "MCTS simulation failed: {s}"),
            Self::Evaluation(s) => write!(f, "MCTS evaluation failed: {s}"),
            Self::UnsupportedCheckpoint(s) => write!(f, "MCTS checkpoint unsupported: {s}"),
            Self::Seed(s) => write!(f, "MCTS automatic seeding failed: {s}"),
            Self::ResourceLimit => write!(f, "MCTS resource limit reached"),
            Self::BudgetExhausted => write!(f, "MCTS simulation budget exhausted"),
            Self::Cancelled => write!(f, "MCTS decision cancelled"),
            Self::Overflow => write!(f, "MCTS counter or size overflow"),
            Self::EvaluationRequired => write!(
                f,
                "MCTS requires external evaluation; use the resumable search interface"
            ),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
