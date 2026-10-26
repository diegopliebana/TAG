//! Programs that choose an action for a player.
//!
//! Every agent receives the game rules, acting seat, player observation and offered
//! action list. It returns an index into that list, not a permanent network-output ID.
//! Each agent owns its RNG, even when its policy does not need to draw random numbers.
mod first_available;
mod osla;
mod osla_python;
mod random;
use crate::core::random::AgentRng;
use crate::core::{ForwardModel, PlayerId};
pub use crate::search::mcts::MctsAgent;
pub use first_available::FirstAvailableAgent;
pub use osla::{OslaAgent, OslaError, OslaRestoreError, OslaSnapshot};
mod sampled_osla;
pub(crate) use osla_python::validate_values;
pub use osla_python::{
    BatchEvaluator, CandidateBatch, EvaluationLimits, EvaluationWork, EvaluatorIdentity,
    ExternalEvaluation, OpenLinesBatchEvaluator, OslaPythonAgent, OslaPythonConfig,
    OslaPythonError, OslaPythonSnapshot, PreparedEvaluationAgent,
};
pub use random::RandomAgent;
pub use sampled_osla::{SampledOslaAgent, SampledOslaSnapshot};

/// An action selection request must contain at least one available move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: fieldless error; PartialEq/Eq: exact variant equality.
pub struct EmptyActionList;

impl std::fmt::Display for EmptyActionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "action selection requires a nonempty action list")
    }
}
impl std::error::Error for EmptyActionList {}

/// All agents receive rules, the acting player, that player's observation and the action list.
/// Each agent has a separate RNG. Returned indices refer to this call's action list.
pub trait Agent<F: ForwardModel> {
    /// Controller failures, separate from game outcomes and runner failures.
    type Error;
    /// Borrow this controller's RNG for inspection.
    fn rng(&self) -> &AgentRng;
    /// Exclusively borrow this controller's RNG; sampling changes its live progress.
    fn rng_mut(&mut self) -> &mut AgentRng;

    /// Choose one entry from the action list supplied for this decision.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference so the agent can update
    /// its memory/RNG. `model` is read-only rule access; `actor` is the player choosing.
    /// `observation` contains only that player's allowed information. `actions` is a
    /// read-only slice of available choices, in the order provided by the game.
    ///
    /// # Returns
    /// Ok(index) selects that list entry. For [drop_in_2, drop_in_5], index 1 chooses column
    /// 5. This is not a stable neural-policy ID; it only indexes this call's list.
    ///
    /// # Errors
    /// Return the agent's Error if it cannot choose, including an empty list. A forced
    /// pass must be an explicit game action rather than an empty-list convention.
    fn choose(
        &mut self,                    // Exclusive controller borrow.
        model: &F,                    // Shared rules borrow.
        actor: PlayerId,              // Copied seat.
        observation: &F::Observation, // Shared permitted view.
        actions: &[F::Action],        // Shared actions.
    ) -> Result<usize, Self::Error>; // Copied index or owned controller error.

    /// Prepare a choice that may yield while retaining its temporary search state.
    /// The default calls ordinary move selection. Search agents may override this
    /// to let a single-threaded application process input between iterations.
    /// Dropping the future releases its temporary work; callers decide whether
    /// the agent itself is a live controller or an independently restored copy.
    fn choose_async<'a>(
        &'a mut self,
        model: &'a F,
        actor: PlayerId,
        observation: &'a F::Observation,
        actions: &'a [F::Action],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, Self::Error>> + 'a>> {
        Box::pin(async move { self.choose(model, actor, observation, actions) })
    }
}
