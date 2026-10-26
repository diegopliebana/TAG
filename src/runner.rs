//! Run sequential games by alternating agent choices and rule updates.
//!
//! The runner obtains actions directly from state and creates the next player's observation,
//! asks that player's agent for a list index, checks it, and applies the move.
//! Recording is optional. Earlier accepted moves remain in place if a later call fails.

use crate::agents::Agent;
use crate::core::{Decision, ForwardModel, PlayerId};
use std::fmt;

/// Owned player view and choices for one sequential decision.
pub(crate) struct OfferedDecision<F: ForwardModel> {
    pub observation: F::Observation,
    pub actions: Vec<F::Action>,
}

/// Borrow rules/state and copy the actor; return owned permitted input or an execution error.
pub(crate) fn offered_decision<F: ForwardModel, A>(
    model: &F,        // Shared rules borrow.
    state: &F::State, // Shared state borrow.
    actor: PlayerId,  // Copied acting seat.
) -> Result<OfferedDecision<F>, RunnerError<F::Error, A>> {
    let actions = decision_actions(model, state, actor)?;
    let observation = model.observe(state, actor).map_err(RunnerError::Game)?;
    Ok(OfferedDecision {
        observation,
        actions,
    })
}

/// Obtain the acting player's choices without constructing an observation.
pub(crate) fn decision_actions<F: ForwardModel, A>(
    model: &F,
    state: &F::State,
    actor: PlayerId,
) -> Result<Vec<F::Action>, RunnerError<F::Error, A>> {
    let actions = model
        .available_actions(state, actor)
        .map_err(RunnerError::Game)?;
    if actions.is_empty() {
        return Err(RunnerError::NoAvailableActions { player: actor });
    }
    Ok(actions)
}

/// Borrow rules/controller, copy actor and consume decision input; return the owned chosen action.
pub(crate) fn choose_action<F: ForwardModel, A: Agent<F>>(
    model: &F,                 // Shared rules borrow.
    agent: &mut A,             // Exclusive controller borrow.
    actor: PlayerId,           // Copied acting seat.
    input: OfferedDecision<F>, // Consumed view and action list.
) -> Result<F::Action, RunnerError<F::Error, A::Error>> {
    let index = agent
        .choose(model, actor, &input.observation, &input.actions)
        .map_err(|error| RunnerError::Agent {
            player: actor,
            error,
        })?;
    let action_count = input.actions.len();
    // Transfer the selected move out of the list; actions can own data.
    input
        .actions
        .into_iter()
        .nth(index)
        .ok_or(RunnerError::InvalidSelection {
            player: actor,
            index,
            action_count,
        })
}

/// Execution failures are separate from player results and scores.
#[derive(Debug, PartialEq, Eq)] // PartialEq/Eq: compare failure details when both nested errors support equality.
// Errors move to the caller.
pub enum RunnerError<G, A> {
    MissingController {
        player: PlayerId,
    },
    NoAvailableActions {
        player: PlayerId,
    },
    InvalidSelection {
        player: PlayerId,
        index: usize,
        action_count: usize,
    },
    Game(G),
    Agent {
        player: PlayerId,
        error: A,
    },
}

impl<G: fmt::Display, A: fmt::Display> fmt::Display for RunnerError<G, A> {
    fn fmt(
        &self,                              // Shared error borrow; borrowed for formatting.
        formatter: &mut fmt::Formatter<'_>, // Exclusive output borrow; inferred lifetime.
    ) -> fmt::Result {
        // Success or owned formatting error.
        match self {
            Self::MissingController { player } => {
                write!(formatter, "no controller for player {}", player.index())
            }
            Self::NoAvailableActions { player } => write!(
                formatter,
                "player {} must act but was offered no actions",
                player.index()
            ),
            Self::InvalidSelection {
                player,
                index,
                action_count,
            } => write!(
                formatter,
                "player {} selected index {index} from {action_count} actions",
                player.index()
            ),
            Self::Agent { player, error } => write!(
                formatter,
                "player {} controller failed: {error}",
                player.index()
            ),
            Self::Game(error) => write!(formatter, "game rule operation failed: {error}"),
        }
    }
}

impl<G: std::error::Error + 'static, A: std::error::Error + 'static> std::error::Error
    for RunnerError<G, A>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Shared self borrow; returns a shared nested-error borrow.
        // 'static bounds nested errors' internal references; the returned borrow still lasts only as long as self.
        match self {
            Self::Game(error) => Some(error),
            Self::Agent { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Run from the supplied state until normal completion or an execution error.
/// Agents are indexed by seat. Accepted earlier moves remain in state on failure.
///
/// # Arguments
/// `model` is passed by a shared (read-only) reference; `state` and seat-indexed `agents` are exclusively
/// borrowed and remain owned by the caller. Agents receive rules and views only.
///
/// # Returns
/// `Ok(())` means normal completion. Earlier accepted moves and live agent RNG
/// progress are retained if execution stops.
///
/// # Errors
/// Missing controllers, empty offers, invalid choices, rule failures and
/// typed agent failures are reported separately. Agent failure retains the state
/// from before the current decision.
/// An internal rule failure aborts the run and may leave partial transition effects;
/// discard that state rather than retrying it. Invalid action selection is rejected
/// before calling the forward model.
pub fn run_game<F: ForwardModel, A: Agent<F>>(
    model: &F,            // Shared rules borrow; caller retains the model.
    state: &mut F::State, // Exclusive state borrow; caller retains it for inspection.
    agents: &mut [A],     // Exclusive controller slice borrow; each seat owns its controller.
) -> Result<(), RunnerError<F::Error, A::Error>> {
    run_game_impl(model, state, agents, |_, _, _, _, _| {})
}

/// Run a match and inspect each accepted action without requiring agent checkpoints.
///
/// # Arguments
/// `model` supplies read-only rules; `state` and seat-indexed `agents` are mutated
/// during play. `accepted` receives the updated state, acting player, owned action
/// and read-only agent after each successful transition. It can collect experiment
/// traces and statistics. The callback is never called for a rejected action.
///
/// # Errors
/// The same execution errors as `run_game`. Earlier accepted moves and callbacks
/// remain in place if a later decision fails.
pub fn run_game_observed<F, A, R>(
    model: &F,
    state: &mut F::State,
    agents: &mut [A],
    accepted: R,
) -> Result<(), RunnerError<F::Error, A::Error>>
where
    F: ForwardModel,
    A: Agent<F>,
    R: FnMut(&F, &F::State, PlayerId, F::Action, &A),
{
    run_game_impl(model, state, agents, accepted)
}

/// Run headlessly while transferring accepted moves to a caller-owned recorder.
///
/// # Arguments
/// Rules are passed by a shared (read-only) reference; state/agents accessed through an exclusive mutable reference. The callback is
/// consumed and receives owned actions and acting-agent checkpoints after acceptance.
/// Caller records initial parameters/state/controller snapshots before this call.
///
/// # Returns
/// Normal completion or the same owned execution failures as run_game. Previously
/// accepted records and agent progress survive failures. Recording requires the
/// optional Checkpoint capability; the unrecorded entry point keeps its bounds.
/// Records capture progress immediately after accepted moves. To start a new
/// recording after a failed decision, capture the current state/agent snapshots
/// as its root, including progress retained from that attempt.
pub fn run_game_recorded<F, A, R>(
    model: &F,
    state: &mut F::State,
    agents: &mut [A],
    mut record: R,
) -> Result<(), RunnerError<F::Error, A::Error>>
where
    F: ForwardModel,
    A: Agent<F> + crate::core::checkpoint::Checkpoint,
    R: FnMut(crate::recording::RecordedMove<F::Action, A::Snapshot>),
{
    use crate::recording::{ControllerSnapshot, RecordedMove};
    run_game_impl(
        model,
        state,
        agents,
        |model, state, actor, action, agent| {
            record(RecordedMove {
                actor,
                action,
                decision: model.next_player(state),
                controller_after: ControllerSnapshot::Ai(agent.snapshot()),
            });
        },
    )
}

/// Execute the common loop; recording runs only after a live move is accepted.
fn run_game_impl<F: ForwardModel, A: Agent<F>, R>(
    model: &F,
    state: &mut F::State,
    agents: &mut [A],
    mut accepted: R,
) -> Result<(), RunnerError<F::Error, A::Error>>
where
    R: FnMut(&F, &F::State, PlayerId, F::Action, &A),
{
    // Completion or owned execution error.
    loop {
        let actor = match model.next_player(state) {
            Decision::Finished => return Ok(()),
            Decision::Player(player) => player,
        };
        let agent = agents
            .get_mut(actor.index())
            .ok_or(RunnerError::MissingController { player: actor })?; // Exclusive borrow of just the acting controller.
        let input = offered_decision(model, state, actor)?;
        let action = choose_action(model, agent, actor, input)?;
        model
            .next(state, actor, &action)
            .map_err(RunnerError::Game)?;
        accepted(model, state, actor, action, agent);
    }
}
