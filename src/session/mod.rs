//! Handle one human or AI decision at a time.
//!
//! Unlike run_game, a Session waits for its caller between moves. Position versions
//! identify which board a request was based on, so an old request can be rejected.
//! RecordedSession adds history; BranchSession preserves alternative continuations.
mod branches;
mod controllers;
mod error;
mod evaluation;
mod recorded;
mod recording_error;
mod recording_types;
use crate::agents::Agent;
use crate::core::{Decision, ForwardModel, PlayerId};
use crate::runner::{RunnerError, choose_action, decision_actions, offered_decision};
pub use branches::{
    BranchError, BranchId, BranchInfo, BranchRecordsResult, BranchSession, ControllerTimeline,
};
pub use controllers::{
    BuiltinAgent, BuiltinAgentError, BuiltinAgentSnapshot, BuiltinRestoreError, Controller,
    ControllerSnapshot,
};
pub use error::SessionError;
pub use evaluation::{
    EvaluationKey, EvaluationProposal, EvaluationRequest, EvaluationSession, EvaluationStatus,
};
pub use recorded::RecordedSession;
pub use recording_error::RecordingError;
pub use recording_types::{PreparedAi, PreparedAiResult, RecordingParts, RecordingResult};

/// A player observation and action list with separate storage, labelled with the position version.
#[derive(Debug)] // Debug: inspect the view when game data supports Debug.
pub struct DecisionView<O, A> {
    pub position_version: u64,
    pub observation: O,
    pub actions: Vec<A>,
}

/// An applied move and resulting next-player decision; the caller can retain the action for recording.
#[derive(Debug)] // Debug: inspect accepted move data when the action supports Debug.
pub struct AcceptedMove<A> {
    pub position_version: u64,
    pub actor: PlayerId,
    pub action: A,
    pub decision: Decision,
}

/// Either an accepted move or an error identifying the failed game/controller operation.
pub type SessionMoveResult<T, G, A> = Result<AcceptedMove<T>, SessionError<G, A>>;

/// A game and its controllers, advanced one decision at a time by the caller.
/// Controllers are indexed by seat; Human seats wait for submitted actions.
pub struct Session<F: ForwardModel, A: Agent<F>> {
    model: F,
    state: F::State,
    controllers: Vec<Controller<A>>,
    position_version: u64,
    stopped: bool,
}

impl<F: ForwardModel, A: Agent<F>> Session<F, A> {
    /// Create a game with one independently owned controller per seat.
    ///
    /// # Arguments
    /// Model and controller vector are consumed. Parameters are passed by a shared (read-only) reference;
    /// player_count is copied and checked by the game.
    ///
    /// # Returns
    /// An owned session at position version zero.
    ///
    /// # Errors
    /// Controller-count mismatch and game initialization errors.
    pub fn create(
        model: F,                        // Consumed rules; session owns them.
        parameters: &F::Parameters,      // Shared configuration borrow.
        player_count: usize,             // Copied seat count.
        controllers: Vec<Controller<A>>, // Consumed seat collection and AI progress.
    ) -> Result<Self, SessionError<F::Error, A::Error>> {
        if controllers.len() != player_count {
            return Err(SessionError::ControllerCount {
                expected: player_count,
                actual: controllers.len(),
            });
        }
        let state = model
            .init_state(parameters, player_count)
            .map_err(|error| SessionError::Execution(RunnerError::Game(error)))?;
        let actual = model.player_count(&state);
        if controllers.len() != actual {
            return Err(SessionError::ControllerCount {
                expected: actual,
                actual: controllers.len(),
            });
        }
        Ok(Self {
            model,
            state,
            controllers,
            position_version: 0,
            stopped: false,
        })
    }

    /// Consume caller-validated state, rules and controllers into a new session.
    /// Returns an owned session at version zero. Game-specific loaders validate
    /// state invariants before calling this; controller count is checked here.
    ///
    /// # Arguments
    /// `model`, validated `state` and seat-ordered `controllers` are consumed.
    /// Callers must validate state invariants through game-specific loading first.
    ///
    /// # Returns
    /// Owned session at position version zero using those exact state/controller values.
    ///
    /// # Errors
    /// Controller count differing from the model's player count.
    pub fn from_state(
        model: F,
        state: F::State,
        controllers: Vec<Controller<A>>,
    ) -> Result<Self, SessionError<F::Error, A::Error>> {
        let expected = model.player_count(&state);
        if controllers.len() != expected {
            return Err(SessionError::ControllerCount {
                expected,
                actual: controllers.len(),
            });
        }
        Ok(Self {
            model,
            state,
            controllers,
            position_version: 0,
            stopped: false,
        })
    }

    /// Shared rules borrow; the session retains ownership.
    pub fn model(&self) -> &F {
        &self.model
    }
    /// Shared full-state borrow for permitted native simulation/research access.
    /// Player-facing adapters construct decision_view for their observer.
    pub fn state(&self) -> &F::State {
        &self.state
    }
    /// Copied version of the current session position; each accepted move advances it.
    pub fn position_version(&self) -> u64 {
        self.position_version
    }
    /// Copied next actor or game completion, derived from the rules.
    pub fn next_player(&self) -> Decision {
        self.model.next_player(&self.state)
    }

    /// Whether a validated live transition failed and further moves are blocked.
    /// This is an execution error, separate from a normal terminal game result.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
    /// Shared controller borrow for a copied seat; unknown seats return None.
    pub fn controller(&self, player: PlayerId) -> Option<&Controller<A>> {
        self.controllers.get(player.index())
    }

    /// Construct an independent observation and its offered actions.
    ///
    /// # Arguments
    /// Self is passed by a shared (read-only) reference; observer is copied and validated by the rules.
    ///
    /// # Returns
    /// An owned view tagged with the current copied position version. Inactive/finished
    /// views retain their facts the player is allowed to know and have the game's offered-action list.
    ///
    /// # Errors
    /// Invalid observers return the owned game error.
    pub fn decision_view(
        &self,              // Shared session borrow.
        observer: PlayerId, // Copied observing seat.
    ) -> Result<DecisionView<F::Observation, F::Action>, F::Error> {
        let observation = self.model.observe(&self.state, observer)?;
        let actions = if self.next_player() == Decision::Player(observer) {
            self.model.available_actions(&self.state, observer)?
        } else {
            vec![]
        };
        Ok(DecisionView {
            position_version: self.position_version,
            observation,
            actions,
        })
    }

    /// Validate the request before touching a controller or changing the board.
    fn acting_position_version(
        &self,
        expected_position_version: u64,
    ) -> Result<(PlayerId, u64), SessionError<F::Error, A::Error>> {
        if self.stopped {
            return Err(SessionError::GameStopped);
        }
        if expected_position_version != self.position_version {
            return Err(SessionError::StalePositionVersion {
                expected: expected_position_version,
                actual: self.position_version,
            });
        }
        let actor = match self.next_player() {
            Decision::Player(actor) => actor,
            Decision::Finished => return Err(SessionError::GameFinished),
        };
        let next_position_version = self
            .position_version
            .checked_add(1)
            .ok_or(SessionError::PositionVersionExhausted)?;
        Ok((actor, next_position_version))
    }

    /// Submit one human move from a specific displayed position version.
    ///
    /// # Arguments
    /// Self is accessed through an exclusive mutable reference; position version and actor are copied. Action is
    /// consumed and compared with the player's available actions from state.
    ///
    /// # Returns
    /// An owned accepted action and resulting position version/decision; advances one move.
    ///
    /// # Errors
    /// Stale/finished requests, wrong actor/controller, unoffered moves or game
    /// rejection. Invalid requests preserve state, position version and AI progress.
    /// A failure while executing a validated action stops the session and may leave
    /// partial effects. Reset or reconstruct a valid position before continuing.
    pub fn submit_human(
        &mut self,                      // Exclusive session borrow.
        expected_position_version: u64, // Copied position_version from the displayed view.
        actor: PlayerId,                // Copied acting seat.
        action: F::Action, // Consumed move; returned in the accepted record on success.
    ) -> SessionMoveResult<F::Action, F::Error, A::Error>
    where
        F::Action: PartialEq,
    {
        let (expected_actor, next_position_version) =
            self.acting_position_version(expected_position_version)?;
        if actor != expected_actor {
            return Err(SessionError::WrongActor {
                expected: expected_actor,
                actual: actor,
            });
        }
        match self.controller(actor) {
            Some(Controller::Human) => {}
            Some(Controller::Ai(_)) => return Err(SessionError::AiRequired { player: actor }),
            None => {
                return Err(SessionError::Execution(RunnerError::MissingController {
                    player: actor,
                }));
            }
        }
        let actions =
            decision_actions(&self.model, &self.state, actor).map_err(SessionError::Execution)?;
        if !actions.contains(&action) {
            return Err(SessionError::ActionNotOffered { player: actor });
        }
        self.apply(actor, action, next_position_version)
    }

    /// Execute exactly one decision for the current AI seat.
    ///
    /// # Arguments
    /// Self is accessed through an exclusive mutable reference; expected_position_version is copied from a view.
    ///
    /// # Returns
    /// An owned accepted move and resulting position version/decision.
    ///
    /// # Errors
    /// Stale/finished requests, a human seat, game or agent failures. Request
    /// validation happens before agent access. Agent/evaluator progress made
    /// during a failed decision is retained, matching run_game semantics.
    /// A live rule execution error stops the session; agent/simulation errors do not
    /// apply an action to the live game and may be retried after fixing their cause.
    pub fn advance_ai(
        &mut self,                      // Exclusive session/controller access for this decision.
        expected_position_version: u64, // Copied position_version from the displayed view.
    ) -> SessionMoveResult<F::Action, F::Error, A::Error> {
        let (actor, next_position_version) =
            self.acting_position_version(expected_position_version)?;
        // Borrow the acting controller once; the remaining fields stay independently accessible.
        let agent = match self.controllers.get_mut(actor.index()) {
            Some(Controller::Ai(agent)) => agent,
            Some(Controller::Human) => return Err(SessionError::HumanRequired { player: actor }),
            None => {
                return Err(SessionError::Execution(RunnerError::MissingController {
                    player: actor,
                }));
            }
        };
        let input =
            offered_decision(&self.model, &self.state, actor).map_err(SessionError::Execution)?;
        let action =
            choose_action(&self.model, agent, actor, input).map_err(SessionError::Execution)?;
        self.apply(actor, action, next_position_version)
    }

    /// Apply a borrowed move, then transfer its ownership into the accepted record.
    fn apply(
        &mut self,
        actor: PlayerId,
        action: F::Action,
        next_position_version: u64,
    ) -> SessionMoveResult<F::Action, F::Error, A::Error> {
        if let Err(error) = self.model.next(&mut self.state, actor, &action) {
            self.stopped = true;
            return Err(SessionError::Execution(RunnerError::Game(error)));
        }
        self.position_version = next_position_version;
        Ok(AcceptedMove {
            position_version: self.position_version,
            actor,
            action,
            decision: self.next_player(),
        })
    }
}

#[cfg(test)]
mod tests;
