//! Interactive simultaneous decisions with private pending choices.
use super::{Controller, DecisionView};
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::joint::{self, Choice, JointError};
use crate::core::{ForwardModel, PlayerId};
use crate::runner::{RunnerError, choose_action, offered_decision};
use std::fmt;

/// One completed decision and the version of its complete successor state.
#[derive(Debug)]
pub struct AcceptedJointDecision<A> {
    pub position_version: u64,
    pub choices: Vec<Choice<A>>,
}

/// Saved only at a complete decision boundary. Restoring creates a new session
/// and therefore a new request scope in an adapter.
#[derive(Clone, Debug)]
pub struct JointSessionSnapshot<S, C> {
    pub state: S,
    pub controllers: Vec<crate::recording::ControllerSnapshot<C>>,
    pub position_version: u64,
}

#[derive(Debug)]
pub enum JointSessionError<G, A> {
    Invalid(String),
    Game(G),
    Agent(A),
    Stopped,
}

/// Typed result for a joint session using its game and agent error types.
pub type JointSessionResult<T, F, A> =
    Result<T, JointSessionError<<F as ForwardModel>::Error, <A as Agent<F>>::Error>>;

impl<G: fmt::Display, A: fmt::Display> fmt::Display for JointSessionError<G, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid joint session request: {message}"),
            Self::Game(error) => error.fmt(f),
            Self::Agent(error) => error.fmt(f),
            Self::Stopped => write!(f, "game stopped after a rule execution error"),
        }
    }
}

/// Pending actions live in the controller, never in the public game state.
/// A pending decision keeps the position version and all actor observations
/// unchanged. Cancellation increments the version so old requests become stale.
pub struct JointSession<F: ForwardModel, A: Agent<F>> {
    model: F,
    state: F::State,
    controllers: Vec<Controller<A>>,
    position_version: u64,
    pending: Vec<Choice<F::Action>>,
    stopped: bool,
}

impl<F, A> JointSession<F, A>
where
    F: ForwardModel,
    F::Action: PartialEq,
    A: Agent<F>,
    F::Error: fmt::Display,
    A::Error: fmt::Display,
{
    pub fn from_state(
        model: F,
        state: F::State,
        controllers: Vec<Controller<A>>,
    ) -> Result<Self, JointSessionError<F::Error, A::Error>> {
        if controllers.len() != model.player_count(&state) {
            return Err(JointSessionError::Invalid(
                "controller count differs from player count".into(),
            ));
        }
        Ok(Self {
            model,
            state,
            controllers,
            position_version: 0,
            pending: Vec::new(),
            stopped: false,
        })
    }

    pub fn state(&self) -> &F::State {
        &self.state
    }
    pub fn model(&self) -> &F {
        &self.model
    }
    /// Read a seat's controller without exposing its unfinished choice.
    pub fn controller(&self, seat: PlayerId) -> Option<&Controller<A>> {
        self.controllers.get(seat.index())
    }
    pub fn position_version(&self) -> u64 {
        self.position_version
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Replace the authoritative state and cancel all unfinished choices.
    /// Existing input versions become stale even if the new state looks equal.
    pub fn reset(&mut self, state: F::State) -> Result<(), JointSessionError<F::Error, A::Error>> {
        if self.model.player_count(&state) != self.controllers.len() {
            return Err(JointSessionError::Invalid(
                "controller count differs from player count".into(),
            ));
        }
        let next = self
            .position_version
            .checked_add(1)
            .ok_or_else(|| JointSessionError::Invalid("position version exhausted".into()))?;
        self.state = state;
        self.pending.clear();
        self.stopped = false;
        self.position_version = next;
        Ok(())
    }

    /// Save exact state and controller progress only after a complete decision.
    /// A caller can cancel a pending decision first, then save its initial state.
    pub fn snapshot(&self) -> JointSessionResult<JointSessionSnapshot<F::State, A::Snapshot>, F, A>
    where
        F::State: Clone,
        A: Checkpoint,
    {
        if self.stopped || !self.pending.is_empty() {
            return Err(JointSessionError::Invalid(
                "snapshot requires a complete valid decision boundary".into(),
            ));
        }
        Ok(JointSessionSnapshot {
            state: self.state.clone(),
            controllers: self.controllers.iter().map(Checkpoint::snapshot).collect(),
            position_version: self.position_version,
        })
    }

    /// Seats still waiting for a choice, in rule-defined order. It reveals no
    /// submitted actions; callers must decide whether to display participation.
    pub fn waiting_seats(&self) -> Vec<PlayerId> {
        if self.stopped {
            return Vec::new();
        }
        self.model
            .decision_players(&self.state)
            .into_iter()
            .filter(|seat| !self.pending.iter().any(|(p, _)| p == seat))
            .collect()
    }

    pub fn decision_view(
        &self,
        observer: PlayerId,
    ) -> JointSessionResult<DecisionView<F::Observation, F::Action>, F, A> {
        let observation = self
            .model
            .observe(&self.state, observer)
            .map_err(JointSessionError::Game)?;
        let actions = if self.waiting_seats().contains(&observer) {
            self.model
                .available_actions(&self.state, observer)
                .map_err(JointSessionError::Game)?
        } else {
            Vec::new()
        };
        Ok(DecisionView {
            position_version: self.position_version,
            observation,
            actions,
        })
    }

    /// Drop pending choices and invalidate all outstanding input for this state.
    pub fn cancel(&mut self) -> Result<(), JointSessionError<F::Error, A::Error>> {
        let next = self
            .position_version
            .checked_add(1)
            .ok_or_else(|| JointSessionError::Invalid("position version exhausted".into()))?;
        self.pending.clear();
        self.position_version = next;
        Ok(())
    }

    pub fn submit_human(
        &mut self,
        version: u64,
        actor: PlayerId,
        action: F::Action,
    ) -> JointSessionResult<Option<AcceptedJointDecision<F::Action>>, F, A> {
        if !matches!(self.controllers.get(actor.index()), Some(Controller::Human)) {
            return Err(JointSessionError::Invalid(
                "seat is not a human controller".into(),
            ));
        }
        self.submit(version, actor, action)
    }

    /// Commit an independently prepared AI choice and its controller progress.
    /// Rejected input leaves the controller and pending decision unchanged.
    pub fn submit_ai_choice(
        &mut self,
        version: u64,
        actor: PlayerId,
        action: F::Action,
        after: A::Snapshot,
    ) -> JointSessionResult<Option<AcceptedJointDecision<F::Action>>, F, A>
    where
        A: Checkpoint,
        A::RestoreError: fmt::Display,
    {
        if !matches!(self.controllers.get(actor.index()), Some(Controller::Ai(_))) {
            return Err(JointSessionError::Invalid(
                "seat is not an AI controller".into(),
            ));
        }
        self.check(version)?;
        if !self.waiting_seats().contains(&actor) {
            return Err(JointSessionError::Invalid(
                "seat has already chosen or is inactive".into(),
            ));
        }
        if !self
            .model
            .available_actions(&self.state, actor)
            .map_err(JointSessionError::Game)?
            .contains(&action)
        {
            return Err(JointSessionError::Invalid("action is not offered".into()));
        }
        self.position_version
            .checked_add(1)
            .ok_or_else(|| JointSessionError::Invalid("position version exhausted".into()))?;
        let next = Controller::Ai(
            A::from_snapshot(&after)
                .map_err(|error| JointSessionError::Invalid(error.to_string()))?,
        );
        self.controllers[actor.index()] = next;
        self.submit(version, actor, action)
    }

    pub fn advance_ai(
        &mut self,
        version: u64,
    ) -> JointSessionResult<Option<AcceptedJointDecision<F::Action>>, F, A> {
        self.check(version)?;
        let actor = *self
            .waiting_seats()
            .first()
            .ok_or_else(|| JointSessionError::Invalid("no seat is waiting".into()))?;
        let agent = match self.controllers.get_mut(actor.index()) {
            Some(Controller::Ai(agent)) => agent,
            _ => {
                return Err(JointSessionError::Invalid(
                    "next seat requires human input".into(),
                ));
            }
        };
        let input = offered_decision::<F, A::Error>(&self.model, &self.state, actor)
            .map_err(map_runner_error)?;
        let action = choose_action(&self.model, agent, actor, input).map_err(map_runner_error)?;
        self.submit(version, actor, action)
    }

    fn check(&self, version: u64) -> Result<(), JointSessionError<F::Error, A::Error>> {
        if self.stopped {
            return Err(JointSessionError::Stopped);
        }
        if version != self.position_version {
            return Err(JointSessionError::Invalid("stale position version".into()));
        }
        if self.model.decision_players(&self.state).is_empty() {
            return Err(JointSessionError::Invalid("game has finished".into()));
        }
        Ok(())
    }

    fn submit(
        &mut self,
        version: u64,
        actor: PlayerId,
        action: F::Action,
    ) -> JointSessionResult<Option<AcceptedJointDecision<F::Action>>, F, A> {
        self.check(version)?;
        if !self.waiting_seats().contains(&actor) {
            return Err(JointSessionError::Invalid(
                "seat has already chosen or is inactive".into(),
            ));
        }
        let actions = self
            .model
            .available_actions(&self.state, actor)
            .map_err(JointSessionError::Game)?;
        if !actions.contains(&action) {
            return Err(JointSessionError::Invalid("action is not offered".into()));
        }
        let next = self
            .position_version
            .checked_add(1)
            .ok_or_else(|| JointSessionError::Invalid("position version exhausted".into()))?;
        self.pending.push((actor, action));
        if !self.waiting_seats().is_empty() {
            return Ok(None);
        }
        let seats = self.model.decision_players(&self.state);
        let mut choices = std::mem::take(&mut self.pending);
        choices.sort_by_key(|(seat, _)| seats.iter().position(|p| p == seat).unwrap_or(usize::MAX));
        match joint::apply(&self.model, &mut self.state, &choices) {
            Ok(()) => {
                self.position_version = next;
                Ok(Some(AcceptedJointDecision {
                    position_version: next,
                    choices,
                }))
            }
            Err(JointError::Invalid(message)) => Err(JointSessionError::Invalid(message)),
            Err(JointError::Rule(error)) => {
                self.stopped = true;
                Err(JointSessionError::Game(error))
            }
        }
    }
}

fn map_runner_error<G: fmt::Display, A: fmt::Display>(
    error: RunnerError<G, A>,
) -> JointSessionError<G, A> {
    match error {
        RunnerError::Game(error) => JointSessionError::Game(error),
        RunnerError::Agent { error, .. } => JointSessionError::Agent(error),
        other => JointSessionError::Invalid(other.to_string()),
    }
}

impl<F, A> JointSession<F, A>
where
    F: ForwardModel,
    F::State: Clone,
    F::Action: PartialEq,
    A: Agent<F> + Checkpoint,
    F::Error: fmt::Display,
    A::Error: fmt::Display,
    A::RestoreError: fmt::Display,
{
    /// Reconstruct a new session from a boundary snapshot. Adapters must issue
    /// fresh request handles for the new instance.
    pub fn from_snapshot(
        model: F,
        snapshot: &JointSessionSnapshot<F::State, A::Snapshot>,
    ) -> Result<Self, JointSessionError<F::Error, A::Error>> {
        let controllers = snapshot
            .controllers
            .iter()
            .map(|saved| {
                Controller::<A>::from_snapshot(saved)
                    .map_err(|error| JointSessionError::Invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut session = Self::from_state(model, snapshot.state.clone(), controllers)?;
        session.position_version = snapshot.position_version;
        Ok(session)
    }
}
