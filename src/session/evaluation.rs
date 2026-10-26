//! Request state values from an external evaluator, then select and apply a move.
//! Pending evaluations retain their own simulation data. The live game and agent
//! RNG advance only when a completed decision is accepted.
use super::{Controller, Session};
use crate::agents::Agent;
use crate::agents::{
    CandidateBatch, EvaluationWork, EvaluatorIdentity, ExternalEvaluation, OslaPythonAgent,
    OslaPythonError, OslaPythonSnapshot, PreparedEvaluationAgent, validate_values,
};
use crate::core::{Decision, PlayerId, PublicInformationModel, checkpoint::Checkpoint};
use crate::core::{ForwardModel, random::AgentRng};
use crate::learning::LearningGame;
use std::sync::Arc;

/// An opaque session-bound handle. Cloning a handle keeps the same identity.
#[derive(Clone, Debug)]
pub struct EvaluationKey {
    // Arc shares ownership of a marker allocation. Pointer identity binds a key
    // to one session; snapshots allocate a new marker for the independent copy.
    owner: Arc<()>,
    serial: u64,
    position: u64,
    offset: usize,
}
impl PartialEq for EvaluationKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && self.serial == other.serial
            && self.position == other.position
            && self.offset == other.offset
    }
}
impl Eq for EvaluationKey {}
impl EvaluationKey {
    pub fn id(&self) -> u64 {
        self.serial
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// A batch of successor observations and the evaluator that should score them.
/// Return values with this exact key so they can be matched to the original decision.
pub struct EvaluationRequest {
    pub key: EvaluationKey,
    pub evaluator: EvaluatorIdentity,
    pub batch: CandidateBatch,
}

/// The selected action, ready to apply after checking the position is still current.
/// The live agent's RNG remains unchanged until the action is applied.
#[derive(Clone)]
pub struct EvaluationProposal {
    pub key: EvaluationKey,
    pub actor: PlayerId,
    pub action_id: usize,
}
pub enum EvaluationStatus {
    /// Box owns the larger request on the heap, keeping this enum compact.
    Request(Box<EvaluationRequest>),
    Ready(EvaluationProposal),
}
struct Selection {
    index: usize,
    next_agent: OslaPythonSnapshot,
}
struct Pending<F: ForwardModel> {
    key: EvaluationKey,
    work: EvaluationWork<F>,
    before: OslaPythonSnapshot,
    scores: Vec<f64>,
    chunk_len: usize,
    ready: Option<Selection>,
}

/// Retain the external agent and its RNG while a decision waits for evaluation.
/// Synchronous selection returns ExternalEvaluationRequired. Creating this wrapper
/// does not require simulation-state reconstruction, so human actions and ordinary
/// session operations remain available for any supported game.
struct WaitingController(OslaPythonAgent);
impl<F: ForwardModel> Agent<F> for WaitingController {
    type Error = OslaPythonError;
    fn rng(&self) -> &AgentRng {
        self.0.rng_ref()
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        self.0.rng_ref_mut()
    }
    fn choose(
        &mut self,
        _: &F,
        _: PlayerId,
        _: &F::Observation,
        _: &[F::Action],
    ) -> Result<usize, Self::Error> {
        Err(OslaPythonError::ExternalEvaluationRequired)
    }
}
impl Checkpoint for WaitingController {
    type Snapshot = OslaPythonSnapshot;
    type RestoreError = OslaPythonError;
    fn snapshot(&self) -> Self::Snapshot {
        self.0.snapshot()
    }
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Ok(Self(OslaPythonAgent::from_snapshot(snapshot)?))
    }
}

/// Extend Session with external state evaluation for one decision at a time.
/// Candidate and batch limits bound the requested work. Each decision finishes
/// before the next player is asked to act.
pub struct EvaluationSession<F: LearningGame>
where
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    session: Session<F, WaitingController>,
    owner: Arc<()>,
    next_id: u64,
    pending: Option<Pending<F>>,
}
impl<F> EvaluationSession<F>
where
    F: LearningGame + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// Consume a validated state and give each seat a human controller initially.
    pub fn from_state(model: F, state: F::State) -> Result<Self, OslaPythonError> {
        let count = model.player_count(&state);
        let session = Session::from_state(
            model,
            state,
            (0..count).map(|_| Controller::Human).collect(),
        )
        .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
        Ok(Self {
            session,
            owner: Arc::new(()),
            next_id: 1,
            pending: None,
        })
    }
    pub fn model(&self) -> &F {
        self.session.model()
    }
    pub fn state(&self) -> &F::State {
        self.session.state()
    }
    pub fn next_player(&self) -> Decision {
        self.session.next_player()
    }
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Read the complete controller checkpoint for a valid seat (None means human).
    pub fn checkpoint(
        &self,
        player: PlayerId,
    ) -> Result<Option<OslaPythonSnapshot>, OslaPythonError> {
        match self.session.controller(player) {
            Some(Controller::Human) => Ok(None),
            Some(Controller::Ai(agent)) => Ok(Some(agent.snapshot())),
            None => Err(OslaPythonError::Invalid("unknown controller seat".into())),
        }
    }

    /// Install an independently owned controller, invalidating pending work and old positions.
    /// `player` is copied; `controller` is consumed. None selects human input.
    /// Unknown seats and exhausted revisions fail before changing the session.
    pub fn set_controller(
        &mut self,
        player: PlayerId,
        controller: Option<OslaPythonAgent>,
    ) -> Result<(), OslaPythonError> {
        if player.index() >= self.session.controllers.len() {
            return Err(OslaPythonError::Invalid("unknown controller seat".into()));
        }
        let revision = self.next_revision()?;
        self.session.controllers[player.index()] = controller
            .map(|agent| Controller::Ai(WaitingController(agent)))
            .unwrap_or(Controller::Human);
        self.session.position_version = revision;
        self.pending = None;
        Ok(())
    }

    /// Submit a human action through the existing Session checks. AI seats require evaluation.
    pub fn apply_human(
        &mut self,
        actor: PlayerId,
        action: F::Action,
    ) -> Result<(), OslaPythonError> {
        self.session
            .submit_human(self.session.position_version, actor, action)
            .map_err(|e| OslaPythonError::Invalid(e.to_string()))?;
        self.pending = None;
        Ok(())
    }

    /// Prepare the current AI decision using its ordinary rules/actor/observation/actions.
    /// Live state and RNG stay unchanged. A second preparation requires cancellation first.
    ///
    /// # Returns
    /// An owned first chunk and opaque key for this session, position and decision.
    ///
    /// # Errors
    /// Pending work, completion, a human seat, invalid inputs or exceeded limits.
    /// Source reconstruction and candidate simulation failures are also returned.
    pub fn prepare(&mut self) -> Result<EvaluationRequest, OslaPythonError>
    where
        F: PublicInformationModel,
    {
        if self.session.is_stopped() {
            return Err(OslaPythonError::Invalid(
                "game stopped after a rule execution error; reset or restore a valid position"
                    .into(),
            ));
        }
        if self.pending.is_some() {
            return Err(OslaPythonError::Busy);
        }
        let actor = match self.next_player() {
            Decision::Player(p) => p,
            Decision::Finished => return Err(OslaPythonError::Invalid("game has finished".into())),
        };
        let agent = match self.session.controller(actor) {
            Some(Controller::Ai(agent)) => agent,
            _ => return Err(OslaPythonError::Invalid("current seat is human".into())),
        };
        let view = self
            .session
            .decision_view(actor)
            .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
        let work = agent
            .0
            .prepare(self.model(), actor, &view.observation, &view.actions)?;
        let batch = work.batch(self.model(), 0)?;
        let serial = self.next_id;
        let next_id = serial.checked_add(1).ok_or(OslaPythonError::Overflow)?;
        let key = EvaluationKey {
            owner: self.owner.clone(),
            serial,
            position: self.session.position_version,
            offset: 0,
        };
        let before = agent.snapshot();
        let evaluator = before.config.evaluator.clone();
        self.pending = Some(Pending {
            key: key.clone(),
            work,
            before,
            scores: Vec::new(),
            chunk_len: batch.action_ids.len(),
            ready: None,
        });
        self.next_id = next_id;
        Ok(EvaluationRequest {
            key,
            evaluator,
            batch,
        })
    }

    fn checked(&self, key: &EvaluationKey) -> Result<&Pending<F>, OslaPythonError> {
        let pending = self.pending.as_ref().ok_or(OslaPythonError::Stale)?;
        if *key != pending.key
            || key.position != self.session.position_version
            || self.checkpoint(pending.work.actor)?.as_ref() != Some(&pending.before)
        {
            return Err(OslaPythonError::Stale);
        }
        Ok(pending)
    }

    /// Check a response without storing scores or changing a controller.
    /// Coordinators call this for every member before accepting a grouped reply.
    ///
    /// # Arguments
    /// `key` borrows the pending chunk handle. `evaluator` borrows its declared
    /// code and settings. `action_ids` and `values` borrow ordered reply arrays.
    ///
    /// # Errors
    /// Returns an error for stale keys, completed proposals, mismatched identities
    /// or actions, incorrect lengths and nonfinite scores. Pending work is retained.
    pub fn validate_response(
        &self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        action_ids: &[usize],
        values: &[f64],
    ) -> Result<(), OslaPythonError>
    where
        F: PublicInformationModel,
    {
        self.validated_pending(key, evaluator, action_ids, values)
            .map(|_| ())
    }

    fn validated_pending(
        &self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        action_ids: &[usize],
        values: &[f64],
    ) -> Result<&Pending<F>, OslaPythonError>
    where
        F: PublicInformationModel,
    {
        let pending = self.checked(key)?;
        if pending.ready.is_some() {
            return Err(OslaPythonError::NotReady);
        }
        if evaluator != &pending.before.config.evaluator {
            return Err(OslaPythonError::Invalid(
                "reply evaluator identity or settings differ".into(),
            ));
        }
        let expected_ids =
            (key.offset..key.offset + pending.chunk_len).map(|i| pending.work.action_id(i));
        if !expected_ids.eq(action_ids.iter().copied()) {
            return Err(OslaPythonError::Invalid(
                "reply action IDs or order differ".into(),
            ));
        }
        validate_values(values, pending.chunk_len, key.offset)?;
        Ok(pending)
    }

    /// Validate one complete chunk. Return the next chunk or a proposal after all values arrive.
    /// Invalid responses retain the same request for retry and leave live/staged RNG untouched.
    ///
    /// # Arguments
    /// `key` identifies this chunk. `evaluator` must match the saved code/version/settings.
    /// `action_ids` and `values` are borrowed response slices in candidate order.
    /// Scores are copied into owned temporary storage only after validation succeeds.
    ///
    /// # Errors
    /// Stale/wrong-session keys, repeated responses, mismatched evaluator/action IDs,
    /// wrong score counts, nonfinite values or failure building the next chunk.
    pub fn finish(
        &mut self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        action_ids: &[usize],
        values: &[f64],
    ) -> Result<EvaluationStatus, OslaPythonError>
    where
        F: PublicInformationModel,
    {
        let pending = self.validated_pending(key, evaluator, action_ids, values)?;
        let next_offset = pending.scores.len() + values.len();
        if next_offset < pending.work.len() {
            // Build the next chunk before accepting scores so a simulation or
            // allocation-limit error leaves the original request retryable.
            let batch = pending.work.batch(self.model(), next_offset)?;
            let next_key = EvaluationKey {
                offset: next_offset,
                ..key.clone()
            };
            let evaluator = pending.before.config.evaluator.clone();
            let pending = self.pending.as_mut().expect("request checked above");
            pending.key = next_key.clone();
            pending.chunk_len = batch.action_ids.len();
            pending.scores.extend_from_slice(values);
            Ok(EvaluationStatus::Request(Box::new(EvaluationRequest {
                key: next_key,
                evaluator,
                batch,
            })))
        } else {
            // Selection can fail, so assemble its complete input before changing
            // the pending decision. Earlier chunks append directly without a copy.
            let mut scores = pending.scores.clone();
            scores.extend_from_slice(values);
            let mut temporary: OslaPythonAgent = OslaPythonAgent::from_snapshot(&pending.before)?;
            let index = temporary.select(pending.work.len(), &scores)?;
            let proposal = EvaluationProposal {
                key: key.clone(),
                actor: pending.work.actor,
                action_id: pending.work.action_id(index),
            };
            let pending = self.pending.as_mut().expect("request checked above");
            pending.scores = scores;
            pending.ready = Some(Selection {
                index,
                next_agent: temporary.snapshot(),
            });
            Ok(EvaluationStatus::Ready(proposal))
        }
    }

    /// Apply a ready move and install its exact next agent checkpoint after validation succeeds.
    /// `key` borrows the proposal handle. A stale key or unfinished evaluation returns
    /// an error. On success exactly one move is applied and the handle is consumed
    /// logically: subsequent use is rejected. Game errors preserve the pending proposal.
    pub fn commit(&mut self, key: &EvaluationKey) -> Result<(), OslaPythonError> {
        let pending = self.checked(key)?;
        let selected = pending.ready.as_ref().ok_or(OslaPythonError::NotReady)?;
        let actor = pending.work.actor;
        let action = pending.work.actions[selected.index].clone();
        let next_agent = OslaPythonAgent::from_snapshot(&selected.next_agent)?;
        let revision = self.next_revision()?;
        self.session
            .apply(actor, action, revision)
            .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
        self.session.controllers[actor.index()] = Controller::Ai(WaitingController(next_agent));
        self.pending = None;
        Ok(())
    }

    /// Discard this exact request/proposal without changing live board or RNG.
    /// `key` is borrowed and checked before discarding. A stale key leaves current work intact.
    pub fn cancel(&mut self, key: &EvaluationKey) -> Result<(), OslaPythonError> {
        self.checked(key)?;
        self.pending = None;
        Ok(())
    }

    /// Reset to a caller-created initial state and restart each agent from its configured seed.
    pub fn reset_state(&mut self, state: F::State) -> Result<(), OslaPythonError> {
        if self.model().player_count(&state) != self.session.controllers.len() {
            return Err(OslaPythonError::Invalid(
                "reset player count changed".into(),
            ));
        }
        let revision = self.next_revision()?;
        let mut controllers = Vec::new();
        for controller in &self.session.controllers {
            controllers.push(match controller {
                Controller::Human => Controller::Human,
                Controller::Ai(agent) => {
                    let snapshot = agent.snapshot();
                    Controller::Ai(WaitingController(OslaPythonAgent::new(
                        ExternalEvaluation,
                        snapshot.config,
                        snapshot.rng.initial_seed,
                    )?))
                }
            });
        }
        self.session.state = state;
        self.session.controllers = controllers;
        self.session.position_version = revision;
        self.session.stopped = false;
        self.pending = None;
        Ok(())
    }

    /// Copy a settled state and exact controller checkpoints; pending work must finish or cancel first.
    pub fn snapshot(&self) -> Result<Self, OslaPythonError> {
        if self.has_pending() {
            return Err(OslaPythonError::Busy);
        }
        let controllers = self
            .session
            .controllers
            .iter()
            .map(|c| match c {
                Controller::Human => Ok(Controller::Human),
                Controller::Ai(agent) => Ok(Controller::Ai(WaitingController::from_snapshot(
                    &agent.snapshot(),
                )?)),
            })
            .collect::<Result<Vec<_>, OslaPythonError>>()?;
        let mut session =
            Session::from_state(self.model().clone(), self.state().clone(), controllers)
                .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
        session.stopped = self.session.stopped;
        Ok(Self {
            session,
            owner: Arc::new(()),
            next_id: 1,
            pending: None,
        })
    }
    fn next_revision(&self) -> Result<u64, OslaPythonError> {
        self.session
            .position_version
            .checked_add(1)
            .ok_or(OslaPythonError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::tictactoe::{PlaceMark, TicTacToeForwardModel, TicTacToeParameters};

    #[test]
    fn copying_preserves_execution_stop_and_reset_to_valid_state_clears_it() {
        let model = TicTacToeForwardModel;
        let initial = model
            .init_state(&TicTacToeParameters::default(), 2)
            .unwrap();
        let mut original = EvaluationSession::from_state(model, initial.clone()).unwrap();
        // Inject the lifecycle flag; transition-failure tests cover setting it.
        original.session.stopped = true;
        let mut copy = original.snapshot().unwrap();
        assert!(copy.session.is_stopped());
        assert!(matches!(
            copy.session
                .submit_human(0, PlayerId::new(0), PlaceMark { x: 0, y: 0 }),
            Err(super::super::SessionError::GameStopped)
        ));
        original.reset_state(initial).unwrap();
        assert!(!original.session.is_stopped());
        original
            .session
            .submit_human(
                original.session.position_version(),
                PlayerId::new(0),
                PlaceMark { x: 0, y: 0 },
            )
            .unwrap();
        assert!(copy.session.is_stopped());
    }
}
