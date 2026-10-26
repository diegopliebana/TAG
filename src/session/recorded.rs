//! Record accepted moves and calculate AI choices before applying them.
//!
//! Preparation uses a separate restored agent. Discarding the proposal leaves live
//! state and RNG unchanged. Commit checks that it still belongs to this session,
//! position and controller before applying the move and recording its progress.
use super::{
    AcceptedMove, Controller, PreparedAi, PreparedAiResult, RecordingError, RecordingParts,
    RecordingResult, Session, SessionError,
};
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::{ForwardModel, PlayerId};
use crate::recording::{
    ControllerSnapshot, GameHistory, HistoryCachePolicy, HistoryError, RecordedMove,
};
use crate::runner::{choose_action, decision_actions, offered_decision};
use std::sync::Arc;

/// A wrapper around Session that records moves and saves controller progress for restoration.
/// Each instance has an origin token, so one session cannot commit another's proposal.
pub struct RecordedSession<F: ForwardModel, A: Agent<F> + Checkpoint> {
    pub(super) session: Session<F, A>,
    pub(super) history: GameHistory<F::State, F::Action, A::Snapshot>,
    origin: Arc<()>,
}

impl<F, A> RecordedSession<F, A>
where
    F: ForwardModel,
    F::State: Clone,
    A: Agent<F> + Checkpoint,
{
    /// Consume a live session and capture its current state/controller progress as root.
    /// Returns an owned recording wrapper; state Clone must produce independent data.
    pub fn new(session: Session<F, A>) -> Self {
        Self::with_cache_policy(session, HistoryCachePolicy::default())
    }
    /// Consume a session and copied cache settings into an owned recording wrapper.
    /// State/controller progress becomes its root. Cache capacity controls extra
    /// reconstructed state copies; zero disables caching.
    pub fn with_cache_policy(session: Session<F, A>, cache_policy: HistoryCachePolicy) -> Self {
        let history = GameHistory::with_cache_policy(
            session.state.clone(),
            session
                .controllers
                .iter()
                .map(Checkpoint::snapshot)
                .collect(),
            cache_policy,
        );
        Self {
            session,
            history,
            origin: Arc::new(()),
        }
    }
    /// Shared live session access for state, rules, views and controller inspection.
    pub fn session(&self) -> &Session<F, A> {
        &self.session
    }
    /// Shared recording access; owned actions and checkpoints remain in the history.
    pub fn history(&self) -> &GameHistory<F::State, F::Action, A::Snapshot> {
        &self.history
    }
    /// Consume the wrapper and transfer both live session and recording to the caller.
    pub fn into_parts(self) -> RecordingParts<F, A> {
        RecordingParts {
            session: self.session,
            history: self.history,
        }
    }

    /// Browse a recorded position without touching live state or controller progress.
    /// Exclusive self access updates the bounded cache; copied position counts moves
    /// from this recording's root. Returns independent state or a history error.
    pub fn position_at(&mut self, position: usize) -> Result<F::State, HistoryError<F::Error>> {
        self.history.position_at(&self.session.model, position)
    }
    /// Construct an owned permitted historical view for a copied position/observer.
    /// Live state, position version and controller RNGs remain unchanged.
    pub fn view_at(
        &mut self,
        position: usize,
        observer: PlayerId,
    ) -> crate::recording::HistoricalViewResult<F::Observation, F::Action, F::Error> {
        self.history
            .view_at(&self.session.model, position, observer)
    }

    /// Restore an independently owned session at a historical move boundary.
    ///
    /// # Arguments
    /// Exclusive self updates the history cache. Position is copied; Clone on rules
    /// preserves the same ruleset. Agents restore from owned independent snapshots.
    ///
    /// # Returns
    /// A new session at position version zero, with original game counters and exact
    /// controller progress. The source recording/live session stay available.
    ///
    /// # Errors
    /// Invalid history/replay data or a typed controller restoration failure.
    pub fn restore_session_at(
        &mut self,
        position: usize,
    ) -> RecordingResult<Session<F, A>, F::Error, A::Error, A::RestoreError>
    where
        F: Clone,
    {
        let state = self
            .position_at(position)
            .map_err(RecordingError::History)?;
        let snapshots = self
            .history
            .controllers_at(position)
            .map_err(RecordingError::History)?;
        let controllers = snapshots
            .iter()
            .map(Controller::from_snapshot)
            .collect::<Result<Vec<_>, _>>()
            .map_err(RecordingError::Restore)?;
        Ok(Session {
            model: self.session.model.clone(),
            state,
            controllers,
            position_version: 0,
            stopped: false,
        })
    }

    /// Validate and calculate one AI choice on an independently restored controller.
    ///
    /// # Arguments
    /// Shared self access preserves the live state/controllers. Position version
    /// is copied from the displayed live view.
    ///
    /// # Returns
    /// An owned proposal; commit_ai accepts it once for this exact session position.
    ///
    /// # Errors
    /// Stale/finished/wrong-controller requests, restoration or game/agent failures.
    /// All temporary evaluator/RNG work is discarded on failure.
    pub fn prepare_ai(&self, expected_position_version: u64) -> PreparedAiResult<F, A> {
        let (actor, _) = self
            .session
            .acting_position_version(expected_position_version)
            .map_err(RecordingError::Session)?;
        let Some(Controller::Ai(agent)) = self.session.controller(actor) else {
            return Err(RecordingError::Session(SessionError::HumanRequired {
                player: actor,
            }));
        };
        let before = agent.snapshot();
        let mut temporary = A::from_snapshot(&before).map_err(RecordingError::Restore)?;
        let input = offered_decision(&self.session.model, &self.session.state, actor)
            .map_err(|error| RecordingError::Session(SessionError::Execution(error)))?;
        let action = choose_action(&self.session.model, &mut temporary, actor, input)
            .map_err(|error| RecordingError::Session(SessionError::Execution(error)))?;
        Ok(PreparedAi {
            origin: Arc::clone(&self.origin),
            position_version: expected_position_version,
            actor,
            action,
            before,
            after: temporary.snapshot(),
        })
    }

    /// Prepare an already recorded AI choice and its recorded continuation.
    /// Callers validate the recording's policy/RNG transition before calling this.
    /// Commit still checks origin, current controller, position and action legality.
    /// This does not rerun the search or establish that its choice was optimal.
    pub fn recorded_ai_choice(
        &self,
        version: u64,
        action: F::Action,
        after: &A::Snapshot,
    ) -> PreparedAiResult<F, A>
    where
        F::Action: PartialEq,
    {
        let (actor, _) = self
            .session
            .acting_position_version(version)
            .map_err(RecordingError::Session)?;
        let Some(Controller::Ai(agent)) = self.session.controller(actor) else {
            return Err(RecordingError::Session(SessionError::HumanRequired {
                player: actor,
            }));
        };
        let actions = decision_actions(&self.session.model, &self.session.state, actor)
            .map_err(|e| RecordingError::Session(SessionError::Execution(e)))?;
        if !actions.contains(&action) {
            return Err(RecordingError::UnavailableRecordedAction);
        }
        let restored = A::from_snapshot(after).map_err(RecordingError::Restore)?;
        Ok(PreparedAi {
            origin: Arc::clone(&self.origin),
            position_version: version,
            actor,
            action,
            before: agent.snapshot(),
            after: restored.snapshot(),
        })
    }

    /// Create an owned, suspendable calculation from this position and a restored
    /// agent. The task borrows no live session data. Dropping it preserves the
    /// live game and RNG; committing its eventual proposal retains all existing
    /// origin, position and controller checks.
    pub fn prepare_ai_task(
        &self,
        version: u64,
    ) -> RecordingResult<
        impl std::future::Future<Output = PreparedAiResult<F, A>> + use<F, A>,
        F::Error,
        A::Error,
        A::RestoreError,
    >
    where
        F: Clone,
        F::Action: Clone,
    {
        let (actor, _) = self
            .session
            .acting_position_version(version)
            .map_err(RecordingError::Session)?;
        let Some(Controller::Ai(agent)) = self.session.controller(actor) else {
            return Err(RecordingError::Session(SessionError::HumanRequired {
                player: actor,
            }));
        };
        let before = agent.snapshot();
        let mut temporary = A::from_snapshot(&before).map_err(RecordingError::Restore)?;
        let input = offered_decision(&self.session.model, &self.session.state, actor)
            .map_err(|e| RecordingError::Session(SessionError::Execution(e)))?;
        let model = self.session.model.clone();
        let origin = Arc::clone(&self.origin);
        Ok(async move {
            let index = temporary
                .choose_async(&model, actor, &input.observation, &input.actions)
                .await
                .map_err(|error| {
                    RecordingError::Session(SessionError::Execution(
                        crate::runner::RunnerError::Agent {
                            player: actor,
                            error,
                        },
                    ))
                })?;
            let action = input.actions.get(index).cloned().ok_or_else(|| {
                RecordingError::Session(SessionError::Execution(
                    crate::runner::RunnerError::InvalidSelection {
                        player: actor,
                        index,
                        action_count: input.actions.len(),
                    },
                ))
            })?;
            Ok(PreparedAi {
                origin,
                position_version: version,
                actor,
                action,
                before,
                after: temporary.snapshot(),
            })
        })
    }

    /// Consume and commit one prepared move, then record its exact controller progress.
    ///
    /// # Arguments
    /// Exclusive self access applies one move; proposal ownership transfers here.
    /// Action Clone duplicates the accepted action into its history record.
    ///
    /// # Returns
    /// An owned accepted move; live agent progress advances to the prepared checkpoint.
    ///
    /// # Errors
    /// Foreign/stale proposals, changed controller, restoration and rule failures.
    pub fn commit_ai(
        &mut self,
        proposal: PreparedAi<F::Action, A::Snapshot>,
    ) -> RecordingResult<AcceptedMove<F::Action>, F::Error, A::Error, A::RestoreError>
    where
        F::Action: Clone,
        A::Snapshot: PartialEq,
    {
        if !Arc::ptr_eq(&proposal.origin, &self.origin) {
            return Err(RecordingError::ForeignProposal);
        }
        let (actor, next_version) = self
            .session
            .acting_position_version(proposal.position_version)
            .map_err(RecordingError::Session)?;
        let Some(Controller::Ai(agent)) = self.session.controller(actor) else {
            return Err(RecordingError::ControllerChanged);
        };
        if actor != proposal.actor || agent.snapshot() != proposal.before {
            return Err(RecordingError::ControllerChanged);
        }
        let restored = A::from_snapshot(&proposal.after).map_err(RecordingError::Restore)?;
        let accepted = self
            .session
            .apply(actor, proposal.action, next_version)
            .map_err(RecordingError::Session)?;
        self.session.controllers[actor.index()] = Controller::Ai(restored);
        self.record_accepted(&accepted, ControllerSnapshot::Ai(proposal.after));
        Ok(accepted)
    }

    /// Calculate, commit and record one AI move with exclusive session access.
    /// Copies the expected position version; returns an owned accepted move or
    /// a typed failure. Failed preparation leaves the live controller unchanged.
    pub fn advance_ai(
        &mut self,
        expected_position_version: u64,
    ) -> RecordingResult<AcceptedMove<F::Action>, F::Error, A::Error, A::RestoreError>
    where
        F::Action: Clone,
        A::Snapshot: PartialEq,
    {
        let prepared = self.prepare_ai(expected_position_version)?;
        self.commit_ai(prepared)
    }

    /// Submit and record one consumed human action using the session's validation.
    /// Exclusive self borrow; copied actor/version. Clone duplicates the accepted
    /// move into history. Invalid requests preserve live state, controllers and records.
    /// An internal transition error stops the live session with possible partial
    /// effects; it adds no successful move or controller update to the recording.
    pub fn submit_human(
        &mut self,
        expected_position_version: u64,
        actor: PlayerId,
        action: F::Action,
    ) -> RecordingResult<AcceptedMove<F::Action>, F::Error, A::Error, A::RestoreError>
    where
        F::Action: Clone + PartialEq,
    {
        let accepted = self
            .session
            .submit_human(expected_position_version, actor, action)
            .map_err(RecordingError::Session)?;
        self.record_accepted(&accepted, ControllerSnapshot::Human);
        Ok(accepted)
    }
    /// Borrow an accepted move and consume its checkpoint into the shared history.
    /// Clone duplicates the owned action; the caller retains its accepted result.
    fn record_accepted(
        &mut self,
        accepted: &AcceptedMove<F::Action>,
        controller_after: ControllerSnapshot<A::Snapshot>,
    ) where
        F::Action: Clone,
    {
        self.history.record(RecordedMove {
            actor: accepted.actor,
            action: accepted.action.clone(),
            decision: accepted.decision,
            controller_after,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::tictactoe::{
        PlaceMark, TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters,
    };

    #[test]
    fn rejected_prepared_action_preserves_live_state_controller_and_history() {
        let mut game = RecordedSession::new(
            Session::create(
                TicTacToeForwardModel,
                &TicTacToeParameters::default(),
                2,
                vec![Controller::osla(TicTacToeEvaluator, 7), Controller::Human],
            )
            .unwrap(),
        );
        let initial = game.session().state().clone();
        let checkpoint = game
            .session()
            .controller(PlayerId::new(0))
            .unwrap()
            .snapshot();
        let mut proposal = game.prepare_ai(0).unwrap();
        // Inject an invalid coordinate to exercise the rule-rejection path at commitment.
        proposal.action = PlaceMark {
            x: usize::MAX,
            y: 0,
        };
        assert!(matches!(
            game.commit_ai(proposal),
            Err(RecordingError::Session(SessionError::Execution(_)))
        ));
        assert_eq!(game.session().state(), &initial);
        assert_eq!(game.session().position_version(), 0);
        assert_eq!(
            game.session()
                .controller(PlayerId::new(0))
                .unwrap()
                .snapshot(),
            checkpoint
        );
        assert!(game.history().moves().is_empty());
    }
}
