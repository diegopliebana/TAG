//! Apply browser commands to the shared Rust session types.
//!
//! Runtime keeps game settings, recorded branches, displayed position and any AI
//! move calculated but not applied. Rules and controller decisions are delegated
//! to the engine rather than reimplemented for the browser.
use crate::errors::branch_error;
use crate::games::BrowserGame;
use crate::protocol::*;
use serde_json::Value;
use tag_rust::core::PlayerId;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::session::{BranchId, BranchSession, Controller, PreparedAi, Session};

mod files;
mod history;
#[cfg(test)]
mod history_tests;

type PendingProposal<M, C> = (String, PreparedAi<M, C>);
type Preparation<F, A> = std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<
                    PreparedAi<
                        <F as tag_rust::core::ForwardModel>::Action,
                        <A as Checkpoint>::Snapshot,
                    >,
                    BridgeError,
                >,
            >,
    >,
>;
pub(crate) struct Runtime<F: BrowserGame> {
    parameters: F::Parameters,
    pub(crate) models: crate::learned::Registry,
    game: BranchSession<F, F::Agent>,
    pending: Option<PendingProposal<F::Action, <F::Agent as Checkpoint>::Snapshot>>,
    calculation: Option<(String, Preparation<F, F::Agent>)>,
    next_proposal: u64,
    display: Option<(BranchId, usize)>,
    stored_moves: usize,
    origins: Vec<BranchOrigin>,
    view_observer: Option<usize>,
    public_only: bool,
    reveal_hidden_information: bool,
}
fn fail(message: impl Into<String>) -> BridgeError {
    BridgeError::new("execution", message)
}
pub(crate) fn integer(text: &str) -> Result<u64, BridgeError> {
    if text.is_empty() || !text.bytes().all(|c| c.is_ascii_digit()) {
        return Err(BridgeError::new(
            "invalid_request",
            "expected unsigned decimal identifier",
        ));
    }
    text.parse()
        .map_err(|_| BridgeError::new("invalid_request", "identifier exceeds u64 range"))
}
impl<F: BrowserGame> Runtime<F> {
    fn poll_preparation(&mut self, proposal: &str) -> Result<ProposalView, BridgeError> {
        use std::task::{Context, Poll, Waker};
        let Some((id, task)) = &mut self.calculation else {
            return Err(BridgeError::new(
                "invalid_proposal",
                "no pending AI calculation",
            ));
        };
        if id != proposal {
            return Err(BridgeError::new(
                "invalid_proposal",
                "unknown or superseded calculation",
            ));
        }
        // Keep at most sixteen completed iterations in one browser message.
        // Each individual iteration still obeys the search's transition/time budgets.
        let mut result = Poll::Pending;
        let clock = crate::mcts::BrowserClock::default();
        let started = tag_rust::search::mcts::Clock::now_ms(&clock);
        for _ in 0..16 {
            result = task.as_mut().poll(&mut Context::from_waker(Waker::noop()));
            if result.is_ready() {
                break;
            }
            if tag_rust::search::mcts::Clock::now_ms(&clock) - started >= 4. {
                break;
            }
        }
        match result {
            Poll::Pending => Ok(ProposalView {
                id: proposal.into(),
                actor: self.view(None)?.next_player.unwrap_or(0),
                action: Value::Null,
                pending: true,
            }),
            Poll::Ready(result) => {
                self.calculation = None;
                let value = result?;
                let data = ProposalView {
                    id: proposal.into(),
                    actor: value.actor().index(),
                    action: if F::HIDDEN_INFORMATION {
                        Value::Null
                    } else {
                        serde_json::to_value(value.action()).map_err(|e| fail(e.to_string()))?
                    },
                    pending: false,
                };
                self.pending = Some((proposal.into(), value));
                Ok(data)
            }
        }
    }
    /// Create a bounded runtime from browser setup data.
    ///
    /// # Arguments
    /// `model` is consumed. `parameters` consumes optional JSON configuration;
    /// None selects the game's defaults. `seats` consumes optional seat settings;
    /// None selects two human players. Each AI seed is an unsigned decimal string.
    ///
    /// # Returns
    /// An owned runtime with one recorded game and no pending AI proposal.
    ///
    /// # Errors
    /// Invalid parameters/seeds/counts or exceeded browser board/OSLA limits.
    #[cfg(test)]
    pub(crate) fn create(
        model: F,
        parameters: Option<Value>,
        seats: Option<Vec<SeatSetup>>,
    ) -> Result<Self, BridgeError> {
        Self::create_with_models(model, parameters, seats, &crate::learned::Registry::new())
    }
    pub(crate) fn create_with_models(
        model: F,
        parameters: Option<Value>,
        seats: Option<Vec<SeatSetup>>,
        models: &crate::learned::Registry,
    ) -> Result<Self, BridgeError> {
        let parameters = F::parse_parameters(parameters)?;
        let seats = seats.unwrap_or_else(|| vec![SeatSetup::Human, SeatSetup::Human]);
        F::validate_seats(&parameters, &seats, models)?;
        let (width, height) = F::dimensions(&parameters);
        let cells = width
            .checked_mul(height)
            .ok_or_else(|| BridgeError::new("resource_limit", "board dimensions overflow"))?;
        if width > 128 || height > 128 || cells > 4096 {
            return Err(BridgeError::new(
                "resource_limit",
                "browser board limit is 128 per dimension and 4096 cells",
            ));
        }
        if seats.iter().any(|s| matches!(s, SeatSetup::Osla { .. })) && cells > 256 {
            return Err(BridgeError::new(
                "resource_limit",
                "browser OSLA limit is 256 cells",
            ));
        }
        let controllers = seats
            .into_iter()
            .map(|setup| F::controller_with_models(&model, &parameters, setup, models))
            .collect::<Result<Vec<_>, BridgeError>>()?;
        let player_count = controllers.len();
        let game = BranchSession::new(
            Session::create(model, &parameters, player_count, controllers)
                .map_err(|e| BridgeError::new("invalid_setup", e.to_string()))?,
        );
        Ok(Self {
            calculation: None,
            view_observer: None,
            public_only: false,
            reveal_hidden_information: false,
            parameters,
            models: models.clone(),
            game,
            pending: None,
            next_proposal: 0,
            display: None,
            stored_moves: 0,
            origins: vec![BranchOrigin::Original],
        })
    }
    /// Construct the current player's permitted browser view.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; copied `observer` selects a seat. None uses the
    /// acting seat, or seat zero after completion.
    ///
    /// # Returns
    /// Owned observation-derived board/progress, actions, results and scores.
    ///
    /// # Errors
    /// Unknown observer, permitted-state reconstruction or game-query failure.
    pub(crate) fn view(&self, observer: Option<usize>) -> Result<PublicView, BridgeError> {
        let session = self.game.active().session();
        let tip = self
            .game
            .info(self.game.active_branch())
            .map_err(branch_error)?
            .move_count;
        let controllers = (0..session.model().player_count(session.state()))
            .map(|id| match session.controller(PlayerId::new(id)).unwrap() {
                Controller::Human => SeatSetup::Human,
                Controller::Ai(agent) => F::settings(agent),
            })
            .collect();
        self.make_view(
            session.state(),
            observer,
            controllers,
            history::ViewLocation {
                branch: self.game.active_branch(),
                cursor: tip,
                tip,
                read_only: false,
            },
        )
    }
    /// Execute one request against the current recorded game.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; `command` and optional decimal-string
    /// `version` are consumed. Mutations require the current position version.
    ///
    /// # Returns
    /// Owned view containing only information the player is allowed to see and an optional independently prepared AI proposal.
    ///
    /// # Errors
    /// Invalid/stale position, unavailable/obsolete proposal, action or game failure.
    pub(crate) fn execute(
        &mut self,
        command: Command,
        version: Option<String>,
    ) -> Result<(PublicView, Option<ProposalView>), BridgeError> {
        match command {
            Command::Presentation {
                observer,
                public_only,
                reveal_hidden_information,
            } => {
                if observer.is_some_and(|p| {
                    p >= self
                        .game
                        .active()
                        .session()
                        .model()
                        .player_count(self.game.active().session().state())
                }) {
                    return Err(BridgeError::new(
                        "invalid_observer",
                        "observer is outside game seats",
                    ));
                }
                self.view_observer = observer;
                self.public_only = public_only;
                self.reveal_hidden_information = reveal_hidden_information;
                if let Some((branch, cursor)) = self.display {
                    return Ok((self.history_view(branch, cursor, None)?, None));
                }
                return Ok((self.view(None)?, None));
            }
            Command::View { observer } => {
                if let Some((branch, cursor)) = self.display {
                    return Ok((self.history_view(branch, cursor, observer)?, None));
                }
                return Ok((self.view(observer)?, None));
            }
            Command::History {
                branch,
                cursor,
                observer,
            } => {
                self.calculation = None;
                return Ok((self.history_view(BranchId(branch), cursor, observer)?, None));
            }
            _ => {}
        }
        if self.display.is_some()
            && matches!(
                command,
                Command::HumanMove { .. }
                    | Command::PrepareAi
                    | Command::PollAi { .. }
                    | Command::CommitAi { .. }
            )
        {
            return Err(BridgeError::new(
                "historical_position",
                "recorded positions are read-only; fork or return to branch end",
            ));
        }
        let expected = integer(version.as_deref().ok_or_else(|| {
            BridgeError::new("invalid_request", "mutation requires position_version")
        })?)?;
        if expected != self.game.active().session().position_version() {
            return Err(BridgeError::new(
                "stale_position",
                "request position no longer matches",
            ));
        }
        let full_history = matches!(
            command,
            Command::Reset | Command::SelectBranch { .. } | Command::Fork { .. }
        );
        let accepted = matches!(
            command,
            Command::HumanMove { .. } | Command::CommitAi { .. }
        );
        if accepted && self.stored_moves >= 100_000 {
            return Err(BridgeError::new(
                "resource_limit",
                "browser stored move limit is 100,000",
            ));
        }
        // A result step may retain the viewer who experienced the move. This
        // changes only this response; the next View returns normal presentation.
        let response_observer = match &command {
            Command::HumanMove { observer, .. } | Command::CommitAi { observer, .. } => *observer,
            _ => None,
        };
        if response_observer.is_some_and(|p| {
            p >= self
                .game
                .active()
                .session()
                .model()
                .player_count(self.game.active().session().state())
        }) {
            return Err(BridgeError::new(
                "invalid_observer",
                "observer is outside game seats",
            ));
        }
        let proposal = match command {
            Command::HumanMove { actor, action, .. } => {
                let action = serde_json::from_value(action)
                    .map_err(|e| BridgeError::new("invalid_action", e.to_string()))?;
                self.game
                    .submit_human(expected, PlayerId::new(actor), action)
                    .map_err(branch_error)?;
                self.pending = None;
                None
            }
            Command::Reset => {
                self.reveal_hidden_information = false;
                self.calculation = None;
                self.check_branch_capacity(BranchId(0))?;
                self.game.reset().map_err(branch_error)?;
                self.view_observer = None;
                self.public_only = false;
                self.origins.push(BranchOrigin::Reset);
                self.pending = None;
                self.display = None;
                None
            }
            Command::SelectBranch { branch } => {
                self.calculation = None;
                self.game.select(BranchId(branch)).map_err(branch_error)?;
                self.pending = None;
                self.display = None;
                None
            }
            Command::Fork {
                branch,
                cursor,
                replacements,
            } => {
                self.calculation = None;
                self.fork_branch(BranchId(branch), cursor, replacements)?;
                self.origins.push(BranchOrigin::Fork);
                self.pending = None;
                self.display = None;
                None
            }
            Command::PrepareAi => {
                let id = self
                    .next_proposal
                    .checked_add(1)
                    .ok_or_else(|| fail("proposal identifier exhausted"))?;
                let task = self.game.prepare_ai_task(expected).map_err(|error| {
                    branch_error(tag_rust::session::BranchError::Execution(error))
                })?;
                self.next_proposal = id;
                self.pending = None;
                self.calculation = Some((
                    id.to_string(),
                    Box::pin(async move {
                        task.await
                            .map_err(|e| branch_error(tag_rust::session::BranchError::Execution(e)))
                    }),
                ));
                Some(self.poll_preparation(&id.to_string())?)
            }
            Command::PollAi { proposal } => Some(self.poll_preparation(&proposal)?),
            Command::CommitAi { proposal, .. } => {
                if self.pending.as_ref().is_none_or(|(id, _)| *id != proposal) {
                    return Err(BridgeError::new(
                        "invalid_proposal",
                        "unknown or superseded proposal",
                    ));
                }
                let (_, value) = self.pending.take().unwrap();
                self.game.commit_ai(value).map_err(branch_error)?;
                None
            }
            Command::DiscardAi { proposal } => {
                if self
                    .calculation
                    .as_ref()
                    .is_some_and(|(id, _)| *id == proposal)
                {
                    self.calculation = None;
                    return Ok((self.view(None)?, None));
                }
                if self.pending.as_ref().is_none_or(|(id, _)| *id != proposal) {
                    return Err(BridgeError::new(
                        "invalid_proposal",
                        "unknown or superseded proposal",
                    ));
                }
                self.pending = None;
                None
            }
            Command::Create { .. }
            | Command::View { .. }
            | Command::History { .. }
            | Command::Presentation { .. } => {
                unreachable!("creation handled by lifecycle; view handled above")
            }
        };
        if accepted {
            self.stored_moves += 1;
        }
        let mut view = self.view(response_observer)?;
        if full_history {
            view.moves = Some(self.move_list(self.game.active_branch(), view.tip)?);
        }
        if accepted {
            let record = self
                .game
                .moves(self.game.active_branch())
                .map_err(branch_error)?
                .last()
                .expect("accepted move was recorded");
            view.accepted_move = Some(self.move_view(record, view.tip)?);
        }
        Ok((view, proposal))
    }
}
