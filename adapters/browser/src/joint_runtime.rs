//! Browser execution for complete simultaneous decisions.
use crate::{games::BrowserGame, protocol::*, sushigo};
use serde_json::Value;
use tag_rust::{
    agents::Agent,
    core::{ForwardModel, PlayerId, checkpoint::Checkpoint},
    games::sushigo::{Action, State, SushiGoForwardModel},
    recording::RecordedJointDecision,
    replay::JointChoiceData,
    session::{Controller, JointSession, JointSessionSnapshot},
};

type Game = SushiGoForwardModel;
type Player = sushigo::StochasticAgent;
type Snapshot = <Player as Checkpoint>::Snapshot;
type Boundary = JointSessionSnapshot<State, Snapshot>;
type Record = RecordedJointDecision<State, Action, Snapshot>;
type Calculation =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<(Action, Snapshot), BridgeError>>>>;

mod files;
mod history;

fn invalid(code: &'static str, error: impl std::fmt::Display) -> BridgeError {
    BridgeError::new(code, error.to_string())
}

struct Proposal {
    id: String,
    version: u64,
    serial: u64,
    actor: PlayerId,
    action: Action,
    after: Snapshot,
}

#[derive(Clone)]
struct JointBranch {
    parent: Option<usize>,
    fork_position: usize,
    base: Boundary,
    records: Vec<Record>,
    origin: BranchOrigin,
}

pub(crate) struct JointRuntime {
    game: JointSession<Game, Player>,
    root: Boundary,
    boundary: Boundary,
    records: Vec<Record>,
    branches: Vec<JointBranch>,
    active_branch: usize,
    display: Option<(usize, usize)>,
    root_choices: Vec<Vec<JointChoiceData<Action>>>,
    parameters: <Game as ForwardModel>::Parameters,
    models: crate::learned::Registry,
    observer: Option<usize>,
    public_only: bool,
    reveal_hidden_information: bool,
    proposal: Option<Proposal>,
    calculation: Option<(String, u64, u64, PlayerId, Calculation)>,
    next_proposal: u64,
    submission_serial: u64,
}

impl JointRuntime {
    fn poll_preparation(&mut self, id: &str) -> Result<ProposalView, BridgeError> {
        use std::task::{Context, Poll, Waker};
        if let Some((current, version, serial, actor, task)) = &mut self.calculation {
            if current != id {
                return Err(invalid("invalid_proposal", "unknown calculation"));
            }
            let (version, serial, actor) = (*version, *serial, *actor);
            let mut result = Poll::Pending;
            let clock = crate::mcts::BrowserClock::default();
            let started = tag_rust::search::mcts::Clock::now_ms(&clock);
            for _ in 0..16 {
                result = task.as_mut().poll(&mut Context::from_waker(Waker::noop()));
                if result.is_ready()
                    || tag_rust::search::mcts::Clock::now_ms(&clock) - started >= 4.
                {
                    break;
                }
            }
            return match result {
                Poll::Pending => Ok(ProposalView {
                    id: id.into(),
                    actor: actor.index(),
                    action: Value::Null,
                    pending: true,
                }),
                Poll::Ready(result) => {
                    self.calculation = None;
                    let (action, after) = result?;
                    self.proposal = Some(Proposal {
                        id: id.into(),
                        version,
                        serial,
                        actor,
                        action,
                        after,
                    });
                    Ok(ProposalView {
                        id: id.into(),
                        actor: actor.index(),
                        action: Value::Null,
                        pending: false,
                    })
                }
            };
        }
        let proposal = self
            .proposal
            .as_ref()
            .filter(|p| p.id == id)
            .ok_or_else(|| invalid("invalid_proposal", "unknown proposal"))?;
        Ok(ProposalView {
            id: id.into(),
            actor: proposal.actor.index(),
            action: Value::Null,
            pending: false,
        })
    }

    pub(crate) fn create(
        parameters: Option<Value>,
        seats: Option<Vec<SeatSetup>>,
        models: &crate::learned::Registry,
    ) -> Result<Self, BridgeError> {
        let parameters = Game::parse_parameters(parameters)?;
        let seats = seats.unwrap_or_else(|| vec![SeatSetup::Human; 3]);
        Game::validate_seats(&parameters, &seats, models)?;
        let players = seats.len();
        let controllers = seats
            .into_iter()
            .map(|seat| {
                Game::controller_with_models(&SushiGoForwardModel, &parameters, seat, models)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = SushiGoForwardModel
            .init_state(&parameters, players)
            .map_err(|e| invalid("invalid_setup", e))?;
        let game = JointSession::from_state(SushiGoForwardModel, state, controllers)
            .map_err(|e| invalid("invalid_setup", e))?;
        let root = game.snapshot().map_err(|e| invalid("invalid_setup", e))?;
        Ok(Self {
            game,
            boundary: root.clone(),
            root: root.clone(),
            records: Vec::new(),
            branches: vec![JointBranch {
                parent: None,
                fork_position: 0,
                base: root.clone(),
                records: Vec::new(),
                origin: BranchOrigin::Original,
            }],
            active_branch: 0,
            display: None,
            root_choices: Vec::new(),
            parameters,
            models: models.clone(),
            observer: None,
            public_only: false,
            reveal_hidden_information: false,
            proposal: None,
            calculation: None,
            next_proposal: 0,
            submission_serial: 0,
        })
    }

    fn waiting(&self) -> Vec<PlayerId> {
        self.game.waiting_seats()
    }
    fn controller_settings(&self) -> Vec<SeatSetup> {
        (0..self.game.model().player_count(self.game.state()))
            .map(|seat| {
                match self
                    .game
                    .controller(PlayerId::new(seat))
                    .expect("seat exists")
                {
                    Controller::Human => SeatSetup::Human,
                    Controller::Ai(agent) => Game::settings(agent),
                }
            })
            .collect()
    }
    pub(crate) fn view(&self, observer: Option<usize>) -> Result<PublicView, BridgeError> {
        let waiting = self.waiting();
        let seat = observer
            .or(self.observer)
            .unwrap_or_else(|| waiting.first().map_or(0, |p| p.index()));
        if seat >= self.game.model().player_count(self.game.state()) {
            return Err(invalid(
                "invalid_observer",
                "observer is outside game seats",
            ));
        }
        let input = self
            .game
            .decision_view(PlayerId::new(seat))
            .map_err(|e| invalid("execution", e))?;
        let concealed = !matches!(
            self.game.controller(PlayerId::new(seat)),
            Some(Controller::Human)
        ) || self.public_only
            || observer.is_some_and(|requested| self.observer != Some(requested));
        let board = Game::display_board(
            self.game.state(),
            &input.observation,
            self.reveal_hidden_information,
            concealed,
        );
        let mut results = Vec::new();
        let mut scores = Vec::new();
        for p in 0..self.game.model().player_count(self.game.state()) {
            let (result, score) = Game::player_values(&input.observation, PlayerId::new(p))
                .map_err(|e| invalid("execution", e))?;
            results.push(result);
            scores.push(score);
        }
        let actions = if concealed {
            Vec::new()
        } else {
            input
                .actions
                .iter()
                .map(|a| serde_json::to_value(a).expect("portable action"))
                .collect()
        };
        let controllers = self
            .controller_settings()
            .into_iter()
            .map(Game::display_settings)
            .collect();
        Ok(PublicView {
            game: GameKind::Sushigo,
            position_version: self.game.position_version().to_string(),
            branch: self.active_branch,
            observer: seat,
            next_player: waiting.first().map(|p| p.index()),
            waiting_seats: Some(waiting.iter().map(|p| p.index()).collect()),
            controllers,
            board,
            actions,
            results,
            scores,
            cursor: self.records.len(),
            tip: self.records.len(),
            read_only: false,
            branches: self.branch_summaries(),
            moves: None,
            accepted_move: None,
        })
    }
    fn accepted(
        &mut self,
        accepted: Option<tag_rust::session::AcceptedJointDecision<Action>>,
    ) -> Result<(), BridgeError> {
        self.submission_serial = self
            .submission_serial
            .checked_add(1)
            .ok_or_else(|| invalid("resource_limit", "submission serial exhausted"))?;
        self.proposal = None;
        self.calculation = None;
        if let Some(decision) = accepted {
            let after = self.game.snapshot().map_err(|e| invalid("execution", e))?;
            let record = Record {
                choices: decision.choices,
                state_after: after.state.clone(),
                next_seats: self.game.waiting_seats(),
                controllers_after: after.controllers.clone(),
            };
            self.records.push(record.clone());
            self.branches[self.active_branch].records.push(record);
            self.boundary = after;
        }
        Ok(())
    }
    pub(crate) fn execute(
        &mut self,
        command: Command,
        version: Option<String>,
    ) -> Result<(PublicView, Option<ProposalView>), BridgeError> {
        match command {
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
                self.proposal = None;
                self.calculation = None;
                return Ok((self.history_view(branch, cursor, observer)?, None));
            }
            Command::Presentation {
                observer,
                public_only,
                reveal_hidden_information,
            } => {
                if let Some(seat) = observer
                    && seat >= self.game.model().player_count(self.game.state())
                {
                    return Err(invalid(
                        "invalid_observer",
                        "observer is outside game seats",
                    ));
                }
                self.observer = observer;
                self.public_only = public_only;
                self.reveal_hidden_information = reveal_hidden_information;
                if let Some((branch, cursor)) = self.display {
                    return Ok((self.history_view(branch, cursor, None)?, None));
                }
                return Ok((self.view(None)?, None));
            }
            _ => {}
        }
        let expected = version
            .as_deref()
            .ok_or_else(|| invalid("invalid_request", "missing position version"))
            .and_then(crate::runtime::integer)?;
        if expected != self.game.position_version() {
            return Err(invalid("stale_position", "position changed"));
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
            return Err(invalid(
                "historical_position",
                "recorded decisions are read-only; fork or return to branch end",
            ));
        }
        if self.submission_serial == u64::MAX
            && matches!(
                command,
                Command::HumanMove { .. }
                    | Command::CommitAi { .. }
                    | Command::CancelJoint
                    | Command::Reset
                    | Command::SelectBranch { .. }
                    | Command::Fork { .. }
            )
        {
            return Err(invalid("resource_limit", "submission serial exhausted"));
        }
        if self
            .branches
            .iter()
            .map(|branch| branch.records.len().saturating_sub(branch.fork_position))
            .sum::<usize>()
            >= 100_000
            && matches!(
                command,
                Command::HumanMove { .. } | Command::CommitAi { .. }
            )
        {
            return Err(invalid("resource_limit", "record limit reached"));
        }
        let mut proposal = None;
        let accepted_command = matches!(
            command,
            Command::HumanMove { .. } | Command::CommitAi { .. }
        );
        let full_history = matches!(
            command,
            Command::Reset | Command::SelectBranch { .. } | Command::Fork { .. }
        );
        let prior_decisions = self.records.len();
        match command {
            Command::HumanMove { actor, action, .. } => {
                let choice: Action =
                    serde_json::from_value(action).map_err(|e| invalid("invalid_action", e))?;
                let accepted = self
                    .game
                    .submit_human(expected, PlayerId::new(actor), choice)
                    .map_err(|e| invalid("invalid_action", e))?;
                self.accepted(accepted)?;
                self.observer = None;
            }
            Command::PrepareAi => {
                let actor = *self
                    .waiting()
                    .first()
                    .ok_or_else(|| invalid("invalid_request", "no seat waits"))?;
                let Controller::Ai(agent) = self
                    .game
                    .controller(actor)
                    .ok_or_else(|| invalid("invalid_request", "missing seat"))?
                else {
                    return Err(invalid("invalid_request", "next seat is human"));
                };
                let mut copy = Player::from_snapshot(&agent.snapshot())
                    .map_err(|e| invalid("execution", e))?;
                let input = self
                    .game
                    .decision_view(actor)
                    .map_err(|e| invalid("execution", e))?;
                let next = self
                    .next_proposal
                    .checked_add(1)
                    .ok_or_else(|| invalid("resource_limit", "proposal IDs exhausted"))?;
                self.next_proposal = next;
                let task = async move {
                    let index = copy
                        .choose_async(
                            &SushiGoForwardModel,
                            actor,
                            &input.observation,
                            &input.actions,
                        )
                        .await
                        .map_err(|e| invalid("execution", e))?;
                    let action = *input
                        .actions
                        .get(index)
                        .ok_or_else(|| invalid("execution", "AI returned invalid choice"))?;
                    Ok((action, copy.snapshot()))
                };
                self.calculation = Some((
                    next.to_string(),
                    expected,
                    self.submission_serial,
                    actor,
                    Box::pin(task),
                ));
                self.proposal = None;
                proposal = Some(self.poll_preparation(&next.to_string())?);
            }
            Command::PollAi { proposal: id } => {
                proposal = Some(self.poll_preparation(&id)?);
            }
            Command::CommitAi { proposal: id, .. } => {
                let current = self
                    .proposal
                    .take()
                    .filter(|p| {
                        p.id == id && p.version == expected && p.serial == self.submission_serial
                    })
                    .ok_or_else(|| invalid("invalid_proposal", "stale or unknown proposal"))?;
                let accepted = self
                    .game
                    .submit_ai_choice(expected, current.actor, current.action, current.after)
                    .map_err(|e| invalid("invalid_proposal", e))?;
                self.accepted(accepted)?;
                self.observer = None;
            }
            Command::DiscardAi { proposal: id } => {
                if self.calculation.as_ref().is_some_and(|p| p.0 == id) {
                    self.calculation = None;
                    return Ok((self.view(None)?, None));
                }
                if self.proposal.as_ref().is_none_or(|p| p.id != id) {
                    return Err(invalid("invalid_proposal", "unknown proposal"));
                }
                self.proposal = None;
            }
            Command::CancelJoint => {
                let next = self
                    .game
                    .position_version()
                    .checked_add(1)
                    .ok_or_else(|| invalid("resource_limit", "position versions exhausted"))?;
                let mut snapshot = self.boundary.clone();
                snapshot.position_version = next;
                self.game = JointSession::from_snapshot(SushiGoForwardModel, &snapshot)
                    .map_err(|e| invalid("execution", e))?;
                self.boundary = snapshot;
                self.proposal = None;
                self.calculation = None;
                self.observer = None;
                self.submission_serial += 1;
            }
            Command::Reset => {
                if self.branches.len() >= 256 {
                    return Err(invalid("resource_limit", "browser limit is 256 branches"));
                }
                let next = self
                    .game
                    .position_version()
                    .checked_add(1)
                    .ok_or_else(|| invalid("resource_limit", "position versions exhausted"))?;
                let mut snapshot = self.root.clone();
                snapshot.position_version = next;
                self.game = JointSession::from_snapshot(SushiGoForwardModel, &snapshot)
                    .map_err(|e| invalid("execution", e))?;
                self.boundary = snapshot;
                self.records.clear();
                self.branches.push(JointBranch {
                    parent: Some(0),
                    fork_position: 0,
                    base: self.boundary.clone(),
                    records: Vec::new(),
                    origin: BranchOrigin::Reset,
                });
                self.active_branch = self.branches.len() - 1;
                self.display = None;
                self.proposal = None;
                self.calculation = None;
                self.reveal_hidden_information = false;
                self.observer = None;
                self.submission_serial += 1;
            }
            Command::SelectBranch { branch } => self.select_branch(branch)?,
            Command::Fork {
                branch,
                cursor,
                replacements,
            } => self.fork_branch(branch, cursor, replacements)?,
            Command::Create { .. }
            | Command::View { .. }
            | Command::Presentation { .. }
            | Command::History { .. } => {
                unreachable!()
            }
        }
        let mut view = self.view(None)?;
        if full_history {
            view.moves = Some(
                self.records
                    .iter()
                    .enumerate()
                    .map(|(position, record)| self.move_view(record, position + 1))
                    .collect(),
            );
        }
        if accepted_command
            && self.records.len() > prior_decisions
            && let Some(record) = self.records.last()
        {
            view.accepted_move = Some(self.move_view(record, self.records.len()));
        }
        Ok((view, proposal))
    }
    pub(crate) fn update_models(&mut self, models: &crate::learned::Registry) {
        self.models = models.clone();
    }
    pub(crate) fn file_data(&mut self) -> Result<(PublicView, PublicView, Value), BridgeError> {
        let live = self.view(None)?;
        let view = if let Some((branch, cursor)) = self.display {
            self.history_view(branch, cursor, None)?
        } else {
            let mut view = self.view(None)?;
            view.moves = Some(
                self.records
                    .iter()
                    .enumerate()
                    .map(|(position, record)| self.move_view(record, position + 1))
                    .collect(),
            );
            view
        };
        Ok((view, live, Game::parameters_view(&self.parameters)?))
    }
    pub(crate) fn predict_model(&self, id: &str, observer: usize) -> Result<Value, BridgeError> {
        let model = self
            .models
            .get(id)
            .ok_or_else(|| invalid("invalid_model", "missing model"))?;
        crate::learned::prediction(
            &SushiGoForwardModel,
            self.game.state(),
            &self.parameters,
            model,
            observer,
        )
    }
}
