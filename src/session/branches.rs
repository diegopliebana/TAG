//! Keep alternative lines of play without erasing the original continuation.
//!
//! A branch can reuse moves recorded by its parent and add its own later moves.
//! Forking restores the selected position and agent progress; switching branches
//! restores that branch's end. Branch IDs belong to one session or replay file.
use super::{AcceptedMove, Controller, RecordedSession, RecordingError, Session};
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::{Decision, ForwardModel, PlayerId};
use crate::recording::{
    ControllerSnapshot, GameHistory, HistoricalView, HistoryError, RecordedMove,
};

/// A branch identifier whose meaning stays fixed within this session or file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: index; PartialEq/Eq: exact identity.
pub struct BranchId(pub usize);
/// A branch's parent/fork position and total move count, including inherited moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: index metadata; PartialEq/Eq: exact fields.
pub struct BranchInfo {
    pub id: BranchId,
    pub parent: Option<(BranchId, usize)>,
    pub move_count: usize,
}
/// Invalid branch navigation, controller replacement or continuation.
#[derive(Debug)] // Debug: transferred contextual errors.
pub enum BranchError<G, A, C> {
    UnknownBranch(BranchId),
    PositionVersionExhausted,
    InvalidPosition { requested: usize, last: usize },
    ControllerCount { expected: usize, actual: usize },
    Execution(RecordingError<G, A, C>),
    History(HistoryError<G>),
    Restore(C),
}
impl<G: std::fmt::Display, A: std::fmt::Display, C: std::fmt::Display> std::fmt::Display
    for BranchError<G, A, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PositionVersionExhausted => {
                write!(f, "branch session position version limit reached")
            }
            Self::UnknownBranch(id) => write!(f, "unknown branch {}", id.0),
            Self::InvalidPosition { requested, last } => {
                write!(f, "branch position {requested} exceeds {last}")
            }
            Self::ControllerCount { expected, actual } => write!(
                f,
                "expected {expected} replacement seats, received {actual}"
            ),
            Self::Execution(e) => e.fmt(f),
            Self::History(e) => e.fmt(f),
            Self::Restore(e) => e.fmt(f),
        }
    }
}
impl<G: std::error::Error + 'static, A: std::error::Error + 'static, C: std::error::Error + 'static>
    std::error::Error for BranchError<G, A, C>
{
}
/// Owned branch result with the selected rules/controller error types.
pub type BranchExecutionResult<T, F, A> = Result<
    T,
    BranchError<
        <F as ForwardModel>::Error,
        <A as Agent<F>>::Error,
        <A as Checkpoint>::RestoreError,
    >,
>;

/// Shared continuation records or typed branch failure.
pub type BranchMovesResult<'a, F, A> = BranchExecutionResult<
    &'a [RecordedMove<<F as ForwardModel>::Action, <A as Checkpoint>::Snapshot>],
    F,
    A,
>;

/// Owned list of borrowed accepted records including a branch's inherited prefix.
pub type BranchRecordsResult<'a, F, A> = BranchExecutionResult<
    Vec<&'a RecordedMove<<F as ForwardModel>::Action, <A as Checkpoint>::Snapshot>>,
    F,
    A,
>;
/// Owned chronological controller checkpoints at absolute recording boundaries.
pub type ControllerTimeline<C> = Vec<(usize, Vec<ControllerSnapshot<C>>)>;
struct Branch<M, C> {
    parent: Option<(BranchId, usize)>,
    controllers: Vec<ControllerSnapshot<C>>,
    moves: Vec<RecordedMove<M, C>>,
    changes: ControllerTimeline<C>,
}
/// Own one live continuation and a tree of preserved accepted-move continuations.
/// Inactive branches retain their local records and controller roots; inherited moves
/// remain in their ancestors. Only the active continuation owns a live state/cache.
pub struct BranchSession<F: ForwardModel, A: Agent<F> + Checkpoint> {
    model: F,
    root: F::State,
    branches: Vec<Branch<F::Action, A::Snapshot>>,
    active_id: BranchId,
    active: RecordedSession<F, A>,
}
impl<F, A> BranchSession<F, A>
where
    F: ForwardModel + Clone,
    F::State: Clone,
    A: Agent<F> + Checkpoint,
{
    /// Consume a session into branch zero, rooted at its current validated state.
    /// Returns an owned branch collection; initial agent progress is captured exactly.
    ///
    /// # Arguments
    /// `session` is consumed. Its current state and controllers become branch zero;
    /// state/rules/snapshot copies preserve independent owned data.
    ///
    /// # Returns
    /// An owned branch collection with the supplied session active.
    pub fn new(session: Session<F, A>) -> Self {
        let model = session.model().clone();
        let root = session.state().clone();
        let active = RecordedSession::new(session);
        let controllers = active.history().initial_controllers().to_vec();
        Self {
            model,
            root,
            branches: vec![Branch {
                parent: None,
                controllers,
                moves: vec![],
                changes: vec![],
            }],
            active_id: BranchId(0),
            active,
        }
    }
    /// Consume caller-trusted headless history into branch zero with a live tip.
    /// Rules/state reconstruction and controller restoration are checked before return.
    /// External files use the replay loader's additional version/AI validation.
    ///
    /// # Arguments
    /// `model` and caller-trusted `history` are consumed. The model must match the
    /// recorded rules. History includes root state, initial seat snapshots and accepted moves.
    ///
    /// # Returns
    /// An owned branch-zero session restored at the recording tip.
    ///
    /// # Errors
    /// Invalid recorded moves/decisions, controller counts or checkpoint restoration.
    pub fn from_history(
        model: F,
        mut history: GameHistory<F::State, F::Action, A::Snapshot>,
    ) -> BranchExecutionResult<Self, F, A> {
        let last = history.moves().len();
        let tip_state = history
            .position_at(&model, last)
            .map_err(BranchError::History)?;
        let tip_controllers = history
            .controllers_at(last)
            .map_err(BranchError::History)?
            .iter()
            .map(Controller::from_snapshot)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BranchError::Restore)?;
        let initial_controllers = history
            .initial_controllers()
            .iter()
            .map(Controller::from_snapshot)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BranchError::Restore)?;
        let tip = Session::from_state(model.clone(), tip_state, tip_controllers)
            .map_err(|e| BranchError::Execution(RecordingError::Session(e)))?;
        let root = Session::from_state(model, history.initial_state().clone(), initial_controllers)
            .map_err(|e| BranchError::Execution(RecordingError::Session(e)))?;
        let mut result = Self::new(root);
        result.active.history = history;
        result.active.session = tip;
        Ok(result)
    }
    /// Preserve existing continuations and start a new branch from the original root.
    /// Original root controller settings/progress are restored; returns its branch ID.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference to add and activate a new branch.
    ///
    /// # Returns
    /// The new branch ID, rooted at the original state and controller progress.
    ///
    /// # Errors
    /// History/controller restoration failure or exhausted position version.
    pub fn reset(&mut self) -> BranchExecutionResult<BranchId, F, A> {
        let seats = self.model.player_count(&self.root);
        self.fork(BranchId(0), 0, (0..seats).map(|_| None).collect())
    }
    /// Shared active recorded session; mutation uses the checked branch operations.
    pub fn active(&self) -> &RecordedSession<F, A> {
        &self.active
    }
    /// Copied current branch identifier.
    pub fn active_branch(&self) -> BranchId {
        self.active_id
    }
    /// Shared full root state for native export.
    pub fn root(&self) -> &F::State {
        &self.root
    }
    /// Copied branch count.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
    /// Copy checked ancestry and absolute tip position for this branch.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` is a copied branch identifier.
    ///
    /// # Returns
    /// Copied ancestry and total accepted moves, including inherited moves.
    ///
    /// # Errors
    /// Unknown branch ID.
    pub fn info(&self, id: BranchId) -> BranchExecutionResult<BranchInfo, F, A> {
        let b = self
            .branches
            .get(id.0)
            .ok_or(BranchError::UnknownBranch(id))?;
        Ok(BranchInfo {
            id,
            parent: b.parent,
            move_count: b.parent.map_or(0, |(_, p)| p) + self.moves(id)?.len(),
        })
    }
    /// Borrow only the continuation records owned by the selected branch.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` is copied.
    ///
    /// # Returns
    /// A shared slice of that branch's local moves, borrowed for the lifetime of `self`.
    /// Inherited moves are reached through its parent; no records are copied.
    ///
    /// # Errors
    /// Unknown branch ID.
    pub fn moves(&self, id: BranchId) -> BranchMovesResult<'_, F, A> {
        let b = self
            .branches
            .get(id.0)
            .ok_or(BranchError::UnknownBranch(id))?;
        Ok(if id == self.active_id {
            self.active.history().moves()
        } else {
            &b.moves
        })
    }

    /// Read accepted records up to an absolute position, including shared ancestors.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference. `id` and `position` are copied branch/recording
    /// coordinates; position zero is the original recording root.
    ///
    /// # Returns
    /// An owned vector of shared record references in chronological order. Records
    /// and checkpoints remain owned by this session; no agent or state is copied.
    /// The references' inferred lifetime is bounded by the shared session borrow.
    ///
    /// # Errors
    /// Unknown branch or position beyond its accepted continuation.
    pub fn recorded_prefix(&self, id: BranchId, position: usize) -> BranchRecordsResult<'_, F, A> {
        let mut records = Vec::new();
        for (branch, count) in self.route(id, position)? {
            records.extend(self.moves(branch)?.iter().take(count));
        }
        Ok(records)
    }
    /// Borrow controller snapshots at the selected branch's continuation root.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` is copied.
    ///
    /// # Returns
    /// Shared seat snapshots at this branch's continuation root, in seat order.
    ///
    /// # Errors
    /// Unknown branch ID.
    pub fn initial_controllers(
        &self,
        id: BranchId,
    ) -> BranchExecutionResult<&[ControllerSnapshot<A::Snapshot>], F, A> {
        Ok(&self
            .branches
            .get(id.0)
            .ok_or(BranchError::UnknownBranch(id))?
            .controllers)
    }
    /// Resolve shared ancestors and the required local move counts.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` and absolute `position` are copied.
    /// Position zero is the collection's original root; counts include inherited moves.
    ///
    /// # Returns
    /// Owned ancestor IDs and local move counts, ordered from root to requested position.
    ///
    /// # Errors
    /// Unknown branch or out-of-range position.
    fn route(
        &self,
        mut id: BranchId,
        mut position: usize,
    ) -> BranchExecutionResult<Vec<(BranchId, usize)>, F, A> {
        let last = self.info(id)?.move_count;
        if position > last {
            return Err(BranchError::InvalidPosition {
                requested: position,
                last,
            });
        }
        let mut route = Vec::new();
        loop {
            let branch = &self.branches[id.0];
            match branch.parent {
                Some((parent, fork)) => {
                    if position >= fork {
                        route.push((id, position - fork));
                        position = fork;
                    }
                    id = parent;
                }
                None => {
                    route.push((id, position));
                    break;
                }
            }
        }
        route.reverse();
        Ok(route)
    }
    /// Reconstruct independent state at an absolute branch position through real rules.
    /// Rules/records are passed by a shared (read-only) reference; owned state or contextual errors are returned.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference to update the active reconstruction cache.
    /// `id` selects a branch; `position` counts accepted moves from the original root,
    /// including inherited moves. Both values are copied.
    ///
    /// # Returns
    /// Independently owned state at that position; live state and agents retain their progress.
    ///
    /// # Errors
    /// Unknown branch, out-of-range position, rule failure or recorded decision mismatch.
    pub fn position_at(
        &mut self,
        id: BranchId,
        position: usize,
    ) -> BranchExecutionResult<F::State, F, A> {
        let last = self.info(id)?.move_count;
        if position > last {
            return Err(BranchError::InvalidPosition {
                requested: position,
                last,
            });
        }
        let fork = self.branches[id.0].parent.map_or(0, |(_, p)| p);
        if id == self.active_id && position >= fork {
            return self
                .active
                .position_at(position - fork)
                .map_err(BranchError::History);
        }
        let mut state = self.root.clone();
        let mut completed = 0;
        for (branch, count) in self.route(id, position)? {
            for record in self.moves(branch)?.iter().take(count) {
                completed += 1;
                self.model
                    .next(&mut state, record.actor, &record.action)
                    .map_err(|error| {
                        BranchError::History(HistoryError::Game {
                            position: completed,
                            error,
                        })
                    })?;
                if self.model.next_player(&state) != record.decision {
                    return Err(BranchError::History(HistoryError::DecisionMismatch {
                        position: completed,
                    }));
                }
            }
        }
        Ok(state)
    }
    /// Return owned permitted historical facts/actions for a copied observer and position.
    /// Live state/controllers remain unchanged; invalid observers use the game's error.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference for reconstruction caching. `id` and `position`
    /// select a branch and absolute move count. `observer` is the copied player whose
    /// information the player is allowed to know and offered moves are requested.
    ///
    /// # Returns
    /// Owned observation/actions for the historical position.
    ///
    /// # Errors
    /// Invalid branch/position, reconstruction failure or an invalid observer.
    pub fn view_at(
        &mut self,
        id: BranchId,
        position: usize,
        observer: PlayerId,
    ) -> BranchExecutionResult<HistoricalView<F::Observation, F::Action>, F, A> {
        let state = self.position_at(id, position)?;
        let observation = self
            .model
            .observe(&state, observer)
            .map_err(|error| BranchError::History(HistoryError::Game { position, error }))?;
        let actions = if self.model.next_player(&state) == Decision::Player(observer) {
            self.model
                .available_actions(&state, observer)
                .map_err(|error| BranchError::History(HistoryError::Game { position, error }))?
        } else {
            vec![]
        };
        Ok(HistoricalView {
            position,
            observation,
            actions,
        })
    }
    /// Copy each seat's latest progress at an absolute branch position.
    /// Branch-root snapshots account for controller replacements; ancestors supply moves.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference. Copied `id` and absolute `position` select a historical
    /// move boundary, including inherited moves and controller replacements.
    ///
    /// # Returns
    /// Owned seat snapshots in player order; each final snapshot is copied once.
    ///
    /// # Errors
    /// Unknown branch/position or inconsistent recorded acting seat.
    pub fn controllers_at(
        &self,
        id: BranchId,
        position: usize,
    ) -> BranchExecutionResult<Vec<ControllerSnapshot<A::Snapshot>>, F, A> {
        let mut seats = Vec::new();
        for (branch, count) in self.route(id, position)? {
            seats = self.branches[branch.0].controllers.iter().collect();
            let offset = self.branches[branch.0].parent.map_or(0, |(_, p)| p);
            let mut changes = self.branches[branch.0].changes.iter().peekable();
            for (index, record) in self.moves(branch)?.iter().take(count).enumerate() {
                while changes.peek().is_some_and(|(p, _)| *p <= offset + index) {
                    seats = changes.next().unwrap().1.iter().collect();
                }
                let expected = seats.len();
                let seat =
                    seats
                        .get_mut(record.actor.index())
                        .ok_or(BranchError::ControllerCount {
                            expected,
                            actual: record.actor.index() + 1,
                        })?;
                *seat = &record.controller_after;
            }
            while changes.peek().is_some_and(|(p, _)| *p <= offset + count) {
                seats = changes.next().unwrap().1.iter().collect();
            }
        }
        Ok(seats.into_iter().cloned().collect())
    }
    /// Copy controller boundaries along one selected line, without fork relationships.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` identifies the selected branch.
    ///
    /// # Returns
    /// Owned chronological (absolute position, seat snapshots) including move zero.
    /// Repeated boundaries at the same position retain the last selected replacement.
    ///
    /// # Errors
    /// Unknown branch or invalid inherited position.
    pub fn controller_timeline(
        &self,
        id: BranchId,
    ) -> BranchExecutionResult<ControllerTimeline<A::Snapshot>, F, A> {
        let mut result: ControllerTimeline<A::Snapshot> = Vec::new();
        for (branch, count) in self.route(id, self.info(id)?.move_count)? {
            let data = &self.branches[branch.0];
            let start = data.parent.map_or(0, |(_, p)| p);
            for (position, seats) in std::iter::once((start, &data.controllers)).chain(
                data.changes
                    .iter()
                    .filter(|(p, _)| *p <= start + count)
                    .map(|(p, s)| (*p, s)),
            ) {
                if result.last().is_some_and(|(p, _)| *p == position) {
                    result.pop();
                }
                result.push((position, seats.clone()));
            }
        }
        Ok(result)
    }
    /// Borrow local controller changes for portable recording export.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `id` selects a branch, excluding inherited changes.
    ///
    /// # Returns
    /// Borrowed chronological checkpoint boundaries owned by this branch.
    ///
    /// # Errors
    /// Unknown branch identifier.
    pub fn controller_changes(
        &self,
        id: BranchId,
    ) -> BranchExecutionResult<&ControllerTimeline<A::Snapshot>, F, A> {
        Ok(&self
            .branches
            .get(id.0)
            .ok_or(BranchError::UnknownBranch(id))?
            .changes)
    }
    /// Install validated controller replacements at the current recorded boundary.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; owned `controllers` supplies every seat.
    /// Caller validates policy/configuration compatibility before transferring ownership.
    ///
    /// # Returns
    /// Unit after recording the boundary and invalidating old position requests.
    ///
    /// # Errors
    /// Wrong seat count or exhausted position version, before mutation.
    pub fn replace_controllers(
        &mut self,
        controllers: Vec<Controller<A>>,
    ) -> BranchExecutionResult<(), F, A> {
        let expected = self.model.player_count(self.active.session().state());
        if controllers.len() != expected {
            return Err(BranchError::ControllerCount {
                expected,
                actual: controllers.len(),
            });
        }
        let version = self
            .active
            .session()
            .position_version()
            .checked_add(1)
            .ok_or(BranchError::PositionVersionExhausted)?;
        let position = self.info(self.active_id)?.move_count;
        let snapshots = controllers.iter().map(Controller::snapshot).collect();
        if self.branches[self.active_id.0]
            .changes
            .last()
            .is_some_and(|(p, _)| *p == position)
        {
            self.branches[self.active_id.0].changes.pop();
        }
        self.branches[self.active_id.0]
            .changes
            .push((position, snapshots));
        self.active.session.controllers = controllers;
        self.active.session.position_version = version;
        Ok(())
    }
    /// Build an independent live session at a recorded branch position.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference for reconstruction caching. `id` and absolute
    /// `position` are copied; the rules are cloned for the returned session.
    ///
    /// # Returns
    /// Independent owned session with exact historical controller progress.
    ///
    /// # Errors
    /// History, controller-count or checkpoint restoration failure.
    fn restore(
        &mut self,
        id: BranchId,
        position: usize,
    ) -> BranchExecutionResult<Session<F, A>, F, A> {
        let state = self.position_at(id, position)?;
        let seats = self
            .controllers_at(id, position)?
            .iter()
            .map(Controller::from_snapshot)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BranchError::Restore)?;
        Session::from_state(self.model.clone(), state, seats)
            .map_err(|e| BranchError::Execution(RecordingError::Session(e)))
    }
    /// Preserve the old continuation and install a validated active session.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference. `id` is the copied destination branch ID;
    /// `next` is consumed and must already contain its validated state/history.
    ///
    /// # Returns
    /// Unit after transferring the previous continuation's records to its branch
    /// and installing the new live continuation.
    fn replace_active(&mut self, id: BranchId, next: RecordedSession<F, A>) {
        let old = std::mem::replace(&mut self.active, next).into_parts();
        self.branches[self.active_id.0].moves = old.history.into_moves();
        self.active_id = id;
    }
    /// Fork at a checked historical position, consuming optional replacement controllers.
    /// None preserves that seat's exact progress; Some starts the supplied controller.
    /// Returns a new active branch ID; source continuation remains available on failure/success.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference. `id` selects the source branch; copied
    /// `position` is an absolute move count from the original root.
    /// `replacements` is consumed and must contain one entry per seat: `None` preserves
    /// that seat's checkpoint; `Some(controller)` transfers the supplied controller
    /// into the new branch, including its current seed/progress.
    ///
    /// # Returns
    /// The new active branch ID. Its parent continuation remains available.
    ///
    /// # Errors
    /// Wrong replacement count, unknown branch/position, history or checkpoint failure,
    /// or exhausted position version. Rejections preserve live execution.
    pub fn fork(
        &mut self,
        id: BranchId,
        position: usize,
        replacements: Vec<Option<Controller<A>>>,
    ) -> BranchExecutionResult<BranchId, F, A> {
        let expected = self.model.player_count(&self.root);
        if replacements.len() != expected {
            return Err(BranchError::ControllerCount {
                expected,
                actual: replacements.len(),
            });
        }
        let mut session = self.restore(id, position)?;
        for (seat, replacement) in replacements.into_iter().enumerate() {
            if let Some(c) = replacement {
                session.controllers[seat] = c;
            }
        }
        let version = self
            .active
            .session()
            .position_version()
            .checked_add(1)
            .ok_or(BranchError::PositionVersionExhausted)?;
        let mut next =
            RecordedSession::with_cache_policy(session, self.active.history().cache_policy());
        // A branch change advances the version to reject delayed input from the previous branch.
        next.session.position_version = version;
        let new_id = BranchId(self.branches.len());
        let controllers = next.history().initial_controllers().to_vec();
        self.branches.push(Branch {
            parent: Some((id, position)),
            controllers,
            moves: vec![],
            changes: vec![],
        });
        self.replace_active(new_id, next);
        Ok(new_id)
    }
    /// Select a preserved branch and restore its tip with exact controller progress.
    /// Validation/restoration completes before replacing the live session.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; copied `id` identifies the branch to activate.
    ///
    /// # Returns
    /// Unit after restoring that branch's recorded tip and advancing position version.
    /// Selecting the active branch keeps the current session.
    ///
    /// # Errors
    /// Unknown branch, restoration failure or exhausted position version.
    pub fn select(&mut self, id: BranchId) -> BranchExecutionResult<(), F, A> {
        if id == self.active_id {
            return Ok(());
        }
        let last = self.info(id)?.move_count;
        let fork = self.branches[id.0].parent.map_or(0, |(_, p)| p);
        let root = self.restore(id, fork)?;
        // Restore the selected tip while retaining its local root history.
        let snapshots = self.controllers_at(id, last)?;
        // Start at this branch's restored root and apply only its local continuation.
        let mut state = root.state().clone();
        for (index, record) in self.moves(id)?.iter().enumerate() {
            let position = fork + index + 1;
            self.model
                .next(&mut state, record.actor, &record.action)
                .map_err(|error| BranchError::History(HistoryError::Game { position, error }))?;
            if self.model.next_player(&state) != record.decision {
                return Err(BranchError::History(HistoryError::DecisionMismatch {
                    position,
                }));
            }
        }
        let controllers = snapshots
            .iter()
            .map(Controller::from_snapshot)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BranchError::Restore)?;
        let mut tip = Session::from_state(self.model.clone(), state, controllers)
            .map_err(|e| BranchError::Execution(RecordingError::Session(e)))?;
        tip.position_version = self
            .active
            .session()
            .position_version()
            .checked_add(1)
            .ok_or(BranchError::PositionVersionExhausted)?;
        let records = std::mem::take(&mut self.branches[id.0].moves);
        let mut next =
            RecordedSession::with_cache_policy(root, self.active.history().cache_policy());
        for record in records {
            next.history.record(record);
        }
        next.session = tip;
        self.replace_active(id, next);
        Ok(())
    }
    /// Submit a consumed human action to the active continuation and record acceptance.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference. `version` is copied from the live displayed
    /// position; `actor` is the copied acting player. `action` is consumed.
    ///
    /// # Returns
    /// Owned accepted move with its resulting version/decision; action is cloned into history.
    ///
    /// # Errors
    /// Stale version, wrong actor/controller, terminal game, unoffered action or rule rejection.
    pub fn submit_human(
        &mut self,
        version: u64,
        actor: PlayerId,
        action: F::Action,
    ) -> BranchExecutionResult<AcceptedMove<F::Action>, F, A>
    where
        F::Action: Clone + PartialEq,
    {
        self.active
            .submit_human(version, actor, action)
            .map_err(BranchError::Execution)
    }
    /// Borrow the active session and copy its version to prepare an independent AI proposal.
    /// Returns owned temporary progress or a typed error; live progress is preserved.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `version` is copied from the current live view.
    ///
    /// # Returns
    /// Owned proposed action and temporary controller progress; dropping it preserves live progress.
    ///
    /// # Errors
    /// Stale version, human/finished decision, checkpoint restoration or game/agent failure.
    pub fn prepare_ai(&self, version: u64) -> super::PreparedAiResult<F, A> {
        self.active.prepare_ai(version)
    }
    /// Own an independent suspendable AI calculation for the active continuation.
    /// Dropping the task cancels preparation without changing recorded progress.
    pub fn prepare_ai_task(
        &self,
        version: u64,
    ) -> super::RecordingResult<
        impl std::future::Future<Output = super::PreparedAiResult<F, A>> + use<F, A>,
        F::Error,
        A::Error,
        A::RestoreError,
    >
    where
        F: Clone,
        F::Action: Clone,
    {
        self.active.prepare_ai_task(version)
    }
    /// Apply a caller-validated recorded choice without repeating the original
    /// computation. Its accepted action and restored controller are recorded together.
    pub fn apply_recorded_ai(
        &mut self,
        version: u64,
        action: F::Action,
        after: &A::Snapshot,
    ) -> BranchExecutionResult<AcceptedMove<F::Action>, F, A>
    where
        F::Action: Clone + PartialEq,
        A::Snapshot: PartialEq,
    {
        let proposal = self
            .active
            .recorded_ai_choice(version, action, after)
            .map_err(BranchError::Execution)?;
        self.commit_ai(proposal)
    }
    /// Consume a prepared action into the active continuation with origin/version checks.
    /// Returns an accepted move or typed failure; branch changes invalidate old proposals.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; `proposal` is consumed. It must originate from
    /// the same active recorded session and still match its version/controller progress.
    ///
    /// # Returns
    /// Owned accepted move, recorded with the prepared agent checkpoint.
    ///
    /// # Errors
    /// Foreign/stale proposal, changed controller, restoration or rule failure.
    pub fn commit_ai(
        &mut self,
        proposal: super::PreparedAi<F::Action, A::Snapshot>,
    ) -> BranchExecutionResult<AcceptedMove<F::Action>, F, A>
    where
        F::Action: Clone,
        A::Snapshot: PartialEq,
    {
        self.active
            .commit_ai(proposal)
            .map_err(BranchError::Execution)
    }
    /// Advance one active AI move, preserving proposal/recording semantics.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; `version` is copied from the live position.
    ///
    /// # Returns
    /// Owned accepted AI move after preparation and commitment, with history updated.
    ///
    /// # Errors
    /// The same request, preparation, restoration and commitment failures as the two-step API.
    pub fn advance_ai(
        &mut self,
        version: u64,
    ) -> BranchExecutionResult<AcceptedMove<F::Action>, F, A>
    where
        F::Action: Clone,
        A::Snapshot: PartialEq,
    {
        self.active
            .advance_ai(version)
            .map_err(BranchError::Execution)
    }
}
