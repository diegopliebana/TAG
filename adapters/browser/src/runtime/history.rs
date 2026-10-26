//! Build display data for earlier positions and branch/move labels.
//!
//! Use the requested player's observation. Do not expose full agent checkpoints or
//! future hidden-game information merely because the recording contains it.
use super::{Runtime, fail};
use crate::errors::branch_error;
use crate::games::BrowserGame;
use crate::protocol::*;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::{Decision, PlayerId};
use tag_rust::recording::{ControllerSnapshot, RecordedMove};
use tag_rust::session::{BranchId, Controller};

pub(super) struct ViewLocation {
    pub branch: BranchId,
    pub cursor: usize,
    pub tip: usize,
    pub read_only: bool,
}
impl<F: BrowserGame> Runtime<F> {
    /// Convert a checkpoint to permitted controller identity without exposing RNG progress.
    ///
    /// # Arguments
    /// `self` and `snapshot` are passed by a shared (read-only) reference; the historical record retains ownership.
    ///
    /// # Returns
    /// Owned controller label/initial seed.
    ///
    /// # Errors
    /// Snapshot restoration failure, without changing live controllers.
    fn snapshot_settings(
        &self,
        snapshot: &ControllerSnapshot<<F::Agent as Checkpoint>::Snapshot>,
    ) -> Result<SeatSetup, BridgeError> {
        match snapshot {
            ControllerSnapshot::Human => Ok(SeatSetup::Human),
            ControllerSnapshot::Ai(data) => {
                let agent = F::Agent::from_snapshot(data)
                    .map_err(|e| BridgeError::new("restoration_failed", e.to_string()))?;
                Ok(F::settings(&agent))
            }
        }
    }
    /// Build display facts through the state's player-visible observation.
    ///
    /// # Arguments
    /// `self` and `state` are passed by a shared (read-only) reference. `observer` is a copied optional seat.
    /// `controllers` and `location` are consumed owned display identity/settings.
    ///
    /// # Returns
    /// Owned public view; historical locations omit actionable choices.
    ///
    /// # Errors
    /// Invalid observer or observation/value/record metadata failure.
    pub(super) fn make_view(
        &self,
        state: &F::State,
        observer: Option<usize>,
        controllers: Vec<SeatSetup>,
        location: ViewLocation,
    ) -> Result<PublicView, BridgeError> {
        let session = self.game.active().session();
        let model = session.model();
        let next = match model.next_player(state) {
            Decision::Player(p) => Some(p.index()),
            Decision::Finished => None,
        };
        let public_only = F::HIDDEN_INFORMATION
            && (self.public_only
                || (observer.is_none()
                    && self.view_observer.is_none()
                    && !controllers.iter().any(|s| matches!(s, SeatSetup::Human))));
        let automatic = if F::HIDDEN_INFORMATION {
            next.filter(|&p| matches!(controllers[p], SeatSetup::Human))
                .or_else(|| {
                    self.game
                        .recorded_prefix(location.branch, location.cursor)
                        .ok()?
                        .iter()
                        .rev()
                        .find(|m| {
                            matches!(m.controller_after, ControllerSnapshot::Human)
                                && matches!(controllers[m.actor.index()], SeatSetup::Human)
                        })
                        .map(|m| m.actor.index())
                })
                .or_else(|| {
                    controllers
                        .iter()
                        .position(|s| matches!(s, SeatSetup::Human))
                })
        } else {
            next
        };
        let observer = observer.or(self.view_observer).or(automatic).unwrap_or(0);
        let observation = model
            .observe(state, PlayerId::new(observer))
            .map_err(|e| BridgeError::new("invalid_observer", e.to_string()))?;
        let actions = if !location.read_only
            && next == Some(observer)
            && (!F::HIDDEN_INFORMATION
                || (!public_only && matches!(controllers[observer], SeatSetup::Human)))
        {
            model
                .available_actions(state, PlayerId::new(observer))
                .map_err(|e| fail(e.to_string()))?
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| fail(e.to_string()))?
        } else {
            vec![]
        };
        let mut results = Vec::new();
        let mut scores = Vec::new();
        for seat in 0..model.player_count(state) {
            let (result, score) = F::player_values(&observation, PlayerId::new(seat))
                .map_err(|e| fail(e.to_string()))?;
            results.push(result);
            scores.push(score);
        }
        let mut branches = Vec::new();
        for id in 0..self.game.branch_count() {
            let info = self.game.info(BranchId(id)).map_err(branch_error)?;
            branches.push(BranchSummary {
                id,
                parent: info.parent.map(|(parent, _)| parent.0),
                fork_position: info.parent.map_or(0, |(_, position)| position),
                move_count: info.move_count,
                origin: self.origins[id],
            });
        }
        Ok(PublicView {
            game: F::KIND,
            position_version: session.position_version().to_string(),
            branch: location.branch.0,
            observer,
            next_player: next,
            controllers: controllers.into_iter().map(F::display_settings).collect(),
            board: F::display_board(
                state,
                &observation,
                self.reveal_hidden_information,
                public_only,
            ),
            actions,
            results,
            scores,
            cursor: location.cursor,
            tip: location.tip,
            read_only: location.read_only,
            branches,
            moves: None,
            accepted_move: None,
        })
    }
    /// Copy permitted move/source metadata without transferring the checkpoint.
    ///
    /// # Arguments
    /// `self` and `record` are passed by a shared (read-only) reference; `position` is the copied recording index.
    ///
    /// # Returns
    /// Owned transport metadata with actor, action and controller label/seed.
    ///
    /// # Errors
    /// Snapshot restoration or action serialization failure.
    pub(super) fn move_view(
        &self,
        record: &RecordedMove<F::Action, <F::Agent as Checkpoint>::Snapshot>,
        position: usize,
    ) -> Result<MoveView, BridgeError> {
        Ok(MoveView {
            position,
            actor: record.actor.index(),
            action: serde_json::to_value(&record.action).map_err(|e| fail(e.to_string()))?,
            controller: F::display_settings(self.snapshot_settings(&record.controller_after)?),
        })
    }
    /// Read chronological permitted move metadata, including shared ancestors.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; `branch` and `position` are copied branch/prefix coordinates.
    ///
    /// # Returns
    /// Owned metadata vector; original records and checkpoints stay in the session.
    ///
    /// # Errors
    /// Unknown branch/position or controller/serialization failure.
    pub(super) fn move_list(
        &self,
        branch: BranchId,
        position: usize,
    ) -> Result<Vec<MoveView>, BridgeError> {
        self.game
            .recorded_prefix(branch, position)
            .map_err(branch_error)?
            .into_iter()
            .enumerate()
            .map(|(index, record)| self.move_view(record, index + 1))
            .collect()
    }
    /// Reconstruct a recorded position without rewinding authoritative state or RNG.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference for caching/display selection. `branch`, `cursor`
    /// and optional `observer` are copied recording/seat coordinates.
    ///
    /// # Returns
    /// Owned view containing only information the player is allowed to see/list; the adapter remembers whether it is read-only.
    ///
    /// # Errors
    /// Invalid branch/cursor/observer or reconstruction/metadata failure; previous
    /// display selection and authoritative progress remain intact.
    pub(super) fn history_view(
        &mut self,
        branch: BranchId,
        cursor: usize,
        observer: Option<usize>,
    ) -> Result<PublicView, BridgeError> {
        let tip = self.game.info(branch).map_err(branch_error)?.move_count;
        let state = self
            .game
            .position_at(branch, cursor)
            .map_err(branch_error)?;
        let controllers = self
            .game
            .controllers_at(branch, cursor)
            .map_err(branch_error)?
            .iter()
            .map(|snapshot| self.snapshot_settings(snapshot))
            .collect::<Result<Vec<_>, _>>()?;
        let read_only = branch != self.game.active_branch() || cursor != tip;
        let mut view = self.make_view(
            &state,
            observer,
            controllers,
            ViewLocation {
                branch,
                cursor,
                tip,
                read_only,
            },
        )?;
        view.moves =
            Some(self.move_list(branch, if F::HIDDEN_INFORMATION { cursor } else { tip })?);
        self.display = read_only.then_some((branch, cursor));
        Ok(view)
    }
    /// Check collection capacity before creating a continuation.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference and `parent` is the copied proposed ancestor ID.
    ///
    /// # Returns
    /// Unit when the proposed branch fits 256 branches and depth 64 (including root).
    ///
    /// # Errors
    /// Unknown branch or a browser resource limit.
    pub(super) fn check_branch_capacity(&self, parent: BranchId) -> Result<(), BridgeError> {
        let mut info = self.game.info(parent).map_err(branch_error)?;
        let mut depth = 2;
        while let Some((id, _)) = info.parent {
            depth += 1;
            info = self.game.info(id).map_err(branch_error)?;
        }
        if self.game.branch_count() >= 256 || depth > 64 {
            return Err(BridgeError::new(
                "resource_limit",
                "browser limit is 256 branches and ancestry depth 64",
            ));
        }
        Ok(())
    }
    /// Install a validated continuation while preserving every source branch.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference. `branch` and `cursor` are copied source
    /// coordinates. `replacements` is consumed: None preserves a checkpoint;
    /// Some supplies settings for a newly owned controller.
    ///
    /// # Returns
    /// Unit after activating the new branch.
    ///
    /// # Errors
    /// Invalid branch/position, controller count/settings/restoration or board/branch
    /// resource limit, checked before installing a replacement.
    pub(super) fn fork_branch(
        &mut self,
        branch: BranchId,
        cursor: usize,
        replacements: Vec<Option<SeatSetup>>,
    ) -> Result<(), BridgeError> {
        self.check_branch_capacity(branch)?;
        let state = self
            .game
            .position_at(branch, cursor)
            .map_err(branch_error)?;
        let model = self.game.active().session().model();
        if replacements.len() != model.player_count(&state) {
            return Err(BridgeError::new(
                "invalid_setup",
                "expected one replacement entry per seat",
            ));
        }
        let setups: Vec<_> = replacements
            .iter()
            .cloned()
            .map(|s| s.unwrap_or(SeatSetup::Human))
            .collect();
        F::validate_seats(&self.parameters, &setups, &self.models)?;
        let observation = model
            .observe(&state, PlayerId::new(0))
            .map_err(|e| fail(e.to_string()))?;
        let board = F::board(&observation);
        if replacements
            .iter()
            .any(|s| matches!(s, Some(SeatSetup::Osla { .. })))
            && board.width * board.height > 256
        {
            return Err(BridgeError::new(
                "resource_limit",
                "browser OSLA limit is 256 cells",
            ));
        }
        let controllers = replacements
            .into_iter()
            .map(|setting| {
                setting
                    .map(|setup| {
                        F::controller_with_models(
                            self.game.active().session().model(),
                            &self.parameters,
                            setup,
                            &self.models,
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<Option<Controller<F::Agent>>>, _>>()?;
        self.game
            .fork(branch, cursor, controllers)
            .map_err(branch_error)?;
        Ok(())
    }
}
