//! Complete-decision history. Branches store their shared prefix for quick views.
use super::*;

impl JointRuntime {
    pub(super) fn move_view(&self, record: &Record, position: usize) -> MoveView {
        MoveView {
            position,
            actor: 0,
            action: serde_json::to_value(record.choices.iter().map(|(actor, action)| serde_json::json!({"actor":actor.index(),"action":action})).collect::<Vec<_>>()).expect("portable joint action"),
            controller: SeatSetup::Human,
        }
    }
    pub(super) fn branch_summaries(&self) -> Vec<BranchSummary> {
        self.branches
            .iter()
            .enumerate()
            .map(|(id, branch)| BranchSummary {
                id,
                parent: branch.parent,
                fork_position: branch.fork_position,
                move_count: branch.records.len(),
                origin: branch.origin,
            })
            .collect()
    }

    pub(super) fn snapshot_at(
        &self,
        branch: usize,
        cursor: usize,
    ) -> Result<Boundary, BridgeError> {
        let branch_data = self
            .branches
            .get(branch)
            .ok_or_else(|| invalid("invalid_branch", "unknown branch"))?;
        if cursor > branch_data.records.len() {
            return Err(invalid("invalid_position", "position exceeds branch end"));
        }
        if cursor == branch_data.fork_position {
            return Ok(branch_data.base.clone());
        }
        if cursor == 0 {
            return Ok(self.root.clone());
        }
        let record = &branch_data.records[cursor - 1];
        Ok(Boundary {
            state: record.state_after.clone(),
            controllers: record.controllers_after.clone(),
            position_version: 0,
        })
    }

    pub(super) fn history_view(
        &mut self,
        branch: usize,
        cursor: usize,
        observer: Option<usize>,
    ) -> Result<PublicView, BridgeError> {
        if self.game.snapshot().is_err() {
            return Err(invalid(
                "invalid_request",
                "finish or cancel pending choices before viewing history",
            ));
        }
        let mut snapshot = self.snapshot_at(branch, cursor)?;
        snapshot.position_version = self.game.position_version();
        let past = JointSession::from_snapshot(SushiGoForwardModel, &snapshot)
            .map_err(|e| invalid("invalid_file", e))?;
        let live = std::mem::replace(&mut self.game, past);
        let result = self.view(observer);
        self.game = live;
        let mut view = result?;
        let tip = self.branches[branch].records.len();
        view.branch = branch;
        view.cursor = cursor;
        view.tip = tip;
        view.read_only = branch != self.active_branch || cursor != tip;
        if view.read_only {
            view.actions.clear();
        }
        view.moves = Some(
            self.branches[branch]
                .records
                .iter()
                .take(cursor)
                .enumerate()
                .map(|(position, record)| self.move_view(record, position + 1))
                .collect(),
        );
        self.display = view.read_only.then_some((branch, cursor));
        Ok(view)
    }

    pub(super) fn select_branch(&mut self, branch: usize) -> Result<(), BridgeError> {
        let tip = self
            .branches
            .get(branch)
            .ok_or_else(|| invalid("invalid_branch", "unknown branch"))?
            .records
            .len();
        let mut snapshot = self.snapshot_at(branch, tip)?;
        snapshot.position_version = self
            .game
            .position_version()
            .checked_add(1)
            .ok_or_else(|| invalid("resource_limit", "position versions exhausted"))?;
        self.game = JointSession::from_snapshot(SushiGoForwardModel, &snapshot)
            .map_err(|e| invalid("invalid_branch", e))?;
        self.records = self.branches[branch].records.clone();
        self.boundary = snapshot;
        self.active_branch = branch;
        self.display = None;
        self.proposal = None;
        self.calculation = None;
        self.observer = None;
        self.submission_serial = self
            .submission_serial
            .checked_add(1)
            .ok_or_else(|| invalid("resource_limit", "submission serial exhausted"))?;
        Ok(())
    }

    pub(super) fn fork_branch(
        &mut self,
        branch: usize,
        cursor: usize,
        replacements: Vec<Option<SeatSetup>>,
    ) -> Result<(), BridgeError> {
        if self.game.snapshot().is_err() {
            return Err(invalid(
                "invalid_request",
                "finish or cancel pending choices before forking",
            ));
        }
        if self.branches.len() >= 256 {
            return Err(invalid("resource_limit", "browser limit is 256 branches"));
        }
        let mut depth = 1;
        let mut ancestor = Some(branch);
        while let Some(id) = ancestor {
            depth += 1;
            if depth > 64 {
                return Err(invalid("resource_limit", "branch depth exceeds 64"));
            }
            ancestor = self
                .branches
                .get(id)
                .ok_or_else(|| invalid("invalid_branch", "unknown branch"))?
                .parent;
        }
        let mut snapshot = self.snapshot_at(branch, cursor)?;
        if replacements.len() != snapshot.controllers.len() {
            return Err(invalid(
                "invalid_setup",
                "expected one replacement per seat",
            ));
        }
        let mut settings = Vec::with_capacity(replacements.len());
        for (seat, replacement) in replacements.iter().enumerate() {
            let setting = if let Some(setting) = replacement {
                setting.clone()
            } else {
                match Controller::<Player>::from_snapshot(&snapshot.controllers[seat])
                    .map_err(|e| invalid("invalid_setup", e))?
                {
                    Controller::Human => SeatSetup::Human,
                    Controller::Ai(agent) => Game::settings(&agent),
                }
            };
            settings.push(setting);
        }
        Game::validate_seats(&self.parameters, &settings, &self.models)?;
        for (seat, replacement) in replacements.into_iter().enumerate() {
            if let Some(setting) = replacement {
                let controller = Game::controller_with_models(
                    &SushiGoForwardModel,
                    &self.parameters,
                    setting,
                    &self.models,
                )?;
                snapshot.controllers[seat] = controller.snapshot();
            }
        }
        snapshot.position_version = self
            .game
            .position_version()
            .checked_add(1)
            .ok_or_else(|| invalid("resource_limit", "position versions exhausted"))?;
        let game = JointSession::from_snapshot(SushiGoForwardModel, &snapshot)
            .map_err(|e| invalid("invalid_setup", e))?;
        let records = self.branches[branch]
            .records
            .iter()
            .take(cursor)
            .cloned()
            .collect();
        self.branches.push(JointBranch {
            parent: Some(branch),
            fork_position: cursor,
            base: snapshot.clone(),
            records,
            origin: BranchOrigin::Fork,
        });
        self.active_branch = self.branches.len() - 1;
        self.records = self.branches[self.active_branch].records.clone();
        self.game = game;
        self.boundary = snapshot;
        self.display = None;
        self.proposal = None;
        self.calculation = None;
        self.observer = None;
        self.submission_serial = self
            .submission_serial
            .checked_add(1)
            .ok_or_else(|| invalid("resource_limit", "submission serial exhausted"))?;
        Ok(())
    }
}
