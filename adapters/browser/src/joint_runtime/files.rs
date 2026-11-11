//! Actual-origin state proofs and versioned complete-decision replay files.
use super::*;
use tag_rust::{
    core::joint,
    learning::LearningGame,
    recording::ControllerSnapshot,
    replay::{
        JointBranchData, JointChoiceData, JointDecisionData, JointFile, ReplayCodec, Versions,
    },
};

fn versions() -> Versions {
    Versions {
        format: 3,
        game: "sushigo".into(),
        rules: 1,
        agents: 3,
        evaluators: 1,
        rng: "chacha8-rand_chacha-0.9.0-rand_core-0.9.5".into(),
    }
}
fn choice_data(choices: &[(PlayerId, Action)]) -> Vec<JointChoiceData<Action>> {
    choices
        .iter()
        .map(|(actor, action)| JointChoiceData {
            actor: actor.index(),
            action: *action,
        })
        .collect()
}
fn state_fingerprint(state: &State) -> String {
    // The versioned format records a compact check of the complete rule state.
    // Its authority comes from replaying the seed and legal joint choices.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in state.verification_facts().bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
fn verify_choice_group(
    state: &mut State,
    data: &[JointChoiceData<Action>],
) -> Result<(), BridgeError> {
    let players = SushiGoForwardModel.decision_players(state);
    if data.len() != players.len()
        || data
            .iter()
            .enumerate()
            .any(|(i, c)| c.actor != players[i].index())
    {
        return Err(invalid(
            "invalid_file",
            "joint choices differ from decision seat order",
        ));
    }
    let choices: Vec<_> = data
        .iter()
        .map(|c| (PlayerId::new(c.actor), c.action))
        .collect();
    joint::apply(&SushiGoForwardModel, state, &choices).map_err(|e| invalid("invalid_file", e))
}

impl JointRuntime {
    pub(crate) fn export(
        &mut self,
        scope: SaveScope,
        cursor: usize,
        delay_ms: u32,
    ) -> Result<String, BridgeError> {
        let mut source = self
            .display
            .map_or(self.active_branch, |(branch, _)| branch);
        if matches!(scope, SaveScope::Branch) {
            while cursor < self.branches[source].fork_position {
                source = self.branches[source].parent.expect("root starts at zero");
            }
        }
        if delay_ms > 5000 || cursor > self.branches[source].records.len() {
            return Err(invalid(
                "invalid_request",
                "invalid file cursor or playback delay",
            ));
        }
        if self.game.snapshot().is_err() {
            return Err(invalid(
                "invalid_request",
                "finish or cancel pending joint choices before saving",
            ));
        }
        let codec = sushigo::codec(&self.models);
        let mut proof = self.root_choices.clone();
        let encode_branch = |id, branch: &JointBranch| JointBranchData {
            id,
            origin: match branch.origin {
                BranchOrigin::Original => "original",
                BranchOrigin::Fork => "fork",
                BranchOrigin::Reset => "reset",
            }
            .into(),
            parent: branch.parent,
            parent_position: branch.fork_position,
            controllers: branch
                .base
                .controllers
                .iter()
                .map(|s| codec.encode(s))
                .collect(),
            decisions: branch
                .records
                .iter()
                .skip(branch.fork_position)
                .map(|record| JointDecisionData {
                    choices: choice_data(&record.choices),
                    state_after: state_fingerprint(&record.state_after),
                    next_seats: record.next_seats.iter().map(|p| p.index()).collect(),
                    controllers_after: record
                        .controllers_after
                        .iter()
                        .map(|s| codec.encode(s))
                        .collect(),
                })
                .collect(),
        };
        let (kind, branches, selected, position, delay) = if matches!(scope, SaveScope::State) {
            let source = self
                .display
                .map_or(self.active_branch, |(branch, _)| branch);
            proof.extend(
                self.branches[source]
                    .records
                    .iter()
                    .take(cursor)
                    .map(|r| choice_data(&r.choices)),
            );
            ("tag-state", Vec::new(), 0, 0, 0)
        } else if matches!(scope, SaveScope::Branch) {
            let branch = &self.branches[source];
            proof.extend(
                branch
                    .records
                    .iter()
                    .take(branch.fork_position)
                    .map(|r| choice_data(&r.choices)),
            );
            let mut data = encode_branch(0, branch);
            data.parent = None;
            data.parent_position = 0;
            data.origin = "original".into();
            (
                "tag-replay",
                vec![data],
                0,
                cursor - branch.fork_position,
                delay_ms,
            )
        } else {
            let branches = self
                .branches
                .iter()
                .enumerate()
                .map(|(id, branch)| encode_branch(id, branch))
                .collect();
            ("tag-replay", branches, source, cursor, delay_ms)
        };
        let document = JointFile {
            kind: kind.into(),
            versions: versions(),
            parameters: Game::parameters_view(&self.parameters)?,
            players: self.root.controllers.len(),
            root_choices: proof,
            root_state: if matches!(scope, SaveScope::State) {
                let source = self
                    .display
                    .map_or(self.active_branch, |(branch, _)| branch);
                let snapshot = self.snapshot_at(source, cursor)?;
                state_fingerprint(&snapshot.state)
            } else if matches!(scope, SaveScope::Branch) {
                state_fingerprint(&self.branches[source].base.state)
            } else {
                state_fingerprint(&self.root.state)
            },
            branches,
            selected_branch: selected,
            cursor: position,
            delay_ms: delay,
        };
        let text = serde_json::to_string(&document).map_err(|e| invalid("invalid_file", e))?;
        if text.len() > 10 * 1024 * 1024 {
            return Err(invalid("resource_limit", "export exceeds 10 MiB"));
        }
        let references = crate::learned::references(&text)?;
        if matches!(scope, SaveScope::State) || references.is_empty() {
            return Ok(text);
        }
        let mut used = crate::learned::Registry::new();
        for (id, model) in &self.models {
            if references.contains(id) {
                used.insert(id.clone(), model.clone());
            }
        }
        let bundle = serde_json::json!({"kind":"tag-learned-browser","version":1,
            "models":crate::learned::documents(&used),"replay":document});
        let text = serde_json::to_string(&bundle).map_err(|e| invalid("invalid_file", e))?;
        if text.len() > 10 * 1024 * 1024 {
            return Err(invalid("resource_limit", "model bundle exceeds 10 MiB"));
        }
        Ok(text)
    }

    pub(crate) fn import(
        text: &str,
        controllers: Option<Vec<SeatSetup>>,
        models: &crate::learned::Registry,
    ) -> Result<(Self, u32, bool), BridgeError> {
        if text.len() > 10 * 1024 * 1024 {
            return Err(invalid("resource_limit", "file exceeds 10 MiB"));
        }
        let file: JointFile<Action> =
            serde_json::from_str(text).map_err(|e| invalid("invalid_file", e))?;
        if file.versions != versions()
            || !(2..=5).contains(&file.players)
            || file.delay_ms > 5000
            || file.root_choices.len() > 100_000
            || !matches!(file.kind.as_str(), "tag-state" | "tag-replay")
        {
            return Err(invalid(
                "invalid_file",
                "unsupported joint file identity or limits",
            ));
        }
        if file.parameters["seed"].as_str().is_none() {
            return Err(invalid(
                "invalid_file",
                "file requires a resolved decimal seed",
            ));
        }
        let parameters = Game::parse_parameters(Some(file.parameters.clone()))
            .map_err(|e| invalid("invalid_file", e.message))?;
        let mut state = SushiGoForwardModel
            .init_state(&parameters, file.players)
            .map_err(|e| invalid("invalid_file", e))?;
        for choices in &file.root_choices {
            verify_choice_group(&mut state, choices)?;
        }
        if state_fingerprint(&state) != file.root_state {
            return Err(invalid(
                "invalid_file",
                format!(
                    "root state differs from actual-origin proof: expected {}, observed {}",
                    file.root_state,
                    state_fingerprint(&state)
                ),
            ));
        }
        if file.kind == "tag-state" {
            if !file.branches.is_empty()
                || file.cursor != 0
                || file.selected_branch != 0
                || file.delay_ms != 0
            {
                return Err(invalid("invalid_file", "state file contains replay data"));
            }
            let seats = controllers.unwrap_or_else(|| vec![SeatSetup::Human; file.players]);
            Game::validate_seats(&parameters, &seats, models)?;
            let agents = seats
                .into_iter()
                .map(|seat| {
                    Game::controller_with_models(&SushiGoForwardModel, &parameters, seat, models)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let game = JointSession::from_state(SushiGoForwardModel, state, agents)
                .map_err(|e| invalid("invalid_file", e))?;
            let root = game.snapshot().map_err(|e| invalid("invalid_file", e))?;
            return Ok((
                Self {
                    game,
                    root: root.clone(),
                    boundary: root.clone(),
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
                    root_choices: file.root_choices,
                    parameters,
                    models: models.clone(),
                    observer: None,
                    public_only: false,
                    reveal_hidden_information: false,
                    proposal: None,
                    calculation: None,
                    next_proposal: 0,
                    submission_serial: 0,
                },
                0,
                true,
            ));
        }
        if controllers.is_some()
            || file.branches.is_empty()
            || file.branches.len() > 256
            || file.selected_branch >= file.branches.len()
        {
            return Err(invalid(
                "invalid_file",
                "invalid recorded branches or replacement controllers",
            ));
        }
        for id in crate::learned::references(text)? {
            let model = models
                .get(&id)
                .ok_or_else(|| invalid("invalid_file", "missing referenced model"))?;
            let (_, count) =
                sushigo::model_parameters(model).map_err(|e| invalid("invalid_file", e))?;
            if count != file.players {
                return Err(invalid(
                    "invalid_file",
                    "model player count differs from replay",
                ));
            }
            model
                .validate_spec(
                    &SushiGoForwardModel
                        .learning_spec_for_players(
                            &parameters,
                            count,
                            tag_rust::learning::LearningLimits::default(),
                        )
                        .map_err(|e| invalid("invalid_file", e))?,
                )
                .map_err(|e| invalid("invalid_file", e))?;
        }
        let codec = sushigo::codec(models);
        let mut branches: Vec<JointBranch> = Vec::with_capacity(file.branches.len());
        let mut original_root = None;
        let mut total_records = 0usize;
        for (id, branch) in file.branches.iter().enumerate() {
            if branch.id != id
                || branch.controllers.len() != file.players
                || branch.decisions.len() > 100_000
            {
                return Err(invalid(
                    "invalid_file",
                    "invalid joint branch identity or size",
                ));
            }
            total_records = total_records
                .checked_add(branch.decisions.len())
                .ok_or_else(|| invalid("resource_limit", "record count overflow"))?;
            if total_records > 100_000 {
                return Err(invalid(
                    "resource_limit",
                    "stored joint decision limit exceeded",
                ));
            }
            let origin = match branch.origin.as_str() {
                "original" if id == 0 => BranchOrigin::Original,
                "fork" if id != 0 => BranchOrigin::Fork,
                "reset" if id != 0 => BranchOrigin::Reset,
                _ => return Err(invalid("invalid_file", "invalid branch origin")),
            };
            let (base_state, mut records, origin) = if id == 0 {
                if branch.parent.is_some() || branch.parent_position != 0 {
                    return Err(invalid("invalid_file", "root branch has a parent"));
                }
                (state.clone(), Vec::new(), origin)
            } else {
                let parent = branch
                    .parent
                    .ok_or_else(|| invalid("invalid_file", "branch has no parent"))?;
                if parent >= id {
                    return Err(invalid("invalid_file", "branch parent must precede child"));
                }
                let mut depth = 2usize;
                let mut ancestor = Some(parent);
                while let Some(index) = ancestor {
                    if depth > 64 {
                        return Err(invalid("resource_limit", "branch ancestry depth exceeded"));
                    }
                    ancestor = branches[index].parent;
                    depth += 1;
                }
                let source = &branches[parent];
                if branch.parent_position > source.records.len() {
                    return Err(invalid("invalid_file", "fork position exceeds parent"));
                }
                let before = if branch.parent_position == source.fork_position {
                    &source.base.state
                } else if branch.parent_position == 0 {
                    &state
                } else {
                    &source.records[branch.parent_position - 1].state_after
                };
                (
                    before.clone(),
                    source
                        .records
                        .iter()
                        .take(branch.parent_position)
                        .cloned()
                        .collect(),
                    origin,
                )
            };
            let agents = branch
                .controllers
                .iter()
                .map(|d| codec.decode(d).map_err(|e| invalid("invalid_file", e)))
                .collect::<Result<Vec<_>, _>>()?;
            let mut game = JointSession::from_state(SushiGoForwardModel, base_state, agents)
                .map_err(|e| invalid("invalid_file", e))?;
            let base = game.snapshot().map_err(|e| invalid("invalid_file", e))?;
            if id == 0 {
                original_root = Some(base.clone());
            }
            for entry in &branch.decisions {
                let players = game.waiting_seats();
                if entry.choices.len() != players.len()
                    || entry.controllers_after.len() != file.players
                    || entry
                        .choices
                        .iter()
                        .enumerate()
                        .any(|(i, c)| c.actor != players[i].index())
                {
                    return Err(invalid("invalid_file", "joint record has wrong seats"));
                }
                let version = game.position_version();
                let mut completed = None;
                for choice in &entry.choices {
                    let actor = PlayerId::new(choice.actor);
                    let after = &entry.controllers_after[choice.actor];
                    completed = Some(
                        match game
                            .controller(actor)
                            .ok_or_else(|| invalid("invalid_file", "missing controller"))?
                        {
                            Controller::Human => {
                                if *after != tag_rust::replay::SeatData::Human {
                                    return Err(invalid(
                                        "invalid_file",
                                        "human checkpoint changed",
                                    ));
                                }
                                game.submit_human(version, actor, choice.action)
                                    .map_err(|e| invalid("invalid_file", e))?
                            }
                            Controller::Ai(before) => {
                                let prior =
                                    codec.encode(&ControllerSnapshot::Ai(before.snapshot()));
                                let saved = codec
                                    .decode(after)
                                    .map_err(|e| invalid("invalid_file", e))?;
                                let Controller::Ai(saved) = saved else {
                                    return Err(invalid("invalid_file", "AI checkpoint missing"));
                                };
                                if !codec
                                    .recorded_choice(&prior, after)
                                    .map_err(|e| invalid("invalid_file", e))?
                                {
                                    let input = game
                                        .decision_view(actor)
                                        .map_err(|e| invalid("invalid_file", e))?;
                                    let mut replayed = Player::from_snapshot(&before.snapshot())
                                        .map_err(|e| invalid("invalid_file", e))?;
                                    let index = replayed
                                        .choose(
                                            game.model(),
                                            actor,
                                            &input.observation,
                                            &input.actions,
                                        )
                                        .map_err(|e| invalid("invalid_file", e))?;
                                    if input.actions.get(index) != Some(&choice.action)
                                        || codec
                                            .encode(&ControllerSnapshot::Ai(replayed.snapshot()))
                                            != *after
                                    {
                                        return Err(invalid(
                                            "invalid_file",
                                            "AI action or progress differs from replay",
                                        ));
                                    }
                                }
                                game.submit_ai_choice(
                                    version,
                                    actor,
                                    choice.action,
                                    saved.snapshot(),
                                )
                                .map_err(|e| invalid("invalid_file", e))?
                            }
                        },
                    );
                }
                let completed = completed
                    .flatten()
                    .ok_or_else(|| invalid("invalid_file", "joint decision did not complete"))?;
                let next: Vec<_> = game.waiting_seats().iter().map(|p| p.index()).collect();
                if next != entry.next_seats {
                    return Err(invalid("invalid_file", "next decision seats differ"));
                }
                let after = game.snapshot().map_err(|e| invalid("invalid_file", e))?;
                if state_fingerprint(&after.state) != entry.state_after {
                    return Err(invalid(
                        "invalid_file",
                        "resolved state differs from replay",
                    ));
                }
                if after
                    .controllers
                    .iter()
                    .map(|s| codec.encode(s))
                    .collect::<Vec<_>>()
                    != entry.controllers_after
                {
                    return Err(invalid("invalid_file", "controller checkpoints differ"));
                }
                records.push(Record {
                    choices: completed.choices,
                    state_after: after.state.clone(),
                    next_seats: game.waiting_seats(),
                    controllers_after: after.controllers,
                });
            }
            branches.push(JointBranch {
                parent: branch.parent,
                fork_position: branch.parent_position,
                base,
                records,
                origin,
            });
        }
        let root = original_root.expect("nonempty branches");
        let active_branch = file.selected_branch;
        let chosen = &branches[active_branch];
        if file.cursor > chosen.records.len() {
            return Err(invalid("invalid_file", "cursor exceeds selected branch"));
        }
        let end = if chosen.records.len() == chosen.fork_position {
            chosen.base.clone()
        } else {
            let last = chosen.records.last().expect("nonempty branch");
            Boundary {
                state: last.state_after.clone(),
                controllers: last.controllers_after.clone(),
                position_version: 0,
            }
        };
        let game = JointSession::from_snapshot(SushiGoForwardModel, &end)
            .map_err(|e| invalid("invalid_file", e))?;
        let records = chosen.records.clone();
        let display = (file.cursor < records.len()).then_some((active_branch, file.cursor));
        let boundary = game.snapshot().map_err(|e| invalid("invalid_file", e))?;
        Ok((
            Self {
                game,
                root,
                boundary,
                records,
                branches,
                active_branch,
                display,
                root_choices: file.root_choices,
                parameters,
                models: models.clone(),
                observer: None,
                public_only: false,
                reveal_hidden_information: false,
                proposal: None,
                calculation: None,
                next_proposal: 0,
                submission_serial: 0,
            },
            file.delay_ms,
            false,
        ))
    }
}
