//! Save/load recorded lines and branches with their controller settings/progress.
//!
//! Import replays accepted actions and checks the resulting decisions and agent data.
//! A branch-only export includes one complete line, not relationships to other branches.
use super::common::{check_storage, check_versions, invalid, json, versions};
use super::controllers::{branch_has_extension, branch_has_random, check_fresh_controller};
use super::*;
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::evaluation::Evaluator;
use crate::core::{Decision, PlayerId};
use crate::session::{BranchId, BranchSession, BuiltinAgent, Controller, Session};
use serde::{Serialize, de::DeserializeOwned};

/// Reconstruct chance only for games that record it. This never advances the
/// live session or controller streams; inherited moves precede local records.
fn chance_events<F, A>(
    session: &BranchSession<F, A>,
    id: BranchId,
    start: usize,
    tip: usize,
) -> Result<Vec<Option<serde_json::Value>>, PersistenceError>
where
    F: ReplayGame,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug,
    A: Agent<F> + Checkpoint,
    A::Snapshot: PartialEq,
    A::Error: std::fmt::Debug,
    A::RestoreError: std::fmt::Debug,
{
    if !F::HAS_CHANCE {
        return Ok(vec![None; tip - start]);
    }
    let model = session.active().session().model();
    let mut state = session.root().clone();
    let mut events = Vec::new();
    for (i, r) in session
        .recorded_prefix(id, tip)
        .map_err(|e| invalid(format!("chance history: {e:?}")))?
        .iter()
        .enumerate()
    {
        model
            .next(&mut state, r.actor, &r.action)
            .map_err(|e| invalid(format!("chance transition: {e:?}")))?;
        if i >= start {
            events.push(model.chance_event(&r.action, &state));
        }
    }
    Ok(events)
}

/// Export one complete line with navigation and controller changes, without ancestry.
///
/// # Arguments
/// `session` and matching `parameters` are borrowed; `cursor` is the displayed move
/// in the active branch, and `delay_ms` is the saved playback delay.
/// `codec` borrows the adapter controller format without changing model state.
///
/// # Returns
/// Owned replay JSON containing only branch zero and its complete recorded line.
///
/// # Errors
/// Invalid selection/parameters, checkpoints or serialization failure.
pub fn export_branch_with<F, A>(
    session: &BranchSession<F, A>,
    parameters: &F::Parameters,
    cursor: usize,
    delay_ms: u32,
    codec: &impl ReplayCodec<A>,
) -> Result<String, PersistenceError>
where
    F: ReplayGame,
    F::State: Clone + PartialEq,
    F::Action: Clone + Serialize,
    F::Parameters: Clone + Serialize,
    F::Error: std::fmt::Debug,
    A: Agent<F> + Checkpoint,
    A::Snapshot: PartialEq,
    A::Error: std::fmt::Debug,
    A::RestoreError: std::fmt::Debug,
{
    let id = session.active_branch();
    let timeline = session
        .controller_timeline(id)
        .map_err(|e| invalid(format!("controllers: {e:?}")))?;
    let controllers: Vec<_> = timeline[0].1.iter().map(|s| codec.encode(s)).collect();
    let controller_changes: Vec<_> = timeline
        .iter()
        .skip(1)
        .map(|(position, seats)| ControllerChange {
            position: *position,
            controllers: seats.iter().map(|s| codec.encode(s)).collect(),
        })
        .collect();
    let tip = session
        .info(id)
        .map_err(|e| invalid(format!("branch: {e:?}")))?
        .move_count;
    if cursor > tip {
        return Err(invalid("cursor exceeds selected branch"));
    }
    let chance = chance_events(session, id, 0, tip)?;
    let moves: Vec<_> = session
        .recorded_prefix(id, tip)
        .map_err(|e| invalid(format!("records: {e:?}")))?
        .into_iter()
        .enumerate()
        .map(|(i, r)| MoveData {
            chance: chance[i].clone(),
            actor: r.actor.index(),
            action: r.action.clone(),
            next_player: match r.decision {
                Decision::Player(p) => Some(p.index()),
                Decision::Finished => None,
            },
            controller_after: codec.encode(&r.controller_after),
        })
        .collect();
    let branch = BranchData {
        id: 0,
        parent: None,
        parent_position: 0,
        controllers,
        moves,
        controller_changes,
    };
    let mut identities = versions::<F>();
    if branch_has_random(&branch) {
        identities.agents = 2;
    }
    if !branch.controller_changes.is_empty() {
        identities.format = 2;
    }
    if branch_has_extension(&branch) {
        identities.agents = 3;
    }
    let model = session.active().session().model();
    let initial_state = model.state_data(session.root());
    if model.restore_state(parameters, &initial_state)? != *session.root() {
        return Err(invalid("export parameters differ from recorded root rules"));
    }
    let file = ReplayFile {
        kind: "tag-replay".into(),
        versions: identities,
        parameters: parameters.clone(),
        initial_state,
        branches: vec![branch],
        selected_branch: 0,
        cursor,
        delay_ms,
    };
    Ok(serde_json::to_string_pretty(&file)?)
}

/// Apply a decoded linear replacement at this boundary after validating fresh seeds.
///
/// # Arguments
/// `session` is accessed through an exclusive mutable reference; `changes` borrows the remaining file events;
/// `position` selects the current absolute boundary; `codec` restores the file
/// policy using its own borrowed immutable model registry.
///
/// # Returns
/// Unit, preserving unchanged checkpoints and installing explicit replacements.
///
/// # Errors
/// Unsupported checkpoints, nonfresh replacement RNG or session validation failure.
fn apply_changes<F, A>(
    session: &mut BranchSession<F, A>,
    changes: &mut std::iter::Peekable<std::slice::Iter<'_, ControllerChange>>,
    position: usize,
    codec: &impl ReplayCodec<A>,
) -> Result<(), PersistenceError>
where
    F: ReplayGame,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug,
    A: Agent<F> + Checkpoint,
    A::Snapshot: PartialEq,
    A::Error: std::fmt::Debug,
    A::RestoreError: std::fmt::Debug,
{
    // Consume only an event at this boundary; unmatched future events stay queued.
    let Some(change) = changes.next_if(|change| change.position == position) else {
        return Ok(());
    };
    let previous = (0..change.controllers.len())
        .map(|seat| {
            session
                .active()
                .session()
                .controller(PlayerId::new(seat))
                .map(Controller::snapshot)
                .ok_or_else(|| invalid("unknown replacement seat"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut seats = Vec::new();
    for (data, old) in change.controllers.iter().zip(&previous) {
        if *data != codec.encode(old) {
            check_fresh_controller(data)?;
            codec.validate_fresh(data)?;
        }
        seats.push(codec.decode(data)?);
    }
    session
        .replace_controllers(seats)
        .map_err(|e| invalid(format!("controllers: {e:?}")))
}

/// Validated replay and its saved selection; adapters load it paused.
pub struct LoadedReplay<F: PersistableGame, A: crate::agents::Agent<F> + Checkpoint> {
    pub session: BranchSession<F, A>,
    pub parameters: F::Parameters,
    pub cursor: usize,
    pub delay_ms: u32,
}
/// Borrow current branch data and parameters; return owned versioned JSON text.
/// The supplied codec interprets adapter policy settings and restores their models.
///
/// # Arguments
/// `session` and `parameters` are shared borrows. Parameters must match the root
/// rules. `cursor` is the copied absolute move count to display in the selected
/// branch; `delay_ms` is the copied global AI delay in milliseconds.
///
/// # Returns
/// Owned pretty-printed JSON, including initial state, ancestry, actions and checkpoints.
///
/// # Errors
/// Invalid cursor, parameter/state mismatch, branch data or JSON conversion failure.
pub fn export_replay_with<F, A>(
    session: &BranchSession<F, A>,
    parameters: &F::Parameters,
    cursor: usize,
    delay_ms: u32,
    codec: &impl ReplayCodec<A>,
) -> Result<String, PersistenceError>
where
    F: ReplayGame,
    F::State: Clone + PartialEq,
    F::Action: Clone + Serialize,
    F::Parameters: Clone + Serialize,
    F::Error: std::fmt::Debug,
    A: Agent<F> + Checkpoint,
    A::Snapshot: PartialEq,
    A::Error: std::fmt::Debug,
    A::RestoreError: std::fmt::Debug,
{
    let mut branches = Vec::new();
    for index in 0..session.branch_count() {
        let id = BranchId(index);
        let info = session
            .info(id)
            .map_err(|e| invalid(format!("branch: {e:?}")))?;
        let controllers = session
            .initial_controllers(id)
            .map_err(|e| invalid(format!("branch: {e:?}")))?
            .iter()
            .map(|s| codec.encode(s))
            .collect();
        let chance = chance_events(
            session,
            id,
            info.parent.map_or(0, |(_, p)| p),
            info.move_count,
        )?;
        let moves = session
            .moves(id)
            .map_err(|e| invalid(format!("branch: {e:?}")))?
            .iter()
            .enumerate()
            .map(|(i, r)| MoveData {
                chance: chance[i].clone(),
                actor: r.actor.index(),
                action: r.action.clone(),
                next_player: match r.decision {
                    Decision::Finished => None,
                    Decision::Player(p) => Some(p.index()),
                },
                controller_after: codec.encode(&r.controller_after),
            })
            .collect();
        branches.push(BranchData {
            id: index,
            parent: info.parent.map(|(id, _)| id.0),
            parent_position: info.parent.map_or(0, |(_, p)| p),
            controllers,
            moves,
            controller_changes: session
                .controller_changes(id)
                .map_err(|e| invalid(format!("controllers: {e:?}")))?
                .iter()
                .map(|(position, seats)| ControllerChange {
                    position: *position,
                    controllers: seats.iter().map(|s| codec.encode(s)).collect(),
                })
                .collect(),
        });
    }
    if cursor
        > session
            .info(session.active_branch())
            .map_err(|e| invalid(format!("branch: {e:?}")))?
            .move_count
    {
        return Err(invalid("cursor exceeds selected branch"));
    }
    let mut file = ReplayFile {
        kind: "tag-replay".into(),
        versions: versions::<F>(),
        parameters: parameters.clone(),
        initial_state: session
            .active()
            .session()
            .model()
            .state_data(session.root()),
        branches,
        selected_branch: session.active_branch().0,
        cursor,
        delay_ms,
    };
    if file
        .branches
        .iter()
        .any(|b| !b.controller_changes.is_empty())
    {
        file.versions.format = 2;
    }
    if file.branches.iter().any(branch_has_random) {
        file.versions.agents = 2;
    }
    if file.branches.iter().any(branch_has_extension) {
        file.versions.agents = 3;
    }
    // Verify parameters agree with the root even for caller-supplied export arguments.
    let restored = session
        .active()
        .session()
        .model()
        .restore_state(parameters, &file.initial_state)?;
    if &restored != session.root() {
        return Err(invalid("export parameters differ from recorded root rules"));
    }
    Ok(serde_json::to_string_pretty(&file)?)
}
/// Borrow text and rules, consume validation limits; return independent validated branches.
/// Every accepted AI move is reproduced and compared with its stored action/checkpoint.
/// Errors leave any caller-owned existing session unchanged.
///
/// # Arguments
/// `text` is passed by a shared (read-only) reference JSON. `model` is consumed and must support the
/// file's game/rules version. `limits` is a copied resource policy. `codec` borrows
/// the adapter controller representation and model registry for restoration.
///
/// # Returns
/// Owned validated branch session, parameters, saved cursor and delay.
/// Adapters use the saved cursor and open playback paused.
///
/// # Errors
/// Malformed/oversized input, unsupported versions, invalid ancestry/state/moves,
/// AI action/checkpoint disagreement or excessive validation work.
pub fn import_replay_with<F, A>(
    text: &str,
    model: F,
    limits: ReplayLimits,
    codec: &impl ReplayCodec<A>,
) -> Result<LoadedReplay<F, A>, PersistenceError>
where
    F: ReplayGame,
    F::State: Clone,
    F::Action: Clone + PartialEq + DeserializeOwned,
    F::Parameters: DeserializeOwned,
    F::Error: std::fmt::Debug,
    A: Agent<F> + Checkpoint,
    A::Snapshot: PartialEq,
    A::Error: std::fmt::Debug,
    A::RestoreError: std::fmt::Debug,
{
    let file: ReplayFile<F::Parameters, F::SavedState, F::Action> = json(text, limits)?;
    if file.kind != "tag-replay" {
        return Err(invalid("expected a replay document"));
    }
    let mut compatible = file.versions.clone();
    if !matches!(compatible.format, 1 | 2) {
        return Err(invalid(format!(
            "unsupported replay format {}; supported versions: 1 and 2",
            compatible.format
        )));
    }
    compatible.format = 1;
    if !matches!(compatible.agents, 1..=3) {
        return Err(invalid(format!(
            "unsupported agents version/identity: file {}; supported 1, 2 and adapter scheme 3",
            compatible.agents
        )));
    }
    if compatible.agents == 1 && file.branches.iter().any(branch_has_random) {
        return Err(invalid("Random controllers require agent scheme version 2"));
    }
    if compatible.agents < 3 && file.branches.iter().any(branch_has_extension) {
        return Err(invalid(
            "adapter controllers require agent scheme version 3",
        ));
    }
    compatible.agents = 1; // Explicit support for legacy policies and the additive Random scheme.
    check_versions::<F>(&compatible)?;
    if file.versions.format == 1
        && file
            .branches
            .iter()
            .any(|b| !b.controller_changes.is_empty())
    {
        return Err(invalid(
            "linear controller changes require replay format version 2",
        ));
    }
    check_storage::<F>(&file.parameters, limits)?;
    let ancestry_work = super::file_validation::replay_work::<F>(&file, limits)?;
    let root = model.restore_state(&file.parameters, &file.initial_state)?;
    let expected = model.player_count(&root);
    let first = &file.branches[0];
    if first.id != 0
        || first.parent.is_some()
        || first.parent_position != 0
        || first.controllers.len() != expected
    {
        return Err(invalid("invalid replay root branch"));
    }
    let controllers = first
        .controllers
        .iter()
        .map(|s| codec.decode(s))
        .collect::<Result<Vec<_>, _>>()?;
    let initial = Session::from_state(model, root, controllers)
        .map_err(|e| invalid(format!("session: {e:?}")))?;
    let mut session = BranchSession::new(initial);
    let mut work = ancestry_work;
    for (index, branch) in file.branches.iter().enumerate() {
        if branch.id != index || branch.controllers.len() != expected {
            return Err(invalid(
                "branch IDs must be ordered and unique; seat counts must match",
            ));
        }
        if index > 0 {
            let parent = branch
                .parent
                .ok_or_else(|| invalid("child branch lacks parent"))?;
            if parent >= index {
                return Err(invalid("parent must precede child; cycles are prohibited"));
            }
            let preserved = session
                .controllers_at(BranchId(parent), branch.parent_position)
                .map_err(|e| invalid(format!("fork: {e:?}")))?;
            let mut replacements = Vec::new();
            for (data, previous) in branch.controllers.iter().zip(preserved.iter()) {
                if *data == codec.encode(previous) {
                    replacements.push(None);
                } else {
                    let replacement = codec.decode(data)?;
                    check_fresh_controller(data)?;
                    codec.validate_fresh(data)?;
                    replacements.push(Some(replacement));
                }
            }
            session
                .fork(BranchId(parent), branch.parent_position, replacements)
                .map_err(|e| invalid(format!("fork: {e:?}")))?;
        }
        let mut changes = branch.controller_changes.iter().peekable();
        for (offset, record) in branch.moves.iter().enumerate() {
            apply_changes::<F, A>(
                &mut session,
                &mut changes,
                branch.parent_position + offset,
                codec,
            )?;
            let actor = PlayerId::new(record.actor);
            let version = session.active().session().position_version();
            let ai = matches!(
                session.active().session().controller(actor),
                Some(Controller::Ai(_))
            );
            // A conservative placement-game work estimate charges all moves for state scans,
            // and OSLA for up to one candidate per cell plus line evaluation per candidate.
            let seat = codec.encode(
                &session
                    .active()
                    .session()
                    .controller(actor)
                    .ok_or_else(|| invalid("unknown actor"))?
                    .snapshot(),
            );
            let cost = F::verification_work(&file.parameters, codec.lookahead(&seat))
                .and_then(|n| n.checked_add(codec.extra_work(&seat)))
                .ok_or_else(|| invalid("verification work overflow"))?;
            work = work
                .checked_add(cost)
                .ok_or_else(|| invalid("verification work overflow"))?;
            if work > limits.max_verification_work {
                return Err(invalid(
                    "AI/move verification exceeds configured work limit",
                ));
            }
            let recorded_choice = ai && codec.recorded_choice(&seat, &record.controller_after)?;
            let accepted = if recorded_choice {
                let controller = codec.decode(&record.controller_after)?;
                let Controller::Ai(agent) = controller else {
                    return Err(invalid(
                        "recorded AI continuation requires an AI controller",
                    ));
                };
                session.apply_recorded_ai(version, record.action.clone(), &agent.snapshot())
            } else if ai {
                session.advance_ai(version)
            } else {
                session.submit_human(version, actor, record.action.clone())
            }
            .map_err(|e| invalid(format!("move validation: {e:?}")))?;
            let decision = match accepted.decision {
                Decision::Player(p) => Some(p.index()),
                Decision::Finished => None,
            };
            if accepted.actor != actor
                || accepted.action != record.action
                || decision != record.next_player
            {
                return Err(invalid(
                    "recorded action/actor/result disagrees with execution",
                ));
            }
            if session
                .active()
                .session()
                .model()
                .chance_event(&record.action, session.active().session().state())
                != record.chance
            {
                return Err(invalid("recorded chance event disagrees with execution"));
            }
            let progress = session
                .active()
                .session()
                .controller(actor)
                .ok_or_else(|| invalid("unknown acting seat"))?
                .snapshot();
            if codec.encode(&progress) != record.controller_after {
                return Err(invalid(
                    "recorded controller progress disagrees with execution",
                ));
            }
        }
        apply_changes::<F, A>(
            &mut session,
            &mut changes,
            branch.parent_position + branch.moves.len(),
            codec,
        )?;
    }
    let selected = BranchId(file.selected_branch);
    if file.cursor
        > session
            .info(selected)
            .map_err(|e| invalid(format!("selection: {e:?}")))?
            .move_count
    {
        return Err(invalid("cursor exceeds selected branch"));
    }
    session
        .select(selected)
        .map_err(|e| invalid(format!("selection: {e:?}")))?;
    Ok(LoadedReplay {
        session,
        parameters: file.parameters,
        cursor: file.cursor,
        delay_ms: file.delay_ms,
    })
}

/// Export built-in controllers using the original portable file scheme.
/// The borrowed session/parameters are unchanged; cursor/delay select display state.
pub fn export_branch<F, E>(
    session: &BranchSession<F, BuiltinAgent<E>>,
    parameters: &F::Parameters,
    cursor: usize,
    delay_ms: u32,
) -> Result<String, PersistenceError>
where
    F: ReplayGame<SavedEvaluator = E> + crate::core::PublicInformationModel,
    F::State: Clone + PartialEq,
    F::Action: Clone + Serialize,
    F::Parameters: Clone + Serialize,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: std::fmt::Debug,
    E::RestoreError: std::fmt::Debug,
{
    export_branch_with(session, parameters, cursor, delay_ms, &BuiltinCodec)
}

/// Export built-in controllers using the original portable file scheme.
/// The borrowed session/parameters are unchanged; cursor/delay select display state.
pub fn export_replay<F, E>(
    session: &BranchSession<F, BuiltinAgent<E>>,
    parameters: &F::Parameters,
    cursor: usize,
    delay_ms: u32,
) -> Result<String, PersistenceError>
where
    F: ReplayGame<SavedEvaluator = E> + crate::core::PublicInformationModel,
    F::State: Clone + PartialEq,
    F::Action: Clone + Serialize,
    F::Parameters: Clone + Serialize,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: std::fmt::Debug,
    E::RestoreError: std::fmt::Debug,
{
    export_replay_with(session, parameters, cursor, delay_ms, &BuiltinCodec)
}

/// Validate built-in replay controllers and restore an independent branch session.
/// Borrowed text and copied limits bound validation; model ownership transfers.
pub fn import_replay<F, E>(
    text: &str,
    model: F,
    limits: ReplayLimits,
) -> Result<LoadedReplay<F, BuiltinAgent<E>>, PersistenceError>
where
    F: ReplayGame<SavedEvaluator = E> + crate::core::PublicInformationModel,
    F::State: Clone,
    F::Action: Clone + PartialEq + DeserializeOwned,
    F::Parameters: DeserializeOwned,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: std::fmt::Debug,
    E::RestoreError: std::fmt::Debug,
{
    import_replay_with(text, model, limits, &BuiltinCodec)
}
