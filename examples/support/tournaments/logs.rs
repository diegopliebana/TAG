//! Save one completed game's research metadata and a portable browser recording.
use super::{Result, config::ResolvedTournament, execution::PlayedGame};
use crate::matches::report::write_new;
use serde_json::json;
use std::path::{Component, Path, PathBuf};

/// References are relative to the main report, so moving the whole results folder
/// preserves its links. Windows paths on different drives require absolute paths.
fn reference(base: &Path, target: &Path) -> Result<PathBuf> {
    let base = normal(base)?;
    let target = normal(target)?;
    let a: Vec<_> = base.components().collect();
    let b: Vec<_> = target.components().collect();
    let common = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
    if common == 0 {
        return Ok(target);
    }
    let mut relative = PathBuf::new();
    for _ in common..a.len() {
        relative.push("..");
    }
    for part in &b[common..] {
        relative.push(part);
    }
    Ok(relative)
}

pub(super) fn normal(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for part in std::path::absolute(path)?.components() {
        match part {
            Component::ParentDir => {
                result.pop();
            }
            Component::CurDir => {}
            _ => result.push(part),
        }
    }
    Ok(result)
}

pub fn save(
    settings: &ResolvedTournament,
    tournament_id: &str,
    record: &mut PlayedGame,
    manifest: &super::execution::GameManifest,
) -> Result<()> {
    let directory = settings
        .config
        .game_logs
        .as_ref()
        .and_then(|logs| logs.directory.as_ref())
        .expect("separate logs have a directory");
    let game = &settings.config.games[record.entry_index];
    let outcome = record
        .outcome
        .as_ref()
        .ok_or("cannot record an unfinished game")?;
    let replay =
        crate::matches::execution::export_action_replay(&game.name, &game.parameters, outcome)?;
    let replay_path = directory.join(format!("game-{:06}.tag-replay.json", record.index));
    let log_path = directory.join(format!("game-{:06}.json", record.index));
    let parent = settings
        .config
        .results_file
        .parent()
        .unwrap_or(Path::new("."));
    // Save the replay first. If metadata writing fails, the completed recording
    // still exists and the main report retains its reference and the save error.
    let replay_data = if let Some(replay) = &replay {
        write_new(&replay_path, replay)?;
        record.replay_file = Some(reference(parent, &replay_path)?);
        Some(serde_json::from_str::<serde_json::Value>(replay)?)
    } else {
        None
    };
    // Stochastic action/chance data remains in the research log; only games
    // with browser replay support move those actions into a companion file.
    let native_actions = if replay.is_none() {
        record
            .outcome
            .as_ref()
            .expect("played game")
            .actions
            .clone()
    } else {
        record.outcome.as_mut().expect("played game").actions = None;
        None
    };
    let mut native_parameters = manifest.parameters.clone();
    if let Some(seed) = record.outcome.as_ref().and_then(|o| o.game_seed) {
        native_parameters["seed"] = json!(seed);
    }
    let metadata = json!({
        "schema_version": 1, "tournament_id": tournament_id,
        "engine_version": env!("CARGO_PKG_VERSION"),
        "master_seed": settings.master_seed,
        "results_file": reference(directory, &settings.config.results_file)?,
        "game": game.name,
        "parameters": replay_data.as_ref().map_or(native_parameters, |data| data["parameters"].clone()),
        "rules_version": replay_data.as_ref().map_or(json!(manifest.rules_version), |data| data["versions"]["rules"].clone()),
        "replay_file": replay.as_ref().map(|_| replay_path.file_name().expect("generated filename").to_string_lossy()),
        "actions": native_actions,
        "record": record,
        "agents": settings.agents.iter().map(|agent| {
            let (algorithm, effective) = crate::matches::report::describe_agent(agent)?;
            Ok(json!({"name":agent.name,"algorithm":algorithm,"settings":effective,"base_seed":agent.base_seed}))
        }).collect::<Result<Vec<_>>>()?
    });
    write_new(&log_path, &serde_json::to_string_pretty(&metadata)?)?;
    record.game_log = Some(reference(parent, &log_path)?);
    record.outcome.as_mut().expect("played game").actions = None;
    Ok(())
}
