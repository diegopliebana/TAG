//! Machine-readable experiment results preserve agent identity across seat rotation.
use super::Result;
use serde::Serialize;
use serde_json::{Value, json};
use std::path::Path;

#[derive(Debug, Serialize, PartialEq)]
pub struct MatchReport {
    pub schema_version: u32,
    pub seed_derivation_version: u32,
    pub master_seed: u64,
    pub game: String,
    pub parameters: Value,
    pub positions: super::config::Positions,
    pub ranking: Option<String>,
    pub requested_games: usize,
    pub completed_games: usize,
    pub truncated_games: usize,
    pub agents: Vec<AgentReport>,
    pub games: Vec<GameReport>,
    pub failure: Option<Failure>,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct AgentReport {
    pub name: String,
    pub algorithm: String,
    pub base_seed: u64,
    pub settings: Value,
    pub combined: Outcomes,
    pub by_seat: Vec<Outcomes>,
}
#[derive(Debug, Serialize, Default, PartialEq)]
pub struct Outcomes {
    pub games: usize,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub truncated: usize,
}
impl Outcomes {
    pub fn add(&mut self, result: &str) {
        self.games += 1;
        match result {
            "win" => self.wins += 1,
            "draw" => self.draws += 1,
            "loss" => self.losses += 1,
            "truncated" => self.truncated += 1,
            _ => unreachable!("validated terminal result"),
        }
    }
}
#[derive(Debug, Serialize, PartialEq)]
pub struct GameReport {
    pub index: usize,
    /// None for the deterministic registered games, which have no game RNG.
    pub game_seed: Option<u64>,
    /// Entries use configuration order, independent of each agent's current seat.
    pub agent_seeds: Vec<u64>,
    pub seats: Vec<usize>,
    pub results: Vec<String>,
    pub ranks: Option<Vec<usize>>,
    pub truncated: bool,
    pub max_actions: Option<usize>,
    pub accepted_moves: usize,
    pub mcts: Vec<SearchTotals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actions: Option<Vec<ActionRecord>>,
    pub recording: Option<String>,
}
#[derive(Debug, Serialize, Default, PartialEq)]
pub struct SearchTotals {
    pub completed_iterations: u64,
    pub forward_model_calls: u64,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ActionRecord {
    pub seat: usize,
    pub action: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chance: Option<Value>,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct Failure {
    pub game_index: usize,
    pub message: String,
}

/// Record effective defaults and agent identities before gameplay. MCTS seeds in
/// this manifest are base seeds; each game's actual seeds are recorded separately.
pub fn create(settings: &super::config::ResolvedMatch, parameters: Value) -> Result<MatchReport> {
    let mut agents = Vec::new();
    for player in &settings.players {
        let (algorithm, effective) = describe_agent(player)?;
        agents.push(AgentReport {
            name: player.name.clone(),
            algorithm: algorithm.into(),
            base_seed: player.base_seed,
            settings: effective,
            combined: Outcomes::default(),
            by_seat: (0..settings.players.len())
                .map(|_| Outcomes::default())
                .collect(),
        });
    }
    Ok(MatchReport {
        schema_version: 1,
        seed_derivation_version: 1,
        master_seed: settings.master_seed,
        game: settings.config.game.name.clone(),
        parameters,
        positions: settings.config.positions,
        ranking: settings.config.game.ranking.clone(),
        requested_games: settings.config.games,
        completed_games: 0,
        truncated_games: 0,
        agents,
        games: Vec::new(),
        failure: None,
    })
}

/// Describe the same effective agent settings in match and tournament manifests.
pub fn describe_agent(player: &super::config::ResolvedPlayer) -> Result<(&'static str, Value)> {
    use super::config::AgentConfig;
    let algorithm = match player.kind {
        AgentConfig::FirstAvailable {} => "first_available",
        AgentConfig::Random {} => "random",
        AgentConfig::Osla { .. } => "osla",
        AgentConfig::Mcts { .. } => "mcts",
        #[cfg(feature = "inference")]
        AgentConfig::NeuralMcts { .. } => "neural_mcts",
    };
    #[allow(unused_mut)]
    let mut effective = if let Some(template) = &player.mcts {
        let mut template = template.clone();
        template.seed = Some(player.base_seed);
        serde_json::from_str(&template.to_json()?)?
    } else {
        json!({"evaluator": if algorithm == "osla" { "game_heuristic" } else { "none" }})
    };
    if let AgentConfig::Osla {
        repetitions_per_action,
    } = player.kind
    {
        effective["repetitions_per_action"] = json!(repetitions_per_action);
    }
    #[cfg(feature = "inference")]
    if let AgentConfig::NeuralMcts { bindings, .. } = &player.kind {
        effective = json!({"search":effective,"bindings":bindings,
            "models":player.models.iter().map(|(entry,model)| {
                let document = model.as_document();
                (entry, json!({"model_id": model.id(), "contract": document["contract"],
                    "model_context":document["provenance"]["model_context"]}))
            }).collect::<std::collections::BTreeMap<_,_>>()});
    }
    Ok((algorithm, effective))
}

impl MatchReport {
    pub fn summary(&self) -> String {
        let mut text = format!(
            "{}: completed {}/{} games; truncated={}; master_seed={}\n",
            self.game,
            self.completed_games,
            self.requested_games,
            self.truncated_games,
            self.master_seed
        );
        for agent in &self.agents {
            text.push_str(&format!(
                "{} ({}): wins={} draws={} losses={} truncated={}\n",
                agent.name,
                agent.algorithm,
                agent.combined.wins,
                agent.combined.draws,
                agent.combined.losses,
                agent.combined.truncated
            ));
            for (seat, counts) in agent.by_seat.iter().enumerate() {
                text.push_str(&format!(
                    "  seat {seat}: games={} wins={} draws={} losses={} truncated={}\n",
                    counts.games, counts.wins, counts.draws, counts.losses, counts.truncated
                ));
            }
        }
        if let Some(failure) = &self.failure {
            text.push_str(&format!(
                "Stopped at game {}: {}\n",
                failure.game_index, failure.message
            ));
        }
        text
    }
}

/// Never overwrite an existing result or recording. A failed write removes only
/// the new file created by this call. Existing user files remain unchanged.
pub fn write_new(path: &Path, text: &str) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    if let Err(error) = file
        .write_all(text.as_bytes())
        .and_then(|()| file.sync_all())
    {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    Ok(())
}
