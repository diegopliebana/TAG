//! Tournament JSON names an agent pool, game entries and a required result file.
use super::Result;
use crate::matches::config::{
    GameConfig, MatchConfig, PlayerConfig, Positions, Reporting, ResolvedPlayer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tag_rust::search::mcts::automatic_seed;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Exhaustive,
    Random,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SeatAllocation {
    #[default]
    Permutations,
    Random,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)] // Names are established JSON metric identifiers.
pub enum Metric {
    #[default]
    WinRate,
    PointsRate,
    RankPointsRate,
    AverageFinishingPosition,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogMode {
    Off,
    Inline,
    Separate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GameLogs {
    pub mode: LogMode,
    pub directory: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TournamentGame {
    #[serde(default)]
    pub ranking: Option<String>,
    pub id: String,
    pub name: String,
    #[serde(default = "two")]
    pub player_count: usize,
    #[serde(default = "object")]
    pub parameters: Value,
    #[serde(default)]
    pub max_actions: Option<usize>,
}
fn two() -> usize {
    2
}
fn one() -> usize {
    1
}
fn object() -> Value {
    json!({})
}
fn default_limit() -> usize {
    100_000
}
impl TournamentGame {
    pub fn game_config(&self) -> GameConfig {
        GameConfig {
            ranking: self.ranking.clone(),
            name: self.name.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TournamentConfig {
    pub schema_version: u32,
    pub agents: Vec<PlayerConfig>,
    pub games: Vec<TournamentGame>,
    pub mode: Mode,
    /// Randomly sampled groups per game entry, before seat permutations/repetitions.
    #[serde(default)]
    pub matchups: Option<usize>,
    #[serde(default)]
    pub seat_allocation: SeatAllocation,
    #[serde(default)]
    pub allow_self_play: bool,
    #[serde(default = "one")]
    pub repetitions: usize,
    #[serde(default)]
    pub ranking_metric: Metric,
    #[serde(default)]
    pub seed: Option<u64>,
    pub results_file: PathBuf,
    #[serde(default)]
    pub include_actions: bool,
    #[serde(default)]
    pub game_logs: Option<GameLogs>,
    /// Bound the actual schedule size across all game entries.
    #[serde(default = "default_limit")]
    pub max_games: usize,
}
pub struct ResolvedTournament {
    pub config: TournamentConfig,
    pub master_seed: u64,
    pub agents: Vec<ResolvedPlayer>,
}

impl TournamentConfig {
    pub fn load(path: &Path) -> Result<ResolvedTournament> {
        let text = crate::matches::files::read_text(
            path,
            crate::matches::files::CONFIG_BYTES,
            "tournament file",
        )?;
        if text.len() > crate::matches::files::CONFIG_BYTES {
            return Err(format!("tournament file '{}' exceeds 1 MiB", path.display()).into());
        }
        let mut config: Self = serde_json::from_str(&text)
            .map_err(|e| format!("invalid tournament file '{}': {e}", path.display()))?;
        let directory = path.parent().unwrap_or(Path::new("."));
        if config.results_file.as_os_str().is_empty() {
            return Err("results_file must name a file".into());
        }
        config.results_file = directory.join(&config.results_file);
        config.resolve(directory)
    }

    pub fn resolve(mut self, directory: &Path) -> Result<ResolvedTournament> {
        if let Some(logs) = &mut self.game_logs {
            if self.include_actions {
                return Err("use game_logs or include_actions=true, not both".into());
            }
            match (&logs.mode, &logs.directory) {
                (LogMode::Separate, Some(path)) if !path.as_os_str().is_empty() => {}
                (LogMode::Off | LogMode::Inline, None) => {}
                _ => return Err("game_logs.directory is required only for separate mode".into()),
            }
            if let Some(path) = &mut logs.directory {
                *path = directory.join(&*path);
            }
            self.include_actions = logs.mode == LogMode::Inline;
        }
        if self.schema_version != 1 {
            return Err("tournament schema_version must be 1".into());
        }
        if self.games.is_empty() || self.agents.is_empty() {
            return Err("tournaments need at least one game entry and one agent".into());
        }
        if self.repetitions == 0 || self.max_games == 0 {
            return Err("repetitions and max_games must be positive".into());
        }
        match (self.mode, self.matchups) {
            (Mode::Random, Some(count)) if count > 0 && count <= self.max_games => {},
            (Mode::Exhaustive, None) => {},
            _ => return Err("random mode requires positive matchups within max_games; exhaustive mode must omit matchups".into()),
        }
        if self.results_file.as_os_str().is_empty() {
            return Err("results_file must name a file".into());
        }
        let mut ids = BTreeSet::new();
        for game in &self.games {
            if game.id.trim().is_empty() || !ids.insert(&game.id) {
                return Err("game entry IDs must be nonempty and unique".into());
            }
            if game.max_actions == Some(0) {
                return Err("max_actions must be positive or null".into());
            }
            if game.player_count == 0 || game.player_count > 64 {
                return Err("scheduler player_count must be between 1 and 64".into());
            }
            if !self.allow_self_play && game.player_count > self.agents.len() {
                return Err(format!(
                    "game '{}': not enough distinct agents; add agents or enable allow_self_play",
                    game.id
                )
                .into());
            }
        }
        let master_seed = self.seed.map(Ok).unwrap_or_else(automatic_seed)?;
        // Reuse the match parser and factory specifications, resolving referenced
        // agent files once relative to the tournament file. No game is played here.
        let template = MatchConfig {
            schema_version: 1,
            game: self.games[0].game_config(),
            players: self.agents.clone(),
            games: 1,
            seed: Some(master_seed),
            positions: Positions::Fixed,
            max_actions: None,
            reporting: Reporting::default(),
        }
        .resolve(directory)?;
        #[cfg(feature = "inference")]
        for agent in &template.players {
            for entry in agent.models.keys().filter(|entry| !entry.is_empty()) {
                if !self.games.iter().any(|game| game.id == *entry) {
                    return Err(format!(
                        "agent '{}': model_files names unknown game entry '{entry}'",
                        agent.name
                    )
                    .into());
                }
            }
        }
        Ok(ResolvedTournament {
            config: self,
            master_seed,
            agents: template.players,
        })
    }
}

impl ResolvedTournament {
    pub fn log_mode(&self) -> LogMode {
        self.config.game_logs.as_ref().map_or(
            if self.config.include_actions {
                LogMode::Inline
            } else {
                LogMode::Off
            },
            |logs| logs.mode,
        )
    }
}
