//! Read strict match JSON and resolve agent files relative to the match file.
use super::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tag_rust::search::mcts::{Settings, automatic_seed};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchConfig {
    pub schema_version: u32,
    pub game: GameConfig,
    pub players: Vec<PlayerConfig>,
    pub games: usize,
    /// None selects operating-system randomness once, before any game begins.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub positions: Positions,
    #[serde(default)]
    pub max_actions: Option<usize>,
    #[serde(default)]
    pub reporting: Reporting,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameConfig {
    #[serde(default)]
    pub ranking: Option<String>,
    pub name: String,
    #[serde(default = "empty_object")]
    pub parameters: Value,
}
fn one_repetition() -> std::num::NonZeroU32 {
    std::num::NonZeroU32::new(1).unwrap()
}
fn empty_object() -> Value {
    serde_json::json!({})
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerConfig {
    /// Stable experiment identity, even when this agent changes seats.
    pub name: String,
    #[serde(default)]
    pub seed: Option<u64>,
    pub agent: AgentConfig,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentConfig {
    FirstAvailable {},
    Random {},
    Osla {
        #[serde(default = "one_repetition")]
        repetitions_per_action: std::num::NonZeroU32,
    },
    Mcts {
        #[serde(default)]
        config: Option<Value>,
        #[serde(default)]
        config_file: Option<PathBuf>,
    },
    #[cfg(feature = "inference")]
    NeuralMcts {
        #[serde(default)]
        config: Option<Value>,
        #[serde(default)]
        config_file: Option<PathBuf>,
        #[serde(default)]
        model_file: Option<PathBuf>,
        #[serde(default)]
        model_files: std::collections::BTreeMap<String, PathBuf>,
        #[serde(default)]
        bindings: tag_rust::inference::ModelBindings,
    },
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Positions {
    #[default]
    Rotate,
    Fixed,
}
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    #[default]
    Summary,
    Json,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Reporting {
    pub format: Format,
    pub path: Option<PathBuf>,
    pub recordings: Option<PathBuf>,
    pub include_actions: bool,
}

#[derive(Clone)]
pub struct ResolvedPlayer {
    pub name: String,
    pub kind: AgentConfig,
    pub mcts: Option<Settings>,
    pub base_seed: u64,
    #[cfg(feature = "inference")]
    pub models: ModelMap,
    #[cfg(feature = "inference")]
    pub selected_model: Option<tag_rust::inference::LocalMctsModel>,
}
impl ResolvedPlayer {
    /// Select an explicitly named tournament game entry without changing pool state.
    #[allow(dead_code)]
    pub fn for_game(&self, game_id: &str) -> Result<Self> {
        #[allow(unused_mut)]
        let mut player = self.clone();
        #[cfg(not(feature = "inference"))]
        let _ = game_id;
        #[cfg(feature = "inference")]
        if let AgentConfig::NeuralMcts { bindings, .. } = &self.kind {
            let model = self
                .models
                .get(game_id)
                .or_else(|| self.models.get(""))
                .ok_or_else(|| {
                    format!(
                        "agent '{}': no model configured for game entry '{game_id}'",
                        self.name
                    )
                })?;
            player.selected_model = Some(tag_rust::inference::LocalMctsModel::new(
                model.clone(),
                bindings.clone(),
            )?);
        }
        Ok(player)
    }
}
pub struct ResolvedMatch {
    pub game_index_offset: usize,
    pub config: MatchConfig,
    pub master_seed: u64,
    pub players: Vec<ResolvedPlayer>,
}

impl MatchConfig {
    /// Read a match file; references and output paths are relative to its directory.
    /// Configuration errors return before gameplay or creation of output files.
    pub fn load(path: &Path) -> Result<ResolvedMatch> {
        let mut config: Self = serde_json::from_str(&read_json(path)?)?;
        let directory = path.parent().unwrap_or(Path::new("."));
        for path in [&mut config.reporting.path, &mut config.reporting.recordings]
            .into_iter()
            .flatten()
        {
            *path = directory.join(&*path);
        }
        config.resolve(directory)
    }

    pub fn resolve(self, directory: &Path) -> Result<ResolvedMatch> {
        if self.schema_version != 1 {
            return Err("match schema_version must be 1".into());
        }
        if self.max_actions == Some(0) {
            return Err("max_actions must be positive or null".into());
        }
        if self.games == 0 || self.games > 10_000 {
            return Err("games must be between 1 and 10000".into());
        }
        if self.players.is_empty() {
            return Err(
                "players must contain at least one agent; the game validates its player count"
                    .into(),
            );
        }
        super::execution::validate_game_name(&self.game.name)?;
        if !self.game.parameters.is_object() {
            return Err("game.parameters must be an object".into());
        }
        let mut names = BTreeSet::new();
        let master_seed = self.seed.map(Ok).unwrap_or_else(automatic_seed)?;
        let mut players = Vec::new();
        for player in &self.players {
            if player.name.trim().is_empty() || !names.insert(&player.name) {
                return Err("player names must be nonempty and unique".into());
            }
            let mcts = match &player.agent {
                AgentConfig::Mcts {
                    config,
                    config_file,
                } => Some(resolve_search(config, config_file, directory, "MCTS")?),
                #[cfg(feature = "inference")]
                AgentConfig::NeuralMcts {
                    config,
                    config_file,
                    ..
                } => Some(resolve_search(
                    config,
                    config_file,
                    directory,
                    "neural MCTS",
                )?),
                _ => None,
            };
            if player.seed.is_some() && mcts.as_ref().is_some_and(|s| s.seed.is_some()) {
                return Err("specify a player seed or an MCTS configuration seed, not both".into());
            }
            if self.reporting.recordings.is_some()
                && matches!(player.agent, AgentConfig::Osla { repetitions_per_action } if repetitions_per_action.get() != 1)
            {
                return Err("sampled OSLA has no basic replay controller representation; use action/chance logs".into());
            }
            if self.reporting.recordings.is_some() && mcts.is_some() {
                return Err("the match runner's reporting.recordings does not support MCTS controllers; use tournament separate game_logs for action replays or browser session saves for resumable MCTS".into());
            }
            #[cfg(feature = "inference")]
            let (models, selected_model) = resolve_models(&player.agent, directory)?;
            players.push(ResolvedPlayer {
                #[cfg(feature = "inference")]
                models,
                #[cfg(feature = "inference")]
                selected_model,
                name: player.name.clone(),
                kind: player.agent.clone(),
                base_seed: player
                    .seed
                    .or_else(|| mcts.as_ref().and_then(|s| s.seed))
                    .unwrap_or(master_seed),
                mcts,
            });
        }
        Ok(ResolvedMatch {
            game_index_offset: 0,
            config: self,
            master_seed,
            players,
        })
    }
}

#[cfg(feature = "inference")]
type ModelMap =
    std::collections::BTreeMap<String, std::sync::Arc<tag_rust::inference::LearnedModel>>;
#[cfg(feature = "inference")]
fn resolve_models(
    agent: &AgentConfig,
    directory: &Path,
) -> Result<(ModelMap, Option<tag_rust::inference::LocalMctsModel>)> {
    use tag_rust::inference::{LearnedModel, LocalMctsModel, MAX_MODEL_BYTES};
    let mut models = std::collections::BTreeMap::new();
    let mut selected = None;
    if let AgentConfig::NeuralMcts {
        model_file,
        model_files,
        bindings,
        ..
    } = agent
    {
        if (model_file.is_some() && !model_files.is_empty())
            || (model_file.is_none() && model_files.is_empty())
        {
            return Err(
                "neural MCTS requires either model_file or a nonempty model_files map".into(),
            );
        }
        if model_files.len() > 16 || model_files.keys().any(|id| id.trim().is_empty()) {
            return Err("model_files supports at most 16 nonempty game-entry IDs".into());
        }
        let files = model_files
            .iter()
            .map(|(id, p)| (id.as_str(), p))
            .chain(model_file.iter().map(|p| ("", p)));
        let mut bytes = 0;
        for (id, path) in files {
            let path = directory.join(path);
            let text = super::files::read_text(&path, MAX_MODEL_BYTES, "neural model")?;
            bytes += text.len();
            if bytes > MAX_MODEL_BYTES {
                return Err("agent model files exceed the combined 10 MiB JSON limit".into());
            }
            let model = std::sync::Arc::new(
                LearnedModel::parse(&text)
                    .map_err(|e| format!("neural model '{}': {e}", path.display()))?,
            );
            let binding = LocalMctsModel::new(model.clone(), bindings.clone())?;
            if id.is_empty() {
                selected = Some(binding);
            }
            models.insert(id.into(), model);
        }
    }
    Ok((models, selected))
}

/// Bound configuration reads before allocating a string, including referenced files.
fn read_json(path: &Path) -> Result<String> {
    let text = super::files::read_text(path, super::files::CONFIG_BYTES, "configuration file")?;
    if text.len() > super::files::CONFIG_BYTES {
        return Err(format!("configuration file '{}' exceeds 1 MiB", path.display()).into());
    }
    Ok(text)
}

fn resolve_search(
    config: &Option<Value>,
    config_file: &Option<PathBuf>,
    directory: &Path,
    description: &str,
) -> Result<Settings> {
    let text = match (config, config_file) {
        (Some(value), None) => serde_json::to_string(value)?,
        (None, Some(path)) => read_json(&directory.join(path))?,
        _ => {
            return Err(
                format!("{description} requires exactly one of config and config_file").into(),
            );
        }
    };
    Ok(Settings::from_json(&text)?)
}
