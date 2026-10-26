//! Data structures used in saved JSON files.
//!
//! These structures describe file contents, not automatically valid live game objects.
//! Game-specific validation runs before file values are accepted as simulation state.
use crate::core::checkpoint::Checkpoint;
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::progress::TurnCounters;
use crate::core::random::{AgentRng, AgentRngSnapshot};
use serde::{Deserialize, Serialize};

/// Saved game outcome using stable field names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)] // Clone: owned transport value; PartialEq/Eq: exact metadata.
#[serde(
    tag = "kind",
    content = "player",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum OutcomeData {
    Ongoing,
    Draw,
    Won(usize),
}
impl From<SingleWinnerOutcome> for OutcomeData {
    fn from(outcome: SingleWinnerOutcome) -> Self {
        match outcome {
            SingleWinnerOutcome::Ongoing => Self::Ongoing,
            SingleWinnerOutcome::Draw => Self::Draw,
            SingleWinnerOutcome::Won(player) => Self::Won(player.index()),
        }
    }
}

/// Convert copied counters into the shared board transport order.
///
/// # Arguments
/// `counters` is copied from a state's or observation's shared progress component.
///
/// # Returns
/// Copied move, cumulative-turn, within-round-turn and completed-round counts.
pub fn counter_data(counters: TurnCounters) -> [usize; 4] {
    [
        counters.move_count(),
        counters.completed_turn_count(),
        counters.turn_count_in_round(),
        counters.round_count(),
    ]
}
/// Complete placement-game metadata; cells use x-major order and optional occupant IDs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)] // Clone: independent vectors; PartialEq/Eq: compare all serialized facts.
#[serde(deny_unknown_fields)]
pub struct BoardStateData {
    pub cells: Vec<Option<usize>>,
    pub current_player: usize,
    pub counters: [usize; 4],
    pub outcome: OutcomeData,
    pub winning_cells: Vec<(usize, usize)>,
}
/// Exact generator continuation, with decimal strings preserving integers too large for JavaScript to represent exactly as Number in JavaScript.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)] // Clone: owned strings; PartialEq/Eq: exact file fields.
#[serde(deny_unknown_fields)]
pub struct RngData {
    pub version: u32,
    pub initial_seed: String,
    pub stream: String,
    pub word_position: String,
}
impl RngData {
    /// Convert an exact generator checkpoint into its file representation.
    ///
    /// # Arguments
    /// `s` is a copied exact RNG checkpoint, including seed, stream and word position.
    ///
    /// # Returns
    /// Owned transport value with integers too large for JavaScript to represent exactly as Number formatted as decimal strings.
    pub fn from_snapshot(s: AgentRngSnapshot) -> Self {
        Self {
            version: s.version,
            initial_seed: s.initial_seed.to_string(),
            stream: s.stream.to_string(),
            word_position: s.word_position.to_string(),
        }
    }
    /// Validate saved RNG fields and reconstruct the exact checkpoint.
    ///
    /// # Arguments
    /// `self` is a shared borrow of decoded RNG fields.
    ///
    /// # Returns
    /// Copied validated checkpoint suitable for restoring an independent generator.
    ///
    /// # Errors
    /// Malformed/out-of-range integers, unsupported version or invalid generator offset.
    pub fn snapshot(&self) -> Result<AgentRngSnapshot, PersistenceError> {
        fn number<T: std::str::FromStr>(s: &str) -> Result<T, PersistenceError> {
            if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
                return Err(PersistenceError::Invalid(
                    "RNG fields require unsigned decimal strings".into(),
                ));
            }
            s.parse()
                .map_err(|_| PersistenceError::Invalid("RNG integer is outside its range".into()))
        }
        let snapshot = AgentRngSnapshot {
            version: self.version,
            initial_seed: number(&self.initial_seed)?,
            stream: number(&self.stream)?,
            word_position: number(&self.word_position)?,
        };
        AgentRng::from_snapshot(&snapshot).map_err(|e| PersistenceError::Invalid(e.to_string()))?;
        Ok(snapshot)
    }
}
/// Built-in seat policy and exact persistent progress. Current evaluators are stateless.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)] // Clone: owned progress; PartialEq/Eq: complete identity and checkpoint equality.
#[serde(
    tag = "kind",
    content = "rng",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum SeatData {
    Human,
    Random(RngData),
    FirstAvailable(RngData),
    Osla(RngData),
    /// A versioned adapter policy; its codec validates the owned settings/model identity.
    Extension {
        rng: RngData,
        policy: serde_json::Value,
    },
}
/// One accepted action and its resulting acting-seat progress.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: independent owned action/progress for transport.
#[serde(deny_unknown_fields)]
pub struct MoveData<M> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chance: Option<serde_json::Value>,
    pub actor: usize,
    pub action: M,
    pub next_player: Option<usize>,
    pub controller_after: SeatData,
}
/// Local continuation; parent_position is an absolute move count in the parent.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: independent local records and root settings.
#[serde(deny_unknown_fields)]
pub struct BranchData<M> {
    pub id: usize,
    pub parent: Option<usize>,
    pub parent_position: usize,
    pub controllers: Vec<SeatData>,
    pub moves: Vec<MoveData<M>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controller_changes: Vec<ControllerChange>,
}
/// Explicit linear controller replacements; supported by replay format version 2.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: owned seat data; Debug: inspect; serde: portable fields.
#[serde(deny_unknown_fields)]
pub struct ControllerChange {
    pub position: usize,
    pub controllers: Vec<SeatData>,
}
/// Strict version identities for exact continued play.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)] // Clone: owned identifiers; PartialEq/Eq: exact supported version equality.
#[serde(deny_unknown_fields)]
pub struct Versions {
    pub format: u32,
    pub game: String,
    pub rules: u32,
    pub agents: u32,
    pub evaluators: u32,
    pub rng: String,
}
/// Branched session document; loading always leaves playback paused in its adapter.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: independently owned file document.
#[serde(deny_unknown_fields)]
pub struct ReplayFile<P, S, M> {
    pub kind: String,
    pub versions: Versions,
    pub parameters: P,
    pub initial_state: S,
    pub branches: Vec<BranchData<M>>,
    pub selected_branch: usize,
    pub cursor: usize,
    pub delay_ms: u32,
}
/// Independent state document, with unchanged parameters and no prior controllers/history.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: independently owned state document.
#[serde(deny_unknown_fields)]
pub struct StateFile<P, S> {
    pub kind: String,
    pub versions: Versions,
    pub parameters: P,
    pub state: S,
}
/// Configurable file-validation work limits, separate from engine parameter domains.
#[derive(Clone, Copy, Debug)] // Clone/Copy: small resource settings.
pub struct ReplayLimits {
    pub max_bytes: usize,
    pub max_depth: usize,
    pub max_branches: usize,
    pub max_moves: usize,
    pub max_cells: usize,
    pub max_verification_work: usize,
}
impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_bytes: 10 * 1024 * 1024,
            max_depth: 64,
            max_branches: 256,
            max_moves: 100_000,
            max_cells: 4096,
            max_verification_work: 100_000_000,
        }
    }
}
/// Owned descriptive validation or JSON conversion failure.
#[derive(Debug)] // Debug: inspect transferred failure details.
pub enum PersistenceError {
    Invalid(String),
    Json(serde_json::Error),
}
impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => s.fmt(f),
            Self::Json(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for PersistenceError {}
impl From<serde_json::Error> for PersistenceError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
