//! Message fields shared by browser JavaScript and the Rust adapter.
//!
//! Large IDs and seeds use decimal strings because JavaScript Number cannot hold
//! every 64-bit integer exactly. Message types describe requests, views and errors.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tag_rust::core::PlayerResult;
/// Browser outcomes preserve every winner without changing older save formats.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", content = "player", rename_all = "snake_case")]
pub enum BoardOutcome {
    Ongoing,
    Draw,
    Won(usize),
    SharedWin(Vec<usize>),
}
impl From<tag_rust::core::outcomes::SingleWinnerOutcome> for BoardOutcome {
    fn from(value: tag_rust::core::outcomes::SingleWinnerOutcome) -> Self {
        use tag_rust::core::outcomes::SingleWinnerOutcome::*;
        match value {
            Ongoing => Self::Ongoing,
            Draw => Self::Draw,
            Won(p) => Self::Won(p.index()),
        }
    }
}
impl From<tag_rust::replay::OutcomeData> for BoardOutcome {
    fn from(value: tag_rust::replay::OutcomeData) -> Self {
        use tag_rust::replay::OutcomeData::*;
        match value {
            Ongoing => Self::Ongoing,
            Draw => Self::Draw,
            Won(p) => Self::Won(p),
        }
    }
}

/// Supported game selection at the adapter boundary.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)] // Clone/Copy: small game tag.
#[serde(rename_all = "snake_case")]
pub enum GameKind {
    Tictactoe,
    Connect4,
    Cantstop,
    Loveletter,
}
/// Explicit seat policy; AI randomness is initialized independently per seat.
#[derive(Clone, Debug, Serialize, Deserialize)] // Clone: owned seed strings.
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SeatSetup {
    Human,
    Random {
        seed: String,
    },
    FirstAvailable {
        seed: String,
    },
    Osla {
        seed: String,
    },
    SampledOsla {
        seed: String,
        repetitions_per_action: u32,
    },
    QLearning {
        seed: String,
        model_id: String,
    },
    NeuralOsla {
        seed: String,
        model_id: String,
    },
    Mcts {
        seed: String,
        configuration: Value,
    },
    NeuralMcts {
        seed: String,
        configuration: Value,
        model_id: String,
        #[serde(default)]
        bindings: tag_rust::inference::ModelBindings,
    },
}
/// One operation. Model rules remain in the engine.
#[derive(Debug, Deserialize)] // Debug: inspect owned request data.
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Create {
        game: GameKind,
        parameters: Option<Value>,
        controllers: Option<Vec<SeatSetup>>,
    },
    View {
        observer: Option<usize>,
    },
    History {
        branch: usize,
        cursor: usize,
        observer: Option<usize>,
    },
    SelectBranch {
        branch: usize,
    },
    Fork {
        branch: usize,
        cursor: usize,
        replacements: Vec<Option<SeatSetup>>,
    },
    HumanMove {
        actor: usize,
        action: Value,
        observer: Option<usize>,
    },
    Presentation {
        observer: Option<usize>,
        public_only: bool,
        reveal_hidden_information: bool,
    },
    Reset,
    PrepareAi,
    PollAi {
        proposal: String,
    },
    CommitAi {
        proposal: String,
        observer: Option<usize>,
    },
    DiscardAi {
        proposal: String,
    },
}
/// Request identity, expected lifecycle/position and owned command.
#[derive(Debug, Deserialize)] // Debug: inspect contextual request fields.
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub generation: String,
    pub position_version: Option<String>,
    pub command: Command,
}
/// Typed adapter failure, transferred with the request identity.
#[derive(Debug, Serialize)] // Debug: inspect failures.
pub struct BridgeError {
    pub code: &'static str,
    pub message: String,
}
impl BridgeError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
/// Board/progress facts built strictly from a player observation.
#[derive(Debug, Serialize)] // Debug: inspect owned permitted display data.
pub struct BoardView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cantstop: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loveletter: Option<Value>,
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Option<usize>>,
    pub current_player: usize,
    pub counters: [usize; 4],
    pub outcome: BoardOutcome,

    pub winning_cells: Vec<(usize, usize)>,
}
/// Owned live player view; actions contain the game's portable move values.
#[derive(Debug, Serialize)] // Debug: inspect permitted view data.
pub struct PublicView {
    pub game: GameKind,
    pub position_version: String,
    pub branch: usize,
    pub observer: usize,
    pub next_player: Option<usize>,
    pub controllers: Vec<SeatSetup>,
    pub board: BoardView,
    pub actions: Vec<Value>,
    pub results: Vec<PlayerResult>,
    pub scores: Vec<f64>,
    pub cursor: usize,
    pub tip: usize,
    pub read_only: bool,
    pub branches: Vec<BranchSummary>,
    /// Full selected-branch list only for navigation/branch changes.
    pub moves: Option<Vec<MoveView>>,
    /// Compact accepted move for incremental live-list updates.
    pub accepted_move: Option<MoveView>,
}

/// Permitted continuation identity; source records/checkpoints remain in Rust.
#[derive(Debug, Serialize)] // Debug: inspect owned branch metadata.
pub struct BranchSummary {
    pub id: usize,
    pub parent: Option<usize>,
    pub fork_position: usize,
    pub move_count: usize,
    pub origin: BranchOrigin,
}
/// UI label category for a locally created continuation.
#[derive(Clone, Copy, Debug, Serialize)] // Clone/Copy: small tag; Debug: inspect metadata.
#[serde(rename_all = "snake_case")]
pub enum BranchOrigin {
    Original,
    Fork,
    Reset,
}
/// One accepted move's permitted actor/action/source, without RNG progress.
#[derive(Debug, Serialize)] // Debug: inspect owned transport metadata.
pub struct MoveView {
    pub position: usize,
    pub actor: usize,
    pub action: Value,
    pub controller: SeatSetup,
}
/// One independently prepared action awaiting explicit commitment.
#[derive(Debug, Serialize)] // Debug: inspect proposal identifiers and portable action.
pub struct ProposalView {
    pub id: String,
    pub actor: usize,
    pub action: Value,
    pub pending: bool,
}
/// Reply echoes request identity and distinguishes request generation from current generation.
#[derive(Debug, Serialize)] // Debug: inspect command results.
pub struct Reply {
    pub request_id: String,
    pub request_generation: String,
    pub generation: String,
    pub view: Option<PublicView>,
    pub proposal: Option<ProposalView>,
    pub error: Option<BridgeError>,
}

/// File operations travel separately from ordinary 32 KiB commands.
#[derive(Debug, Deserialize)] // Debug: inspect; Deserialize: strict file envelope.
#[serde(deny_unknown_fields)]
pub struct FileRequest {
    pub request_id: String,
    pub generation: String,
    pub command: FileCommand,
}
#[derive(Debug, Deserialize)] // Debug: inspect; Deserialize: strict tagged operations.
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FileCommand {
    LoadModel,
    Models,
    PredictModel {
        model_id: String,
        observer: usize,
    },
    Export {
        scope: SaveScope,
        cursor: usize,
        delay_ms: u32,
    },
    Import {
        controllers: Option<Vec<SeatSetup>>,
    },
}
#[derive(Clone, Copy, Debug, Deserialize)] // Clone/Copy: scope tag; Debug: inspect; Deserialize: transport.
#[serde(rename_all = "snake_case")]
pub enum SaveScope {
    Session,
    Branch,
    State,
}
#[derive(Debug, Serialize)] // Debug: inspect; Serialize: owned file results.
pub struct FileReply {
    pub models: Option<Value>,
    /// Rust-serialized model documents after bundle import, retained as opaque text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sources: Option<Vec<String>>,
    pub request_id: String,
    pub request_generation: String,
    pub generation: String,
    pub view: Option<PublicView>,
    pub live_view: Option<PublicView>,
    pub parameters: Option<Value>,
    pub delay_ms: Option<u32>,
    pub needs_controllers: bool,
    pub text: Option<String>,
    pub error: Option<BridgeError>,
}
