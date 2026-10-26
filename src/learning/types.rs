//! Descriptions and returned values for learning inputs.
//!
//! A TensorSpec describes an array's number type, dimensions and meanings. A
//! Float32Tensor holds actual values. LearningSpec describes a game configuration;
//! EncodedObservation holds one player's encoded position. Image/graph descriptions
//! are formats for future providers, not implemented image capture or graph encoding.
// Learning notes: Debug permits inspection. Clone duplicates owned vectors/strings;
// PartialEq compares descriptors/data, and Eq is used only for exact integer/tag data.
// Copy is limited to small policies/tags. Float tensors deliberately do not derive Eq.
use crate::core::PlayerId;
use std::fmt;

/// Maximum cell count and numeric-data size allowed in one encoding operation.
/// These adjustable safety limits are separate from the game rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LearningLimits {
    /// Maximum number of board cells allowed in one operation.
    pub max_cells: usize,
    /// Maximum bytes of numeric values/indices returned, not total process memory.
    pub max_bytes: usize,
}
impl Default for LearningLimits {
    fn default() -> Self {
        Self {
            max_cells: 4096,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}
/// Reasons an observation could not be encoded.
/// Returning one of these errors leaves the observation and game unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LearningError {
    Invalid(String),
    UnsupportedPayload(String),
    ResourceLimit,
    Overflow,
}
impl fmt::Display for LearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid learning input: {message}"),
            Self::UnsupportedPayload(name) => write!(f, "unsupported learning payload: {name}"),
            Self::ResourceLimit => write!(f, "learning data exceeds configured allocation limit"),
            Self::Overflow => write!(f, "learning size or numeric conversion overflow"),
        }
    }
}
impl std::error::Error for LearningError {}

/// Information needed to distinguish two meanings of the same-sized arrays.
/// For example, changing the order of cell values requires a new encoding version,
/// even if the board dimensions stay the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodingIdentity {
    /// Game identifier, such as tictactoe.
    pub game: &'static str,
    /// Version of the rules interpreting this game state.
    pub rules_version: u32,
    /// Version of the numeric layout and value meanings.
    pub encoding_version: u32,
    /// Game-specific configuration. General games need no board dimensions.
    pub configuration: EncodingConfiguration,
    /// Number of player seats described by this encoding.
    pub player_count: usize,
}
/// Settings that distinguish meanings of observation arrays.
/// Placement retains the established two-game settings; Named stores canonical
/// game-defined key/value settings (including any content/version identity).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodingConfiguration {
    Placement {
        width: usize,
        height: usize,
        target: usize,
    },
    Named(std::collections::BTreeMap<String, String>),
}
impl EncodingIdentity {
    /// Return placement columns, or zero when this encoding has no placement board.
    pub fn width(&self) -> usize {
        match self.configuration {
            EncodingConfiguration::Placement { width, .. } => width,
            _ => 0,
        }
    }
    /// Return placement rows, or zero when this encoding has no placement board.
    pub fn height(&self) -> usize {
        match self.configuration {
            EncodingConfiguration::Placement { height, .. } => height,
            _ => 0,
        }
    }
    /// Return placement winning length, or zero for a general encoding.
    pub fn target(&self) -> usize {
        match self.configuration {
            EncodingConfiguration::Placement { target, .. } => target,
            _ => 0,
        }
    }
}
/// The number type an array uses, such as 32-bit floating point or 64-bit integer.
/// Adapters must check this explicitly instead of silently rounding or converting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float64,
    Int8,
    Int64,
    Uint8,
    Bool,
}
/// Description of one named array: number type, dimensions and meaning of its axes.
/// It contains no observation values. For a `[3,height,width]` board array, the axes
/// are channel, row and column; components name the three channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// Name callers request, such as cells or board.
    pub name: &'static str,
    /// Element number type; adapters must not assume every array is float32.
    pub dtype: DType,
    /// Dimension sizes, without a leading batch dimension.
    pub shape: Vec<usize>,
    /// Meaning of each dimension, for example channel, y, x.
    pub axes: Vec<&'static str>,
    /// Channel names or explanations of coded values.
    pub components: Vec<&'static str>,
    /// Inclusive uniform integral bounds, also usable for floating-point arrays.
    /// None means no bounds have been declared; adapters must not invent them.
    pub bounds: Option<(i64, i64)>,
}
/// Description required by a future image encoder.
/// Identifies pixels, colors, number ranges and how the image was rendered.
/// This type does not itself capture or draw an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpec {
    pub pixels: TensorSpec,
    pub colour_space: String,
    pub value_range: String,
    pub renderer_identity: String,
    pub render_settings_identity: String,
}
/// Description required by a future graph encoder.
/// Nodes represent entities; edges describe connections between them. Feature arrays
/// store numbers for nodes, edges or the whole graph. No graph encoder exists yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSpec {
    pub name: &'static str,
    pub node_features: TensorSpec,
    pub edge_features: TensorSpec,
    pub global_features: TensorSpec,
    /// Int64 [2, edge_count], row zero source and row one destination.
    pub edge_index: TensorSpec,
    pub directed: bool,
    /// Ragged batches concatenate nodes/edges and provide independent offsets.
    pub batching: &'static str,
}
/// The different kinds of inputs a game can advertise: features, grids, images or graphs.
/// The existing API calls each named observation representation a payload.
/// Current games encode features and grids. Image and graph specifications describe
/// formats for future encoders; these games do not yet generate those inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PayloadSpec {
    Features(TensorSpec),
    Grid(TensorSpec),
    Image(ImageSpec),
    // Box owns the larger graph descriptor on the heap, keeping ordinary entries
    // smaller. Clone still duplicates its owned data; it does not share mutation.
    Graph(Box<GraphSpec>),
}
impl PayloadSpec {
    /// Return the observation representation's name, such as "cells" or "board".
    pub fn name(&self) -> &'static str {
        match self {
            Self::Features(t) | Self::Grid(t) => t.name,
            Self::Image(image) => image.pixels.name,
            Self::Graph(graph) => graph.name,
        }
    }
    /// Borrow a single-tensor descriptor when this representation has one.
    /// Graph bundles return None because they contain multiple typed tensors.
    pub fn tensor(&self) -> Option<&TensorSpec> {
        match self {
            Self::Features(t) | Self::Grid(t) => Some(t),
            Self::Image(image) => Some(&image.pixels),
            Self::Graph(_) => None,
        }
    }
}

/// How a game represents its policy choices. Fixed domains keep permanent IDs;
/// candidate policies supply decision-local IDs with explicit action features.
/// CellXMajor uses x*height+y; Column uses the column number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionMapping {
    CellXMajor,
    Column,
    /// Game-defined permanent IDs. The game's LearningGame implementation maps actions.
    Indexed,
    /// A variable candidate list, mapped to authoritative actions by the caller.
    /// ActionSpec.size is zero; individual requests supply candidate counts/identities.
    Candidates,
}
/// The set of action IDs for one game configuration, including unavailable actions.
/// For standard Connect 4, size=7 and Column maps IDs 0 through 6 to columns.
/// A separate legal-action mask says which IDs can be chosen now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpec {
    pub size: usize,
    pub mapping: ActionMapping,
}
impl ActionSpec {
    /// Validate borrowed domain/dimension metadata before mapping an action.
    /// Returns unit; inconsistent domains or overflow return a typed error.
    fn validate(&self, identity: &EncodingIdentity) -> Result<(), LearningError> {
        let expected = match self.mapping {
            ActionMapping::CellXMajor => identity
                .width()
                .checked_mul(identity.height())
                .ok_or(LearningError::Overflow)?,
            ActionMapping::Column => identity.width(),
            _ => {
                return Err(LearningError::Invalid(
                    "game-defined actions have no placement coordinates".into(),
                ));
            }
        };
        if identity.width() == 0 || identity.height() == 0 || self.size != expected {
            return Err(LearningError::Invalid(
                "action domain/dimension mismatch".into(),
            ));
        }
        Ok(())
    }

    /// Encode copied coordinates under borrowed domain metadata.
    /// Returns a stable ID; errors reject invalid coordinates or overflow.
    pub fn id(
        &self,
        identity: &EncodingIdentity,
        x: usize,
        y: usize,
    ) -> Result<usize, LearningError> {
        self.validate(identity)?;
        if x >= identity.width() || y >= identity.height() {
            return Err(LearningError::Invalid(
                "action coordinates out of bounds".into(),
            ));
        }
        match self.mapping {
            ActionMapping::CellXMajor => x
                .checked_mul(identity.height())
                .and_then(|v| v.checked_add(y))
                .ok_or(LearningError::Overflow),
            ActionMapping::Column if y == 0 => Ok(x),
            ActionMapping::Column => Err(LearningError::Invalid("column actions use y=0".into())),
            _ => Err(LearningError::Invalid(
                "game-defined actions have no placement coordinates".into(),
            )),
        }
    }
    /// Decode a copied ID under borrowed domain metadata.
    /// Returns (x,y); column actions use y=0. Invalid IDs return an error.
    pub fn coordinates(
        &self,
        identity: &EncodingIdentity,
        id: usize,
    ) -> Result<(usize, usize), LearningError> {
        self.validate(identity)?;
        if id >= self.size || identity.height() == 0 {
            return Err(LearningError::Invalid("action ID out of bounds".into()));
        }
        Ok(match self.mapping {
            ActionMapping::CellXMajor => (id / identity.height(), id % identity.height()),
            ActionMapping::Column => (id, 0),
            _ => {
                return Err(LearningError::Invalid(
                    "game-defined actions have no placement coordinates".into(),
                ));
            }
        })
    }
}
/// The cell indices in each row, column or diagonal group examined by the heuristic.
/// Indices are stored consecutively; window_length is the number of cells per group.
/// For 3 by 3 Tic-Tac-Toe there are eight groups of three indices. No scores are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenLinesSpec {
    pub version: u32,
    pub window_length: usize,
    pub indices: Vec<i64>,
}
/// Description of the learning data a game configuration can produce.
/// It explains available arrays, always-returned progress/results, action IDs and
/// heuristic line groups. Actual values for a position are in EncodedObservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearningSpec {
    /// Game/configuration/version information needed to interpret the arrays.
    pub identity: EncodingIdentity,
    /// Inputs this game actually produces; absent image/graph producers are not advertised.
    pub payloads: Vec<PayloadSpec>,
    /// Descriptions of always-returned progress and results.
    pub metadata: Vec<TensorSpec>,
    /// All representable action IDs and their mapping, including currently illegal moves.
    pub actions: ActionSpec,
    /// Optional ordered cell groups for the placement heuristic. Other games use None.
    pub heuristic: Option<OpenLinesSpec>,
}
/// One actual array of 32-bit floating-point values, with a name and dimensions.
/// The values vector has its own storage. Clone copies it, so edits to a copy do not
/// change the original array or game state.
#[derive(Clone, Debug, PartialEq)]
pub struct Float32Tensor {
    /// Which requested representation these values belong to.
    pub name: &'static str,
    /// How to interpret the flat values vector as an array.
    pub shape: Vec<usize>,
    /// Separate storage for the actual numbers; no alias into live game data.
    pub values: Vec<f32>,
}
/// Labels for each player's current result, not numeric training rewards.
/// For example, Draw=2 identifies the label "draw"; it does not award two points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum ResultTag {
    Ongoing = 0,
    Win = 1,
    Draw = 2,
    Loss = 3,
}
/// Arrays and information describing one player's view of one position.
/// Each payload declares its viewpoint; placement cells are relative to observer.
/// Results use absolute player order.
/// All mutable data is stored separately from the game. A later executor will attach
/// request IDs and position versions; those are not invented by this encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedObservation {
    /// Meaning/version of this encoding.
    pub identity: EncodingIdentity,
    /// Player for whom own/opponent values were calculated.
    pub observer: PlayerId,
    /// Who may act next; None when play has ended.
    pub next_player: Option<PlayerId>,
    /// Requested arrays, each with its own values.
    pub tensors: Vec<Float32Tensor>,
    /// Moves, total completed turns, turns this round, rounds, in that order.
    pub progress: [i64; 4],
    /// One result label per absolute player ID, independent of observer-relative channels.
    pub results: Vec<ResultTag>,
    /// True for each ID currently offered to this observer; false for inactive/finished views.
    pub action_mask: Vec<bool>,
    /// IDs in the same order as the game's compact offered-action list.
    pub offered_action_ids: Vec<usize>,
}

/// Shared progress/outcome metadata used by sequential games.
pub(crate) fn metadata_spec(players: usize) -> Vec<TensorSpec> {
    vec![
        TensorSpec {
            name: "progress",
            bounds: Some((0, i64::MAX)),
            dtype: DType::Int64,
            shape: vec![4],
            axes: vec!["counter"],
            components: vec![
                "move_count",
                "completed_turn_count",
                "turn_count_in_round",
                "round_count",
            ],
        },
        TensorSpec {
            name: "results",
            bounds: Some((0, 3)),
            dtype: DType::Int8,
            shape: vec![players],
            axes: vec!["absolute_seat"],
            components: vec!["ongoing=0", "win=1", "draw=2", "loss=3"],
        },
    ]
}
