//! Portable records for one complete simultaneous decision per replay position.
use super::{SeatData, Versions};
use serde::{Deserialize, Serialize};

/// A portable seat choice; order is the game's declared decision order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointChoiceData<A> {
    pub actor: usize,
    pub action: A,
}

/// One complete successor and every controller's resulting progress.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointDecisionData<A> {
    pub choices: Vec<JointChoiceData<A>>,
    /// Fingerprint of the complete resolved state, including hidden cards and chance progress.
    pub state_after: String,
    pub next_seats: Vec<usize>,
    pub controllers_after: Vec<SeatData>,
}

/// One branch of complete decisions. Parent coordinates count joint decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointBranchData<A> {
    pub id: usize,
    /// Original root, user fork or reset continuation.
    pub origin: String,
    pub parent: Option<usize>,
    pub parent_position: usize,
    pub controllers: Vec<SeatData>,
    pub decisions: Vec<JointDecisionData<A>>,
}

/// An actual-origin state proof or a branched replay of joint decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointFile<A> {
    pub kind: String,
    pub versions: Versions,
    pub parameters: serde_json::Value,
    pub players: usize,
    pub root_choices: Vec<Vec<JointChoiceData<A>>>,
    /// Fingerprint of the state reached by the actual-origin root proof.
    pub root_state: String,
    pub branches: Vec<JointBranchData<A>>,
    pub selected_branch: usize,
    pub cursor: usize,
    pub delay_ms: u32,
}
