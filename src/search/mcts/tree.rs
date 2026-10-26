//! An open-loop node represents an action history and stores search statistics.
//! Each iteration simulates a state by applying actions from the root.
use crate::core::PlayerId;
use std::collections::BTreeMap;

pub(super) struct Edge<A> {
    pub action: A,
    pub child: usize,
    /// Number of completed iterations that selected this edge.
    pub visits: u64,
    /// Sum of evaluated values for each player, indexed by absolute player ID.
    pub value_sums: Vec<f64>,
}
/// Visits and explored actions for one deciding player at an action history.
pub(super) struct PlayerStatistics<A> {
    pub visits: u64,
    pub edges: Vec<Edge<A>>,
}
impl<A> Default for PlayerStatistics<A> {
    fn default() -> Self {
        Self {
            visits: 0,
            edges: Vec::new(),
        }
    }
}
pub(super) struct Node<A> {
    // A stochastic history may lead to different deciding players. Keep their
    // selection statistics separate; each edge stores values for every player.
    pub decisions: BTreeMap<usize, PlayerStatistics<A>>,
}
impl<A> Default for Node<A> {
    fn default() -> Self {
        Self {
            decisions: BTreeMap::new(),
        }
    }
}
pub(super) struct PathEntry {
    pub node: usize,
    pub actor: PlayerId,
    pub edge: usize,
}
/// Insert this edge only after evaluation succeeds within the time budget.
pub(super) struct PendingExpansion<A> {
    pub node: usize,
    pub actor: PlayerId,
    pub action: A,
}
