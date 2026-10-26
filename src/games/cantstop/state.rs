//! Exact state copies retain RNG progress; public positions contain no RNG data.
use super::{COLUMN_COUNT, CantStopError, CantStopPhase, CantStopRules, types::column_index};
use crate::core::{
    Decision, PlayerId, PlayerResult,
    checkpoint::Checkpoint,
    outcomes::SingleWinnerOutcome,
    players::validate_player,
    progress::TurnCounters,
    random::{ChanceRng, ChanceRngSnapshot},
};

/// Public facts shared by players, simulations and state evaluation functions.
/// Positions are immutable to callers; only checked forward-model transitions
/// can construct a different position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CantStopPosition {
    pub(super) rules: CantStopRules,
    pub(super) permanent: Vec<[u32; COLUMN_COUNT]>,
    pub(super) temporary: [Option<u32>; COLUMN_COUNT],
    pub(super) owners: [Option<PlayerId>; COLUMN_COUNT],
    pub(super) dice: Option<[u8; 4]>,
    pub(super) current_player: PlayerId,
    pub(super) phase: CantStopPhase,
    pub(super) counters: TurnCounters,
    pub(super) status: SingleWinnerOutcome,
}
impl CantStopPosition {
    pub fn rules(&self) -> &CantStopRules {
        &self.rules
    }
    pub fn player_count(&self) -> usize {
        self.permanent.len()
    }
    /// Current player is retained as last-actor metadata after a winning Stop.
    pub fn current_player(&self) -> PlayerId {
        self.current_player
    }
    pub fn phase(&self) -> CantStopPhase {
        self.phase
    }
    pub fn dice(&self) -> Option<[u8; 4]> {
        self.dice
    }
    pub fn counters(&self) -> TurnCounters {
        self.counters
    }
    pub fn status(&self) -> SingleWinnerOutcome {
        self.status
    }
    pub fn next_player(&self) -> Decision {
        self.status.next_player(self.current_player)
    }
    pub fn permanent_position(&self, player: PlayerId, column: u8) -> Result<u32, CantStopError> {
        validate_player(player, self.player_count())?;
        Ok(self.permanent[player.index()][column_index(column)?])
    }
    pub fn temporary_position(&self, column: u8) -> Result<Option<u32>, CantStopError> {
        Ok(self.temporary[column_index(column)?])
    }
    pub fn column_owner(&self, column: u8) -> Result<Option<PlayerId>, CantStopError> {
        Ok(self.owners[column_index(column)?])
    }
    pub fn player_score(&self, player: PlayerId) -> Result<f64, CantStopError> {
        validate_player(player, self.player_count())?;
        Ok(self
            .owners
            .iter()
            .filter(|owner| **owner == Some(player))
            .count() as f64)
    }
    pub fn player_result(&self, player: PlayerId) -> Result<PlayerResult, CantStopError> {
        Ok(self.status.player_result(player, self.player_count())?)
    }
}

/// Privileged simulation state. Clone is an exact, independent copy, including
/// the generator position; it must not be used to create diverse search forks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CantStopState {
    pub(super) position: CantStopPosition,
    pub(super) random: ChanceRng,
}
impl CantStopState {
    pub fn position(&self) -> &CantStopPosition {
        &self.position
    }
    /// Privileged audit/save access. Never include this checkpoint in an observation.
    pub fn chance_snapshot(&self) -> ChanceRngSnapshot {
        self.random.snapshot()
    }
}
impl Checkpoint for CantStopState {
    // A typed snapshot cannot be forged from unchecked external fields. Portable
    // file decoding/validation is a separate integration stage.
    type Snapshot = Self;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self {
        self.clone()
    }
    fn from_snapshot(saved: &Self) -> Result<Self, Self::RestoreError> {
        Ok(saved.clone())
    }
}

/// The observing seat and an independent copy of public facts, without RNGs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CantStopObservation {
    pub(super) position: CantStopPosition,
    pub(super) observer: PlayerId,
}
impl CantStopObservation {
    pub fn observer(&self) -> PlayerId {
        self.observer
    }
    pub fn position(&self) -> &CantStopPosition {
        &self.position
    }
}
