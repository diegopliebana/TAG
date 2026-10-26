//! Rule parameters, portable actions and checked configuration domains.
use crate::core::{PlayerId, players::InvalidPlayer};
use std::fmt;

pub const COLUMN_COUNT: usize = 11;

/// Heights count advances from outside the board to the claimed top space.
/// Default geometry is the published board, including the numbered top space.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(deny_unknown_fields)
)]
#[cfg_attr(any(feature = "replay", feature = "mcts-json"), serde(default))]
pub struct CantStopRules {
    pub column_heights: [u32; COLUMN_COUNT],
    pub markers: usize,
    pub columns_to_win: usize,
}
impl Default for CantStopRules {
    fn default() -> Self {
        Self {
            column_heights: [3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3],
            markers: 3,
            columns_to_win: 3,
        }
    }
}
impl CantStopRules {
    pub(super) fn validate(&self, players: usize) -> Result<(), CantStopError> {
        if !(2..=4).contains(&players) {
            return Err(CantStopError::UnsupportedPlayerCount(players));
        }
        if self.column_heights.contains(&0) {
            return Err(CantStopError::Parameters(
                "every column height must be positive",
            ));
        }
        if !(1..=COLUMN_COUNT).contains(&self.markers) {
            return Err(CantStopError::Parameters(
                "temporary markers must be in 1..=11",
            ));
        }
        // Otherwise every column could close without anybody reaching the target,
        // leaving no playable column. Do not invent a draw or change the rules.
        if self.columns_to_win == 0 || self.columns_to_win > COLUMN_COUNT.div_ceil(players) {
            return Err(CantStopError::Parameters(
                "winning target must be positive and prevent all columns closing without a winner",
            ));
        }
        Ok(())
    }
}

/// The seed is privileged setup data. Observations contain rules, never this seed.
/// Native callers supply an integer; adapters resolve automatic seeds later.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(deny_unknown_fields)
)]
#[cfg_attr(any(feature = "replay", feature = "mcts-json"), serde(default))]
pub struct CantStopParameters {
    pub rules: CantStopRules,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "snake_case"))]
pub enum CantStopPhase {
    /// Only Roll is legal, including at the start of every subsequent turn.
    StartTurn,
    Allocation,
    /// A successful allocation permits Stop or another Roll.
    Decision,
}

/// One or two column sums, with unordered pairs stored in increasing order.
/// Private fields prevent malformed values and ambiguous pair identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(deny_unknown_fields)
)]
pub struct Allocation {
    columns: [u8; 2],
    length: u8,
}
#[cfg(any(feature = "replay", feature = "mcts-json"))]
impl<'de> serde::Deserialize<'de> for Allocation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Data {
            columns: [u8; 2],
            length: u8,
        }
        let data = Data::deserialize(deserializer)?;
        let result = match data.length {
            1 if data.columns[1] == 0 => Allocation::single(data.columns[0]),
            2 => Allocation::pair(data.columns[0], data.columns[1]),
            _ => {
                return Err(serde::de::Error::custom(
                    "allocation length/unused column is invalid",
                ));
            }
        }
        .map_err(serde::de::Error::custom)?;
        if result.columns != data.columns {
            return Err(serde::de::Error::custom(
                "allocation pair must be canonical",
            ));
        }
        Ok(result)
    }
}
impl Allocation {
    pub fn single(column: u8) -> Result<Self, CantStopError> {
        column_index(column)?;
        Ok(Self {
            columns: [column, 0],
            length: 1,
        })
    }
    pub fn pair(a: u8, b: u8) -> Result<Self, CantStopError> {
        column_index(a)?;
        column_index(b)?;
        Ok(Self {
            columns: [a.min(b), a.max(b)],
            length: 2,
        })
    }
    pub fn columns(&self) -> &[u8] {
        &self.columns[..self.length as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum CantStopAction {
    Roll,
    Stop,
    Allocate(Allocation),
    Bust,
}

pub(super) fn column_index(column: u8) -> Result<usize, CantStopError> {
    if !(2..=12).contains(&column) {
        return Err(CantStopError::InvalidColumn(column));
    }
    Ok((column - 2) as usize)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CantStopError {
    UnsupportedPlayerCount(usize),
    Parameters(&'static str),
    InvalidPlayer(InvalidPlayer),
    WrongPlayer {
        expected: PlayerId,
        actual: PlayerId,
    },
    Finished,
    IllegalAction {
        action: CantStopAction,
        phase: CantStopPhase,
    },
    InvalidColumn(u8),
    InvalidRoll([u8; 4]),
    CounterOverflow,
}
impl From<InvalidPlayer> for CantStopError {
    fn from(e: InvalidPlayer) -> Self {
        Self::InvalidPlayer(e)
    }
}
impl fmt::Display for CantStopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlayerCount(n) => {
                write!(f, "Can't Stop requires 2-4 players; received {n}")
            }
            Self::Parameters(message) => write!(f, "invalid Can't Stop parameters: {message}"),
            Self::InvalidPlayer(e) => e.fmt(f),
            Self::WrongPlayer { expected, actual } => write!(
                f,
                "expected player {}, received {}",
                expected.index(),
                actual.index()
            ),
            Self::Finished => write!(f, "Can't Stop has already finished"),
            Self::IllegalAction { action, phase } => write!(
                f,
                "action {action:?} is not legal in phase {phase:?} for this position"
            ),
            Self::InvalidColumn(n) => write!(f, "column {n} is outside 2..=12"),
            Self::InvalidRoll(dice) => write!(
                f,
                "dice must contain four values in 1..=6; received {dice:?}"
            ),
            Self::CounterOverflow => write!(f, "Can't Stop action/turn counters would overflow"),
        }
    }
}
impl std::error::Error for CantStopError {}
