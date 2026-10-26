//! Game values, experimental parameters and contextual errors.
use crate::core::PlayerId;
use crate::core::grid::{BoardLimits, GridError};
use crate::core::players::InvalidPlayer;
use std::fmt;
pub const PLAYER_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: small cell tag and seat.
// PartialEq/Eq: compare occupancy/seat; reflexive equality.
pub enum Cell {
    Empty,
    Piece(PlayerId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// Clone/Copy: one column index.
// PartialEq/Eq: exact reflexive integer equality; Hash: action identity by column.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct DropPiece {
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Clone/Copy: dimensions, target and copied limit.
// PartialEq/Eq: compare full requested configuration; reflexive integer equality.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct Connect4Parameters {
    pub width: usize,
    pub height: usize,
    pub winning_length: usize,
    pub limits: BoardLimits,
}
impl Default for Connect4Parameters {
    fn default() -> Self {
        // Returns owned copied parameters.
        Self {
            width: 7,
            height: 6,
            winning_length: 4,
            limits: BoardLimits::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit error duplication.
// PartialEq/Eq: compare variant/details; all fields have reflexive equality.
pub enum Connect4Error {
    UnsupportedPlayerCount {
        requested: usize,
    },
    InvalidWinningLength {
        requested: usize,
    },
    ImpossibleWinningLength {
        requested: usize,
        longest_line: usize,
    },
    InvalidPlayer {
        player: PlayerId,
    },
    MoveAfterGameFinished,
    WrongPlayer {
        expected: PlayerId,
        actual: PlayerId,
    },
    InvalidColumn {
        column: usize,
        width: usize,
    },
    FullColumn {
        column: usize,
    },
    Grid(GridError),
}
impl From<GridError> for Connect4Error {
    fn from(error: GridError) -> Self {
        Self::Grid(error)
    } // Consumes core error into owned wrapper.
}
impl From<InvalidPlayer> for Connect4Error {
    fn from(error: InvalidPlayer) -> Self {
        Self::InvalidPlayer {
            player: error.player,
        }
    } // Consumed error, copied seat.
}
impl fmt::Display for Connect4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error, exclusive output borrow; formatting result out.
        match self {
            Self::UnsupportedPlayerCount { requested } => write!(
                formatter,
                "Connect 4 requires 2 players, received {requested}"
            ),
            Self::InvalidWinningLength { requested } => write!(
                formatter,
                "winning length must be positive, received {requested}"
            ),
            Self::ImpossibleWinningLength {
                requested,
                longest_line,
            } => write!(
                formatter,
                "winning length {requested} exceeds longest possible line {longest_line}"
            ),
            Self::InvalidPlayer { player } => {
                write!(formatter, "unknown Connect 4 player {}", player.index())
            }
            Self::MoveAfterGameFinished => {
                write!(formatter, "cannot move after the game has finished")
            }
            Self::WrongPlayer { expected, actual } => write!(
                formatter,
                "player {} must act, received {}",
                expected.index(),
                actual.index()
            ),
            Self::InvalidColumn { column, width } => {
                write!(formatter, "column {column} is outside board width {width}")
            }
            Self::FullColumn { column } => write!(formatter, "column {column} is full"),
            Self::Grid(error) => error.fmt(formatter),
        }
    }
}
impl std::error::Error for Connect4Error {}
