//! Value types, configuration, constants and typed errors.
use crate::core::PlayerId;
use crate::core::grid::{BoardLimits, GridError};
use crate::core::players::InvalidPlayer;
use std::fmt;

pub const GRID_SIZE: usize = 3;
pub const PLAYER_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone: explicit duplication; Copy: tiny cell tag.
// PartialEq: compare Empty/X/O; Eq: each cell value equals itself.
pub enum Cell {
    Empty,
    X,
    O,
}

/// Coordinates of the cell selected for the acting player's mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// Clone: explicit duplication; Copy: two integers.
// PartialEq: compare both coordinates; Eq: integer equality is reflexive.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct PlaceMark {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Clone: explicit duplication; Copy: one size integer.
// PartialEq: compare grid sizes; Eq: each size equals itself.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct TicTacToeParameters {
    pub grid_size: usize,
    pub limits: BoardLimits, // Copied resource policy, independent of experimental grid size.
}

impl Default for TicTacToeParameters {
    fn default() -> Self {
        // No borrowed inputs; returns a new owned parameter value.
        Self {
            grid_size: GRID_SIZE,
            limits: BoardLimits::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit duplication.

// PartialEq: compare error kind/details in tests; Eq: all current fields equal themselves.
pub enum TicTacToeError {
    UnsupportedPlayerCount {
        requested: usize,
    },
    Grid(GridError),
    InvalidPlayer {
        player: PlayerId,
    },
    OutOfBounds {
        x: usize,
        y: usize,
    },
    WrongPlayer {
        expected: PlayerId,
        actual: PlayerId,
    },
    OccupiedCell {
        x: usize,
        y: usize,
    },
    MoveAfterGameFinished,
}

impl fmt::Display for TicTacToeError {
    fn fmt(
        &self,                              // Shared borrow of error; borrowed for formatting.
        formatter: &mut fmt::Formatter<'_>, // Exclusive borrow of output; '_ lets Rust infer its lifetime.
    ) -> fmt::Result {
        // Returns success () or an owned formatting error.
        match self {
            Self::UnsupportedPlayerCount { requested } => {
                write!(
                    formatter,
                    "Tic-Tac-Toe requires 2 players, received {requested}"
                )
            }
            Self::Grid(error) => error.fmt(formatter),
            Self::InvalidPlayer { player } => {
                write!(formatter, "unknown Tic-Tac-Toe player {}", player.index())
            }
            Self::OutOfBounds { x, y } => {
                write!(formatter, "cell ({x}, {y}) is outside the board")
            }
            Self::WrongPlayer { expected, actual } => {
                write!(
                    formatter,
                    "player {} must act, received player {}",
                    expected.index(),
                    actual.index()
                )
            }
            Self::OccupiedCell { x, y } => {
                write!(formatter, "cell ({x}, {y}) is already occupied")
            }
            Self::MoveAfterGameFinished => write!(formatter, "the game has already finished"),
        }
    }
}

impl std::error::Error for TicTacToeError {}

impl From<GridError> for TicTacToeError {
    fn from(error: GridError) -> Self {
        // Consumes owned core error; returns owned game error.
        match error {
            GridError::OutOfBounds { x, y, .. } => Self::OutOfBounds { x, y },
            other => Self::Grid(other),
        }
    }
}
impl From<InvalidPlayer> for TicTacToeError {
    fn from(error: InvalidPlayer) -> Self {
        // Consumes owned core error; seat is Copy.
        Self::InvalidPlayer {
            player: error.player,
        }
    }
}
