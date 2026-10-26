//! Errors returned for invalid interactive requests or failed game/agent calls.
//!
//! These are execution failures, not wins or losses. Earlier accepted moves remain
//! available; each operation documents whether private agent updates can remain.
use crate::core::PlayerId;
use crate::runner::RunnerError;
use std::fmt;

/// Session failures leave accepted earlier moves available to the caller.
#[derive(Debug)] // Debug: inspect contextual errors, transferred to the caller.
pub enum SessionError<G, A> {
    ControllerCount {
        expected: usize,
        actual: usize,
    },
    StalePositionVersion {
        expected: u64,
        actual: u64,
    },
    PositionVersionExhausted,
    GameFinished,
    GameStopped,
    WrongActor {
        expected: PlayerId,
        actual: PlayerId,
    },
    HumanRequired {
        player: PlayerId,
    },
    AiRequired {
        player: PlayerId,
    },
    ActionNotOffered {
        player: PlayerId,
    },
    Execution(RunnerError<G, A>),
}

impl<G: fmt::Display, A: fmt::Display> fmt::Display for SessionError<G, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error, exclusive formatter.
        match self {
            Self::ControllerCount { expected, actual } => {
                write!(f, "expected {expected} controllers, received {actual}")
            }
            Self::StalePositionVersion { expected, actual } => write!(
                f,
                "requested position version {expected}, current position version is {actual}"
            ),
            Self::PositionVersionExhausted => write!(f, "session position version limit reached"),
            Self::GameFinished => write!(f, "game has finished"),
            Self::GameStopped => write!(
                f,
                "game stopped after a rule execution error; reset or restore a valid position before continuing"
            ),
            Self::WrongActor { expected, actual } => write!(
                f,
                "player {} must act, received player {}",
                expected.index(),
                actual.index()
            ),
            Self::HumanRequired { player } => {
                write!(f, "player {} requires human input", player.index())
            }
            Self::AiRequired { player } => {
                write!(f, "player {} is controlled by AI", player.index())
            }
            Self::ActionNotOffered { player } => {
                write!(f, "move is not offered to player {}", player.index())
            }
            Self::Execution(error) => error.fmt(f),
        }
    }
}
impl<G: std::error::Error + 'static, A: std::error::Error + 'static> std::error::Error
    for SessionError<G, A>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Shared nested-error borrow.
        match self {
            Self::Execution(error) => Some(error),
            _ => None,
        }
    }
}
