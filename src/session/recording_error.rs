//! Errors while recording moves or restoring game/agent progress.
//!
//! An invalid prepared move or failed restoration must not replace the live session.
use super::SessionError;
use crate::recording::HistoryError;
/// History/restoration failures preserve the original live session.
#[derive(Debug)] // Debug: inspect contextual owned errors.
pub enum RecordingError<G, A, C> {
    Session(SessionError<G, A>),
    History(HistoryError<G>),
    Restore(C),
    ForeignProposal,
    ControllerChanged,
    UnavailableRecordedAction,
}
impl<G: std::fmt::Display, A: std::fmt::Display, C: std::fmt::Display> std::fmt::Display
    for RecordingError<G, A, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(error) => error.fmt(f),
            Self::History(error) => error.fmt(f),
            Self::Restore(error) => error.fmt(f),
            Self::ForeignProposal => write!(f, "AI proposal belongs to another session"),
            Self::ControllerChanged => {
                write!(f, "controller progress changed after AI preparation")
            }
            Self::UnavailableRecordedAction => write!(
                f,
                "recorded AI action is not in the player's available action list"
            ),
        }
    }
}
impl<G: std::error::Error + 'static, A: std::error::Error + 'static, C: std::error::Error + 'static>
    std::error::Error for RecordingError<G, A, C>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Session(error) => Some(error),
            Self::History(error) => Some(error),
            Self::Restore(error) => Some(error),
            _ => None,
        }
    }
}
