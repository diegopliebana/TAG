//! Translate game/session errors into browser error categories and explanations.
//!
//! Keep the original cause useful to the caller; a failed request is not a game loss.
use crate::protocol::BridgeError;
use std::fmt::Display;
use tag_rust::runner::RunnerError;
use tag_rust::session::{BranchError, RecordingError, SessionError};

/// Convert a consumed branch failure to its browser category and readable message.
///
/// # Arguments
/// `error` is consumed; nested game/agent/restoration errors provide Display text.
///
/// # Returns
/// Owned typed browser error for input, lifecycle, rules or controller failures.
pub(crate) fn branch_error<G: Display, A: Display, C: Display>(
    error: BranchError<G, A, C>,
) -> BridgeError {
    let code = match &error {
        BranchError::PositionVersionExhausted => "resource_limit",
        BranchError::UnknownBranch(_) | BranchError::InvalidPosition { .. } => "invalid_branch",
        BranchError::ControllerCount { .. } => "invalid_setup",
        BranchError::History(_) => "history_error",
        BranchError::Restore(_) => "restoration_failed",
        BranchError::Execution(recording) => match recording {
            RecordingError::History(_) => "history_error",
            RecordingError::Restore(_) => "restoration_failed",
            RecordingError::UnavailableRecordedAction => "invalid_action",
            RecordingError::ForeignProposal | RecordingError::ControllerChanged => {
                "invalid_proposal"
            }
            RecordingError::Session(session) => match session {
                SessionError::ControllerCount { .. } => "invalid_setup",
                SessionError::StalePositionVersion { .. } => "stale_position",
                SessionError::PositionVersionExhausted => "resource_limit",
                SessionError::GameFinished => "game_finished",
                SessionError::GameStopped => "game_stopped",
                SessionError::WrongActor { .. } => "wrong_actor",
                SessionError::HumanRequired { .. } | SessionError::AiRequired { .. } => {
                    "wrong_controller"
                }
                SessionError::ActionNotOffered { .. } => "action_not_offered",
                SessionError::Execution(execution) => match execution {
                    RunnerError::MissingController { .. } => "wrong_controller",
                    RunnerError::NoAvailableActions { .. } => "no_available_actions",
                    RunnerError::InvalidSelection { .. } => "invalid_selection",
                    RunnerError::Game(_) => "game_error",
                    RunnerError::Agent { .. } => "agent_failure",
                },
            },
        },
    };
    BridgeError::new(code, error.to_string())
}
