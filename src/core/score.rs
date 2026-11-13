//! Optional game-defined scores reported alongside win, draw and loss.
use super::PlayerResult;

/// A single team value or one value per seat, with explicit units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "snake_case"))]
pub enum ScoreScope {
    Team,
    Players,
}

/// A score belongs to the game rules, not to a search evaluator or reward.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize))]
pub struct OutcomeScore {
    pub scope: ScoreScope,
    pub values: Vec<f64>,
    pub unit: &'static str,
    pub maximum: Option<f64>,
}

/// Result categories and an optional game-defined score at one position.
#[derive(Clone, Debug, PartialEq)]
pub struct GameResult {
    pub players: Vec<PlayerResult>,
    pub score: Option<OutcomeScore>,
}
