//! Check whether a player ID refers to an existing seat.
//!
//! Seat 0 is the first player. Existence alone does not mean that player can act now;
//! turn order and participation are checked by the game.
use super::PlayerId;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit error duplication.
// PartialEq/Eq: exact seat/count comparison, reflexive integer equality.
pub struct InvalidPlayer {
    pub player: PlayerId,
    pub player_count: usize,
}

/// Check whether a seat exists; turn eligibility remains a game rule.
///
/// # Arguments
/// `player` and `player_count` are copied values; valid indices are 0..player_count.
///
/// # Returns
/// `Ok(())` for an existing seat.
///
/// # Errors
/// `InvalidPlayer` contains the rejected seat and count; a count of zero permits no seats.
pub fn validate_player(player: PlayerId, player_count: usize) -> Result<(), InvalidPlayer> {
    // Copied seat/count in; success or owned error out. Eligibility is a separate game rule.
    if player.index() >= player_count {
        Err(InvalidPlayer {
            player,
            player_count,
        })
    } else {
        Ok(())
    }
}
impl fmt::Display for InvalidPlayer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Shared error, exclusive formatter; returns formatting success/error.
        write!(
            formatter,
            "player {} is outside {} seats",
            self.player.index(),
            self.player_count
        )
    }
}
impl std::error::Error for InvalidPlayer {}
