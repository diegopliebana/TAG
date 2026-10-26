//! Convenient results for games with a single winner or a draw.
//!
//! This helper suits the two current games. Teams, multiple winners or games with
//! more detailed results may need a different representation.
use super::players::{InvalidPlayer, validate_player};
use super::{Decision, PlayerId, PlayerResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: outcome tag and small seat ID.
// PartialEq/Eq: compare variant/winner; both have reflexive equality.
pub enum SingleWinnerOutcome {
    Ongoing,
    Won(PlayerId),
    Draw,
}

impl SingleWinnerOutcome {
    /// Identify who may act next, or whether play has finished.
    ///
    /// # Arguments
    /// Borrows the state or outcome. A supplied `current_player` is copied historical actor metadata.
    ///
    /// # Returns
    /// `Decision::Player` during play; `Decision::Finished` after completion.
    pub fn next_player(&self, current_player: PlayerId) -> Decision {
        // Shared outcome borrow, copied historical actor; copied decision out.
        match self {
            Self::Ongoing => Decision::Player(current_player),
            _ => Decision::Finished,
        }
    }
    /// Read the outcome for one player.
    ///
    /// # Arguments
    /// `player` is a copied seat ID. State/outcome data is borrowed. If supplied, `player_count` defines the valid seat range.
    ///
    /// # Returns
    /// `Ok(PlayerResult)` with ongoing, win, draw or loss.
    ///
    /// # Errors
    /// An unknown seat is rejected; the shared outcome also rejects an invalid stored winner.
    pub fn player_result(
        &self,
        player: PlayerId,
        player_count: usize,
    ) -> Result<PlayerResult, InvalidPlayer> {
        // Shared outcome and copied seat/count; copied result or owned error out.
        validate_player(player, player_count)?;
        Ok(match *self {
            Self::Ongoing => PlayerResult::Ongoing,
            Self::Draw => PlayerResult::Draw,
            Self::Won(winner) => {
                validate_player(winner, player_count)?;
                if winner == player {
                    PlayerResult::Win
                } else {
                    PlayerResult::Loss
                }
            }
        })
    }
}

/// Convert a result using an optional conventional scoring rule.
///
/// # Arguments
/// `result` is copied.
///
/// # Returns
/// A copied `f64`: win +1, loss -1, ongoing/draw 0. Other games may choose different scores.
pub fn win_draw_loss_score(result: PlayerResult) -> f64 {
    // Copied result in, copied score out; explicit opt-in convention.
    match result {
        PlayerResult::Win => 1.0,
        PlayerResult::Loss => -1.0,
        _ => 0.0,
    }
}
