//! Optional position estimates, kept separate from game rules and actual scores.
use super::{Cell, Connect4Error, Connect4State};
use crate::core::evaluation::{Evaluator, open_line_score};
use crate::core::lines::windows;
use crate::core::{PlayerId, PlayerResult};

/// Return +1/-1/0 for a finished game; otherwise estimate open lines for the requested player.
/// An open line contains no opposing piece and could still become a winning line.
#[derive(Clone, Copy, Debug, Default)] // Clone/Copy: stateless policy.

pub struct Connect4Evaluator;

impl crate::core::checkpoint::Checkpoint for Connect4Evaluator {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    /// Shared stateless evaluator borrow; unit captures all persistent progress.
    fn snapshot(&self) {} // Owned unit value returned.
    /// Shared unit snapshot; create a new stateless evaluator.
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}

impl Evaluator<Connect4State> for Connect4Evaluator {
    type Error = Connect4Error;

    /// Rate this position for the supplied seat, regardless of whose turn is next.
    ///
    /// # Arguments
    /// The state is passed by a shared (read-only) reference; the perspective is copied. The evaluator is
    /// accessed through an exclusive mutable reference.
    ///
    /// # Returns
    /// A copied value: terminal win +1, loss -1, draw 0; otherwise an open-line
    /// estimate in [-0.9,0.9]. Mixed/empty windows contribute zero to the average.
    /// Open windows indicate potential; gravity can prevent an empty cell being playable now.
    ///
    /// # Errors
    /// An unknown perspective returns the owned game error before any scoring.
    fn evaluate(
        &mut self,             // Exclusive policy borrow.
        state: &Connect4State, // Shared state borrow.
        player: PlayerId,      // Copied seat perspective.
    ) -> Result<f64, Self::Error> {
        // Copied value or owned error.
        // Validate the seat even for finished games. A proven outcome takes
        // priority over all estimates of an unfinished position.
        match state.player_result(player)? {
            PlayerResult::Win => return Ok(1.0),
            PlayerResult::Loss => return Ok(-1.0),
            PlayerResult::Draw => return Ok(0.0),
            PlayerResult::Ongoing => {}
        }
        let lines = windows(state.width(), state.height(), state.winning_length());
        Ok(open_line_score(lines, player, |(x, y)| {
            // Identify the player occupying this cell, or report that it is empty.
            match state
                .board
                .get(x, y)
                .expect("window coordinates fit the board")
            {
                Cell::Empty => None,
                Cell::Piece(seat) => Some(*seat), // Copy the occupant ID from the borrowed cell.
            }
        }))
    }
}
