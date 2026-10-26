//! State evaluation functions used to compare simulated positions.
//!
//! An evaluator rates a state for a specified player: larger means preferable for
//! that player, regardless of whose turn comes next. The open-line heuristic examines
//! groups of cells that could make a winning line. Its heuristic value estimates
//! how promising the position is; game outcomes come from the rules.
use super::PlayerId;

/// Maximum distance below the best estimate for a move to count as near-best.
pub const EVALUATION_TIE_TOLERANCE: f64 = 0.001;

/// Estimate how good a state is for the requested player, regardless of whose turn is next.
/// Each implementation defines its value scale and the players for whom values or observations are calculated.
pub trait Evaluator<S> {
    type Error;

    /// Estimate how favorable a position is for the requested player.
    ///
    /// # Arguments
    /// `self` can update private evaluator memory. `state` is a read-only reference;
    /// `player` identifies whose estimate is requested, not necessarily whose turn is next.
    /// An acting agent must supply a state based only on information it may know.
    ///
    /// # Returns
    /// A finite f64 on the evaluator's documented scale. Larger is preferable to player.
    /// A score estimate is separate from the game's actual result or score.
    ///
    /// # Errors
    /// The evaluator's Error, such as an unknown player. Search also rejects NaN or
    /// infinite values even if the evaluator returns them inside Ok.
    fn evaluate(
        &mut self,        // Exclusive evaluator borrow.
        state: &S,        // Shared state borrow.
        player: PlayerId, // Copied perspective.
    ) -> Result<f64, Self::Error>; // Copied value or owned error.
}

/// Evaluate open lines for the requested player and average their contributions.
/// A window is a sequence of cells long enough to form a winning line.
///
/// # Arguments
/// `windows` and `occupant` are consumed.
/// Each window supplies coordinates; the closure returns a copied occupant seat
/// or `None` for an empty cell. `player` is the copied friendly perspective.
/// Any other occupant is treated as opposing. Games validate perspective and
/// terminal results first.
///
/// # Returns
/// A copied finite estimate in [-0.9,0.9], or zero if there are no windows.
/// Friendly-only windows contribute (friendly/length)^2; opposing-only windows
/// contribute its negative. Mixed and empty windows contribute zero but still
/// count in the average. Empty windows supplied by a caller contribute zero.
pub fn open_line_score<W, I, P>(
    windows: W,       // Consumed traversal.
    player: PlayerId, // Copied friendly seat.
    mut occupant: P,  // Consumed closure; may borrow the board.
) -> f64
// Copied estimate.
where
    W: IntoIterator<Item = I>,
    I: IntoIterator<Item = (usize, usize)>,
    P: FnMut((usize, usize)) -> Option<PlayerId>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for window in windows {
        let (mut length, mut friendly, mut opposing) = (0usize, 0usize, 0usize);
        for cell in window {
            length += 1;
            match occupant(cell) {
                Some(seat) if seat == player => friendly += 1,
                Some(_) => opposing += 1,
                None => {}
            }
        }
        // Pieces from both sides block this window for either player.
        let contribution = if length == 0 || (friendly > 0 && opposing > 0) {
            0.0
        } else if friendly > 0 {
            (friendly as f64 / length as f64).powi(2)
        } else {
            -(opposing as f64 / length as f64).powi(2)
        };
        sum += contribution;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        0.9 * sum / count as f64
    }
}
