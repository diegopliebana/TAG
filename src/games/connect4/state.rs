//! Independent simulation state with read-only public queries.
use super::PLAYER_COUNT;
use super::{Cell, Connect4Error};
use crate::core::PlayerId;
use crate::core::grid::Grid;
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::outcomes::win_draw_loss_score;
use crate::core::progress::TurnCounters;
use crate::core::{Decision, PlayerResult};

#[derive(Clone, Debug, PartialEq, Eq)] // Clone: explicit independent state.
// PartialEq/Eq: compare all simulation fields; equality is reflexive.
pub struct Connect4State {
    pub(super) board: Grid<Cell>,
    pub(super) winning_length: usize,
    pub(super) current_player: PlayerId,
    pub(super) counters: TurnCounters,
    pub(super) status: SingleWinnerOutcome,
    pub(super) winning_cells: Vec<(usize, usize)>,
}
impl Connect4State {
    /// Return the move, completed-turn and round counts without changing this position.
    pub fn counters(&self) -> crate::core::progress::TurnCounters {
        self.counters
    }

    /// All cells as a flat sequence: cell (x, y) is at x * height() + y.
    /// y = height() - 1 is the bottom row, where pieces start accumulating.
    /// Borrow all cells as `&[Cell]`, a flat x-major slice; nothing is copied or cloned.
    pub fn board(&self) -> &[Cell] {
        self.board.cells()
    } // Shared borrow; borrowed flat x-major cells.
    /// Return the copied `usize` number of board columns.
    pub fn width(&self) -> usize {
        self.board.width()
    } // Shared borrow; copied dimension.
    /// Return the copied `usize` number of board rows.
    pub fn height(&self) -> usize {
        self.board.height()
    } // Shared borrow; copied dimension.
    /// Number of consecutive pieces needed to win, in any allowed line direction.
    /// Return the copied `usize` number of consecutive pieces required to win.
    pub fn winning_length(&self) -> usize {
        self.winning_length
    } // Shared borrow; copied target.
    /// The player to act during play; after completion, the player who made the final move.
    /// Return the copied current seat; after completion this is the final actor.
    pub fn current_player(&self) -> PlayerId {
        self.current_player
    } // Shared borrow; copied historical seat.
    /// Return the copied `PlayerId` of the first seat (seat 0).
    pub fn first_player(&self) -> PlayerId {
        PlayerId::new(0)
    } // Shared borrow; copied fixed first seat.
    /// Return the count of accepted moves, including the terminal move as `usize`, without changing state.
    pub fn move_count(&self) -> usize {
        self.counters.move_count()
    } // Shared borrow; copied count.
    /// Total completed nonterminal turns, retained across round boundaries.
    /// Return the total of completed turns, retained across rounds as `usize`, without changing state.
    pub fn completed_turn_count(&self) -> usize {
        self.counters.completed_turn_count()
    } // Shared borrow; copied total.
    /// Turns completed since the last round boundary; reset when a round completes.
    /// Return the count of completed turns since the last round boundary as `usize`, without changing state.
    pub fn turn_count_in_round(&self) -> usize {
        self.counters.turn_count_in_round()
    } // Shared borrow; copied within-round count.
    /// Return the count of completed rounds as `usize`, without changing state.
    pub fn round_count(&self) -> usize {
        self.counters.round_count()
    } // Shared borrow; copied completed rounds.
    /// Return the copied `SingleWinnerOutcome`.
    pub fn status(&self) -> SingleWinnerOutcome {
        self.status
    } // Shared borrow; copied outcome.
    /// Coordinates of the selected first winning line; empty until a win is recorded.
    /// Borrow the selected winning coordinates as `&[(usize, usize)]`; empty until a win.
    pub fn winning_cells(&self) -> &[(usize, usize)] {
        &self.winning_cells
    } // Shared borrow of owned coordinates.
    /// Read one board cell.
    ///
    /// # Arguments
    /// `x` and `y` are copied zero-based column and row coordinates; `self` is borrowed.
    ///
    /// # Returns
    /// `Ok(Cell)` copies the small cell value; the board remains owned by the state/view.
    ///
    /// # Errors
    /// Coordinates outside the board return the game error.
    pub fn cell(&self, x: usize, y: usize) -> Result<Cell, Connect4Error> {
        // Shared state borrow and copied coordinates; copied cell or owned error.
        Ok(*self.board.get(x, y)?)
    }

    /// Identify who may act next, or whether play has finished.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Decision::Player` during play; `Decision::Finished` after completion.
    pub fn next_player(&self) -> Decision {
        // Shared borrow; copied decision out.
        self.status.next_player(self.current_player)
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
    pub fn player_result(&self, player: PlayerId) -> Result<PlayerResult, Connect4Error> {
        // Shared state and copied seat; copied result or owned error out.
        Ok(self.status.player_result(player, PLAYER_COUNT)?)
    }
    /// Read the score for one player.
    ///
    /// # Arguments
    /// `player` is a copied seat ID; state data is borrowed.
    ///
    /// # Returns
    /// `Ok(f64)`: +1 for a win, -1 for a loss, 0 for ongoing play or a draw.
    ///
    /// # Errors
    /// An unknown player returns the game error.
    pub fn player_score(&self, player: PlayerId) -> Result<f64, Connect4Error> {
        // Shared borrow and copied seat; copied score or owned error out.
        Ok(win_draw_loss_score(self.player_result(player)?))
    }
}
