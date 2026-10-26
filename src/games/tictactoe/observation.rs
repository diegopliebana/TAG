//! Independently owned visible state and observing-player identity.
use super::TicTacToeState;
use super::types::{Cell, TicTacToeError};
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::{Decision, PlayerId, PlayerResult};

/// An independent copy of the public board/state, labelled with the observing player.
/// Both players see the board; only the active observer receives actions to choose.
/// Later moves in the live game do not alter this saved observation.
/// Exposes read-only access to its owned state.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: independent view snapshot.
// PartialEq: compare state and observer; Eq: both have reflexive equality.
pub struct TicTacToeObservation {
    pub(super) state: TicTacToeState,
    pub(super) observer: PlayerId,
}

impl TicTacToeObservation {
    /// Query the winning line using only this independently owned public view.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; no authoritative state is accessed.
    ///
    /// # Returns
    /// An owned coordinate vector, empty unless this observed game was won.
    pub fn winning_cells(&self) -> Vec<(usize, usize)> {
        self.state.winning_cells()
    }

    /// Return the move, completed-turn and round counts without changing this position.
    pub fn counters(&self) -> crate::core::progress::TurnCounters {
        self.state.counters()
    }

    /// All cells as a flat sequence: cell (x, y) is at x * grid_size() + y.
    /// Borrow all cells as `&[Cell]`, a flat x-major slice; nothing is copied or cloned.
    pub fn board(&self) -> &[Cell] {
        // Shared self borrow. Returns a shared borrow of the owned board.
        self.state.board()
    }

    /// Return the copied `usize` side length of the square board.
    pub fn grid_size(&self) -> usize {
        // Shared borrow; copied runtime size.
        self.state.grid_size()
    }

    /// Return the count of completed turns since the last round boundary as `usize`, without changing state.
    pub fn turn_count_in_round(&self) -> usize {
        // Shared borrow; copied counter.
        self.state.turn_count_in_round()
    }

    /// The player to act during play; after completion, the player who made the final move.
    /// Consult next_player() on the model or observation to find whether another move is allowed.
    /// Return the copied current seat; after completion this is the final actor.
    pub fn current_player(&self) -> PlayerId {
        // Shared self borrow. Returns the copied historical player.
        self.state.current_player()
    }

    /// Return the count of accepted moves, including the terminal move as `usize`, without changing state.
    pub fn move_count(&self) -> usize {
        // Shared self borrow. Returns the copied move count.
        self.state.move_count()
    }

    /// Count completed nonterminal turns.
    /// Return the total of completed turns, retained across rounds as `usize`, without changing state.
    pub fn completed_turn_count(&self) -> usize {
        // Shared self borrow. Returns the copied completed-turn count.
        self.state.completed_turn_count()
    }

    /// Return zero, the round count for Tic-Tac-Toe.
    /// Return the count of completed rounds as `usize`, without changing state.
    pub fn round_count(&self) -> usize {
        // Shared self borrow. Returns the copied round count.
        self.state.round_count()
    }

    /// Return the copied `PlayerId` of the first seat (seat 0).
    pub fn first_player(&self) -> PlayerId {
        // Shared self borrow. Returns the copied first seat.
        self.state.first_player()
    }

    /// Return the copied `SingleWinnerOutcome`.
    pub fn status(&self) -> SingleWinnerOutcome {
        // Shared self borrow. Returns the copied status.
        self.state.status()
    }

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
    pub fn cell(
        &self,    // Shared self borrow.
        x: usize, // Copied coordinate.
        y: usize, // Copied coordinate.
    ) -> Result<Cell, TicTacToeError> {
        // Copied cell or owned error.
        self.state.cell(x, y)
    }

    /// Return the copied `PlayerId` identifying whose view this observation represents.
    pub fn observer(&self) -> PlayerId {
        // Shared borrow; returns a copied seat.
        self.observer
    }

    /// Identify who may act next, or whether play has finished.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Decision::Player` during play; `Decision::Finished` after completion.
    pub fn next_player(&self) -> Decision {
        // Shared borrow; returns a copied decision.
        self.state.next_player()
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
        &self,            // Shared observation borrow.
        player: PlayerId, // Copied seat ID.
    ) -> Result<PlayerResult, TicTacToeError> {
        // Copied result or owned error.
        self.state.player_result(player)
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
    pub fn player_score(
        &self,            // Shared observation borrow.
        player: PlayerId, // Copied seat ID.
    ) -> Result<f64, TicTacToeError> {
        // Copied score or owned error.
        self.state.player_score(player)
    }
}
