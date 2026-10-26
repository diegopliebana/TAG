//! Authoritative simulation state and read-only public access.
use super::PLAYER_COUNT;
use super::types::{Cell, TicTacToeError};
use crate::core::PlayerId;
use crate::core::grid::Grid;
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::outcomes::win_draw_loss_score;
use crate::core::progress::TurnCounters;
use crate::core::{Decision, PlayerResult};

/// Owned simulation state with read-only queries and independent snapshots.
/// Board index is x * grid_size + y. No public arbitrary-position constructor exists.
#[derive(Clone, Debug, PartialEq, Eq)] // Clone: independent state snapshot.
// PartialEq: compare the state; Eq: state equality is reflexive.
pub struct TicTacToeState {
    pub(super) board: Grid<Cell>,
    pub(super) current_player: PlayerId,
    pub(super) counters: TurnCounters,
    pub(super) status: SingleWinnerOutcome,
}

impl TicTacToeState {
    /// Query the first winning line without changing game progress.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; the board and outcome remain owned by this state.
    ///
    /// # Returns
    /// An independently owned coordinate vector in rule scan order, empty during
    /// ongoing play or a draw. Multiple completed lines select the first one.
    pub fn winning_cells(&self) -> Vec<(usize, usize)> {
        match self.status {
            SingleWinnerOutcome::Won(player) => super::forward_model::full_line(
                &self.board,
                if player.index() == 0 {
                    Cell::X
                } else {
                    Cell::O
                },
            )
            .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Return the move, completed-turn and round counts without changing this position.
    pub fn counters(&self) -> crate::core::progress::TurnCounters {
        self.counters
    }

    /// All cells as a flat sequence: cell (x, y) is at x * grid_size() + y.
    /// Borrow all cells as `&[Cell]`, a flat x-major slice; nothing is copied or cloned.
    pub fn board(&self) -> &[Cell] {
        // Shared self borrow. Returns a shared borrow of the owned board.
        self.board.cells()
    }

    /// Return the copied `usize` side length of the square board.
    pub fn grid_size(&self) -> usize {
        // Shared borrow; copied runtime size.
        self.board.width()
    }

    /// Return the count of completed turns since the last round boundary as `usize`, without changing state.
    pub fn turn_count_in_round(&self) -> usize {
        // Shared borrow; copied counter.
        self.counters.turn_count_in_round()
    }

    /// The player to act during play; after completion, the player who made the final move.
    /// Consult next_player() on the model or observation to find whether another move is allowed.
    /// Return the copied current seat; after completion this is the final actor.
    pub fn current_player(&self) -> PlayerId {
        // Shared self borrow. Returns the copied historical player.
        self.current_player
    }

    /// Return the count of accepted moves, including the terminal move as `usize`, without changing state.
    pub fn move_count(&self) -> usize {
        // Shared self borrow. Returns the copied move count.
        self.counters.move_count()
    }

    /// Count completed nonterminal turns.
    /// Return the total of completed turns, retained across rounds as `usize`, without changing state.
    pub fn completed_turn_count(&self) -> usize {
        // Shared self borrow. Returns the copied completed-turn count.
        self.counters.completed_turn_count()
    }

    /// Return zero, the round count for Tic-Tac-Toe.
    /// Return the count of completed rounds as `usize`, without changing state.
    pub fn round_count(&self) -> usize {
        // Shared self borrow. Returns the copied round count.
        self.counters.round_count()
    }

    /// Return the copied `PlayerId` of the first seat (seat 0).
    pub fn first_player(&self) -> PlayerId {
        // Shared self borrow. Returns the copied first seat.
        PlayerId::new(0)
    }

    /// Return the copied `SingleWinnerOutcome`.
    pub fn status(&self) -> SingleWinnerOutcome {
        // Shared self borrow. Returns the copied status.
        self.status
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
        // Shared borrow; returns a copied decision.
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
    pub fn player_result(
        &self,            // Shared state borrow.
        player: PlayerId, // Copied seat ID.
    ) -> Result<PlayerResult, TicTacToeError> {
        // Copied result or owned error.
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
    pub fn player_score(
        &self,            // Shared state borrow.
        player: PlayerId, // Copied seat ID.
    ) -> Result<f64, TicTacToeError> {
        // Copied score or owned error.
        Ok(win_draw_loss_score(self.player_result(player)?))
    }
}
