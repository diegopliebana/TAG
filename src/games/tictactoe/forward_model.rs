//! Rule implementation: initialization, visibility, choices and transitions.
use super::{
    Cell, PLAYER_COUNT, PlaceMark, TicTacToeError, TicTacToeObservation, TicTacToeParameters,
    TicTacToeState,
};
use crate::core::grid::Grid;
use crate::core::lines::{first_matching_run, ray};
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::players::validate_player;
use crate::core::progress::TurnCounters;
use crate::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};

/// Enumerate empty cells from a borrowed board, preserving x-major order.
/// Enumerate every empty cell in column-first order.
///
/// # Arguments
/// `board` is passed by a shared (read-only) reference; its contents are unchanged.
///
/// # Returns
/// An owned `Vec<PlaceMark>` containing copied coordinates.
fn empty_cell_actions(board: &Grid<Cell>) -> Vec<PlaceMark> {
    // Shared board borrow; returns an owned list.
    let mut actions = Vec::new();
    // Visit each column from left to right, then its cells in increasing y order.
    for x in 0..board.width() {
        for y in 0..board.height() {
            if *board
                .get(x, y)
                .expect("coordinates come from board dimensions")
                == Cell::Empty
            {
                actions.push(PlaceMark { x, y });
            }
        }
    }
    actions
}

// Rows, columns and the two diagonals must contain this mark from end to end.
/// Check whether this mark fills a row, column or corner-to-corner diagonal.
///
/// # Arguments
/// `board` is passed by a shared (read-only) reference and `mark` is copied; the board is square by game construction.
///
/// # Returns
/// A copied `bool`, true if a full-length line matches.
pub(super) fn has_full_line(board: &Grid<Cell>, mark: Cell) -> bool {
    full_line(board, mark).is_some()
}

/// Find the first full line in the rule's existing column/row/diagonal order.
///
/// # Arguments
/// `board` is passed by a shared (read-only) reference and square; `mark` is the copied player's mark.
///
/// # Returns
/// An owned coordinate vector when a line matches, otherwise `None`.
pub(super) fn full_line(board: &Grid<Cell>, mark: Cell) -> Option<Vec<(usize, usize)>> {
    // Shared board borrow and copied mark.
    let size = board.width();
    let check = |start, step| {
        // Find a full-board-length run of this player's marks along this line.
        first_matching_run(ray(size, size, start, step), size, |(x, y)| {
            // This cell must contain the mark placed by the acting player.
            *board.get(x, y).expect("ray stays within the board") == mark
        })
    };
    for i in 0..size {
        if let Some(line) = check((i, 0), (0, 1)).or_else(|| check((0, i), (1, 0))) {
            return Some(line);
        }
    }
    check((0, 0), (1, 1)).or_else(|| check((size - 1, 0), (-1, 1)))
}

/// Stateless rule implementation.
///
/// Implements the shared ForwardModel interface, including player observations.
#[derive(Clone, Copy, Debug, Default)] // Clone/Copy: stateless model.

pub struct TicTacToeForwardModel;

impl ForwardModel for TicTacToeForwardModel {
    type State = TicTacToeState;
    type Action = PlaceMark;
    type Observation = TicTacToeObservation;
    type Parameters = TicTacToeParameters;
    type Error = TicTacToeError;

    /// Create a new game state from configuration.
    ///
    /// # Arguments
    /// `parameters` is borrowed; `player_count` is copied. The model is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(State)` transfers ownership of the new state to the caller.
    ///
    /// # Errors
    /// Unsupported player counts or invalid/resource-exceeding configuration return the game error.
    fn init_state(
        &self,                            // Shared model borrow.
        parameters: &TicTacToeParameters, // Shared borrow.
        player_count: usize,              // Copied count.
    ) -> Result<TicTacToeState, TicTacToeError> {
        // New owned state or owned error.
        if player_count != PLAYER_COUNT {
            return Err(TicTacToeError::UnsupportedPlayerCount {
                requested: player_count,
            });
        }
        // Start with an empty square board, player 0 to act, and zero completed moves.
        // Grid construction rejects invalid sizes or resource limits before returning a state.
        Ok(TicTacToeState {
            // Newly built state is moved into Ok, then returned to caller.
            board: Grid::filled(
                parameters.grid_size,
                parameters.grid_size,
                Cell::Empty,
                parameters.limits,
            )?,
            current_player: PlayerId::new(0),
            counters: TurnCounters::default(),
            status: SingleWinnerOutcome::Ongoing,
        })
    }

    /// Read how many seats the game state has.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `usize` player count.
    fn player_count(
        &self,                   // Shared model borrow.
        _state: &TicTacToeState, // Shared state borrow; unused for this fixed player count.
    ) -> usize {
        // Returns count by value.
        PLAYER_COUNT
    }

    /// Identify the next decision maker.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `Decision`: a player who may act, or completion.
    fn next_player(
        &self,                  // Shared model borrow.
        state: &TicTacToeState, // Shared state borrow.
    ) -> Decision {
        // Returns a small decision value.
        state.next_player()
    }

    /// Copy only visible information into an independently owned observation.
    /// Build an independently owned view containing only information the player is allowed to know.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; `observer` is the copied seat receiving the view.
    ///
    /// # Returns
    /// `Ok(Observation)` owns a snapshot of the player's visible data.
    ///
    /// # Errors
    /// Unknown observers return the game error.
    fn observe(
        &self,                  // Shared model borrow.
        state: &TicTacToeState, // Shared state borrow; source remains owned by caller.
        observer: PlayerId,     // Copied seat ID.
    ) -> Result<TicTacToeObservation, TicTacToeError> {
        // Returns owned view or error.
        validate_player(observer, PLAYER_COUNT)?;
        Ok(TicTacToeObservation {
            // Explicit clone is safe here because the whole state is player-visible.
            state: state.clone(), // Owned copy.
            observer,
        })
    }

    /// Only the acting player's ongoing observation offers moves.
    /// List choices derived only from information the player is allowed to know.
    ///
    /// # Arguments
    /// `observation` is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// An owned `Vec<Action>`; inactive or terminal views offer no moves in the current games.
    fn offered_actions(
        &self,                              // Shared model borrow.
        observation: &TicTacToeObservation, // Shared view borrow.
    ) -> Vec<PlaceMark> {
        // Returns an owned list of actions.
        // Both players can see the board, but only the player whose turn it is can choose a move.
        // Finished games return an empty action list.
        if observation.next_player() != Decision::Player(observation.observer) {
            return Vec::new();
        }
        empty_cell_actions(&observation.state.board)
    }

    /// Every empty cell, in x-major order. Finished games offer no moves.
    /// List simulation choices for the requested actor.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; planning must use a state it is permitted to inspect. `actor` is copied.
    ///
    /// # Returns
    /// `Ok(Vec<Action>)` owns the actions; valid actors receive an empty list after completion.
    ///
    /// # Errors
    /// Unknown seats and a wrong ongoing actor return the game error.
    fn available_actions(
        &self,                  // Shared model borrow.
        state: &TicTacToeState, // Shared state borrow; enumeration changes nothing.
        actor: PlayerId,        // Copied seat ID.
    ) -> Result<Vec<PlaceMark>, TicTacToeError> {
        // Returns an owned list of coordinate values, or an owned error.
        validate_player(actor, PLAYER_COUNT)?;
        if state.status != SingleWinnerOutcome::Ongoing {
            return Ok(Vec::new());
        }
        if actor != state.current_player {
            return Err(TicTacToeError::WrongPlayer {
                expected: state.current_player,
                actual: actor,
            });
        }
        Ok(empty_cell_actions(&state.board)) // Returns owned actions; only borrows the board.
    }

    /// Apply a valid move; all rejection checks happen before any mutation.
    /// Apply an action after validating it, leaving rejected moves unchanged.
    ///
    /// # Arguments
    /// `state` is accessed through an exclusive mutable reference and remains owned by the caller.
    /// `actor` is copied; `action` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(())` means the state was advanced. A terminal move retains its actor and preterminal turn counts.
    ///
    /// # Errors
    /// Invalid actor, terminal state, wrong turn or invalid action return the game error before mutation.
    ///
    /// An empty in-range cell receives X for seat 0 or O for seat 1.
    /// A full winning line takes precedence over a full-board draw.
    fn next(
        &self,                      // Shared model borrow; stateless rules.
        state: &mut TicTacToeState, // Exclusive mutable borrow; caller retains ownership.
        actor: PlayerId,            // Copied seat ID, checked against the next decision maker.
        action: &PlaceMark,         // Shared action borrow; caller retains the action.
    ) -> Result<(), TicTacToeError> {
        // Returns `Ok(())` or an owned error.
        // Reject invalid players, finished games, wrong turns and unavailable cells
        // before placing a mark: a rejected move must leave the game unchanged.
        validate_player(actor, PLAYER_COUNT)?;
        if state.status != SingleWinnerOutcome::Ongoing {
            return Err(TicTacToeError::MoveAfterGameFinished);
        }
        if actor != state.current_player {
            return Err(TicTacToeError::WrongPlayer {
                expected: state.current_player,
                actual: actor,
            });
        }
        if state.cell(action.x, action.y)? != Cell::Empty {
            return Err(TicTacToeError::OccupiedCell {
                x: action.x,
                y: action.y,
            });
        }

        // Seat 0 always places X; seat 1 always places O.
        let mark = if actor.index() == 0 { Cell::X } else { Cell::O };
        *state.board.get_mut(action.x, action.y)? = mark; // Copy mark into owned board.
        state.counters.record_move();
        // Check a win before a draw: filling the last empty cell can also complete a line.
        let won = has_full_line(&state.board, mark);
        if won {
            state.status = SingleWinnerOutcome::Won(actor);
        } else if state
            .board
            .cells()
            .iter()
            // Every cell must contain a mark for the board to be full.
            .all(|&cell| cell != Cell::Empty)
        {
            state.status = SingleWinnerOutcome::Draw;
        } else {
            // Play continues: finish this turn and hand the next move to the other seat.
            state.counters.complete_turn();
            state.current_player = PlayerId::new(1 - actor.index());
        }
        // A terminal move retains the final actor and preterminal turn counters.
        Ok(())
    }

    /// Query the result for a particular player.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; `player` is a copied seat ID.
    ///
    /// # Returns
    /// `Ok(PlayerResult)` returns a copied outcome category, separate from score and execution errors.
    ///
    /// # Errors
    /// Unknown players return the game error.
    fn player_result(
        &self,                  // Shared model borrow.
        state: &TicTacToeState, // Shared state borrow.
        player: PlayerId,       // Copied seat; caller may reuse its ID.
    ) -> Result<PlayerResult, TicTacToeError> {
        // Returns result by value, or owned error.
        state.player_result(player)
    }

    /// Query the game-defined score for a particular player.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; `player` is a copied seat ID.
    ///
    /// # Returns
    /// `Ok(f64)` returns that player's game-defined score.
    ///
    /// # Errors
    /// Unknown players return the game error.
    fn player_score(
        &self,                  // Shared model borrow.
        state: &TicTacToeState, // Shared state borrow.
        player: PlayerId,       // Copied seat value.
    ) -> Result<f64, TicTacToeError> {
        // Returns score by value or owned error.
        state.player_score(player)
    }
}

impl PublicInformationModel for TicTacToeForwardModel {
    /// Copy this player's complete public view into an independent simulation state.
    ///
    /// # Arguments
    /// `observation` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(TicTacToeState)` owns an exact snapshot of the view, including configuration
    /// and counters.
    ///
    /// # Errors
    /// No validation error for a valid view; ordinary clone allocation can still fail.
    fn simulation_state(
        &self,                              // Shared rule borrow.
        observation: &TicTacToeObservation, // Shared borrowed public view.
    ) -> Result<TicTacToeState, TicTacToeError> {
        // All facts in this game are public. Each clone owns its board and progress.
        Ok(observation.state.clone())
    }
}
