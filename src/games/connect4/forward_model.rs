//! Game rules, checked moves and player-visible observations.
use super::{
    Cell, Connect4Error, Connect4Observation, Connect4Parameters, Connect4State, DropPiece,
    PLAYER_COUNT,
};
use crate::core::grid::Grid;
use crate::core::lines::{first_matching_run, ray};
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::players::validate_player;
use crate::core::progress::TurnCounters;
use crate::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};

#[derive(Clone, Copy, Debug, Default)] // Clone/Copy: stateless rules.

pub struct Connect4ForwardModel;
impl ForwardModel for Connect4ForwardModel {
    type State = Connect4State;
    type Action = DropPiece;
    type Observation = Connect4Observation;
    type Parameters = Connect4Parameters;
    type Error = Connect4Error;

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
        &self,                           // Shared rules borrow.
        parameters: &Connect4Parameters, // Shared configuration borrow.
        player_count: usize,             // Copied count.
    ) -> Result<Connect4State, Connect4Error> {
        // Owned state or owned contextual error.
        if player_count != PLAYER_COUNT {
            return Err(Connect4Error::UnsupportedPlayerCount {
                requested: player_count,
            });
        }
        if parameters.winning_length == 0 {
            return Err(Connect4Error::InvalidWinningLength { requested: 0 });
        }
        // The longest possible line runs along the wider or taller board dimension.
        // The longest possible line sets the maximum winning target.
        let longest_line = parameters.width.max(parameters.height);
        if parameters.winning_length > longest_line {
            return Err(Connect4Error::ImpossibleWinningLength {
                requested: parameters.winning_length,
                longest_line,
            });
        }
        // Create an empty board with player 0 to act.
        // Grid construction also checks positive dimensions, overflow and the cell limit.
        Ok(Connect4State {
            board: Grid::filled(
                parameters.width,
                parameters.height,
                Cell::Empty,
                parameters.limits,
            )?,
            winning_length: parameters.winning_length,
            current_player: PlayerId::new(0),
            counters: TurnCounters::default(),
            status: SingleWinnerOutcome::Ongoing,
            winning_cells: Vec::new(),
        })
    }
    /// Create a view containing the complete public board and progress.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference; `observer` is a copied seat ID.
    ///
    /// # Returns
    /// `Ok(Connect4Observation)` owns a state clone and observer identity.
    /// Both players see the same game facts in their independently owned snapshots.
    ///
    /// # Errors
    /// An unknown observer returns `Connect4Error::InvalidPlayer`.
    fn observe(
        &self,
        state: &Connect4State,
        observer: PlayerId,
    ) -> Result<Connect4Observation, Connect4Error> {
        validate_player(observer, PLAYER_COUNT)?;
        // All Connect 4 facts are public, so copying the full state is appropriate here.
        Ok(Connect4Observation {
            state: state.clone(),
            observer,
        }) // Explicit owned snapshot.
    }

    /// List the observing player's choices using only their view.
    ///
    /// # Arguments
    /// `observation` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// An owned `Vec<DropPiece>` in left-to-right order, one per nonfull column.
    /// Nonacting observers and finished games receive an empty list.
    fn offered_actions(&self, observation: &Connect4Observation) -> Vec<DropPiece> {
        // Only the acting player's view offers moves.
        if observation.next_player() != Decision::Player(observation.observer) {
            return Vec::new();
        }
        nonfull_column_actions(&observation.state)
    }

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
    ///
    /// Columns are offered left to right, one per nonfull column.
    fn available_actions(
        &self,
        state: &Connect4State,
        actor: PlayerId,
    ) -> Result<Vec<DropPiece>, Connect4Error> {
        // Shared state/model borrows and copied actor; returns owned column actions.
        validate_player(actor, PLAYER_COUNT)?;
        if state.status != SingleWinnerOutcome::Ongoing {
            return Ok(Vec::new());
        }
        if actor != state.current_player {
            return Err(Connect4Error::WrongPlayer {
                expected: state.current_player,
                actual: actor,
            });
        }
        Ok(nonfull_column_actions(state))
    }

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
    /// Rejection order is unknown actor, finished game, wrong actor, invalid column, full column.
    /// A valid piece lands in the lowest empty cell; wins are checked before draws.
    fn next(
        &self,
        state: &mut Connect4State,
        actor: PlayerId,
        action: &DropPiece,
    ) -> Result<(), Connect4Error> {
        // Exclusive state borrow, copied actor and shared action.
        // Validate every reason to reject before changing cells or counters.
        validate_player(actor, PLAYER_COUNT)?;
        if state.status != SingleWinnerOutcome::Ongoing {
            return Err(Connect4Error::MoveAfterGameFinished);
        }
        if actor != state.current_player {
            return Err(Connect4Error::WrongPlayer {
                expected: state.current_player,
                actual: actor,
            });
        }
        if action.column >= state.width() {
            return Err(Connect4Error::InvalidColumn {
                column: action.column,
                width: state.width(),
            });
        }
        let mut landing = None;
        // Search from the bottom upward; the first empty cell is where the piece lands.
        for y in (0..state.height()).rev() {
            if state.cell(action.column, y)? == Cell::Empty {
                landing = Some(y);
                break;
            }
        }
        let y = landing.ok_or(Connect4Error::FullColumn {
            column: action.column,
        })?;
        *state.board.get_mut(action.column, y)? = Cell::Piece(actor);
        state.counters.record_move();
        // A win takes precedence over a full-board draw, including on the last move.
        if let Some(cells) = winning_line(state, actor) {
            state.winning_cells = cells;
            state.status = SingleWinnerOutcome::Won(actor);
        } else if state.board.cells().iter().all(|cell| {
            // A draw requires every board cell to contain a piece.
            *cell != Cell::Empty
        }) {
            state.status = SingleWinnerOutcome::Draw;
        } else {
            // Only continuing games finish the turn; two completed turns finish a round.
            state.counters.complete_turn();
            state.current_player = PlayerId::new(1 - actor.index());
            if state.current_player.index() == 0 {
                state.counters.complete_round();
            }
        }
        // Terminal moves retain the final actor and preterminal turn/round counters.
        Ok(())
    }

    /// Read how many seats the game state has.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `usize` player count.
    fn player_count(&self, _state: &Connect4State) -> usize {
        PLAYER_COUNT
    } // Shared borrows; copied count.
    /// Identify the next decision maker.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `Decision`: a player who may act, or completion.
    fn next_player(&self, state: &Connect4State) -> Decision {
        // Shared rules/state borrows; copied decision.
        state.next_player()
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
        &self,
        state: &Connect4State,
        player: PlayerId,
    ) -> Result<PlayerResult, Connect4Error> {
        // Shared rules/state and copied seat; copied result or owned error out.
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
    fn player_score(&self, state: &Connect4State, player: PlayerId) -> Result<f64, Connect4Error> {
        // Shared rules/state and copied seat; copied score or owned error out.
        state.player_score(player)
    }
}

// Select the first winning line in the scan order below, preserving its cell order.
/// Find the first winning line in the game's defined scan order.
///
/// # Arguments
/// `state` is passed by a shared (read-only) reference; `actor` is the copied player whose pieces are tested.
///
/// # Returns
/// `Some(Vec<(usize,usize)>)` owns the selected target-length winning coordinates; otherwise `None`.
pub(super) fn winning_line(state: &Connect4State, actor: PlayerId) -> Option<Vec<(usize, usize)>> {
    // Shared state and copied seat; returns only the owned coordinates of a winning run.
    let (width, height) = (state.width(), state.height());
    let check = |start, step| {
        // Search this ordered line for consecutive pieces belonging to the acting player.
        first_matching_run(
            ray(width, height, start, step),
            state.winning_length,
            |(x, y)| {
                // A cell extends the run only if it contains this player's piece.
                state.cell(x, y).expect("ray stays within the board") == Cell::Piece(actor)
            },
        )
    };
    // Columns left to right, each from bottom to top; then rows bottom to top.
    for x in 0..width {
        if let Some(line) = check((x, height - 1), (0, -1)) {
            return Some(line);
        }
    }
    for y in (0..height).rev() {
        if let Some(line) = check((0, y), (1, 0)) {
            return Some(line);
        }
    }
    // Rising-right diagonals from the left edge, starting at its bottom cell.
    for y in (0..height).rev() {
        if let Some(line) = check((0, y), (1, -1)) {
            return Some(line);
        }
    }
    // For each remaining bottom cell, check rising-right then rising-left.
    for x in 1..width {
        if let Some(line) = check((x, height - 1), (1, -1)) {
            return Some(line);
        }
        if let Some(line) = check((x, height - 1), (-1, -1)) {
            return Some(line);
        }
    }
    // Remaining rising-left diagonals start on the right edge above its bottom cell.
    for y in (0..height - 1).rev() {
        if let Some(line) = check((width - 1, y), (-1, -1)) {
            return Some(line);
        }
    }
    None
}

/// Enumerate nonfull columns from the borrowed state.
///
/// # Arguments
/// `state` is a shared borrow of an actual state or an owned public observation copy.
///
/// # Returns
/// An owned `Vec<DropPiece>` in increasing column order.
fn nonfull_column_actions(state: &Connect4State) -> Vec<DropPiece> {
    let mut actions = Vec::new();
    // Gravity guarantees that a column has space exactly when its top cell is empty.
    for column in 0..state.width() {
        if state
            .cell(column, 0)
            .expect("column comes from board dimensions")
            == Cell::Empty
        {
            actions.push(DropPiece { column });
        }
    }
    actions
}

impl PublicInformationModel for Connect4ForwardModel {
    /// Copy this player's complete public view into an independent simulation state.
    ///
    /// # Arguments
    /// `observation` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(Connect4State)` owns an exact snapshot of the view, including configuration
    /// and counters.
    ///
    /// # Errors
    /// No validation error for a valid view; ordinary clone allocation can still fail.
    fn simulation_state(
        &self,                             // Shared rule borrow.
        observation: &Connect4Observation, // Shared borrowed public view.
    ) -> Result<Connect4State, Connect4Error> {
        // All facts in this game are public. Each clone owns its board and progress.
        Ok(observation.state.clone())
    }
}
