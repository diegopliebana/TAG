//! Validated state transport; rules remain authoritative for outcomes and continuation.
use super::*;
use crate::core::outcomes::SingleWinnerOutcome;
use crate::core::{ForwardModel, PlayerId};
use crate::replay::{
    BoardStateData, OutcomeData, PersistableGame, PersistenceError, ReplayGame, counter_data,
    placement_counters, placement_validation_work, validate_placement, validate_placement_outcome,
};
impl PersistableGame for TicTacToeForwardModel {
    type SavedState = BoardStateData;
    const GAME_ID: &'static str = "tictactoe";
    const RULES_VERSION: u32 = 1;
    /// Calculate the cell count using checked dimension arithmetic.
    ///
    /// # Arguments
    /// `p` is a shared board-configuration borrow.
    ///
    /// # Returns
    /// Copied cell count, or `None` when dimension multiplication overflows.
    fn storage_units(p: &TicTacToeParameters) -> Option<usize> {
        p.grid_size.checked_mul(p.grid_size)
    }
    /// Estimate board validation and terminal predecessor checking work.
    ///
    /// # Arguments
    /// `parameters` and decoded board `data` are shared borrows.
    ///
    /// # Returns
    /// Copied estimate including terminal predecessor candidates, or `None` on overflow.
    fn validation_work(parameters: &Self::Parameters, data: &BoardStateData) -> Option<usize> {
        placement_validation_work(Self::storage_units(parameters)?, data)
    }
    /// Borrow state and copy complete board/progress facts into an owned file value.
    /// Allocates independent cell/winning-coordinate vectors for the caller.
    ///
    /// # Arguments
    /// `self` and complete simulation `state` are shared borrows.
    ///
    /// # Returns
    /// Owned board, actor, counters, outcome and winning-cell data.
    fn state_data(&self, state: &TicTacToeState) -> BoardStateData {
        BoardStateData {
            cells: state
                .board()
                .iter()
                .map(|cell| match cell {
                    Cell::Empty => None,
                    Cell::X => Some(0),
                    Cell::O => Some(1),
                })
                .collect(),
            current_player: state.current_player().index(),
            counters: counter_data(state.counters()),
            outcome: state.status().into(),
            winning_cells: Vec::new(),
        }
    }
    /// Restore private state after checking saved facts against these game rules.
    ///
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference. `parameters` supplies the selected board rules;
    /// `data` supplies decoded cells and progress. Both inputs are shared borrows.
    ///
    /// # Returns
    /// Independently owned state with its validated simulation metadata restored.
    ///
    /// # Errors
    /// Invalid parameters, occupancy/actor/counters, outcome/winning metadata or
    /// a terminal position lacking a legal final-move predecessor.
    fn restore_state(
        &self,
        parameters: &TicTacToeParameters,
        data: &BoardStateData,
    ) -> Result<TicTacToeState, PersistenceError> {
        let invalid = |s: &str| PersistenceError::Invalid(s.into());
        let mut state = self
            .init_state(parameters, 2)
            .map_err(|e| PersistenceError::Invalid(e.to_string()))?;
        let counters = validate_placement(data, state.board().len(), false)?;
        for (i, cell) in data.cells.iter().enumerate() {
            *state
                .board
                .get_mut(i / parameters.grid_size, i % parameters.grid_size)
                .map_err(|e| PersistenceError::Invalid(e.to_string()))? = match cell {
                None => Cell::Empty,
                Some(0) => Cell::X,
                Some(_) => Cell::O,
            };
        }
        state.current_player = PlayerId::new(data.current_player);
        state.counters = counters;
        state.status = match data.outcome {
            OutcomeData::Ongoing => SingleWinnerOutcome::Ongoing,
            OutcomeData::Draw => SingleWinnerOutcome::Draw,
            OutcomeData::Won(p) => SingleWinnerOutcome::Won(PlayerId::new(p)),
        };
        let winner0 = super::forward_model::has_full_line(&state.board, Cell::X);
        let winner1 = super::forward_model::has_full_line(&state.board, Cell::O);
        let full = data.cells.iter().all(Option::is_some);
        validate_placement_outcome(&data.outcome, [winner0, winner1], full)?;
        if !data.winning_cells.is_empty() {
            return Err(invalid(
                "Tic-Tac-Toe state has unexpected winning-cell metadata",
            ));
        }
        // A terminal board must have a legal final move from an ongoing predecessor.
        // Removing candidate final pieces also rejects play that continued after an earlier win.
        if data.outcome != OutcomeData::Ongoing {
            let previous_counters = placement_counters(state.move_count() - 1, false, false);
            let mut witnessed = false;
            for index in 0..data.cells.len() {
                if data.cells[index] != Some(data.current_player) {
                    continue;
                }

                let mut previous = state.clone();
                *previous
                    .board
                    .get_mut(index / parameters.grid_size, index % parameters.grid_size)
                    .unwrap() = Cell::Empty;
                previous.status = SingleWinnerOutcome::Ongoing;
                previous.current_player = PlayerId::new(data.current_player);
                previous.counters = previous_counters;

                if super::forward_model::has_full_line(&previous.board, Cell::X)
                    || super::forward_model::has_full_line(&previous.board, Cell::O)
                {
                    continue;
                }
                let action = PlaceMark {
                    x: index / parameters.grid_size,
                    y: index % parameters.grid_size,
                };
                if self
                    .next(&mut previous, PlayerId::new(data.current_player), &action)
                    .is_ok()
                    && previous == state
                {
                    witnessed = true;
                    break;
                }
            }
            if !witnessed {
                return Err(invalid(
                    "terminal state has no legal final-move predecessor",
                ));
            }
        }
        Ok(state)
    }
}

impl ReplayGame for TicTacToeForwardModel {
    type SavedEvaluator = TicTacToeEvaluator;
    /// Estimate transition or OSLA work before replay validation.
    ///
    /// # Arguments
    /// `parameters` is passed by a shared (read-only) reference; `osla` is copied and chooses AI-decision
    /// versus single-transition cost estimation.
    ///
    /// # Returns
    /// Copied conservative work estimate, or `None` on arithmetic overflow.
    fn verification_work(parameters: &Self::Parameters, osla: bool) -> Option<usize> {
        let cells = Self::storage_units(parameters)?;
        if osla {
            cells
                .checked_mul(cells)?
                .checked_mul(parameters.grid_size)?
                .checked_mul(16)
        } else {
            cells.checked_mul(16)
        }
    }
}
