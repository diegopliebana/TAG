//! Validation helpers for the current alternating two-player placement games.
//!
//! They check shared constraints such as counts and outcome consistency. They are
//! not universal rules for all games; each game retains its own position checks.
use super::{BoardStateData, OutcomeData, PersistenceError, counter_data};
use crate::core::progress::TurnCounters;
/// Build copied counters using accepted-move/turn/round semantics of these placement games.
///
/// # Arguments
/// `moves` is the copied accepted-move count. `terminal` indicates a finished game.
/// Copied `rounds` enables a round boundary after two completed turns.
///
/// # Returns
/// Owned counters using the shared rules of the two current placement games.
pub(crate) fn placement_counters(moves: usize, terminal: bool, rounds: bool) -> TurnCounters {
    let mut counters = TurnCounters::default();
    for index in 0..moves {
        counters.record_move();
        if !(terminal && index + 1 == moves) {
            counters.complete_turn();
            if rounds && (index + 1).is_multiple_of(2) {
                counters.complete_round();
            }
        }
    }
    counters
}
/// Validate alternating occupancy counts and saved counters/actor before state construction.
///
/// # Arguments
/// `data` is a shared board/progress borrow. `cells` is the copied expected
/// board length. Copied `rounds` selects the two-turn-round counter convention.
///
/// # Returns
/// Owned verified counters matching alternating occupancy and saved progress.
///
/// # Errors
/// Invalid cell length/occupants, alternating counts, actor, counters or final winner.
pub(crate) fn validate_placement(
    data: &BoardStateData,
    cells: usize,
    rounds: bool,
) -> Result<TurnCounters, PersistenceError> {
    let invalid = |s: &str| PersistenceError::Invalid(s.into());
    if data.cells.len() != cells {
        return Err(invalid("saved board length differs from parameters"));
    }
    let mut count = [0usize; 2];
    for player in data.cells.iter().flatten() {
        if *player >= 2 {
            return Err(invalid("unknown board occupant"));
        }
        count[*player] += 1;
    }
    if !(count[0] == count[1] || count[0] == count[1] + 1) {
        return Err(invalid("board violates alternating player counts"));
    }
    let moves = count[0] + count[1];
    let terminal = data.outcome != OutcomeData::Ongoing;
    if terminal && moves == 0 {
        return Err(invalid("empty state cannot be terminal"));
    }
    let actor = if terminal { (moves - 1) % 2 } else { moves % 2 };
    if data.current_player != actor {
        return Err(invalid("saved acting player disagrees with move count"));
    }
    let counters = placement_counters(moves, terminal, rounds);
    if data.counters != counter_data(counters) {
        return Err(invalid("saved counters disagree with board/outcome"));
    }
    if let OutcomeData::Won(player) = data.outcome
        && player != actor
    {
        return Err(invalid("winner must be the final acting player"));
    }
    Ok(counters)
}

/// Estimate board scans and possible final-move predecessor checks.
///
/// # Arguments
/// `cells` is the copied board length; `data` is a shared decoded state borrow.
///
/// # Returns
/// Copied conservative work units, or None on arithmetic overflow.
pub(crate) fn placement_validation_work(cells: usize, data: &BoardStateData) -> Option<usize> {
    let candidates = if data.outcome == OutcomeData::Ongoing {
        1
    } else {
        data.cells
            .iter()
            .filter(|cell| **cell == Some(data.current_player))
            .count()
            .checked_add(1)?
    };
    cells.checked_mul(candidates)?.checked_mul(64)
}

/// Check a saved outcome against alternating placement-game board facts.
///
/// # Arguments
/// `outcome` is passed by a shared (read-only) reference; `winners` contains each seat's copied winning-line
/// presence and `full` indicates whether every cell is occupied.
///
/// # Returns
/// Unit when the saved ongoing/draw/win metadata agrees with the board.
///
/// # Errors
/// Contradictory outcomes or an unsupported winning seat.
pub(crate) fn validate_placement_outcome(
    outcome: &OutcomeData,
    winners: [bool; 2],
    full: bool,
) -> Result<(), PersistenceError> {
    let message = match outcome {
        OutcomeData::Ongoing if winners[0] || winners[1] || full => {
            Some("ongoing state already has a win or full board")
        }
        OutcomeData::Draw if winners[0] || winners[1] || !full => {
            Some("draw requires a full board without any winner")
        }
        OutcomeData::Won(player) if *player >= 2 || !winners[*player] || winners[1 - *player] => {
            Some("saved winner differs from board lines")
        }
        _ => None,
    };
    match message {
        Some(message) => Err(PersistenceError::Invalid(message.into())),
        None => Ok(()),
    }
}
