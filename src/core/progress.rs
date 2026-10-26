//! Counts of accepted moves, completed turns and completed rounds.
//!
//! Games decide when to increment each count. A winning move can count as a move
//! without completing a normal turn. A round boundary resets the within-round count
//! but retains the total number of completed turns.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)] // Clone/Copy: four small counts.
// PartialEq/Eq: exact counter comparison with reflexive integer equality.
pub struct TurnCounters {
    moves: usize,
    completed_turns: usize,
    turns_in_round: usize,
    rounds: usize,
}
impl TurnCounters {
    /// Restore externally decoded counts after basic ordering checks. Games must
    /// additionally validate their own turn/round and terminal-action semantics.
    #[cfg(feature = "replay")]
    pub(crate) fn from_counts(counts: [usize; 4]) -> Option<Self> {
        let [moves, completed_turns, turns_in_round, rounds] = counts;
        (completed_turns <= moves && turns_in_round <= completed_turns && rounds <= completed_turns)
            .then_some(Self {
                moves,
                completed_turns,
                turns_in_round,
                rounds,
            })
    }
    /// Return the count of accepted moves, including the terminal move as `usize`, without changing state.
    pub fn move_count(&self) -> usize {
        self.moves
    } // Shared borrow; copied count.
    /// Return the total of completed turns, retained across rounds as `usize`, without changing state.
    pub fn completed_turn_count(&self) -> usize {
        self.completed_turns
    } // Shared borrow; copied total.
    /// Return the count of completed turns since the last round boundary as `usize`, without changing state.
    pub fn turn_count_in_round(&self) -> usize {
        self.turns_in_round
    } // Shared borrow; copied count.
    /// Return the count of completed rounds as `usize`, without changing state.
    pub fn round_count(&self) -> usize {
        self.rounds
    } // Shared borrow; copied count.
    /// Record one accepted move independently of turn boundaries.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; rules decide when to call this.
    ///
    /// # Returns
    /// Returns `()` after increasing moves; turn and round counters are unchanged.
    pub fn record_move(&mut self) {
        self.moves += 1;
    } // Exclusive borrow.
    /// Record one completed turn.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; rules decide when a turn finishes.
    ///
    /// # Returns
    /// Returns `()` after increasing total and within-round turns; moves and rounds are unchanged.
    pub fn complete_turn(&mut self) {
        // Exclusive borrow; caller decides the boundary.
        self.completed_turns += 1;
        self.turns_in_round += 1;
    }
    /// Record a round boundary.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; rules decide when a round finishes.
    ///
    /// # Returns
    /// Returns `()` after increasing rounds and resetting within-round turns to zero; total turns are retained.
    pub fn complete_round(&mut self) {
        // Exclusive borrow.
        self.rounds += 1;
        self.turns_in_round = 0;
    }
}
