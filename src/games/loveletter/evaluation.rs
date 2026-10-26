//! A small baseline: public match progress plus the evaluated player's own hand.
use super::*;
use crate::core::{PlayerId, PlayerResult, checkpoint::Checkpoint, evaluation::Evaluator};

/// Values are independent per player: terminal Win=+1, Loss=-1. During play,
/// 0.7 * tokens/target + 0.3 * highest own rank/9, with zero hand contribution
/// while eliminated from the round. This is a heuristic, not a win probability.
/// Planning callers evaluate sampled worlds, never the real hidden state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoveLetterHeuristic;
impl LoveLetterHeuristic {
    pub fn evaluate_observation(&self, o: &LoveLetterObservation) -> Result<f64, LoveLetterError> {
        match o.public.player_result(o.observer)? {
            PlayerResult::Win => return Ok(1.0),
            PlayerResult::Loss => return Ok(-1.0),
            _ => {}
        }
        let target = o.public.rules.winning_tokens[o.public.player_count() - 2];
        let hand = o.hand.iter().map(|c| c.rank()).max().unwrap_or(0);
        Ok(
            0.7 * f64::from(o.public.tokens[o.observer.index()]) / f64::from(target)
                + 0.3 * f64::from(hand) / 9.0,
        )
    }
}
impl Evaluator<LoveLetterState> for LoveLetterHeuristic {
    type Error = LoveLetterError;
    fn evaluate(&mut self, s: &LoveLetterState, p: PlayerId) -> Result<f64, Self::Error> {
        use crate::core::ForwardModel;
        self.evaluate_observation(&LoveLetterForwardModel.observe(s, p)?)
    }
}
impl Checkpoint for LoveLetterHeuristic {
    type Snapshot = Self;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self {
        *self
    }
    fn from_snapshot(s: &Self) -> Result<Self, Self::RestoreError> {
        Ok(*s)
    }
}
