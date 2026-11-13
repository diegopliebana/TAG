//! A bounded team estimate for sampled positions. Scores remain rule results.
use super::*;
use crate::core::checkpoint::Checkpoint;
use crate::core::evaluation::Evaluator;

#[derive(Clone, Copy, Debug, Default)]
pub struct TheGameHeuristic;

impl Checkpoint for TheGameHeuristic {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self::Snapshot {}
    fn from_snapshot(_: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}

impl Evaluator<State> for TheGameHeuristic {
    type Error = Error;

    fn evaluate(&mut self, state: &State, player: PlayerId) -> Result<f64, Error> {
        if player.index() >= state.hands.len() {
            return Err(Error::InvalidPlayer(player));
        }
        Ok(match state.public.outcome {
            TeamOutcome::Won => 1.,
            TeamOutcome::Lost => -1.,
            TeamOutcome::Ongoing => {
                // This describes current progress, not a win probability.
                0.9 * state.public.placements.len() as f64 / CARD_COUNT as f64
            }
        })
    }
}
