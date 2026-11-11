//! A public score estimate for native OSLA and MCTS.
use super::*;
use crate::core::{checkpoint::Checkpoint, evaluation::Evaluator};

/// Current score plus visible direct card points. Terminal values use game outcomes.
/// This estimate is in points during play and outcome units after the game.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SushiGoHeuristic;

impl SushiGoHeuristic {
    pub fn evaluate_observation(&self, observation: &Observation) -> Result<f64, Error> {
        let seat = observation.observer.index();
        if seat >= observation.public.scores.len() {
            return Err(Error::InvalidPlayer(observation.observer));
        }
        if observation.public.finished {
            let best = *observation.public.scores.iter().max().expect("players");
            let pudding = observation
                .public
                .scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score == best)
                .map(|(p, _)| observation.public.pudding[p])
                .max()
                .expect("winner");
            return Ok(
                if observation.public.scores[seat] == best
                    && observation.public.pudding[seat] == pudding
                {
                    1.
                } else {
                    -1.
                },
            );
        }
        Ok(f64::from(
            observation.public.scores[seat] + score_cards(&observation.public.played[seat]),
        ))
    }
}

impl Evaluator<State> for SushiGoHeuristic {
    type Error = Error;
    fn evaluate(&mut self, state: &State, player: PlayerId) -> Result<f64, Error> {
        self.evaluate_observation(&SushiGoForwardModel.observe(state, player)?)
    }
}

impl Checkpoint for SushiGoHeuristic {
    type Snapshot = Self;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self {
        *self
    }
    fn from_snapshot(snapshot: &Self) -> Result<Self, Self::RestoreError> {
        Ok(*snapshot)
    }
}
