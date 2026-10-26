//! Rust-owned game execution. Python is involved only when a stop needs its attention.
use crate::engine::{Engine, Error, Game};
use tag_rust::{
    agents::{BatchEvaluator, EvaluatorIdentity, OslaPythonError},
    core::{Decision, PublicInformationModel},
    learning::LearningGame,
    session::{EvaluationKey, EvaluationRequest, EvaluationStatus},
};

pub enum Stop {
    Finished,
    Human(usize),
    MoveLimit(usize),
    Evaluation(Box<EvaluationRequest>),
}

/// Accepted moves are reported even if a later decision fails. Callers must
/// update their position token before exposing that error to Python.
pub struct Progress {
    pub moves: u64,
    pub stop: Result<Stop, Error>,
}

impl<F> Game<F>
where
    F: LearningGame + PublicInformationModel + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Parameters: Clone,
    F::Error: std::fmt::Display,
{
    fn advance(&mut self, limit: u64) -> Progress {
        let mut moves = 0;
        let stop = self.run_until_stop(limit, &mut moves);
        Progress { moves, stop }
    }

    fn run_until_stop(&mut self, limit: u64, moves: &mut u64) -> Result<Stop, Error> {
        if self.session.has_pending() {
            return Err(OslaPythonError::Busy.into());
        }
        loop {
            let actor = match self.session.next_player() {
                Decision::Finished => return Ok(Stop::Finished),
                Decision::Player(actor) => actor,
            };
            if self.session.checkpoint(actor)?.is_none() {
                return Ok(Stop::Human(actor.index()));
            }
            if *moves == limit {
                return Ok(Stop::MoveLimit(actor.index()));
            }
            let request = self.session.prepare()?;
            if self.native_evaluators[actor.index()].is_none() {
                return Ok(Stop::Evaluation(Box::new(request)));
            }
            self.complete_native(request)?;
            *moves += 1;
        }
    }

    /// Complete all chunks of one native decision through the existing checked
    /// prepare/finish/commit path. Failed native work is cancelled without a move.
    fn complete_native(&mut self, mut request: EvaluationRequest) -> Result<(), Error> {
        loop {
            let result = self.native_evaluators[request.batch.observer.index()]
                .as_mut()
                .expect("native backend checked above")
                .evaluate(&request.batch)
                .and_then(|values| {
                    self.session.finish(
                        &request.key,
                        &request.evaluator,
                        &request.batch.action_ids,
                        &values,
                    )
                });
            match result {
                Ok(EvaluationStatus::Request(next)) => request = *next,
                Ok(EvaluationStatus::Ready(proposal)) => {
                    if let Err(error) = self.session.commit(&proposal.key) {
                        self.session.cancel(&proposal.key)?;
                        return Err(error.into());
                    }
                    return Ok(());
                }
                Err(error) => {
                    self.session.cancel(&request.key)?;
                    return Err(error.into());
                }
            }
        }
    }

    fn resume(
        &mut self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        ids: &[usize],
        values: &[f64],
        limit: u64,
    ) -> Progress {
        let mut moves = 0;
        let stop = (|| {
            if limit == 0 {
                return Err(Error::Invalid("max_moves must be positive".into()));
            }
            match self.session.finish(key, evaluator, ids, values)? {
                EvaluationStatus::Request(request) => Ok(Stop::Evaluation(request)),
                EvaluationStatus::Ready(proposal) => {
                    self.session.commit(&proposal.key)?;
                    moves = 1;
                    self.run_until_stop(limit, &mut moves)
                }
            }
        })();
        Progress { moves, stop }
    }
}

impl Engine {
    pub fn advance(&mut self, max_moves: u64) -> Progress {
        if max_moves == 0 {
            return Progress {
                moves: 0,
                stop: Err(Error::Invalid("max_moves must be positive".into())),
            };
        }
        match self {
            Self::TicTacToe(game) => game.advance(max_moves),
            Self::Connect4(game) => game.advance(max_moves),
            Self::CantStop(game) => Progress {
                moves: 0,
                stop: Ok(match game.session.next_player() {
                    Decision::Finished => Stop::Finished,
                    Decision::Player(actor) => Stop::Human(actor.index()),
                }),
            },
            Self::LoveLetter(game) => Progress {
                moves: 0,
                stop: Ok(match game.session.next_player() {
                    Decision::Finished => Stop::Finished,
                    Decision::Player(actor) => Stop::Human(actor.index()),
                }),
            },
        }
    }
    pub fn resume(
        &mut self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        ids: &[usize],
        values: &[f64],
        max_moves: u64,
    ) -> Progress {
        match self {
            Self::TicTacToe(game) => game.resume(key, evaluator, ids, values, max_moves),
            Self::Connect4(game) => game.resume(key, evaluator, ids, values, max_moves),
            Self::CantStop(_) => Progress {
                moves: 0,
                stop: Err(crate::engine::unsupported("external evaluation")),
            },
            Self::LoveLetter(_) => Progress {
                moves: 0,
                stop: Err(crate::engine::unsupported_game(
                    "Love Letter",
                    "external evaluation",
                )),
            },
        }
    }
}
