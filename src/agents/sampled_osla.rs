//! Estimate each action by averaging independently sampled successor values.
use super::{Agent, OslaError};
use crate::core::random::{AgentRng, AgentRngSnapshot, SimulationRng, SimulationRngSnapshot};
use crate::core::simulation::SimulationModel;
use crate::core::{PlayerId, checkpoint::Checkpoint, evaluation::Evaluator};
use std::num::NonZeroU32;

/// OSLA for stochastic games. Every repetition obtains a new simulation fork.
/// The original deciding player remains the evaluation perspective throughout.
pub struct SampledOslaAgent<E> {
    evaluator: E,
    rng: AgentRng,
    simulation_rng: SimulationRng,
    repetitions_per_action: NonZeroU32,
    last_transitions: u64,
}

/// Complete persistent state between decisions; diagnostics are not persistent.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledOslaSnapshot<E> {
    pub evaluator: E,
    pub rng: AgentRngSnapshot,
    pub simulation_rng: SimulationRngSnapshot,
    pub repetitions_per_action: NonZeroU32,
}

impl<E> SampledOslaAgent<E> {
    /// Low-level construction takes an already resolved seed and positive work count.
    pub fn new(evaluator: E, seed: u64, repetitions_per_action: NonZeroU32) -> Self {
        Self {
            evaluator,
            rng: AgentRng::new(seed),
            simulation_rng: SimulationRng::new(seed),
            repetitions_per_action,
            last_transitions: 0,
        }
    }
    /// Number of successfully simulated transitions in the most recent decision.
    pub fn last_transitions(&self) -> u64 {
        self.last_transitions
    }
}

impl<E: Checkpoint> Checkpoint for SampledOslaAgent<E>
where
    E::RestoreError: std::fmt::Display,
{
    type Snapshot = SampledOslaSnapshot<E::Snapshot>;
    type RestoreError = String;
    fn snapshot(&self) -> Self::Snapshot {
        SampledOslaSnapshot {
            evaluator: self.evaluator.snapshot(),
            rng: self.rng.snapshot(),
            simulation_rng: self.simulation_rng.snapshot(),
            repetitions_per_action: self.repetitions_per_action,
        }
    }
    fn from_snapshot(s: &Self::Snapshot) -> Result<Self, String> {
        Ok(Self {
            evaluator: E::from_snapshot(&s.evaluator).map_err(|e| e.to_string())?,
            rng: AgentRng::from_snapshot(&s.rng).map_err(|e| e.to_string())?,
            simulation_rng: SimulationRng::from_snapshot(&s.simulation_rng)
                .map_err(|e| e.to_string())?,
            repetitions_per_action: s.repetitions_per_action,
            last_transitions: 0,
        })
    }
}

impl<F: SimulationModel, E: Evaluator<F::State>> Agent<F> for SampledOslaAgent<E> {
    type Error = OslaError<F::Error, E::Error>;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        self.last_transitions = 0;
        if actions.is_empty() {
            return Err(OslaError::EmptyActions);
        }
        let mut means = Vec::with_capacity(actions.len());
        let count = f64::from(self.repetitions_per_action.get());
        for (candidate, action) in actions.iter().enumerate() {
            let mut mean = 0.0;
            for _ in 0..self.repetitions_per_action.get() {
                let mut successor = model
                    .simulation_fork(observation, &mut self.simulation_rng)
                    .map_err(OslaError::Reconstruction)?;
                model
                    .next(&mut successor, actor, action)
                    .map_err(|error| OslaError::Transition { candidate, error })?;
                self.last_transitions += 1;
                let value = self
                    .evaluator
                    .evaluate(&successor, actor)
                    .map_err(|error| OslaError::Evaluation { candidate, error })?;
                if !value.is_finite() {
                    return Err(OslaError::NonfiniteValue { candidate, value });
                }
                // Divide before accumulation to avoid overflowing a finite sum.
                mean += value / count;
            }
            if !mean.is_finite() {
                return Err(OslaError::NonfiniteValue {
                    candidate,
                    value: mean,
                });
            }
            means.push(mean);
        }
        let best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let eligible: Vec<_> = means
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v == best).then_some(i))
            .collect();
        Ok(eligible[self
            .rng
            .index(eligible.len())
            .expect("finite maximum exists")])
    }
}
