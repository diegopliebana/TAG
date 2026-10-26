//! Construct simulations without exposing a real game's future randomness.
use super::random::SimulationRng;
use super::{ForwardModel, PublicInformationModel};

/// Construct simulations from a player's permitted information, including memory.
/// Hidden facts may be sampled subject to that information; future random events
/// use an independent simulation stream.
/// Unlike an exact state snapshot, a fork deliberately changes future chance.
/// The caller owns/checkpoints `random`; neither observations nor forks receive
/// the live game's RNG. Failed construction must leave `random` unchanged.
pub trait SimulationModel: ForwardModel {
    /// Return an independent state consistent with the supplied observation.
    /// Stochastic implementations consume only the caller's simulation stream;
    /// deterministic implementations need not consume any random draws.
    /// Invalid observations return the game's error before advancing the stream.
    fn simulation_fork(
        &self,
        observation: &Self::Observation,
        random: &mut SimulationRng,
    ) -> Result<Self::State, Self::Error>;
}

// Deterministic public games need no new random draws. Retaining this path keeps
// their existing seeded search trajectories unchanged when callers adopt forks.
impl<F: PublicInformationModel> SimulationModel for F {
    fn simulation_fork(
        &self,
        observation: &Self::Observation,
        _: &mut SimulationRng,
    ) -> Result<Self::State, Self::Error> {
        self.simulation_state(observation)
    }
}
