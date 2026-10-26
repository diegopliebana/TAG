//! Independent seeded random streams for agents, real games and simulations.
//!
//! Each agent owns its generator. A seed starts a sequence; a checkpoint also stores
//! the stream and current position so later draws can be reproduced. This is separate
//! from game dice/shuffle randomness and an agent's hidden-state guesses.
mod seed;
use super::checkpoint::Checkpoint;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
/// Obtain an automatic seed from OS randomness. Applications must report/store
/// the resolved integer; failures return an error without a clock fallback.
/// Browsers supply Web Crypto seeds through their adapter instead.
pub use seed::automatic_seed;

/// Saved seed, stream and draw position for the specifically versioned ChaCha8 RNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: integer snapshot; PartialEq/Eq: exact reflexive field equality.
pub struct AgentRngSnapshot {
    pub version: u32,
    pub initial_seed: u64,
    pub stream: u64,
    pub word_position: u128,
}

/// Errors for unsupported saved RNG versions or a draw position beyond its 68-bit range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: small error values; PartialEq/Eq: exact variant/field equality.
pub enum RngSnapshotError {
    UnsupportedVersion(u32),
    InvalidWordPosition(u128),
}
impl std::fmt::Display for RngSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported ChaCha8 snapshot version {version}")
            }
            Self::InvalidWordPosition(position) => write!(
                f,
                "ChaCha8 word position {position} exceeds its 68-bit range"
            ),
        }
    }
}
impl std::error::Error for RngSnapshotError {}

/// Independent ChaCha8 stream with an explicit initial seed.
#[derive(Debug)] // Debug: inspection.

pub struct AgentRng {
    initial_seed: u64,
    generator: ChaCha8Rng,
}

impl Checkpoint for AgentRng {
    type Snapshot = AgentRngSnapshot;
    type RestoreError = RngSnapshotError;

    /// Capture the seed, stream and precise buffered-generator position.
    fn snapshot(&self) -> AgentRngSnapshot {
        // Shared generator borrow; copied owned snapshot.
        AgentRngSnapshot {
            version: 1,
            initial_seed: self.initial_seed,
            stream: self.generator.get_stream(),
            word_position: self.generator.get_word_pos(),
        }
    }

    /// Restore an independent pinned ChaCha8 generator from shared snapshot data.
    /// Version must be one and the word offset must fit 68 bits.
    fn from_snapshot(snapshot: &AgentRngSnapshot) -> Result<Self, RngSnapshotError> {
        if snapshot.version != 1 {
            return Err(RngSnapshotError::UnsupportedVersion(snapshot.version));
        }
        if snapshot.word_position >= (1u128 << 68) {
            return Err(RngSnapshotError::InvalidWordPosition(
                snapshot.word_position,
            ));
        }
        let mut rng = Self::new(snapshot.initial_seed);
        rng.generator.set_stream(snapshot.stream);
        rng.generator.set_word_pos(snapshot.word_position);
        Ok(rng)
    }
}
impl AgentRng {
    /// Create an independent RNG from the supplied seed.
    ///
    /// # Arguments
    /// `seed` is copied; equal seeds intentionally produce equal initial streams.
    ///
    /// # Returns
    /// An owned generator. Reproducibility also requires the same dependency versions
    /// and sequence of sampling requests.
    pub fn new(seed: u64) -> Self {
        Self {
            initial_seed: seed,
            generator: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Return the copied original `u64` seed.
    pub fn initial_seed(&self) -> u64 {
        self.initial_seed
    }

    /// Draw a floating-point value in [0,1) for probability sampling.
    ///
    /// # Arguments
    /// `self` is borrowed exclusively; one generator word is consumed.
    ///
    /// # Returns
    /// A copied f64 formed from 53 random bits. This cannot return one.
    pub fn unit_interval(&mut self) -> f64 {
        (self.generator.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Select a uniformly distributed index using rejection sampling.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference; `count` is the copied number of choices.
    ///
    /// # Returns
    /// `None` for zero choices, `Some(0)` for one, otherwise an index in 0..count.
    /// Zero/one choices retain the RNG position. Larger bounds can consume more
    /// than one generator output when rejection is necessary.
    pub fn index(&mut self, count: usize) -> Option<usize> {
        if count == 0 {
            return None;
        }
        if count == 1 {
            return Some(0);
        }
        let bound = count as u64; // Copied fixed-width bound on supported 32/64-bit targets.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.generator.next_u64();
            // Discard the short remainder interval so each index has equal representation.
            if value >= threshold {
                return Some((value % bound) as usize);
            }
        }
    }
}

// Version-one stream names separate game dice, simulated dice and simulation
// seed generation even when a caller supplies the same initial integer seed.
const GAME_CHANCE: u64 = 0x5441475f47414d45;
const SIMULATED_CHANCE: u64 = 0x5441475f53494d43;
const SIMULATION_SEEDS: u64 = 0x5441475f53494d53;

/// Exact environment randomness, separate from an agent's action selection.
/// Equality compares the seed, stream and precise generator position.
#[derive(Debug)]
pub struct ChanceRng(AgentRng);

/// Saved environment randomness. This is privileged snapshot data, never part
/// of a player's observation. Stream constants and algorithm are version one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChanceRngSnapshot {
    /// Existing versioned ChaCha8 state format; the stream identifies its use.
    pub generator: AgentRngSnapshot,
}

impl Clone for ChanceRng {
    fn clone(&self) -> Self {
        Self::from_snapshot(&self.snapshot())
            .expect("an existing chance RNG has a valid checkpoint")
    }
}
impl PartialEq for ChanceRng {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}
impl Eq for ChanceRng {}

/// Snapshot validation failures for an unsupported stream or generator version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChanceRngError {
    Generator(RngSnapshotError),
    InvalidStream(u64),
}
impl std::fmt::Display for ChanceRngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generator(e) => e.fmt(f),
            Self::InvalidStream(stream) => write!(f, "unsupported chance RNG stream {stream}"),
        }
    }
}
impl std::error::Error for ChanceRngError {}

fn named_stream(seed: u64, stream: u64) -> AgentRng {
    let mut rng = AgentRng::new(seed);
    rng.generator.set_stream(stream);
    rng
}
impl ChanceRng {
    /// Start a real game's chance stream. No operating-system or clock access
    /// occurs here: applications are responsible for resolving automatic seeds.
    pub fn new(seed: u64) -> Self {
        Self(named_stream(seed, GAME_CHANCE))
    }

    /// Draw a uniform outcome in 0..count; zero/one choices consume no randomness.
    pub fn index(&mut self, count: usize) -> Option<usize> {
        self.0.index(count)
    }
}
impl Checkpoint for ChanceRng {
    type Snapshot = ChanceRngSnapshot;
    type RestoreError = ChanceRngError;
    fn snapshot(&self) -> Self::Snapshot {
        ChanceRngSnapshot {
            generator: self.0.snapshot(),
        }
    }
    fn from_snapshot(saved: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        if ![GAME_CHANCE, SIMULATED_CHANCE].contains(&saved.generator.stream) {
            return Err(ChanceRngError::InvalidStream(saved.generator.stream));
        }
        AgentRng::from_snapshot(&saved.generator)
            .map(Self)
            .map_err(ChanceRngError::Generator)
    }
}

/// Search-owned source of new environment RNGs, one per simulated trajectory.
/// Checkpoint this alongside an agent when a reproducible search is resumed.
#[derive(Debug)]
pub struct SimulationRng(AgentRng);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulationRngSnapshot {
    /// Exact seed-source progress, separate from all generated chance streams.
    pub generator: AgentRngSnapshot,
}

impl SimulationRng {
    pub fn new(seed: u64) -> Self {
        Self(named_stream(seed, SIMULATION_SEEDS))
    }
    /// Supply a new chance stream for one trajectory, consuming only this source.
    pub fn fork_chance(&mut self) -> ChanceRng {
        let seed = self.0.generator.next_u64();
        ChanceRng(named_stream(seed, SIMULATED_CHANCE))
    }
}
impl Checkpoint for SimulationRng {
    type Snapshot = SimulationRngSnapshot;
    type RestoreError = ChanceRngError;
    fn snapshot(&self) -> Self::Snapshot {
        SimulationRngSnapshot {
            generator: self.0.snapshot(),
        }
    }
    fn from_snapshot(saved: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        if saved.generator.stream != SIMULATION_SEEDS {
            return Err(ChanceRngError::InvalidStream(saved.generator.stream));
        }
        AgentRng::from_snapshot(&saved.generator)
            .map(Self)
            .map_err(ChanceRngError::Generator)
    }
}
