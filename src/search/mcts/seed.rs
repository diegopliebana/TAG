//! Compatibility with MCTS error reporting; entropy is shared across applications.
use super::{Error, Result};
pub(super) fn system_seed() -> Result<u64> {
    crate::core::random::automatic_seed().map_err(|error| Error::Seed(error.to_string()))
}
