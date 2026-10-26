//! Application-level match configuration and execution. Game rules and search
//! remain in the library; only this application resolves paths and writes files.
pub(crate) mod agents;
pub(crate) mod config;
pub(crate) mod execution;
pub(crate) mod files;
pub(crate) mod report;
pub use config::{Format, MatchConfig};
pub use execution::run;
pub use report::MatchReport;
pub use report::write_new as write_report;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests;

/// Derive a fresh seed for a game and role, independently of RNG consumption.
/// Version 1 uses SplitMix64's finaliser with explicit wrapping u64 arithmetic.
/// `role=0` is reserved for future game randomness; agent k uses role k+1 in
/// configuration order. The current deterministic games have no game RNG.
pub fn derive_seed(base: u64, game: u64, role: u64) -> u64 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    mix(base
        ^ mix(game.wrapping_add(0x9e3779b97f4a7c15))
        ^ mix(role.wrapping_add(0xd1b54a32d192ed03)))
}
