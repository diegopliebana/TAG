//! Interactive sessions exercised across games and generic controller/state types.
mod execution;
mod history;
mod validation;

mod branches;
#[cfg(feature = "replay")]
mod persistence;
