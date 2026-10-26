//! Convert saved controller entries to and from the supported agents.
//!
//! Human entries contain no AI. AI entries identify the policy and saved RNG progress.
//! A newly installed controller must start at its seed; a preserved controller must
//! keep the progress needed to make the same next choice.
use super::common::invalid;
use super::*;
use crate::agents::{FirstAvailableAgent, OslaAgent, OslaSnapshot, RandomAgent};
use crate::core::checkpoint::Checkpoint;
use crate::recording::ControllerSnapshot;
use crate::session::{BuiltinAgent, BuiltinAgentSnapshot, Controller};

/// Inspect borrowed controller entries in the file data to select the minimum supported agent scheme.
/// `branch` supplies root, linear changes and accepted acting-seat checkpoints.
/// Returns true if any recorded controller uses Random; no data is changed.
pub(super) fn branch_has_random<M>(branch: &BranchData<M>) -> bool {
    branch
        .controllers
        .iter()
        .chain(
            branch
                .controller_changes
                .iter()
                .flat_map(|c| &c.controllers),
        )
        .chain(branch.moves.iter().map(|m| &m.controller_after))
        .any(|seat| matches!(seat, SeatData::Random(_)))
}

/// Validate that a changed controller starts a new sequence at its entered seed.
///
/// # Arguments
/// `controller` borrows decoded replacement data. Preserved controllers must not
/// call this check: their nonzero RNG progress is necessary for exact continuation.
///
/// # Returns
/// Unit for a human or an AI at stream zero and word position zero.
///
/// # Errors
/// Invalid RNG metadata or a replacement containing already-consumed randomness.
pub(super) fn check_fresh_controller(controller: &SeatData) -> Result<(), PersistenceError> {
    let rng = match controller {
        SeatData::Human => return Ok(()),
        SeatData::Random(data) | SeatData::FirstAvailable(data) | SeatData::Osla(data) => {
            data.snapshot()?
        }
        SeatData::Extension { rng, .. } => rng.snapshot()?,
    };
    if rng.stream != 0 || rng.word_position != 0 {
        return Err(invalid(
            "replacement agents must start at their entered seed",
        ));
    }
    Ok(())
}

/// Convert an accepted boundary's supported controller progress to file data.
///
/// # Arguments
/// `s` is a shared tagged human/AI checkpoint borrow.
///
/// # Returns
/// Owned seat identity and exact RNG progress, with integers too large for JavaScript to represent exactly as Number encoded as strings.
pub(super) fn seat_data(s: &ControllerSnapshot<BuiltinAgentSnapshot<()>>) -> SeatData {
    match s {
        ControllerSnapshot::Human => SeatData::Human,
        ControllerSnapshot::Ai(BuiltinAgentSnapshot::Random(r)) => {
            SeatData::Random(RngData::from_snapshot(*r))
        }
        ControllerSnapshot::Ai(BuiltinAgentSnapshot::FirstAvailable(r)) => {
            SeatData::FirstAvailable(RngData::from_snapshot(*r))
        }
        ControllerSnapshot::Ai(BuiltinAgentSnapshot::Osla(s)) => {
            SeatData::Osla(RngData::from_snapshot(s.rng))
        }
    }
}
/// Restore one supported human or AI seat from saved progress.
///
/// # Arguments
/// `s` is a shared saved seat identity/progress borrow; `E` is the supported evaluator.
///
/// # Returns
/// Independently owned human/AI controller restored from that checkpoint.
///
/// # Errors
/// Invalid RNG data or evaluator/agent restoration failure.
pub(super) fn seat<E: Checkpoint<Snapshot = ()>>(
    s: &SeatData,
) -> Result<Controller<BuiltinAgent<E>>, PersistenceError>
where
    E::RestoreError: std::fmt::Debug,
{
    Ok(match s {
        SeatData::Extension { .. } => {
            return Err(invalid(
                "adapter controller requires its matching replay codec",
            ));
        }
        SeatData::Human => Controller::Human,
        SeatData::Random(r) => Controller::Ai(BuiltinAgent::Random(
            RandomAgent::from_snapshot(&r.snapshot()?).map_err(|e| invalid(e.to_string()))?,
        )),
        SeatData::FirstAvailable(r) => Controller::Ai(BuiltinAgent::FirstAvailable(
            FirstAvailableAgent::from_snapshot(&r.snapshot()?)
                .map_err(|e| invalid(e.to_string()))?,
        )),
        SeatData::Osla(r) => Controller::Ai(BuiltinAgent::Osla(
            OslaAgent::from_snapshot(&OslaSnapshot {
                evaluator: (),
                rng: r.snapshot()?,
            })
            .map_err(|e| invalid(format!("agent restoration: {e:?}")))?,
        )),
    })
}

/// Convert one adapter's controller checkpoints to portable replay entries.
/// The codec owns any immutable model registry needed to restore its agents.
pub trait ReplayCodec<A: Checkpoint> {
    /// Check additional fresh-controller streams at an explicit replacement.
    /// The common importer separately checks the ordinary action RNG.
    fn validate_fresh(&self, _data: &SeatData) -> Result<(), PersistenceError> {
        Ok(())
    }
    /// Borrow a complete seat checkpoint and return owned portable data.
    fn encode(&self, snapshot: &ControllerSnapshot<A::Snapshot>) -> SeatData;
    /// Validate borrowed file data and restore an independent controller.
    fn decode(&self, data: &SeatData) -> Result<Controller<A>, PersistenceError>;
    /// Validate a recorded AI continuation that must not rerun its computation.
    /// Return true only after checking matching policy identity and compatible RNG
    /// progress. The importer still checks legality, next player and recorded state.
    /// The default independently repeats agent selection as in earlier formats.
    fn recorded_choice(
        &self,
        _before: &SeatData,
        _after: &SeatData,
    ) -> Result<bool, PersistenceError> {
        Ok(false)
    }
    /// Whether replay verification must charge for candidate simulations.
    fn lookahead(&self, data: &SeatData) -> bool {
        matches!(data, SeatData::Osla(_))
    }
    /// Additional bounded inference work for a decision, beyond game simulation.
    fn extra_work(&self, _data: &SeatData) -> usize {
        0
    }
}

/// Codec for the existing built-in stateless evaluator format.
pub struct BuiltinCodec;
impl<E: Checkpoint<Snapshot = ()>> ReplayCodec<BuiltinAgent<E>> for BuiltinCodec
where
    E::RestoreError: std::fmt::Debug,
{
    fn encode(&self, snapshot: &ControllerSnapshot<BuiltinAgentSnapshot<()>>) -> SeatData {
        seat_data(snapshot)
    }
    fn decode(&self, data: &SeatData) -> Result<Controller<BuiltinAgent<E>>, PersistenceError> {
        seat(data)
    }
}

pub(super) fn branch_has_extension<M>(branch: &BranchData<M>) -> bool {
    branch
        .controllers
        .iter()
        .chain(
            branch
                .controller_changes
                .iter()
                .flat_map(|c| &c.controllers),
        )
        .chain(branch.moves.iter().map(|m| &m.controller_after))
        .any(|s| matches!(s, SeatData::Extension { .. }))
}
