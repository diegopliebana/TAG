//! Shared numeric encoding for the current piece-placement games.
//!
//! Game registrations supply cells from a player's observation. This code produces
//! a relative-value vector or three-channel grid, action masks, results and progress.
//! It also describes groups of cells the future Python heuristic will inspect.
use super::*;
use crate::core::{Decision, PlayerId, PlayerResult, progress::TurnCounters};

/// Check placement dimensions and cell limits before allocating output arrays.
/// `identity` is read through a reference; `limits` is a small copied value.
/// Returns width*height, or an error for zero dimensions/target, other than two
/// players, arithmetic overflow or a cell count above the configured maximum.
pub(crate) fn check_placement_size(
    identity: &EncodingIdentity,
    limits: LearningLimits,
) -> Result<usize, LearningError> {
    if identity.width() == 0
        || identity.height() == 0
        || identity.target() == 0
        || identity.player_count != 2
    {
        return Err(LearningError::Invalid(
            "placement dimensions/target must be positive and seats must be two".into(),
        ));
    }
    let count = identity
        .width()
        .checked_mul(identity.height())
        .ok_or(LearningError::Overflow)?;
    if count > limits.max_cells {
        return Err(LearningError::ResourceLimit);
    }
    Ok(count)
}
fn budget(bytes: usize, limits: LearningLimits) -> Result<(), LearningError> {
    if bytes > limits.max_bytes {
        Err(LearningError::ResourceLimit)
    } else {
        Ok(())
    }
}
fn add_bytes(
    total: &mut usize,
    count: usize,
    width: usize,
    limits: LearningLimits,
) -> Result<(), LearningError> {
    *total = total
        .checked_add(count.checked_mul(width).ok_or(LearningError::Overflow)?)
        .ok_or(LearningError::Overflow)?;
    budget(*total, limits)
}
/// Build the description shared by Tic-Tac-Toe and Connect 4 learning inputs.
/// `identity` supplies game/dimensions/target; `mapping` chooses cell or column IDs;
/// `limits` restricts array/index size. These arguments are passed by value.
/// Returns descriptions of cells, board, progress and results, plus ordered winning
/// line groups. Exceeding a limit or integer range returns LearningError.
pub(crate) fn placement_spec(
    identity: EncodingIdentity,
    mapping: ActionMapping,
    limits: LearningLimits,
) -> Result<LearningSpec, LearningError> {
    let count = check_placement_size(&identity, limits)?;
    let action_count = match mapping {
        ActionMapping::CellXMajor => count,
        ActionMapping::Column => identity.width(),
        _ => {
            return Err(LearningError::Invalid(
                "placement encoder requires cell or column actions".into(),
            ));
        }
    };
    let mut indices = Vec::new();
    let mut bytes = 0;
    for window in
        crate::core::lines::windows(identity.width(), identity.height(), identity.target())
    {
        for (x, y) in window {
            add_bytes(&mut bytes, 1, 8, limits)?;
            indices.push(
                i64::try_from(x * identity.height() + y).map_err(|_| LearningError::Overflow)?,
            );
        }
    }
    Ok(LearningSpec {
        payloads: vec![
            PayloadSpec::Features(TensorSpec {
                name: "cells",
                dtype: DType::Float32,
                shape: vec![count],
                axes: vec!["cell_x_major"],
                components: vec!["own=1", "opponent=-1", "empty=0"],
                bounds: Some((-1, 1)),
            }),
            PayloadSpec::Grid(TensorSpec {
                name: "board",
                dtype: DType::Float32,
                shape: vec![3, identity.height(), identity.width()],
                axes: vec!["channel", "y", "x"],
                components: vec!["own", "opponent", "empty"],
                bounds: Some((0, 1)),
            }),
        ],
        metadata: metadata_spec(identity.player_count),
        actions: ActionSpec {
            size: action_count,
            mapping,
        },
        heuristic: Some(OpenLinesSpec {
            version: 1,
            window_length: identity.target(),
            indices,
        }),
        identity,
    })
}
/// Information read from a placement-game observation, ready for the shared encoder.
/// The generic P is a read-only cell-access function. It can borrow the original
/// observation without allocating a second board. The encoder never retains it.
pub(crate) struct PlacementView<P> {
    pub identity: EncodingIdentity,
    pub mapping: ActionMapping,
    pub observer: PlayerId,
    pub decision: Decision,
    pub counters: TurnCounters,
    pub results: Vec<PlayerResult>,
    pub cell_count: usize,
    pub occupant: P,
    pub offered_ids: Vec<usize>,
}
/// Create numeric arrays and action masks from a placement-game observation.
/// `view` supplies player-visible facts and a read-only cell-access function;
/// `payloads` lists array names; `limits` restricts how much numeric data is returned.
/// The result has its own arrays. Errors report unknown/repeated names, inconsistent
/// player/action data, numbers outside the output range or exceeded size limits.
pub(crate) fn encode_placement<P: Fn(usize) -> Option<PlayerId>>(
    view: PlacementView<P>,
    payloads: &[&str],
    limits: LearningLimits,
) -> Result<EncodedObservation, LearningError> {
    let count = check_placement_size(&view.identity, limits)?;
    if view.cell_count != count
        || view.observer.index() >= 2
        || view.results.len() != 2
        || (0..count)
            .filter_map(&view.occupant)
            .any(|p| p.index() >= 2)
    {
        return Err(LearningError::Invalid(
            "placement observation metadata mismatch".into(),
        ));
    }
    let domain = match view.mapping {
        ActionMapping::CellXMajor => count,
        ActionMapping::Column => view.identity.width(),
        _ => {
            return Err(LearningError::Invalid(
                "placement encoder requires cell or column actions".into(),
            ));
        }
    };
    let mut bytes = 0;
    add_bytes(&mut bytes, domain, 1, limits)?;
    add_bytes(
        &mut bytes,
        view.offered_ids.len(),
        std::mem::size_of::<usize>(),
        limits,
    )?;
    add_bytes(&mut bytes, 4, 8, limits)?;
    add_bytes(&mut bytes, 2, 1, limits)?;
    for (index, &name) in payloads.iter().enumerate() {
        if payloads[..index].contains(&name) {
            return Err(LearningError::Invalid(format!("duplicate payload: {name}")));
        }
        let multiplier = match name {
            "cells" => 1,
            "board" => 3,
            _ => return Err(LearningError::UnsupportedPayload(name.into())),
        };
        add_bytes(&mut bytes, count, 4 * multiplier, limits)?;
    }
    let mut mask = vec![false; domain];
    for &id in &view.offered_ids {
        let entry = mask
            .get_mut(id)
            .ok_or_else(|| LearningError::Invalid("offered action outside domain".into()))?;
        if *entry {
            return Err(LearningError::Invalid("duplicate offered action".into()));
        }
        *entry = true;
    }
    let next_player = match view.decision {
        Decision::Player(p) => Some(p),
        Decision::Finished => None,
    };
    if next_player != Some(view.observer) && !view.offered_ids.is_empty() {
        return Err(LearningError::Invalid(
            "inactive observation offers actions".into(),
        ));
    }
    let mut tensors = Vec::new();
    for &name in payloads {
        let (shape, values) = if name == "cells" {
            (
                vec![count],
                (0..count)
                    .map(|index| match (view.occupant)(index) {
                        None => 0.0,
                        Some(p) if p == view.observer => 1.0,
                        Some(_) => -1.0,
                    })
                    .collect(),
            )
        } else {
            let mut values = vec![0.0; count * 3];
            for index in 0..count {
                let channel = match (view.occupant)(index) {
                    Some(p) if p == view.observer => 0,
                    Some(_) => 1,
                    None => 2,
                };
                let (x, y) = (
                    index / view.identity.height(),
                    index % view.identity.height(),
                );
                values[channel * count + y * view.identity.width() + x] = 1.0;
            }
            (
                vec![3, view.identity.height(), view.identity.width()],
                values,
            )
        };
        tensors.push(Float32Tensor {
            name: if name == "cells" { "cells" } else { "board" },
            shape,
            values,
        });
    }
    let c = view.counters;
    let mut progress = [0; 4];
    for (output, input) in progress.iter_mut().zip([
        c.move_count(),
        c.completed_turn_count(),
        c.turn_count_in_round(),
        c.round_count(),
    ]) {
        *output = i64::try_from(input).map_err(|_| LearningError::Overflow)?;
    }
    Ok(EncodedObservation {
        identity: view.identity,
        observer: view.observer,
        next_player,
        tensors,
        progress,
        results: view
            .results
            .iter()
            .map(|r| match r {
                PlayerResult::Ongoing => ResultTag::Ongoing,
                PlayerResult::Win => ResultTag::Win,
                PlayerResult::Draw => ResultTag::Draw,
                PlayerResult::Loss => ResultTag::Loss,
            })
            .collect(),
        action_mask: mask,
        offered_action_ids: view.offered_ids,
    })
}
