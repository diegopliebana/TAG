//! Check branch relationships and estimate replay work before restoring states.
//!
//! A branch must refer to an earlier parent, not form a cycle. Work limits bound
//! reconstruction and verification before a malformed or oversized file can trigger
//! excessive simulation.
use super::common::invalid;
use super::*;

/// Validate ordered ancestry and estimate work for roots and inherited moves.
///
/// # Arguments
/// `file` is a shared decoded replay borrow. `limits` is a copied resource policy.
///
/// # Returns
/// Copied work estimate charged before any branch/AI reconstruction.
///
/// # Errors
/// Excessive size/depth/work, arithmetic overflow or invalid branch ancestry.
pub(super) fn replay_work<F: ReplayGame>(
    file: &ReplayFile<F::Parameters, F::SavedState, F::Action>,
    limits: ReplayLimits,
) -> Result<usize, PersistenceError> {
    if file.branches.is_empty() || file.branches.len() > limits.max_branches {
        return Err(invalid("invalid or excessive branch count"));
    }
    let moves = file
        .branches
        .iter()
        .try_fold(0usize, |total, b| total.checked_add(b.moves.len()))
        .ok_or_else(|| invalid("move count overflow"))?;
    if moves > limits.max_moves {
        return Err(invalid("replay exceeds configured move limit"));
    }
    let transition_work = F::verification_work(&file.parameters, false)
        .ok_or_else(|| invalid("verification work overflow"))?;
    let mut ancestry_work = 0usize;
    let mut depths = Vec::new();
    let mut lengths = Vec::new();
    for (index, branch) in file.branches.iter().enumerate() {
        let mut previous = None;
        if branch.controller_changes.len() > branch.moves.len() + 1 {
            return Err(invalid("excessive controller changes"));
        }
        for change in &branch.controller_changes {
            if change.position < branch.parent_position
                || change.position > branch.parent_position.saturating_add(branch.moves.len())
                || previous.is_some_and(|p| p >= change.position)
                || change.controllers.len() != branch.controllers.len()
            {
                return Err(invalid("invalid controller change boundary"));
            }
            previous = Some(change.position);
        }
        ancestry_work = ancestry_work
            .checked_add(
                branch
                    .controller_changes
                    .len()
                    .checked_mul(transition_work)
                    .ok_or_else(|| invalid("controller validation work overflow"))?,
            )
            .ok_or_else(|| invalid("controller validation work overflow"))?;
        if branch.id != index {
            return Err(invalid("branch IDs must be ordered and unique"));
        }
        let depth = match branch.parent {
            None if index == 0 && branch.parent_position == 0 => 0,
            Some(parent) if parent < index => {
                if branch.parent_position > lengths[parent] {
                    return Err(invalid("fork position exceeds parent"));
                }
                ancestry_work = ancestry_work
                    .checked_add(
                        branch
                            .parent_position
                            .checked_mul(transition_work)
                            .and_then(|v| v.checked_mul(3))
                            .ok_or_else(|| invalid("ancestry work overflow"))?,
                    )
                    .ok_or_else(|| invalid("ancestry work overflow"))?;
                depths[parent] + 1
            }
            _ => return Err(invalid("invalid branch ancestry")),
        };
        if depth > limits.max_depth {
            return Err(invalid("branch ancestry exceeds configured depth limit"));
        }
        depths.push(depth);
        lengths.push(
            branch
                .parent_position
                .checked_add(branch.moves.len())
                .ok_or_else(|| invalid("branch length overflow"))?,
        );
    }
    if ancestry_work > limits.max_verification_work {
        return Err(invalid(
            "branch reconstruction exceeds configured work limit",
        ));
    }
    let root_work = F::validation_work(&file.parameters, &file.initial_state)
        .ok_or_else(|| invalid("state validation work overflow"))?;
    ancestry_work = ancestry_work
        .checked_add(root_work)
        .ok_or_else(|| invalid("validation work overflow"))?;
    if ancestry_work > limits.max_verification_work {
        return Err(invalid(
            "root state validation exceeds configured work limit",
        ));
    }
    Ok(ancestry_work)
}
