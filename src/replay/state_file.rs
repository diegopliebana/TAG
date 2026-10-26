//! Save/load one complete game position without recording or controllers.
//!
//! Parameters and all rule-relevant state are preserved. Earlier moves are absent,
//! so loading creates a new recording root rather than recovering prior navigation.
use super::common::{check_state_versions, check_storage, invalid, json, versions};
use super::*;
use serde::{Serialize, de::DeserializeOwned};

/// Borrow parameters and state; return owned standalone JSON after invariant validation.
///
/// # Arguments
/// `model`, selected `parameters` and complete simulation `state` are shared
/// borrows. Parameters must reproduce this state's rules.
///
/// # Returns
/// Owned standalone-state JSON, preserving parameters and sufficient state.
///
/// # Errors
/// State invariant failure, parameter mismatch or JSON conversion failure.
pub fn export_state<F>(
    model: &F,
    parameters: &F::Parameters,
    state: &F::State,
) -> Result<String, PersistenceError>
where
    F: PersistableGame,
    F::State: PartialEq,
    F::Parameters: Clone + Serialize,
{
    let data = model.state_data(state);
    if model.restore_state(parameters, &data)? != *state {
        return Err(invalid("export parameters differ from state rules"));
    }
    Ok(serde_json::to_string_pretty(&StateFile {
        kind: "tag-state".into(),
        versions: versions::<F>(),
        parameters: parameters.clone(),
        state: data,
    })?)
}
/// Parse/validate a saved game state without a move recording into owned parameters/state; controllers are selected later.
///
/// # Arguments
/// `text` and `model` are shared borrows. `limits` is a copied input/work policy.
///
/// # Returns
/// Owned `(parameters, state)` after game-specific invariant validation.
/// Callers select controllers before passing the state to `Session::from_state`.
///
/// # Errors
/// Malformed input, wrong document kind/version, invalid parameters/state or exceeded limits.
pub fn import_state<F>(
    text: &str,
    model: &F,
    limits: ReplayLimits,
) -> Result<(F::Parameters, F::State), PersistenceError>
where
    F: PersistableGame,
    F::Parameters: DeserializeOwned,
{
    let file: StateFile<F::Parameters, F::SavedState> = json(text, limits)?;
    if file.kind != "tag-state" {
        return Err(invalid("expected a state document"));
    }
    check_state_versions::<F>(&file.versions)?;
    check_storage::<F>(&file.parameters, limits)?;
    let work = F::validation_work(&file.parameters, &file.state)
        .ok_or_else(|| invalid("state validation work overflow"))?;
    if work > limits.max_verification_work {
        return Err(invalid("state validation exceeds configured work limit"));
    }
    let state = model.restore_state(&file.parameters, &file.state)?;
    Ok((file.parameters, state))
}
