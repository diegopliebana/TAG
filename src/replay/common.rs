//! Shared checks on JSON size, required storage and supported file versions.
//!
//! Version checks prevent loading data whose fields, rules, agents or RNG algorithm
//! have a different meaning from the supported implementation.
use super::*;
use serde::de::DeserializeOwned;

pub(super) fn invalid(s: impl Into<String>) -> PersistenceError {
    PersistenceError::Invalid(s.into())
}
/// Construct the current compatibility metadata for game F.
/// There are no value arguments; F supplies game/rules identities.
/// Returns owned metadata, retaining the version-1 state and replay JSON schema.
pub(super) fn versions<F: PersistableGame>() -> Versions {
    Versions {
        format: 1,
        game: F::GAME_ID.into(),
        rules: F::RULES_VERSION,
        agents: 1,
        evaluators: 1,
        rng: "chacha8-rand_chacha-0.9.0-rand_core-0.9.5".into(),
    }
}
/// Compare saved compatibility identities with the supported game scheme.
///
/// # Arguments
/// `v` is a shared borrow of saved version identities; `F` identifies supported rules.
///
/// # Returns
/// Unit when all identities match.
///
/// # Errors
/// Any unsupported identity/version.
pub(super) fn check_versions<F: PersistableGame>(v: &Versions) -> Result<(), PersistenceError> {
    let supported = versions::<F>();
    for (name, saved, expected) in [
        ("format", v.format.to_string(), supported.format.to_string()),
        ("game", v.game.clone(), supported.game),
        ("rules", v.rules.to_string(), supported.rules.to_string()),
        ("agents", v.agents.to_string(), supported.agents.to_string()),
        (
            "evaluators",
            v.evaluators.to_string(),
            supported.evaluators.to_string(),
        ),
        ("RNG", v.rng.clone(), supported.rng),
    ] {
        if saved != expected {
            return Err(invalid(format!(
                "unsupported {name} version/identity: file {saved}; supported {expected}"
            )));
        }
    }
    Ok(())
}
/// Check state storage requirements before game allocation.
///
/// # Arguments
/// `p` is a shared parameter borrow; `limits` is a copied storage policy.
///
/// # Returns
/// Copied required storage units for these parameters.
///
/// # Errors
/// Overflow or a configured storage limit violation.
pub(super) fn check_storage<F: PersistableGame>(
    p: &F::Parameters,
    limits: ReplayLimits,
) -> Result<usize, PersistenceError> {
    let cells = F::storage_units(p).ok_or_else(|| invalid("state dimensions overflow"))?;
    if cells > limits.max_cells {
        return Err(invalid("state exceeds configured import storage limit"));
    }
    Ok(cells)
}
/// Decode file data after enforcing byte and JSON nesting limits.
///
/// # Arguments
/// `text` is passed by a shared (read-only) reference input; `limits` is a copied byte/nesting policy.
/// `T` is the expected type that holds its own file or message data.
///
/// # Returns
/// Owned decoded data after byte/depth checks.
///
/// # Errors
/// Excessive size/depth or JSON/schema errors.
pub(super) fn json<T: DeserializeOwned>(
    text: &str,
    limits: ReplayLimits,
) -> Result<T, PersistenceError> {
    if text.len() > limits.max_bytes {
        return Err(invalid("file exceeds configured byte limit"));
    }
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for c in text.bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                quoted = false;
            }
            continue;
        }
        match c {
            b'"' => quoted = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > limits.max_depth {
                    return Err(invalid("JSON exceeds configured nesting limit"));
                }
            }
            b']' | b'}' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }
    Ok(serde_json::from_str(text)?)
}

/// Check sufficient-state compatibility independently of replay controller identities.
///
/// # Arguments
/// `v` is shared saved metadata; `F` supplies the game and rules identity.
///
/// # Returns
/// Unit when state format/game/rules match; legacy replay metadata remains in the schema.
///
/// # Errors
/// Unsupported state format, game or rules version.
pub(super) fn check_state_versions<F: PersistableGame>(
    v: &Versions,
) -> Result<(), PersistenceError> {
    let supported = versions::<F>();
    for (name, saved, expected) in [
        (
            "state format",
            v.format.to_string(),
            supported.format.to_string(),
        ),
        ("game", v.game.clone(), supported.game),
        ("rules", v.rules.to_string(), supported.rules.to_string()),
    ] {
        if saved != expected {
            return Err(invalid(format!(
                "unsupported {name} version/identity: file {saved}; supported {expected}"
            )));
        }
    }
    Ok(())
}
