//! Bounded application file reads. Schema and combined-file limits stay with callers.
use std::{io::Read, path::Path};

pub(crate) const CONFIG_BYTES: usize = 1_048_576;

/// Read at most the permitted bytes plus one sentinel byte. The caller checks
/// the resulting length against its individual or combined-file budget.
pub(crate) fn read_text(path: &Path, maximum: usize, description: &str) -> std::io::Result<String> {
    let contextual = |operation: &str, error: std::io::Error| {
        std::io::Error::new(
            error.kind(),
            format!(
                "could not {operation} {description} '{}': {error}",
                path.display()
            ),
        )
    };
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|e| contextual("open", e))?
        .take(maximum.saturating_add(1) as u64)
        .read_to_string(&mut text)
        .map_err(|e| contextual("read", e))?;
    Ok(text)
}
