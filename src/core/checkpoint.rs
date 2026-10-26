//! Save enough information to resume an object with the same future behavior.
//!
//! A checkpoint includes changing memory and random-generator progress, not merely
//! the original seed. Restoration creates an independent object; it must not advance
//! the source or share mutable data with it.

/// Saved data sufficient to recreate a separate object with the same future behavior.
/// Include every field that affects future behavior, including RNG position. Mutable
/// data in a snapshot must be independently owned. Per-decision search trees
/// are rebuilt by agents under the selected search design.
pub trait Checkpoint: Sized {
    /// Clone duplicates snapshot data for independent restoration.
    type Snapshot: Clone;
    type RestoreError;

    /// Read self through a shared reference and return a separate copy of its saved progress.
    fn snapshot(&self) -> Self::Snapshot;

    /// Read the saved data through a shared reference and construct a new independent object.
    /// Return RestoreError if the saved version or values are unsupported.
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError>;
}
