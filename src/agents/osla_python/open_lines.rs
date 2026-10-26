//! Score the existing two-player placement encodings entirely in Rust.
use super::{BatchEvaluator, CandidateBatch, OslaPythonError};
use crate::core::checkpoint::Checkpoint;
use crate::learning::{DType, EncodingIdentity, LearningSpec, OpenLinesSpec};

/// The native open-line formula applied to candidate `cells` arrays.
///
/// Supports the two-player cells encoding: own=1, opponent=-1, empty=0.
/// Line order and terminal scores match the game evaluators and Python example.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenLinesBatchEvaluator {
    identity: EncodingIdentity,
    lines: OpenLinesSpec,
    cells: usize,
}

impl OpenLinesBatchEvaluator {
    /// Copy one configuration's identity and ordered winning-line groups.
    /// Returns an error for unsupported line versions, cell shapes or indices.
    pub fn new(spec: &LearningSpec) -> Result<Self, OslaPythonError> {
        let cells = spec
            .payloads
            .iter()
            .find(|p| p.name() == "cells")
            .and_then(|p| p.tensor())
            .ok_or_else(|| invalid("the heuristic requires the cells encoding"))?;
        if cells.dtype != DType::Float32 || cells.shape.len() != 1 {
            return Err(invalid("cells must be a one-dimensional float32 vector"));
        }
        let result = Self {
            identity: spec.identity.clone(),
            lines: spec
                .heuristic
                .clone()
                .ok_or_else(|| invalid("the heuristic requires open-line support"))?,
            cells: cells.shape[0],
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(), OslaPythonError> {
        if self.cells == 0
            || self.lines.version != 1
            || self.lines.window_length == 0
            || !self
                .lines
                .indices
                .len()
                .is_multiple_of(self.lines.window_length)
            || self
                .lines
                .indices
                .iter()
                .any(|&i| i < 0 || i as u64 >= self.cells as u64)
        {
            return Err(invalid(
                "invalid open-line version, window length or cell indices",
            ));
        }
        Ok(())
    }
}

fn invalid(message: &str) -> OslaPythonError {
    OslaPythonError::Invalid(message.into())
}

impl BatchEvaluator for OpenLinesBatchEvaluator {
    fn evaluate(&mut self, batch: &CandidateBatch) -> Result<Vec<f64>, OslaPythonError> {
        let rows = batch.action_ids.len();
        let cells = batch
            .payloads
            .iter()
            .find(|p| p.name == "cells")
            .ok_or_else(|| invalid("the heuristic requires cells in every batch"))?;
        let expected_cells = rows
            .checked_mul(self.cells)
            .ok_or(OslaPythonError::Overflow)?;
        let expected_results = rows.checked_mul(2).ok_or(OslaPythonError::Overflow)?;
        if batch.identity != self.identity
            || batch.observer.index() >= 2
            || cells.shape != [rows, self.cells]
            || cells.values.len() != expected_cells
            || batch.results.len() != expected_results
            || batch.results.iter().any(|&r| !(0..=3).contains(&r))
            || cells
                .values
                .iter()
                .any(|&c| c != -1.0 && c != 0.0 && c != 1.0)
        {
            return Err(invalid(
                "batch must match this configuration's two-player cells encoding",
            ));
        }
        let mut values = Vec::with_capacity(rows);
        for (row, board) in cells.values.chunks(self.cells).enumerate() {
            // Results use absolute seats; the cells already use the observer's view.
            let result = batch.results[row * 2 + batch.observer.index()];
            let value = match result {
                1 => 1.0,
                2 => 0.0,
                3 => -1.0,
                _ => {
                    let mut sum = 0.0;
                    let length = self.lines.window_length;
                    for line in self.lines.indices.chunks(length) {
                        let mut friendly = 0;
                        let mut opposing = 0;
                        for &index in line {
                            match board[index as usize] {
                                1.0 => friendly += 1,
                                -1.0 => opposing += 1,
                                _ => {}
                            }
                        }
                        let contribution = if friendly > 0 && opposing > 0 {
                            0.0
                        } else if friendly > 0 {
                            (friendly as f64 / length as f64).powi(2)
                        } else {
                            -(opposing as f64 / length as f64).powi(2)
                        };
                        sum += contribution;
                    }
                    let count = self.lines.indices.len() / length;
                    if count == 0 {
                        0.0
                    } else {
                        0.9 * sum / count as f64
                    }
                }
            };
            values.push(value);
        }
        Ok(values)
    }
}

impl Checkpoint for OpenLinesBatchEvaluator {
    type Snapshot = Self;
    type RestoreError = OslaPythonError;
    fn snapshot(&self) -> Self {
        self.clone()
    }
    fn from_snapshot(snapshot: &Self) -> Result<Self, Self::RestoreError> {
        snapshot.validate()?;
        Ok(snapshot.clone())
    }
}
