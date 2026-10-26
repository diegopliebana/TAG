//! Python calls into the Rust heuristic; numeric inputs are copied before detaching.
use crate::arrays::field;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use tag_rust::{
    agents::{BatchEvaluator, CandidateBatch, OpenLinesBatchEvaluator},
    learning::{EncodingIdentity, Float32Tensor, LearningLimits, LearningSpec},
};

/// A Python-callable scorer owning Rust's line groups independently of a live game.
#[pyclass(frozen, module = "tag_rust._engine")]
pub struct OpenLinesEvaluator {
    backend: OpenLinesBatchEvaluator,
    identity: EncodingIdentity,
}
impl OpenLinesEvaluator {
    pub fn from_spec(spec: &LearningSpec) -> PyResult<Self> {
        let backend = OpenLinesBatchEvaluator::new(spec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            backend,
            identity: spec.identity.clone(),
        })
    }
}
#[pymethods]
impl OpenLinesEvaluator {
    /// Copy supplied NumPy inputs, detach for Rust scoring, then return owned float64 values.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        identity: &Bound<'py, PyDict>,
        actor: usize,
        cells: PyReadonlyArray2<'_, f32>,
        results: PyReadonlyArray2<'_, i8>,
        action_ids: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use pyo3::exceptions::{PyMemoryError, PyValueError};
        let id = self.identity.clone();
        let settings = identity
            .get_item("configuration")?
            .map(|v| v.extract::<std::collections::BTreeMap<String, String>>())
            .transpose()?
            .unwrap_or_default();
        let settings_match = match &id.configuration {
            tag_rust::learning::EncodingConfiguration::Placement { .. } => settings.is_empty(),
            tag_rust::learning::EncodingConfiguration::Named(expected) => &settings == expected,
        };
        if field(identity, "game")?.extract::<String>()? != id.game
            || !settings_match
            || field(identity, "rules_version")?.extract::<u32>()? != id.rules_version
            || field(identity, "encoding_version")?.extract::<u32>()? != id.encoding_version
            || field(identity, "width")?.extract::<usize>()? != id.width()
            || field(identity, "height")?.extract::<usize>()? != id.height()
            || field(identity, "target")?.extract::<usize>()? != id.target()
            || field(identity, "player_count")?.extract::<usize>()? != id.player_count
        {
            return Err(PyValueError::new_err(
                "candidate encoding differs from this evaluator's configuration",
            ));
        }
        let rows = action_ids.len();
        let bytes = cells
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(results.len()))
            .and_then(|n| rows.checked_mul(16).and_then(|extra| n.checked_add(extra)))
            .ok_or_else(|| PyMemoryError::new_err("scoring buffer size overflow"))?;
        if rows > 4096 || bytes > LearningLimits::default().max_bytes {
            return Err(PyMemoryError::new_err(
                "Rust scoring supports at most 4096 rows and 64 MiB of numeric buffers",
            ));
        }
        if results.shape() != [rows, 2] || cells.shape()[0] != rows {
            return Err(PyValueError::new_err(
                "cells, results and action IDs must have the same candidate count; results needs two seats",
            ));
        }
        let ids = action_ids
            .as_array()
            .iter()
            .map(|&id| {
                usize::try_from(id)
                    .map_err(|_| PyValueError::new_err("action IDs must be nonnegative"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        // Owned copies permit releasing Python's lock without accessing NumPy buffers.
        // Iterating the array views also supports arrays with noncontiguous strides.
        let batch = CandidateBatch {
            identity: id,
            observer: tag_rust::core::PlayerId::new(actor),
            offset: 0,
            total: rows,
            action_ids: ids,
            progress: vec![],
            next_players: vec![],
            payloads: vec![Float32Tensor {
                name: "cells",
                shape: cells.shape().to_vec(),
                values: cells.as_array().iter().copied().collect(),
            }],
            results: results.as_array().iter().copied().collect(),
        };
        let mut backend = self.backend.clone();
        let values = py
            .detach(move || backend.evaluate(&batch))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_vec(py, values))
    }
}
