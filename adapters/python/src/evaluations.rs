//! Convert opaque evaluation handles, controller checkpoints and candidate batches.
use crate::arrays::field;
use crate::game::Position;
use numpy::{
    PyArray, PyArray1, PyArrayMethods, PyReadonlyArray1,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use tag_rust::{
    agents::{
        EvaluationLimits, EvaluatorIdentity, ExternalEvaluation, OslaPythonAgent, OslaPythonConfig,
        OslaPythonSnapshot,
    },
    core::random::AgentRngSnapshot,
    learning::LearningLimits,
    session::{EvaluationKey, EvaluationRequest, EvaluationStatus},
};

#[pyclass(frozen, skip_from_py_object, module = "tag_rust._engine")]
#[derive(Clone)]
pub struct EvaluationHandle {
    pub key: EvaluationKey,
    pub position: Position,
}
#[pymethods]
impl EvaluationHandle {
    #[getter]
    fn id(&self) -> u64 {
        self.key.id()
    }
    #[getter]
    fn offset(&self) -> usize {
        self.key.offset()
    }
}

pub fn read_evaluator(dict: &Bound<'_, PyDict>) -> PyResult<EvaluatorIdentity> {
    Ok(EvaluatorIdentity {
        name: field(dict, "name")?.extract()?,
        version: field(dict, "version")?.extract()?,
        configuration: field(dict, "configuration")?.extract()?,
    })
}

/// Copy an external reply before detached Rust work. Shared by manual finish and resume.
pub fn read_response(
    values: PyReadonlyArray1<'_, f64>,
    action_ids: PyReadonlyArray1<'_, i64>,
    evaluator: &Bound<'_, PyDict>,
) -> PyResult<(Vec<f64>, Vec<usize>, EvaluatorIdentity)> {
    let values = values.as_array().iter().copied().collect();
    let ids = action_ids
        .as_array()
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("negative or overflowing action ID")
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((values, ids, read_evaluator(evaluator)?))
}

fn read_config(dict: &Bound<'_, PyDict>) -> PyResult<OslaPythonConfig> {
    Ok(OslaPythonConfig {
        evaluator: EvaluatorIdentity {
            name: field(dict, "evaluator_name")?.extract()?,
            version: field(dict, "evaluator_version")?.extract()?,
            configuration: field(dict, "evaluator_configuration")?.extract()?,
        },
        payloads: field(dict, "payloads")?.extract()?,
        limits: EvaluationLimits {
            learning: LearningLimits {
                max_cells: field(dict, "max_cells")?.extract()?,
                max_bytes: field(dict, "max_bytes")?.extract()?,
            },
            max_rows: field(dict, "max_rows")?.extract()?,
            max_candidates: field(dict, "max_candidates")?.extract()?,
        },
    })
}

/// Initialize policy/RNG versions in Rust, then expose a complete setup checkpoint.
#[pyfunction]
pub fn create_controller<'py>(
    py: Python<'py>,
    settings: &Bound<'py, PyDict>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    use tag_rust::core::checkpoint::Checkpoint;
    let agent = OslaPythonAgent::new(ExternalEvaluation, read_config(settings)?, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    snapshot(py, agent.snapshot())
}

/// Read a complete checkpoint into ordinary Rust values before any detached work.
pub fn read_snapshot(dict: &Bound<'_, PyDict>) -> PyResult<OslaPythonSnapshot> {
    Ok(OslaPythonSnapshot {
        policy_version: field(dict, "policy_version")?.extract()?,
        config: read_config(dict)?,
        rng: AgentRngSnapshot {
            version: field(dict, "rng_version")?.extract()?,
            initial_seed: field(dict, "seed")?.extract()?,
            stream: field(dict, "stream")?.extract()?,
            word_position: field(dict, "word_position")?.extract()?,
        },
        evaluator: (),
    })
}

pub fn snapshot<'py>(py: Python<'py>, value: OslaPythonSnapshot) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("policy_version", value.policy_version)?;
    out.set_item("evaluator_name", value.config.evaluator.name)?;
    out.set_item("evaluator_version", value.config.evaluator.version)?;
    out.set_item(
        "evaluator_configuration",
        value.config.evaluator.configuration,
    )?;
    out.set_item("payloads", value.config.payloads)?;
    out.set_item("max_cells", value.config.limits.learning.max_cells)?;
    out.set_item("max_bytes", value.config.limits.learning.max_bytes)?;
    out.set_item("max_rows", value.config.limits.max_rows)?;
    out.set_item("max_candidates", value.config.limits.max_candidates)?;
    out.set_item("rng_version", value.rng.version)?;
    out.set_item("seed", value.rng.initial_seed)?;
    out.set_item("stream", value.rng.stream)?;
    out.set_item("word_position", value.rng.word_position)?;
    Ok(out)
}

/// Convert candidate numbers without adding a particular game or group handle.
pub(super) fn batch_arrays<'py>(
    py: Python<'py>,
    batch: tag_rust::agents::CandidateBatch,
    evaluator: tag_rust::agents::EvaluatorIdentity,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let rows = batch.action_ids.len();
    let players = batch.identity.player_count;
    out.set_item("identity", crate::arrays::identity(py, batch.identity)?)?;
    out.set_item("evaluator_name", evaluator.name)?;
    out.set_item("evaluator_version", evaluator.version)?;
    out.set_item("evaluator_configuration", evaluator.configuration)?;
    let ids = batch
        .action_ids
        .into_iter()
        .map(i64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    out.set_item("action_ids", PyArray1::from_vec(py, ids))?;
    let payloads = PyDict::new(py);
    for tensor in batch.payloads {
        let array = ArrayD::from_shape_vec(IxDyn(&tensor.shape), tensor.values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        payloads.set_item(tensor.name, PyArray::from_owned_array(py, array))?;
    }
    out.set_item("payloads", payloads)?;
    out.set_item(
        "progress",
        PyArray1::from_vec(py, batch.progress).reshape([rows, 4])?,
    )?;
    out.set_item(
        "results",
        PyArray1::from_vec(py, batch.results).reshape([rows, players])?,
    )?;
    out.set_item("next_players", PyArray1::from_vec(py, batch.next_players))?;
    Ok(out)
}

pub fn request<'py>(
    py: Python<'py>,
    request: EvaluationRequest,
    position: Position,
) -> PyResult<Bound<'py, PyDict>> {
    let actor = request.batch.observer.index();
    let offset = request.batch.offset;
    let total = request.batch.total;
    let out = batch_arrays(py, request.batch, request.evaluator)?;
    out.set_item("kind", "request")?;
    out.set_item(
        "handle",
        EvaluationHandle {
            key: request.key,
            position: position.clone(),
        },
    )?;
    out.set_item("position", position)?;
    out.set_item("actor", actor)?;
    out.set_item("offset", offset)?;
    out.set_item("total", total)?;
    Ok(out)
}

pub fn status<'py>(
    py: Python<'py>,
    value: EvaluationStatus,
    position: Position,
) -> PyResult<Bound<'py, PyDict>> {
    match value {
        EvaluationStatus::Request(value) => request(py, *value, position),
        EvaluationStatus::Ready(proposal) => {
            let out = PyDict::new(py);
            out.set_item("kind", "proposal")?;
            out.set_item(
                "handle",
                EvaluationHandle {
                    key: proposal.key,
                    position: position.clone(),
                },
            )?;
            out.set_item("position", position)?;
            out.set_item("actor", proposal.actor.index())?;
            out.set_item("action_id", proposal.action_id)?;
            Ok(out)
        }
    }
}
