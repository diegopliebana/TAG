//! Python conversion and lifecycle for Rust-owned single-learner episodes.
use crate::{
    arrays,
    episode::{Config, Output, Runtime},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::sync::Arc;
use tag_rust::learning::LearningLimits;

#[pyclass(module = "tag_rust._engine")]
pub struct Episode {
    pub(super) config: Config,
    pub(super) description: tag_rust::learning::LearningSpec,
    pub(super) runtime: Option<Runtime>,
    pub(super) failed: bool,
    closed: bool,
    owner: Arc<()>,
    pub(super) generation: u64,
}
/// Own fresh reset work until Python has constructed the complete return value.
#[pyclass(module = "tag_rust._engine")]
pub struct ResetCandidate {
    runtime: Option<Runtime>,
    owner: Arc<()>,
    generation: u64,
    #[pyo3(get)]
    output: Py<PyDict>,
}
pub(super) fn convert<'py>(py: Python<'py>, output: Output) -> PyResult<Bound<'py, PyDict>> {
    convert_output(py, output, true)
}

/// Convert `output` into member metadata after its numeric payloads were stacked.
///
/// # Arguments
/// `py` permits Python allocation. `output` transfers one completed learner return.
///
/// # Returns
/// An owned info dictionary with player metadata, counters, reward, flags and RNG.
///
/// # Errors
/// Python allocation or dictionary construction can fail.
pub(super) fn convert_info<'py>(py: Python<'py>, output: Output) -> PyResult<Bound<'py, PyDict>> {
    convert_output(py, output, false)
}

/// Consume one return and select full observation or metadata-only conversion.
/// `include_arrays` is true for individual calls; batch arrays are already stacked.
fn convert_output<'py>(
    py: Python<'py>,
    output: Output,
    include_arrays: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = if include_arrays {
        arrays::observation(py, output.observation)?
    } else {
        arrays::observation_metadata(py, &output.observation)?
    };
    out.set_item("learner_steps", output.steps)?;
    out.set_item("underlying_moves", output.moves)?;
    out.set_item("moves_this_call", output.delta_moves)?;
    out.set_item("reward", output.reward)?;
    out.set_item("terminated", output.terminated)?;
    out.set_item("truncated", output.truncated)?;
    out.set_item("action", output.action)?;
    out.set_item(
        "teacher_rng",
        output
            .teacher_rng
            .map(|rng| (rng.version, rng.initial_seed, rng.stream, rng.word_position)),
    )?;
    if let Some(intermediate) = output.intermediate {
        out.set_item("intermediate", arrays::observation(py, intermediate)?)?;
    } else {
        out.set_item("intermediate", py.None())?;
    }
    out.set_item(
        "opponent_rng",
        (
            output.rng.version,
            output.rng.initial_seed,
            output.rng.stream,
            output.rng.word_position,
        ),
    )?;
    Ok(out)
}
impl Episode {
    pub(super) fn open(&self) -> PyResult<()> {
        if self.closed {
            Err(PyRuntimeError::new_err("episode is closed"))
        } else {
            Ok(())
        }
    }
    /// Check `action` and lifecycle without changing the episode or opponent RNG.
    /// Returned input errors are retryable, unlike execution failures.
    pub(super) fn validate_step(&self, action: usize) -> PyResult<()> {
        self.open()?;
        if self.failed {
            return Err(PyRuntimeError::new_err("episode failed; reset is required"));
        }
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("reset is required before step"))?;
        self.generation
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("episode generation exhausted"))?;
        runtime.validate(action).map_err(PyValueError::new_err)
    }

    /// Execute prevalidated `action` in place. Keep failed=true until complete
    /// Rust/Python output conversion succeeds; accepted moves can survive errors.
    pub(super) fn run_step(&mut self, action: usize) -> Result<Output, String> {
        self.failed = true;
        self.generation += 1;
        self.runtime.as_mut().expect("step preflight").step(action)
    }
    /// Preflight the configured teacher without advancing game or RNG state.
    /// Returns a retryable lifecycle/availability error, or unit for a ready learner.
    pub(super) fn validate_teacher(&self) -> PyResult<()> {
        self.open()?;
        if self.failed {
            return Err(PyRuntimeError::new_err("episode failed; reset is required"));
        }
        self.generation
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("episode generation exhausted"))?;
        self.runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("reset is required before step"))?
            .validate_teacher()
            .map_err(PyValueError::new_err)
    }
    /// Run a prevalidated native teacher decision and ordinary step in place.
    /// Accepted work survives failure; conversion must clear failed after success.
    pub(super) fn run_teacher(&mut self) -> Result<Output, String> {
        self.failed = true;
        self.generation += 1;
        self.runtime
            .as_mut()
            .expect("teacher preflight")
            .step_teacher()
    }
}
#[pymethods]
impl Episode {
    #[new]
    fn new(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let f = |key| arrays::field(config, key);
        let config = Config {
            name: f("game")?.extract()?,
            parameters: f("parameters")?.extract()?,
            learner: f("learner")?.extract()?,
            opponent: f("opponent")?.extract()?,
            settings: f("opponent_settings")?.extract()?,
            payloads: f("payloads")?.extract()?,
            horizon: f("max_episode_steps")?.extract()?,
            limits: LearningLimits {
                max_cells: f("max_cells")?.extract()?,
                max_bytes: f("max_bytes")?.extract()?,
            },
            include_intermediate: config
                .get_item("include_intermediate")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false),
            teacher: config
                .get_item("teacher")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false),
        };
        let description = py
            .detach(|| config.create(0).map(|runtime| runtime.spec()))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            config,
            description,
            runtime: None,
            failed: false,
            closed: false,
            owner: Arc::new(()),
            generation: 0,
        })
    }
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.open()?;
        let spec = py.detach(|| self.description.clone());
        arrays::spec(py, spec)
    }
    #[pyo3(signature=(seed,teacher_seed=None))]
    fn prepare_reset(
        &self,
        py: Python<'_>,
        seed: u64,
        teacher_seed: Option<u64>,
    ) -> PyResult<ResetCandidate> {
        self.open()?;
        self.generation
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("episode generation exhausted"))?;
        let (runtime, output) = py
            .detach(|| {
                let mut runtime = self.config.create_seeded(seed, teacher_seed)?;
                let output = runtime.opening()?;
                Ok::<_, String>((runtime, output))
            })
            .map_err(PyRuntimeError::new_err)?;
        let output = convert(py, output)?.unbind();
        Ok(ResetCandidate {
            runtime: Some(runtime),
            output,
            owner: Arc::clone(&self.owner),
            generation: self.generation,
        })
    }
    fn commit_reset(&mut self, candidate: &mut ResetCandidate) -> PyResult<()> {
        self.open()?;
        if !Arc::ptr_eq(&self.owner, &candidate.owner)
            || self.generation != candidate.generation
            || candidate.runtime.is_none()
        {
            return Err(PyValueError::new_err("stale or foreign reset candidate"));
        }
        self.generation += 1;
        self.runtime = candidate.runtime.take();
        self.failed = false;
        Ok(())
    }
    fn step<'py>(&mut self, py: Python<'py>, action: usize) -> PyResult<Bound<'py, PyDict>> {
        py.detach(|| self.validate_step(action))?;
        let output = py
            .detach(|| self.run_step(action))
            .map_err(PyRuntimeError::new_err)?;
        let out = convert(py, output)?;
        self.failed = false;
        Ok(out)
    }
    fn step_teacher<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        py.detach(|| self.validate_teacher())?;
        let output = py
            .detach(|| self.run_teacher())
            .map_err(PyRuntimeError::new_err)?;
        let out = convert(py, output)?;
        self.failed = false;
        Ok(out)
    }
    fn fail(&mut self) {
        self.failed = true;
    }
    fn close(&mut self) {
        self.runtime = None;
        self.closed = true;
    }
    #[getter]
    pub(super) fn status(&self) -> &'static str {
        if self.closed {
            "closed"
        } else if self.failed {
            "failed"
        } else if self.runtime.is_none() {
            "new"
        } else if self.runtime.as_ref().is_some_and(Runtime::ended) {
            "ended"
        } else {
            "ready"
        }
    }
}

#[cfg(test)]
pub(super) fn test_episode(config: Config) -> Episode {
    let mut runtime = config.create(5).unwrap();
    runtime.opening().unwrap();
    Episode {
        description: runtime.spec(),
        config,
        runtime: Some(runtime),
        failed: false,
        closed: false,
        owner: Arc::new(()),
        generation: 0,
    }
}
