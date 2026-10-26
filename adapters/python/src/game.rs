//! Own one Rust game and reject action requests from other instances or old positions.
use crate::evaluations::{self, EvaluationHandle};
use crate::{
    arrays,
    engine::{Engine, Error},
};
use numpy::PyReadonlyArray1;
use pyo3::{
    create_exception,
    exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};
use tag_rust::agents::OslaPythonError;
use tag_rust::learning::{LearningError, LearningLimits};

create_exception!(_engine, StalePositionError, PyRuntimeError);
create_exception!(_engine, ClosedGameError, PyRuntimeError);
create_exception!(_engine, EvaluationError, PyRuntimeError);
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

fn instance_id() -> PyResult<u64> {
    NEXT_INSTANCE
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| PyOverflowError::new_err("game instance IDs exhausted"))
}

pub(super) fn convert(error: Error) -> PyErr {
    match error {
        Error::Agent(OslaPythonError::Stale) => StalePositionError::new_err(error.to_string()),
        Error::Agent(OslaPythonError::ResourceLimit) => PyMemoryError::new_err(error.to_string()),
        Error::Agent(OslaPythonError::Overflow) => PyOverflowError::new_err(error.to_string()),
        Error::Agent(
            OslaPythonError::Busy
            | OslaPythonError::NotReady
            | OslaPythonError::ExternalEvaluationRequired,
        ) => EvaluationError::new_err(error.to_string()),
        Error::Agent(_) => PyValueError::new_err(error.to_string()),
        Error::Learning(LearningError::ResourceLimit) => PyMemoryError::new_err(error.to_string()),
        Error::Learning(LearningError::Overflow) => PyOverflowError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// A token identifying one instance and its accepted-move/reset revision.
#[pyclass(frozen, eq, skip_from_py_object, module = "tag_rust._engine")]
#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    #[pyo3(get)]
    instance: u64,
    #[pyo3(get)]
    revision: u64,
}

#[pymethods]
impl Position {
    fn __repr__(&self) -> String {
        format!(
            "Position(instance={}, revision={})",
            self.instance, self.revision
        )
    }
}

/// Rust owns the simulation. PyO3 borrows this object while detached computations run.
#[pyclass(module = "tag_rust._engine")]
pub struct Game {
    pub(super) engine: Option<Engine>,
    instance: u64,
    pub(super) revision: u64,
}

impl Game {
    pub(super) fn engine(&self) -> PyResult<&Engine> {
        self.engine
            .as_ref()
            .ok_or_else(|| ClosedGameError::new_err("this game is closed"))
    }
    pub(super) fn token(&self) -> Position {
        Position {
            instance: self.instance,
            revision: self.revision,
        }
    }
    fn next_revision(&self) -> PyResult<u64> {
        self.revision
            .checked_add(1)
            .ok_or_else(|| PyOverflowError::new_err("position revisions exhausted"))
    }

    pub(super) fn execution_budget(&self, max_moves: u64) -> PyResult<()> {
        self.engine()?;
        if max_moves == 0 {
            return Err(PyValueError::new_err("max_moves must be positive"));
        }
        self.revision.checked_add(max_moves).ok_or_else(|| {
            PyOverflowError::new_err("move budget exceeds remaining position revisions")
        })?;
        Ok(())
    }

    fn execution_result<'py>(
        &mut self,
        py: Python<'py>,
        progress: crate::execution::Progress,
    ) -> PyResult<Bound<'py, PyDict>> {
        // A later failed decision must not hide earlier accepted moves from tokens.
        self.revision += progress.moves;
        let stop = progress.stop.map_err(convert)?;
        let out = PyDict::new(py);
        out.set_item("moves", progress.moves)?;
        out.set_item("position", self.token())?;
        let (reason, actor, request) = match stop {
            crate::execution::Stop::Finished => ("finished", None, None),
            crate::execution::Stop::Human(actor) => ("human", Some(actor), None),
            crate::execution::Stop::MoveLimit(actor) => ("move_limit", Some(actor), None),
            crate::execution::Stop::Evaluation(request) => {
                let actor = request.batch.observer.index();
                (
                    "evaluation",
                    Some(actor),
                    Some(evaluations::request(py, *request, self.token())?),
                )
            }
        };
        out.set_item("reason", reason)?;
        out.set_item("actor", actor)?;
        out.set_item("request", request)?;
        Ok(out)
    }
}

#[pymethods]
impl Game {
    /// Read a fresh position token. Reset and accepted moves invalidate previous tokens.
    #[getter]
    fn position(&self) -> PyResult<Position> {
        self.engine()?;
        Ok(self.token())
    }

    #[pyo3(signature = (settings_json, observers, payloads, values_name, policy_name, max_rows=4096, max_bytes=67108864))]
    #[allow(clippy::too_many_arguments)]
    fn start_mcts(
        &self,
        py: Python<'_>,
        settings_json: &str,
        observers: &str,
        payloads: Vec<String>,
        values_name: String,
        policy_name: String,
        max_rows: usize,
        max_bytes: usize,
    ) -> PyResult<crate::mcts::MctsSearch> {
        use tag_rust::search::mcts::{Settings, ValueObservers};
        let settings =
            Settings::from_json(settings_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let observers = match observers {
            "all_players" => ValueObservers::AllPlayers,
            "root_player" => ValueObservers::RootPlayer,
            "deciding_player" => ValueObservers::DecidingPlayer,
            _ => return Err(PyValueError::new_err("unknown value observer mode")),
        };
        if max_rows == 0 || max_bytes == 0 {
            return Err(PyValueError::new_err(
                "search row/byte limits must be positive",
            ));
        }
        let options = crate::mcts::SearchOptions {
            settings,
            observers,
            payloads,
            values_name,
            policy_name,
            max_rows,
            max_bytes,
        };
        let engine = self.engine()?;
        py.detach(|| engine.start_mcts(options))
            .map_err(PyValueError::new_err)
    }

    /// Resolved real-game seed, outside player-visible observations.
    #[getter]
    fn seed(&self) -> PyResult<Option<u64>> {
        Ok(self.engine()?.seed())
    }
    fn payload_names(&self) -> PyResult<Vec<&'static str>> {
        Ok(self.engine()?.payload_names())
    }

    /// Identify the next acting seat; None means the game has finished.
    #[getter]
    fn next_player(&self) -> PyResult<Option<usize>> {
        Ok(self.engine()?.next_player())
    }

    /// Return fresh format descriptions using the game's Rust learning implementation.
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let engine = self.engine()?;
        let spec = py.detach(|| engine.spec()).map_err(convert)?;
        arrays::spec(py, spec)
    }

    /// Capture this configuration's Rust heuristic for later calls from Python.
    fn open_lines_evaluator(&self, py: Python<'_>) -> PyResult<crate::scoring::OpenLinesEvaluator> {
        let engine = self.engine()?;
        let spec = py.detach(|| engine.spec()).map_err(convert)?;
        crate::scoring::OpenLinesEvaluator::from_spec(&spec)
    }

    /// Encode requested inputs while detached, then create Python arrays after reattaching.
    fn observe<'py>(
        &self,
        py: Python<'py>,
        observer: usize,
        payloads: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let engine = self.engine()?;
        let value = py
            .detach(|| {
                let names = payloads.iter().map(String::as_str).collect::<Vec<_>>();
                engine.observe(observer, &names)
            })
            .map_err(convert)?;
        let out = arrays::observation(py, value)?;
        out.set_item("position", self.token())?;
        Ok(out)
    }

    /// Apply one decision after checking that its observation belongs to this live position.
    #[pyo3(signature = (action_id, position, actor=None))]
    fn apply(
        &mut self,
        py: Python<'_>,
        action_id: usize,
        position: &Position,
        actor: Option<usize>,
    ) -> PyResult<Position> {
        self.engine()?;
        if *position != self.token() {
            return Err(StalePositionError::new_err(
                "observation belongs to another game or an older position",
            ));
        }
        let next_revision = self.next_revision()?;
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| ClosedGameError::new_err("this game is closed"))?;
        py.detach(|| engine.apply(action_id, actor))
            .map_err(convert)?;
        self.revision = next_revision;
        Ok(self.token())
    }

    /// Restore the initial board with the original parameters and invalidate old observations.
    fn reset(&mut self, py: Python<'_>) -> PyResult<Position> {
        self.engine()?;
        let next_revision = self.next_revision()?;
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| ClosedGameError::new_err("this game is closed"))?;
        py.detach(|| engine.reset()).map_err(convert)?;
        self.revision = next_revision;
        Ok(self.token())
    }

    /// Return an independent state copy with its own instance ID and initial revision.
    fn clone(&self, py: Python<'_>) -> PyResult<Self> {
        let engine = self.engine()?;
        let instance = instance_id()?;
        let engine = py.detach(|| engine.snapshot()).map_err(convert)?;
        Ok(Self {
            engine: Some(engine),
            instance,
            revision: 0,
        })
    }

    /// Install a human seat or an independently restored OSLAPython checkpoint.
    fn set_controller(
        &mut self,
        py: Python<'_>,
        player: usize,
        checkpoint: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Position> {
        self.engine()?;
        let revision = self.next_revision()?;
        let snapshot = checkpoint.map(evaluations::read_snapshot).transpose()?;
        let execution = checkpoint
            .map(|d| {
                d.get_item("execution")
                    .and_then(|v| v.map(|v| v.extract::<String>()).transpose())
            })
            .transpose()?
            .flatten()
            .unwrap_or_else(|| "external".into());
        let engine = self.engine.as_mut().expect("engine checked above");
        py.detach(|| {
            if execution == "external" {
                engine.set_controller(player, snapshot)
            } else {
                engine.set_controller_mode(player, snapshot, &execution)
            }
        })
        .map_err(convert)?;
        self.revision = revision;
        Ok(self.token())
    }

    /// Read persistent settings and RNG progress; pending work stays separate.
    fn controller_checkpoint<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let engine = self.engine()?;
        let (snapshot, execution) = py
            .detach(|| Ok::<_, Error>((engine.checkpoint(player)?, engine.execution_mode(player)?)))
            .map_err(convert)?;
        snapshot
            .map(|s| {
                let out = evaluations::snapshot(py, s)?;
                out.set_item("execution", execution)?;
                Ok(out)
            })
            .transpose()
    }

    /// Simulate candidate actions for the current player and return the first
    /// evaluation batch, subject to the configured row and byte limits.
    fn prepare_evaluation<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| ClosedGameError::new_err("this game is closed"))?;
        let request = py.detach(|| engine.prepare()).map_err(convert)?;
        evaluations::request(py, request, self.token())
    }

    /// Copy scores and IDs while attached, then validate/finish in Rust with no live RNG changes.
    fn finish_evaluation<'py>(
        &mut self,
        py: Python<'py>,
        handle: &EvaluationHandle,
        values: PyReadonlyArray1<'_, f64>,
        action_ids: PyReadonlyArray1<'_, i64>,
        evaluator: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.engine()?;
        if handle.position != self.token() {
            return Err(StalePositionError::new_err("evaluation position changed"));
        }
        let (values, ids, evaluator) = evaluations::read_response(values, action_ids, evaluator)?;
        let engine = self.engine.as_mut().expect("engine checked above");
        let status = py
            .detach(|| engine.finish(&handle.key, &evaluator, &ids, &values))
            .map_err(convert)?;
        evaluations::status(py, status, self.token())
    }

    /// Check that the game, position and controller still match the request, then
    /// apply the selected action and save the RNG progress used to choose it.
    fn commit_evaluation(
        &mut self,
        py: Python<'_>,
        handle: &EvaluationHandle,
    ) -> PyResult<Position> {
        self.engine()?;
        if handle.position != self.token() {
            return Err(StalePositionError::new_err("evaluation position changed"));
        }
        let revision = self.next_revision()?;
        let engine = self.engine.as_mut().expect("engine checked above");
        py.detach(|| engine.commit(&handle.key)).map_err(convert)?;
        self.revision = revision;
        Ok(self.token())
    }

    /// Discard a pending request or ready proposal without advancing the live RNG.
    fn cancel_evaluation(&mut self, py: Python<'_>, handle: &EvaluationHandle) -> PyResult<()> {
        self.engine()?;
        if handle.position != self.token() {
            return Err(StalePositionError::new_err("evaluation position changed"));
        }
        let engine = self.engine.as_mut().expect("engine checked above");
        py.detach(|| engine.cancel(&handle.key)).map_err(convert)
    }

    /// Run Rust decisions until completion, human input, external values or the move limit.
    #[pyo3(signature = (max_moves=1024))]
    fn advance<'py>(&mut self, py: Python<'py>, max_moves: u64) -> PyResult<Bound<'py, PyDict>> {
        self.execution_budget(max_moves)?;
        let engine = self.engine.as_mut().expect("engine checked above");
        let progress = py.detach(|| engine.advance(max_moves));
        self.execution_result(py, progress)
    }

    /// Accept one batch of values. Once all candidates have values, apply the
    /// selected move and continue Rust decisions up to max_moves or the next stop.
    #[pyo3(signature = (handle, values, action_ids, evaluator, max_moves=1024))]
    fn resume<'py>(
        &mut self,
        py: Python<'py>,
        handle: &EvaluationHandle,
        values: PyReadonlyArray1<'_, f64>,
        action_ids: PyReadonlyArray1<'_, i64>,
        evaluator: &Bound<'_, PyDict>,
        max_moves: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.execution_budget(max_moves)?;
        if handle.position != self.token() {
            return Err(StalePositionError::new_err("evaluation position changed"));
        }
        let (values, ids, evaluator) = evaluations::read_response(values, action_ids, evaluator)?;
        let engine = self.engine.as_mut().expect("engine checked above");
        let progress =
            py.detach(|| engine.resume(&handle.key, &evaluator, &ids, &values, max_moves));
        self.execution_result(py, progress)
    }

    /// Release the state. Already returned arrays remain owned by Python.
    fn close(&mut self, py: Python<'_>) {
        let engine = self.engine.take();
        py.detach(move || drop(engine));
    }
}

/// Discover names from the factories that construct Python-accessible games.
#[pyfunction]
pub fn registered_games() -> Vec<&'static str> {
    crate::engine::registered_games()
}

/// Construct a game from owned, checked inputs before detaching from Python.
#[pyfunction]
#[pyo3(signature = (name, parameters=None, players=2, max_cells=4096, max_bytes=67108864))]
pub fn create(
    py: Python<'_>,
    name: String,
    parameters: Option<BTreeMap<String, usize>>,
    players: usize,
    max_cells: usize,
    max_bytes: usize,
) -> PyResult<Game> {
    let instance = instance_id()?;
    let engine = py
        .detach(move || {
            Engine::create(
                &name,
                parameters.unwrap_or_default(),
                players,
                LearningLimits {
                    max_cells,
                    max_bytes,
                },
            )
        })
        .map_err(convert)?;
    Ok(Game {
        engine: Some(engine),
        instance,
        revision: 0,
    })
}

/// Structured settings for games with arrays/nested rule parameters. Rust owns validation.
#[pyfunction]
#[pyo3(signature = (name, parameters_json, players=2, max_cells=4096, max_bytes=67108864))]
pub fn create_json(
    py: Python<'_>,
    name: String,
    parameters_json: String,
    players: usize,
    max_cells: usize,
    max_bytes: usize,
) -> PyResult<Game> {
    if parameters_json.len() > 1024 * 1024 {
        return Err(PyMemoryError::new_err("game parameters exceed 1 MiB"));
    }
    let parameters =
        serde_json::from_str(&parameters_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let instance = instance_id()?;
    let engine = py
        .detach(move || {
            Engine::create_value(
                &name,
                parameters,
                players,
                LearningLimits {
                    max_cells,
                    max_bytes,
                },
            )
        })
        .map_err(convert)?;
    Ok(Game {
        engine: Some(engine),
        instance,
        revision: 0,
    })
}
