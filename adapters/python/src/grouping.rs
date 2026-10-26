//! Coordinate original game objects and combine compatible candidate chunks in Rust.
use crate::group_arrays::{Member, Output};
use crate::{
    engine::Engine,
    evaluations,
    execution::{Progress, Stop},
    game::{EvaluationError, Game, StalePositionError, convert},
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyMemoryError, PyOverflowError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::{collections::BTreeMap, sync::Arc};
use tag_rust::agents::EvaluatorIdentity;

#[pyclass(frozen, skip_from_py_object, module = "tag_rust._engine")]
#[derive(Clone)]
pub struct EvaluationGroupHandle {
    owner: Arc<()>,
    #[pyo3(get)]
    id: u64,
}
struct Pending {
    evaluator: EvaluatorIdentity,
    members: Vec<Member>,
    rows: usize,
}
/// Own references to original games. Mutable borrow guards prevent concurrent
/// Python access while Rust runs detached; games remain independently inspectable.
#[pyclass(module = "tag_rust._engine")]
pub struct GameBatch {
    games: Vec<Py<Game>>,
    owner: Arc<()>,
    next_id: u64,
    pending: BTreeMap<u64, Pending>,
    max_rows: usize,
    max_bytes: usize,
    closed: bool,
}
impl GameBatch {
    fn game_references(&self, py: Python<'_>) -> Vec<Py<Game>> {
        // Independent reference handles let output update coordinator bookkeeping
        // while the original games remain exclusively borrowed.
        self.games.iter().map(|game| game.clone_ref(py)).collect()
    }

    fn check_open(&self) -> PyResult<()> {
        if self.closed {
            return Err(EvaluationError::new_err("this GameBatch is closed"));
        }
        Ok(())
    }
    fn lookup(&self, handle: &EvaluationGroupHandle) -> PyResult<&Pending> {
        self.check_open()?;
        if !Arc::ptr_eq(&self.owner, &handle.owner) {
            return Err(StalePositionError::new_err(
                "group belongs to another coordinator",
            ));
        }
        self.pending
            .get(&handle.id)
            .ok_or_else(|| StalePositionError::new_err("group was consumed or cancelled"))
    }
    fn preflight(
        &self,
        games: &[PyRefMut<'_, Game>],
        indices: &[usize],
        limit: u64,
    ) -> PyResult<()> {
        self.next_id
            .checked_add(indices.len() as u64)
            .ok_or_else(|| PyOverflowError::new_err("group IDs exhausted"))?;
        let mut bytes = 0usize;
        for &index in indices {
            games[index].execution_budget(limit)?;
            let (rows, size) = games[index].engine()?.group_capacity().map_err(convert)?;
            if rows > self.max_rows {
                return Err(PyMemoryError::new_err(
                    "a controller chunk can exceed max_group_rows; reduce its max_rows",
                ));
            }
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| PyOverflowError::new_err("group byte count overflow"))?;
        }
        if bytes > self.max_bytes {
            return Err(PyMemoryError::new_err(
                "possible returned arrays exceed GameBatch max_bytes; reduce controller max_rows or game count",
            ));
        }
        Ok(())
    }
    fn output<'py>(
        &mut self,
        py: Python<'py>,
        games: &[PyRefMut<'_, Game>],
        results: Vec<(usize, Progress)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut groups: Vec<Output> = Vec::new();
        let mut stops = Vec::new();
        for (index, progress) in results {
            let position = games[index].token();
            let stop = PyDict::new(py);
            stop.set_item("game_index", index)?;
            stop.set_item("moves", progress.moves)?;
            stop.set_item("position", position.clone())?;
            let (reason, actor, error) = match progress.stop {
                Ok(Stop::Finished) => ("finished", None, None),
                Ok(Stop::Human(a)) => ("human", Some(a), None),
                Ok(Stop::MoveLimit(a)) => ("move_limit", Some(a), None),
                Err(e) => ("error", None, Some(e.to_string())),
                Ok(Stop::Evaluation(request)) => {
                    let actor = request.batch.observer.index();
                    if let Some(group) = groups
                        .iter_mut()
                        .find(|g| g.compatible(&request, self.max_rows))
                    {
                        group.append(index, position, *request);
                    } else {
                        groups.push(Output::new(index, position, *request));
                    }
                    ("evaluation", Some(actor), None)
                }
            };
            stop.set_item("reason", reason)?;
            stop.set_item("actor", actor)?;
            stop.set_item("error", error)?;
            stops.push(stop);
        }
        let mut converted = Vec::new();
        for group in groups {
            let handle = EvaluationGroupHandle {
                owner: self.owner.clone(),
                id: self.next_id,
            };
            self.next_id += 1;
            let rows = group.request.batch.action_ids.len();
            let evaluator = group.request.evaluator.clone();
            let out = evaluations::batch_arrays(py, group.request.batch, group.request.evaluator)?;
            out.set_item("handle", handle.clone())?;
            out.set_item("observers", PyArray1::from_vec(py, group.observers))?;
            out.set_item("game_indices", PyArray1::from_vec(py, group.indices))?;
            let segments = group
                .members
                .iter()
                .map(|member| member.metadata(py))
                .collect::<PyResult<Vec<_>>>()?;
            out.set_item("segments", segments)?;
            self.pending.insert(
                handle.id,
                Pending {
                    evaluator,
                    members: group.members,
                    rows,
                },
            );
            converted.push(out);
        }
        let out = PyDict::new(py);
        out.set_item("groups", converted)?;
        out.set_item("stops", stops)?;
        Ok(out)
    }
}
#[pymethods]
impl GameBatch {
    #[new]
    #[pyo3(signature=(games,max_group_rows=4096,max_bytes=67108864))]
    fn new(
        py: Python<'_>,
        games: Vec<Py<Game>>,
        max_group_rows: usize,
        max_bytes: usize,
    ) -> PyResult<Self> {
        if games.is_empty() || games.len() > 256 || max_group_rows == 0 || max_bytes == 0 {
            return Err(PyValueError::new_err(
                "require 1..256 games and positive row/byte limits",
            ));
        }
        for (i, g) in games.iter().enumerate() {
            if games[..i].iter().any(|other| other.as_ptr() == g.as_ptr()) {
                return Err(PyValueError::new_err("duplicate game instance"));
            }
            let game = g.try_borrow(py)?;
            if game.engine()?.pending() {
                return Err(EvaluationError::new_err(
                    "finish or cancel existing game work first",
                ));
            }
        }
        Ok(Self {
            games,
            owner: Arc::new(()),
            next_id: 1,
            pending: BTreeMap::new(),
            max_rows: max_group_rows,
            max_bytes,
            closed: false,
        })
    }
    #[getter]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    #[pyo3(signature=(max_moves_per_game=1024))]
    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        max_moves_per_game: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check_open()?;
        if !self.pending.is_empty() {
            return Err(EvaluationError::new_err(
                "resume or cancel pending groups before advance",
            ));
        }
        let owned = self.game_references(py);
        let mut games = borrow_games(py, &owned)?;
        let indices = (0..games.len()).collect::<Vec<_>>();
        self.preflight(&games, &indices, max_moves_per_game)?;
        if games.iter().any(|g| g.engine().is_ok_and(Engine::pending)) {
            return Err(EvaluationError::new_err("a game already has pending work"));
        }
        let mut refs = games.iter_mut().map(|g| &mut **g).collect::<Vec<_>>();
        let results = py.detach(|| {
            refs.iter_mut()
                .enumerate()
                .map(|(i, g)| {
                    let progress = g
                        .engine
                        .as_mut()
                        .expect("preflight")
                        .advance(max_moves_per_game);
                    g.revision += progress.moves;
                    (i, progress)
                })
                .collect()
        });
        self.output(py, &games, results)
    }

    #[pyo3(signature=(handle,values,action_ids,evaluator,max_moves_per_game=1024))]
    fn resume<'py>(
        &mut self,
        py: Python<'py>,
        handle: &EvaluationGroupHandle,
        values: PyReadonlyArray1<'_, f64>,
        action_ids: PyReadonlyArray1<'_, i64>,
        evaluator: &Bound<'_, PyDict>,
        max_moves_per_game: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pending = self.lookup(handle)?;
        if values.len()? != pending.rows || action_ids.len()? != pending.rows {
            return Err(PyValueError::new_err("group response length differs"));
        }
        let (values, ids, evaluator) = evaluations::read_response(values, action_ids, evaluator)?;
        if evaluator != pending.evaluator {
            return Err(PyValueError::new_err("group evaluator differs"));
        }
        let owned = self.game_references(py);
        let mut games = borrow_games(py, &owned)?;
        let indices = pending.members.iter().map(|m| m.index).collect::<Vec<_>>();
        self.preflight(&games, &indices, max_moves_per_game)?;
        // Validate every original request before storing a single score.
        for member in &pending.members {
            let game = &games[member.index];
            if game.token() != member.position {
                return Err(StalePositionError::new_err(
                    "a grouped game position changed",
                ));
            }
            if ids[member.start..member.stop] != member.ids {
                return Err(PyValueError::new_err("group action IDs or order differ"));
            }
            game.engine()?
                .validate_reply(
                    &member.key,
                    &evaluator,
                    &member.ids,
                    &values[member.start..member.stop],
                )
                .map_err(convert)?;
        }
        let pending = self.pending.remove(&handle.id).expect("group checked");
        let mut refs = games.iter_mut().map(|g| &mut **g).collect::<Vec<_>>();
        let results = py.detach(|| {
            pending
                .members
                .into_iter()
                .map(|member| {
                    let game = &mut refs[member.index];
                    let engine = game.engine.as_mut().expect("preflight");
                    let progress = engine.resume(
                        &member.key,
                        &evaluator,
                        &member.ids,
                        &values[member.start..member.stop],
                        max_moves_per_game,
                    );
                    if progress.stop.is_err() {
                        let _ = engine.cancel(&member.key);
                    }
                    game.revision += progress.moves;
                    (member.index, progress)
                })
                .collect()
        });
        self.output(py, &games, results)
    }

    /// Cancel matching outstanding work; changed/closed games keep their newer work.
    fn cancel(&mut self, py: Python<'_>, handle: &EvaluationGroupHandle) -> PyResult<()> {
        self.lookup(handle)?;
        let owned = self.game_references(py);
        let mut games = borrow_games(py, &owned)?;
        let pending = self.pending.remove(&handle.id).expect("group checked");
        cancel_members(&mut games, pending.members);
        Ok(())
    }
    fn cancel_all(&mut self, py: Python<'_>) -> PyResult<()> {
        self.check_open()?;
        if self.pending.is_empty() {
            return Ok(());
        }
        let owned = self.game_references(py);
        let mut games = borrow_games(py, &owned)?;
        // Acquire every game first, then discard all groups in one pass.
        for pending in std::mem::take(&mut self.pending).into_values() {
            cancel_members(&mut games, pending.members);
        }
        Ok(())
    }
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.closed {
            self.cancel_all(py)?;
            self.games.clear();
            self.closed = true;
        }
        Ok(())
    }
}

/// Borrow all games before changing any of them. A conflicting Python borrow
/// therefore fails before accepting scores or discarding pending work.
fn borrow_games<'py>(
    py: Python<'py>,
    games: &'py [Py<Game>],
) -> PyResult<Vec<PyRefMut<'py, Game>>> {
    games
        .iter()
        .map(|game| game.try_borrow_mut(py).map_err(PyErr::from))
        .collect()
}

fn cancel_members(games: &mut [PyRefMut<'_, Game>], members: Vec<Member>) {
    for member in members {
        let game = &mut games[member.index];
        if game.token() == member.position
            && let Some(engine) = game.engine.as_mut()
        {
            // An obsolete key must leave any replacement request alone.
            let _ = engine.cancel(&member.key);
        }
    }
}
