//! Run independent episode operations in one detached Rust call.
use crate::{
    episode::{Output, Runtime},
    episode_arrays,
    episode_binding::Episode,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::sync::Arc;

/// Retain original native episodes; the coordinator never closes its members.
#[pyclass(module = "tag_rust._engine")]
pub struct EpisodeBatch {
    episodes: Vec<Py<Episode>>,
    owner: Arc<()>,
    max_bytes: usize,
    closed: bool,
}
struct Replacement {
    index: usize,
    runtime: Option<Runtime>,
    generation: u64,
}
/// Independent fresh candidates, committed only for successfully formatted rows.
#[pyclass(module = "tag_rust._engine")]
pub struct BatchResetCandidate {
    replacements: Vec<Replacement>,
    owner: Arc<()>,
    #[pyo3(get)]
    output: Py<PyDict>,
}
fn borrow<'a>(py: Python<'a>, episodes: &'a [Py<Episode>]) -> PyResult<Vec<PyRefMut<'a, Episode>>> {
    episodes
        .iter()
        .map(|e| e.bind(py).try_borrow_mut().map_err(PyErr::from))
        .collect()
}
fn selection(indices: &[usize], count: usize) -> PyResult<()> {
    for (offset, &index) in indices.iter().enumerate() {
        if index >= count || indices[..offset].contains(&index) {
            return Err(PyValueError::new_err(
                "indices must be distinct batch member indices",
            ));
        }
    }
    Ok(())
}
fn successful_status(runtime: &Runtime) -> String {
    if runtime.ended() { "ended" } else { "ready" }.into()
}
/// Execute already validated choices without Python objects or callbacks.
/// Each Result belongs to one independent member, so errors do not stop peers.
fn execute_steps(
    episodes: &mut [&mut Episode],
    actions: &[Option<usize>],
    teacher: bool,
) -> (Vec<Option<Output>>, Vec<Option<String>>) {
    let mut rows = Vec::with_capacity(episodes.len());
    let mut errors = Vec::with_capacity(episodes.len());
    for (member, action) in episodes.iter_mut().zip(actions) {
        match action {
            None => {
                rows.push(None);
                errors.push(None);
            }
            Some(id) => match if teacher {
                member.run_teacher()
            } else {
                member.run_step(*id)
            } {
                Ok(out) => {
                    rows.push(Some(out));
                    errors.push(None);
                }
                Err(error) => {
                    rows.push(None);
                    errors.push(Some(error));
                }
            },
        }
    }
    (rows, errors)
}

impl EpisodeBatch {
    fn open(&self) -> PyResult<()> {
        if self.closed {
            Err(PyRuntimeError::new_err("episode batch is closed"))
        } else {
            Ok(())
        }
    }
    fn references(&self, py: Python<'_>) -> Vec<Py<Episode>> {
        self.episodes.iter().map(|e| e.clone_ref(py)).collect()
    }
    fn capacity(&self, episodes: &[PyRefMut<'_, Episode>]) -> PyResult<()> {
        episode_arrays::capacity(
            &episodes[0].description,
            &episodes[0].config.payloads,
            episodes.len(),
            self.max_bytes,
            episodes.iter().any(|e| e.config.include_intermediate),
        )
    }
    fn execute<'py>(
        &self,
        py: Python<'py>,
        actions: Vec<Option<usize>>,
        teacher: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.open()?;
        let owned = self.references(py);
        let mut members = borrow(py, &owned)?;
        if actions.len() != members.len() {
            return Err(PyValueError::new_err(
                "actions must have one entry per batch member",
            ));
        }
        self.capacity(&members)?;
        let mut refs = members.iter_mut().map(|g| &mut **g).collect::<Vec<_>>();
        py.detach(|| {
            for (member, action) in refs.iter().zip(&actions) {
                if let Some(action) = action {
                    if teacher {
                        member.validate_teacher()?;
                    } else {
                        member.validate_step(*action)?;
                    }
                }
            }
            Ok::<_, PyErr>(())
        })?;
        let (rows, errors) = py.detach(|| execute_steps(&mut refs, &actions, teacher));
        let successful = rows.iter().map(Option::is_some).collect::<Vec<_>>();
        let statuses = refs
            .iter()
            .zip(&successful)
            .map(|(member, &ok)| {
                if ok {
                    successful_status(member.runtime.as_ref().expect("executed"))
                } else {
                    member.status().into()
                }
            })
            .collect();
        let out = episode_arrays::convert(
            py,
            &refs[0].description,
            &refs[0].config.payloads,
            rows,
            errors,
            statuses,
            refs.iter().any(|e| e.config.include_intermediate),
        )?;
        for (member, ok) in refs.iter_mut().zip(successful) {
            if ok {
                member.failed = false;
            }
        }
        Ok(out)
    }
}
#[pymethods]
impl EpisodeBatch {
    /// Retain original episodes after checking identity, layout and output bounds.
    ///
    /// # Arguments
    /// `py` permits temporary Python borrows. `episodes` transfers owned reference
    /// handles, not ownership of the callers' game objects. `max_bytes` is the
    /// copied numeric-output limit for each bulk result.
    ///
    /// # Returns
    /// An owned coordinator retaining independently owned member games.
    ///
    /// # Errors
    /// Reject empty/oversized/duplicate collections, borrowing conflicts, closed
    /// members, incompatible descriptions or excessive numeric output bounds.
    #[new]
    #[pyo3(signature=(episodes,max_bytes=67108864))]
    fn new(py: Python<'_>, episodes: Vec<Py<Episode>>, max_bytes: usize) -> PyResult<Self> {
        if episodes.is_empty() || episodes.len() > 256 {
            return Err(PyValueError::new_err(
                "batch requires 1 to 256 distinct episodes",
            ));
        }
        for (index, e) in episodes.iter().enumerate() {
            if episodes[..index]
                .iter()
                .any(|other| other.as_ptr() == e.as_ptr())
            {
                return Err(PyValueError::new_err("duplicate episode"));
            }
        }
        let members = borrow(py, &episodes)?;
        let first = &members[0];
        for member in &members {
            member.open()?;
            if member.description != first.description
                || member.config.payloads != first.config.payloads
            {
                return Err(PyValueError::new_err(
                    "batch requires matching encoding identities and ordered payload layouts",
                ));
            }
        }
        episode_arrays::capacity(
            &first.description,
            &first.config.payloads,
            members.len(),
            max_bytes,
            members.iter().any(|e| e.config.include_intermediate),
        )?;
        drop(members);
        Ok(Self {
            episodes,
            owner: Arc::new(()),
            max_bytes,
            closed: false,
        })
    }

    /// Validate every selected `actions` entry before running any member. Each
    /// runtime error belongs to its member; aggregate conversion failure leaves
    /// all executed members failed so no missing output can become a transition.
    ///
    /// # Arguments
    /// `py` permits exclusive member borrows and output conversion. `actions`
    /// transfers one optional ID per member; None means inactive.
    ///
    /// # Returns
    /// Owned stacked arrays, validity flags, statuses and per-member metadata/errors.
    ///
    /// # Errors
    /// Invalid whole-call input advances nothing. Later member errors appear in
    /// the result; an aggregate conversion error leaves executed members failed.
    fn step<'py>(
        &self,
        py: Python<'py>,
        actions: Vec<Option<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.execute(py, actions, false)
    }
    fn step_teacher<'py>(
        &self,
        py: Python<'py>,
        active: Vec<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.execute(
            py,
            active
                .into_iter()
                .map(|enabled| enabled.then_some(0))
                .collect(),
            true,
        )
    }

    /// Create selected fresh candidates without replacing live episodes. Input
    /// errors reject the whole call; later failures are recorded per member.
    ///
    /// # Arguments
    /// `py` permits temporary member borrows. `indices` transfers the selected
    /// distinct indices; `seeds` transfers one optional derived opponent seed per
    /// index. None records a wrapper seed-staging failure and skips that candidate.
    ///
    /// # Returns
    /// An owned candidate containing fresh runtimes and independently owned output.
    ///
    /// # Errors
    /// Invalid selection/lifecycle/limits reject the call. Runtime member errors
    /// are returned by row. Aggregate conversion errors replace no live episode.
    #[pyo3(signature=(indices,seeds,teacher_seeds=None))]
    fn prepare_resets(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        seeds: Vec<Option<u64>>,
        teacher_seeds: Option<Vec<Option<u64>>>,
    ) -> PyResult<BatchResetCandidate> {
        self.open()?;
        let owned = self.references(py);
        let mut members = borrow(py, &owned)?;
        selection(&indices, members.len())?;
        if indices.len() != seeds.len() {
            return Err(PyValueError::new_err(
                "one seed is required per selected member",
            ));
        }
        self.capacity(&members)?;
        let teacher_seeds = teacher_seeds.unwrap_or_else(|| vec![None; indices.len()]);
        if teacher_seeds.len() != indices.len() {
            return Err(PyValueError::new_err(
                "one teacher seed per selected index is required",
            ));
        }
        for (&i, &teacher_seed) in indices.iter().zip(&teacher_seeds) {
            if members[i].config.teacher != teacher_seed.is_some() {
                return Err(PyValueError::new_err("teacher seed/configuration mismatch"));
            }
        }
        for &i in &indices {
            members[i].open()?;
            members[i]
                .generation
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("episode generation exhausted"))?;
        }
        let refs = members.iter_mut().map(|g| &mut **g).collect::<Vec<_>>();
        let mut rows = (0..refs.len())
            .map(|_| None)
            .collect::<Vec<Option<Output>>>();
        let mut errors = vec![None; refs.len()];
        let mut replacements = Vec::new();
        let mut statuses = refs
            .iter()
            .map(|e| e.status().to_owned())
            .collect::<Vec<_>>();
        py.detach(|| {
            for ((&index, seed), teacher_seed) in indices.iter().zip(seeds).zip(teacher_seeds) {
                // A wrapper seed-staging failure has no candidate. Keep this
                // member in whole-call lifecycle validation but skip execution.
                let Some(seed) = seed else {
                    continue;
                };
                let member = &refs[index];
                let result = (|| {
                    let mut runtime = member.config.create_seeded(seed, teacher_seed)?;
                    let output = runtime.opening()?;
                    Ok::<_, String>((runtime, output))
                })();
                match result {
                    Ok((runtime, output)) => {
                        statuses[index] = successful_status(&runtime);
                        rows[index] = Some(output);
                        replacements.push(Replacement {
                            index,
                            runtime: Some(runtime),
                            generation: member.generation,
                        });
                    }
                    Err(error) => errors[index] = Some(error),
                }
            }
        });
        let output = episode_arrays::convert(
            py,
            &refs[0].description,
            &refs[0].config.payloads,
            rows,
            errors,
            statuses,
            refs.iter().any(|e| e.config.include_intermediate),
        )?
        .unbind();
        Ok(BatchResetCandidate {
            replacements,
            owner: Arc::clone(&self.owner),
            output,
        })
    }

    /// Commit only `indices` with fully constructed outputs. Check all ownership
    /// and generations first, so a stale member cannot partially commit a reset.
    ///
    /// # Arguments
    /// `py` permits exclusive member borrows. `candidate` borrows exclusive access
    /// to staged runtimes; `indices` transfers the subset with complete outputs.
    ///
    /// # Returns
    /// Unit after all selected validated replacements have been installed.
    ///
    /// # Errors
    /// Foreign, stale, consumed or closed selections leave every member unchanged.
    fn commit_resets(
        &self,
        py: Python<'_>,
        candidate: &mut BatchResetCandidate,
        indices: Vec<usize>,
    ) -> PyResult<()> {
        self.open()?;
        if !Arc::ptr_eq(&self.owner, &candidate.owner) {
            return Err(PyValueError::new_err("foreign reset candidates"));
        }
        let owned = self.references(py);
        let mut members = borrow(py, &owned)?;
        selection(&indices, members.len())?;
        for &index in &indices {
            members[index].open()?;
            let replacement = candidate
                .replacements
                .iter()
                .find(|r| r.index == index && r.runtime.is_some())
                .ok_or_else(|| PyValueError::new_err("reset candidate consumed or missing"))?;
            if replacement.generation != members[index].generation {
                return Err(PyValueError::new_err("stale reset candidate"));
            }
        }
        for index in indices {
            let replacement = candidate
                .replacements
                .iter_mut()
                .find(|r| r.index == index)
                .expect("validated");
            let member = &mut members[index];
            member.runtime = replacement.runtime.take();
            member.generation += 1;
            member.failed = false;
        }
        Ok(())
    }

    /// Python output formatting can fail after native execution. Mark exactly
    /// these completed step members failed; reset failures need no state change.
    ///
    /// # Arguments
    /// `py` permits exclusive member borrows; `indices` transfers exactly those
    /// executed members whose Python result could not be returned.
    ///
    /// # Returns
    /// Unit after marking those members failed, without undoing accepted moves.
    ///
    /// # Errors
    /// A closed coordinator, invalid index or borrowing conflict changes no member.
    fn fail(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<()> {
        self.open()?;
        selection(&indices, self.episodes.len())?;
        let mut members = borrow(py, &self.episodes)?;
        for index in indices {
            members[index].failed = true;
        }
        Ok(())
    }
    fn statuses(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.open()?;
        Ok(borrow(py, &self.episodes)?
            .iter()
            .map(|e| e.status().into())
            .collect())
    }
    fn close(&mut self) {
        self.episodes.clear();
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{Config, Runtime};
    use std::collections::BTreeMap;
    use tag_rust::learning::LearningLimits;

    #[test]
    fn runtime_failure_retains_partial_moves_and_does_not_stop_a_peer() {
        let config = Config {
            name: "tictactoe".into(),
            parameters: BTreeMap::new(),
            learner: 0,
            opponent: "first_available".into(),
            settings: BTreeMap::new(),
            payloads: vec!["cells".into()],
            horizon: None,
            limits: LearningLimits::default(),
            include_intermediate: false,
            teacher: false,
        };
        let mut a = crate::episode_binding::test_episode(config.clone());
        let mut b = crate::episode_binding::test_episode(config);
        a.runtime.as_mut().unwrap().inject_output_failure();
        let (rows, errors) = execute_steps(&mut [&mut a, &mut b], &[Some(4), Some(4)], false);
        assert!(errors[0].is_some());
        assert!(rows[0].is_none());
        assert!(errors[1].is_none());
        assert_eq!(rows[1].as_ref().unwrap().moves, 2);
        let Runtime::TicTacToe(game) = a.runtime.as_ref().unwrap() else {
            unreachable!()
        };
        assert_eq!(game.moves, 2);
        assert!(a.failed);
        // The peer also remains guarded until its complete Python output exists.
        assert!(b.failed);
    }
}
