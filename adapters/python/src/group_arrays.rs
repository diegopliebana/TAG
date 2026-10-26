//! Concatenate compatible Rust candidate arrays and retain their original row ranges.
use crate::game::Position;
use pyo3::{prelude::*, types::PyDict};
use tag_rust::session::{EvaluationKey, EvaluationRequest};

/// Retain the original request and its row range for validation and Python display.
pub(super) struct Member {
    pub(super) index: usize,
    pub(super) position: Position,
    pub(super) key: EvaluationKey,
    pub(super) ids: Vec<usize>,
    pub(super) start: usize,
    pub(super) stop: usize,
    actor: usize,
    total: usize,
}
impl Member {
    fn new(index: usize, position: Position, request: &EvaluationRequest, start: usize) -> Self {
        Self {
            index,
            position,
            key: request.key.clone(),
            ids: request.batch.action_ids.clone(),
            start,
            stop: start + request.batch.action_ids.len(),
            actor: request.batch.observer.index(),
            total: request.batch.total,
        }
    }

    /// Give Python named fields for the original game's slice of the combined arrays.
    pub(super) fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("game_index", self.index)?;
        out.set_item("start", self.start)?;
        out.set_item("stop", self.stop)?;
        out.set_item("actor", self.actor)?;
        out.set_item("request_id", self.key.id())?;
        out.set_item("offset", self.key.offset())?;
        out.set_item("total", self.total)?;
        out.set_item("position", self.position.clone())?;
        Ok(out)
    }
}

pub(super) struct Output {
    pub(super) request: EvaluationRequest,
    pub(super) members: Vec<Member>,
    pub(super) observers: Vec<i64>,
    pub(super) indices: Vec<i64>,
}
impl Output {
    pub(super) fn compatible(&self, request: &EvaluationRequest, max_rows: usize) -> bool {
        let a = &self.request.batch;
        let b = &request.batch;
        self.request.evaluator == request.evaluator
            && a.identity == b.identity
            && a.action_ids.len() + b.action_ids.len() <= max_rows
            && a.payloads.len() == b.payloads.len()
            && a.payloads
                .iter()
                .zip(&b.payloads)
                .all(|(a, b)| a.name == b.name && a.shape[1..] == b.shape[1..])
    }
    pub(super) fn new(index: usize, position: Position, request: EvaluationRequest) -> Self {
        let rows = request.batch.action_ids.len();
        let actor = request.batch.observer.index();
        let member = Member::new(index, position, &request, 0);
        Self {
            request,
            members: vec![member],
            observers: vec![actor as i64; rows],
            indices: vec![index as i64; rows],
        }
    }
    pub(super) fn append(&mut self, index: usize, position: Position, request: EvaluationRequest) {
        let start = self.request.batch.action_ids.len();
        let rows = request.batch.action_ids.len();
        let stop = start + rows;
        let actor = request.batch.observer.index();
        self.members
            .push(Member::new(index, position, &request, start));
        self.observers
            .extend(std::iter::repeat_n(actor as i64, rows));
        self.indices.extend(std::iter::repeat_n(index as i64, rows));
        let a = &mut self.request.batch;
        let b = request.batch;
        a.action_ids.extend(b.action_ids);
        a.progress.extend(b.progress);
        a.results.extend(b.results);
        a.next_players.extend(b.next_players);
        for (a, b) in a.payloads.iter_mut().zip(b.payloads) {
            a.shape[0] = stop;
            a.values.extend(b.values);
        }
    }
}
