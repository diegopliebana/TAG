//! Construct evaluation batches of one-step successor observations in action order.
//! Row and byte limits control each batch. Simulation retains the source state
//! and one temporary successor at a time.
use super::{OslaPythonConfig, OslaPythonError};
use crate::core::{Decision, ForwardModel, PlayerId, PublicInformationModel};
use crate::learning::{
    EncodedObservation, EncodingIdentity, Float32Tensor, LearningGame, LearningLimits,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationLimits {
    pub learning: LearningLimits,
    pub max_rows: usize,
    pub max_candidates: usize,
}
impl Default for EvaluationLimits {
    fn default() -> Self {
        Self {
            learning: LearningLimits::default(),
            max_rows: 1024,
            max_candidates: 4096,
        }
    }
}

/// Owned arrays for candidate rows, all observed by the original acting player.
#[derive(Clone, Debug)]
pub struct CandidateBatch {
    pub identity: EncodingIdentity,
    pub observer: PlayerId,
    pub offset: usize,
    pub total: usize,
    pub action_ids: Vec<usize>,
    /// Each tensor starts with the candidate dimension, for example `[N,3,H,W]`.
    pub payloads: Vec<Float32Tensor>,
    /// Four counters per candidate, stored consecutively with conceptual shape `[N,4]`.
    pub progress: Vec<i64>,
    /// One result code per absolute player seat for each candidate: `[N,player_count]`.
    pub results: Vec<i8>,
    /// One next-player ID per candidate, with -1 identifying game completion.
    pub next_players: Vec<i64>,
}

/// The simulation state reconstructed from a player's observation, together with
/// the available actions whose successor positions need evaluation.
pub struct EvaluationWork<F: ForwardModel> {
    pub(crate) actor: PlayerId,
    pub(crate) actions: Vec<F::Action>,
    base: F::State,
    identity: EncodingIdentity,
    ids: Vec<usize>,
    names: Vec<String>,
    limits: EvaluationLimits,
}
impl<F> EvaluationWork<F>
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    /// Validate the agent inputs and settings, then copy the simulation state
    /// and available actions so candidate evaluation can continue independently.
    pub(super) fn new(
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        config: &OslaPythonConfig,
    ) -> Result<Self, OslaPythonError> {
        config.validate()?;
        if actions.is_empty() {
            return Err(OslaPythonError::EmptyActions);
        }
        if actions.len() > config.limits.max_candidates {
            return Err(OslaPythonError::ResourceLimit);
        }
        let metadata = model.encode(observation, &[], config.limits.learning)?;
        if metadata.observer != actor || metadata.next_player != Some(actor) {
            return Err(OslaPythonError::Invalid(
                "preparation requires the acting player's observation".into(),
            ));
        }
        // Check names against the same observation before making successor copies.
        let names = config
            .payloads
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        model.encode(observation, &names, config.limits.learning)?;
        let base = model
            .simulation_state(observation)
            .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
        if model.next_player(&base) != Decision::Player(actor) {
            return Err(OslaPythonError::Invalid(
                "source decision and actor differ".into(),
            ));
        }
        let ids = actions
            .iter()
            .map(|a| model.action_id(observation, a))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            actor,
            actions: actions.to_vec(),
            base,
            identity: metadata.identity,
            ids,
            names: config.payloads.clone(),
            limits: config.limits,
        })
    }
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
    pub fn action_id(&self, index: usize) -> usize {
        self.ids[index]
    }

    /// Build a chunk starting at `offset`, capped by both row and numeric-byte limits.
    /// Every simulated action starts from base; candidate arrays own their storage.
    pub fn batch(&self, model: &F, offset: usize) -> Result<CandidateBatch, OslaPythonError> {
        if offset >= self.len() {
            return Err(OslaPythonError::Invalid(
                "candidate offset out of range".into(),
            ));
        }
        let names = self.names.iter().map(String::as_str).collect::<Vec<_>>();
        let mut batch = CandidateBatch {
            identity: self.identity.clone(),
            observer: self.actor,
            offset,
            total: self.len(),
            action_ids: Vec::new(),
            payloads: Vec::new(),
            progress: Vec::new(),
            results: Vec::new(),
            next_players: Vec::new(),
        };
        let mut bytes = 0usize;
        for candidate in offset..self.len().min(offset.saturating_add(self.limits.max_rows)) {
            let mut successor = self.base.clone();
            model
                .next(&mut successor, self.actor, &self.actions[candidate])
                .map_err(|e| OslaPythonError::Simulation(format!("candidate {candidate}: {e}")))?;
            let observation = model
                .observe(&successor, self.actor)
                .map_err(|e| OslaPythonError::Simulation(e.to_string()))?;
            let row = model.encode(&observation, &names, self.limits.learning)?;
            let row_bytes = numeric_bytes(&row)?;
            let next = bytes
                .checked_add(row_bytes)
                .ok_or(OslaPythonError::Overflow)?;
            if next > self.limits.learning.max_bytes {
                if batch.action_ids.is_empty() {
                    return Err(OslaPythonError::ResourceLimit);
                }
                break;
            }
            bytes = next;
            append(&mut batch, row)?;
            batch.action_ids.push(self.ids[candidate]);
        }
        for tensor in &mut batch.payloads {
            tensor.shape[0] = batch.action_ids.len();
        }
        Ok(batch)
    }
}

fn numeric_bytes(row: &EncodedObservation) -> Result<usize, OslaPythonError> {
    // Output includes action ID, next player, progress and result labels per row.
    let mut bytes = 8usize
        .checked_mul(6)
        .and_then(|b| b.checked_add(row.results.len()))
        .ok_or(OslaPythonError::Overflow)?;
    for tensor in &row.tensors {
        bytes = bytes
            .checked_add(
                tensor
                    .values
                    .len()
                    .checked_mul(4)
                    .ok_or(OslaPythonError::Overflow)?,
            )
            .ok_or(OslaPythonError::Overflow)?;
    }
    Ok(bytes)
}
fn append(batch: &mut CandidateBatch, row: EncodedObservation) -> Result<(), OslaPythonError> {
    if row.identity != batch.identity || row.observer != batch.observer {
        return Err(OslaPythonError::Invalid(
            "candidate encoding identity changed".into(),
        ));
    }
    if batch.action_ids.is_empty() {
        batch.payloads = row
            .tensors
            .into_iter()
            .map(|mut t| {
                t.shape.insert(0, 0);
                t
            })
            .collect();
    } else {
        if row.tensors.len() != batch.payloads.len() {
            return Err(OslaPythonError::Invalid(
                "candidate payload count changed".into(),
            ));
        }
        for (output, tensor) in batch.payloads.iter_mut().zip(row.tensors) {
            if output.name != tensor.name || output.shape[1..] != tensor.shape {
                return Err(OslaPythonError::Invalid(
                    "candidate array format changed".into(),
                ));
            }
            output.values.extend(tensor.values);
        }
    }
    batch.progress.extend(row.progress);
    batch
        .results
        .extend(row.results.into_iter().map(|r| r as i8));
    batch.next_players.push(
        row.next_player
            .map(|p| i64::try_from(p.index()))
            .transpose()
            .map_err(|_| OslaPythonError::Overflow)?
            .unwrap_or(-1),
    );
    Ok(())
}
