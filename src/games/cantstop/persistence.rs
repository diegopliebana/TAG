//! Validated portable real-game state, including exact chance continuation.
use super::*;
use crate::core::{
    ForwardModel, PlayerId,
    checkpoint::Checkpoint,
    evaluation::Evaluator,
    outcomes::SingleWinnerOutcome,
    progress::TurnCounters,
    random::{ChanceRng, ChanceRngSnapshot},
};
use crate::replay::{
    OutcomeData, PersistableGame, PersistenceError, ReplayGame, RngData, counter_data,
};
use serde::{Deserialize, Serialize};

/// Complete decoded game facts. Field values require validation before becoming
/// a usable state; RNG fields remain outside player observations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CantStopSavedState {
    pub permanent: Vec<[u32; 11]>,
    pub temporary: [Option<u32>; 11],
    pub owners: [Option<usize>; 11],
    pub dice: Option<[u8; 4]>,
    pub current_player: usize,
    pub phase: CantStopPhase,
    pub counters: [usize; 4],
    pub outcome: OutcomeData,
    pub chance: RngData,
}
fn invalid(text: impl Into<String>) -> PersistenceError {
    PersistenceError::Invalid(text.into())
}

impl PersistableGame for CantStopForwardModel {
    type SavedState = CantStopSavedState;
    const GAME_ID: &'static str = "cantstop";
    const RULES_VERSION: u32 = 1;
    fn storage_units(_: &CantStopParameters) -> Option<usize> {
        Some(44)
    }
    fn validation_work(_: &CantStopParameters, data: &CantStopSavedState) -> Option<usize> {
        data.permanent.len().checked_mul(11)?.checked_add(128)
    }
    fn state_data(&self, state: &CantStopState) -> CantStopSavedState {
        let p = state.position();
        CantStopSavedState {
            permanent: p.permanent.clone(),
            temporary: p.temporary,
            owners: p.owners.map(|p| p.map(|p| p.index())),
            dice: p.dice,
            current_player: p.current_player.index(),
            phase: p.phase,
            counters: counter_data(p.counters),
            outcome: p.status.into(),
            chance: RngData::from_snapshot(state.chance_snapshot().generator),
        }
    }
    /// Restore owned state from borrowed parameters/data after bounded invariant
    /// validation. This checks rule/phase/score/counter/RNG consistency; complete
    /// historical reachability requires an accompanying replay.
    fn restore_state(
        &self,
        parameters: &CantStopParameters,
        data: &CantStopSavedState,
    ) -> Result<CantStopState, PersistenceError> {
        let n = data.permanent.len();
        let mut state = self
            .init_state(parameters, n)
            .map_err(|e| invalid(e.to_string()))?;
        let [moves, turns, in_round, rounds] = data.counters;
        if data.current_player >= n
            || data.current_player != turns % n
            || rounds != 0
            || in_round != turns
            || moves == usize::MAX
            || turns == usize::MAX
            || turns.checked_mul(2).is_none_or(|v| v > moves)
        {
            return Err(invalid("Can't Stop actor/counters disagree"));
        }
        let counters =
            TurnCounters::from_counts(data.counters).ok_or_else(|| invalid("invalid counters"))?;
        let rng = data.chance.snapshot()?;
        let initial = ChanceRng::new(parameters.seed).snapshot().generator;
        if rng.initial_seed != parameters.seed
            || rng.stream != initial.stream
            || rng.word_position % 2 != 0
            || (moves == 0 && rng.word_position != 0)
            || (moves > 0 && rng.word_position < 8)
            || rng.word_position
                < 8 * (turns as u128 + u128::from(data.phase != CantStopPhase::StartTurn))
        {
            return Err(invalid("real-game chance seed/stream/progress disagrees"));
        }
        if data
            .dice
            .is_some_and(|d| d.iter().any(|d| !(1..=6).contains(d)))
            || (data.dice.is_none() != (moves == 0))
        {
            return Err(invalid("invalid visible dice"));
        }
        let mut claims = vec![0; n];
        let mut advances = 0u128;
        for col in 0..11 {
            let height = parameters.rules.column_heights[col];
            if let Some(owner) = data.owners[col] {
                if owner >= n {
                    return Err(invalid("unknown column owner"));
                }
                claims[owner] += 1;
                for (seat, track) in data.permanent.iter().enumerate() {
                    if track[col] != if seat == owner { height } else { 0 } {
                        return Err(invalid("claimed-column positions disagree"));
                    }
                }
                if data.temporary[col].is_some() {
                    return Err(invalid("claimed column has a temporary marker"));
                }
            } else if data.permanent.iter().any(|track| track[col] >= height) {
                return Err(invalid("unclaimed permanent progress reached the top"));
            }
            advances += data
                .permanent
                .iter()
                .map(|track| u128::from(track[col]))
                .sum::<u128>();
            if let Some(value) = data.temporary[col] {
                let banked = data.permanent[data.current_player][col];
                if value <= banked || value > height {
                    return Err(invalid("temporary progress outside its column"));
                }
                advances += u128::from(value - banked);
            }
        }
        let markers = data.temporary.iter().filter(|v| v.is_some()).count();
        if markers > parameters.rules.markers || advances > 2 * (moves as u128) {
            return Err(invalid("progress exceeds marker/action bounds"));
        }
        let won = match data.outcome {
            OutcomeData::Ongoing => None,
            OutcomeData::Won(w) if w < n && w == data.current_player => Some(w),
            _ => return Err(invalid("invalid Can't Stop outcome")),
        };
        if claims
            .iter()
            .enumerate()
            .any(|(seat, &count)| (count >= parameters.rules.columns_to_win) != (won == Some(seat)))
        {
            return Err(invalid("claimed columns disagree with outcome"));
        }
        if (matches!(data.phase, CantStopPhase::StartTurn) && markers != 0)
            || (matches!(data.phase, CantStopPhase::Decision) && markers == 0 && won.is_none())
            || (won.is_some() && (!matches!(data.phase, CantStopPhase::Decision) || markers != 0))
            || (moves == 0 && (!matches!(data.phase, CantStopPhase::StartTurn) || advances != 0))
            || (moves > 0 && turns == 0 && matches!(data.phase, CantStopPhase::StartTurn))
        {
            return Err(invalid("phase, temporary markers and outcome disagree"));
        }
        let extra = if won.is_some() {
            3
        } else {
            match data.phase {
                CantStopPhase::StartTurn => 0,
                CantStopPhase::Allocation => 1,
                CantStopPhase::Decision => 2,
            }
        };
        if turns
            .checked_mul(2)
            .and_then(|v| v.checked_add(extra))
            .is_none_or(|v| v > moves)
        {
            return Err(invalid("phase exceeds accepted actions"));
        }
        state.position = state::CantStopPosition {
            rules: parameters.rules.clone(),
            permanent: data.permanent.clone(),
            temporary: data.temporary,
            owners: data.owners.map(|p| p.map(PlayerId::new)),
            dice: data.dice,
            current_player: PlayerId::new(data.current_player),
            phase: data.phase,
            counters,
            status: won.map_or(SingleWinnerOutcome::Ongoing, |p| {
                SingleWinnerOutcome::Won(PlayerId::new(p))
            }),
        };
        state.random = ChanceRng::from_snapshot(&ChanceRngSnapshot { generator: rng })
            .map_err(|e| invalid(e.to_string()))?;
        Ok(state)
    }
}
/// Fixed default progress evaluator for the portable game's generic replay trait.
#[derive(Default)]
pub struct CantStopReplayEvaluator;
impl Checkpoint for CantStopReplayEvaluator {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) {}
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}
impl Evaluator<CantStopState> for CantStopReplayEvaluator {
    type Error = CantStopError;
    fn evaluate(&mut self, state: &CantStopState, player: PlayerId) -> Result<f64, Self::Error> {
        CantStopProgress::default().evaluate(state, player)
    }
}
impl ReplayGame for CantStopForwardModel {
    type SavedEvaluator = CantStopReplayEvaluator;
    const HAS_CHANCE: bool = true;
    fn chance_event(
        &self,
        action: &CantStopAction,
        state: &CantStopState,
    ) -> Option<serde_json::Value> {
        matches!(action, CantStopAction::Roll)
            .then(|| serde_json::json!({"dice":state.position().dice().expect("Roll has dice")}))
    }
    fn verification_work(_: &CantStopParameters, osla: bool) -> Option<usize> {
        Some(if osla { 10_000 } else { 128 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Decision, random::AgentRng};
    use crate::replay::{ReplayLimits, export_state, import_state};

    #[test]
    fn portable_positions_preserve_all_phases_and_future_rolls() {
        let model = CantStopForwardModel;
        let mut phases = [false; 3];
        let mut finished = 0;
        for players in 2..=4 {
            for seed in [0, 42, u64::MAX] {
                let parameters = CantStopParameters {
                    seed,
                    ..Default::default()
                };
                let mut state = model.init_state(&parameters, players).unwrap();
                let mut choices = AgentRng::new(seed);
                for _ in 0..2000 {
                    phases[match state.position().phase() {
                        CantStopPhase::StartTurn => 0,
                        CantStopPhase::Allocation => 1,
                        CantStopPhase::Decision => 2,
                    }] = true;
                    let text = export_state(&model, &parameters, &state).unwrap();
                    let (p, mut restored) =
                        import_state(&text, &model, ReplayLimits::default()).unwrap();
                    assert_eq!(p, parameters);
                    assert_eq!(restored, state);
                    let Decision::Player(actor) = model.next_player(&state) else {
                        finished += 1;
                        break;
                    };
                    let actions = model.available_actions(&state, actor).unwrap();
                    let action = &actions[choices.index(actions.len()).unwrap()];
                    model.next(&mut state, actor, action).unwrap();
                    model.next(&mut restored, actor, action).unwrap();
                    assert_eq!(state, restored);
                }
            }
        }
        assert_eq!(phases, [true; 3]);
        assert_eq!(finished, 9);
    }

    #[test]
    fn portable_validation_rejects_inconsistent_facts_and_private_simulation_streams() {
        let model = CantStopForwardModel;
        let p = CantStopParameters::default();
        let state = model.init_state(&p, 3).unwrap();
        let clean = model.state_data(&state);
        let mut invalids = Vec::new();
        let mut d = clean.clone();
        d.current_player = 1;
        invalids.push(d);
        let mut d = clean.clone();
        d.dice = Some([0, 1, 2, 3]);
        invalids.push(d);
        let mut d = clean.clone();
        d.owners[0] = Some(3);
        invalids.push(d);
        let mut d = clean.clone();
        d.permanent[0][0] = 3;
        invalids.push(d);
        let mut d = clean.clone();
        d.temporary[0] = Some(0);
        invalids.push(d);
        let mut d = clean.clone();
        d.phase = CantStopPhase::Decision;
        invalids.push(d);
        let mut d = clean.clone();
        d.outcome = OutcomeData::Won(0);
        invalids.push(d);
        let mut d = clean.clone();
        d.chance.initial_seed = "1".into();
        invalids.push(d);
        let mut d = clean.clone();
        d.chance.stream = "0".into();
        invalids.push(d);
        let mut d = clean.clone();
        d.chance.word_position = "1".into();
        invalids.push(d);
        for data in invalids {
            assert!(model.restore_state(&p, &data).is_err(), "{data:?}");
        }
        assert_eq!(model.restore_state(&p, &clean).unwrap(), state);
        let large = CantStopParameters {
            rules: CantStopRules {
                column_heights: [u32::MAX; 11],
                ..Default::default()
            },
            seed: u64::MAX,
        };
        let state = model.init_state(&large, 4).unwrap();
        assert_eq!(
            model
                .restore_state(&large, &model.state_data(&state))
                .unwrap(),
            state
        );
        assert_eq!(CantStopForwardModel::storage_units(&large), Some(44));
    }
}
