//! Checked rules for rolling, allocation, banking and busting.
use super::{
    Allocation, COLUMN_COUNT, CantStopAction as Action, CantStopError as Error,
    CantStopObservation, CantStopParameters, CantStopPhase as Phase, CantStopPosition,
    CantStopState,
};
use crate::core::{
    Decision, ForwardModel, PlayerId, PlayerResult,
    outcomes::SingleWinnerOutcome,
    players::validate_player,
    progress::TurnCounters,
    random::{ChanceRng, SimulationRng},
    simulation::SimulationModel,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct CantStopForwardModel;

/// Realized chance event returned by native recording/controlled-roll operations.
/// A dice event is part of Roll's transition, not an additional player action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RollEvent {
    pub dice: [u8; 4],
}

impl CantStopForwardModel {
    /// Apply a normal action and return its dice event when it rolls.
    pub fn next_recorded(
        &self,
        state: &mut CantStopState,
        actor: PlayerId,
        action: &Action,
    ) -> Result<Option<RollEvent>, Error> {
        self.apply(state, actor, action, None)
    }

    /// Apply Roll with supplied outcomes for controlled tests/event replay.
    /// The generator still advances through four ordinary die draws, just as a
    /// normal Roll would. Only the realized values are replaced. Invalid input
    /// changes neither the position nor the generator; queries never draw dice.
    pub fn next_with_roll(
        &self,
        state: &mut CantStopState,
        actor: PlayerId,
        dice: [u8; 4],
    ) -> Result<RollEvent, Error> {
        if dice.iter().any(|n| !(1..=6).contains(n)) {
            return Err(Error::InvalidRoll(dice));
        }
        Ok(self
            .apply(state, actor, &Action::Roll, Some(dice))?
            .expect("Roll produces dice"))
    }

    fn apply(
        &self,
        state: &mut CantStopState,
        actor: PlayerId,
        action: &Action,
        supplied: Option<[u8; 4]>,
    ) -> Result<Option<RollEvent>, Error> {
        let p = &state.position;
        validate_player(actor, p.player_count())?;
        if p.status != SingleWinnerOutcome::Ongoing {
            return Err(Error::Finished);
        }
        if actor != p.current_player {
            return Err(Error::WrongPlayer {
                expected: p.current_player,
                actual: actor,
            });
        }
        if !actions(p).contains(action) {
            return Err(Error::IllegalAction {
                action: *action,
                phase: p.phase,
            });
        }
        if p.counters.move_count() == usize::MAX
            || (matches!(action, Action::Stop | Action::Bust)
                && p.counters.completed_turn_count() == usize::MAX)
        {
            return Err(Error::CounterOverflow);
        }
        // All external-input failures have been checked before the first mutation.
        let p = &mut state.position;
        let mut event = None;
        match action {
            Action::Roll => {
                let drawn = std::array::from_fn(|_| {
                    (state.random.index(6).expect("six outcomes") + 1) as u8
                });
                let dice = supplied.unwrap_or(drawn);
                p.dice = Some(dice);
                p.phase = Phase::Allocation;
                event = Some(RollEvent { dice });
            }
            Action::Allocate(allocation) => {
                for column in allocation.columns() {
                    let n = (*column - 2) as usize;
                    let before = p.temporary[n].unwrap_or(p.permanent[actor.index()][n]);
                    p.temporary[n] = Some(before + 1);
                }
                p.phase = Phase::Decision;
            }
            Action::Stop => {
                // Commit all selected columns before testing victory. Temporary
                // arrival at the top never closes a column until this action.
                for n in 0..COLUMN_COUNT {
                    if let Some(value) = p.temporary[n] {
                        p.permanent[actor.index()][n] = value;
                        if value == p.rules.column_heights[n] {
                            p.owners[n] = Some(actor);
                            // Published rules remove opponents' markers from
                            // a claimed column, unlike Java's retained arrays.
                            for player in 0..p.player_count() {
                                if player != actor.index() {
                                    p.permanent[player][n] = 0;
                                }
                            }
                        }
                    }
                }
                p.temporary = [None; COLUMN_COUNT];
                if p.owners
                    .iter()
                    .filter(|owner| **owner == Some(actor))
                    .count()
                    >= p.rules.columns_to_win
                {
                    p.status = SingleWinnerOutcome::Won(actor);
                } else {
                    end_turn(p);
                }
            }
            Action::Bust => {
                p.temporary = [None; COLUMN_COUNT];
                end_turn(p);
            }
        }
        p.counters.record_move();
        Ok(event)
    }
}

fn end_turn(p: &mut CantStopPosition) {
    p.counters.complete_turn();
    p.current_player = PlayerId::new((p.current_player.index() + 1) % p.player_count());
    p.phase = Phase::StartTurn;
    // As in the reference, cycling players is not itself a rules-defined round.
}

/// True only when every sum can be used with sufficient temporary markers.
fn legal(p: &CantStopPosition, allocation: Allocation) -> bool {
    let mut advances = [0u32; COLUMN_COUNT];
    for column in allocation.columns() {
        advances[(*column - 2) as usize] += 1;
    }
    let mut markers = p.temporary.iter().filter(|v| v.is_some()).count();
    for (n, &count) in advances.iter().enumerate() {
        if count == 0 {
            continue;
        }
        if p.owners[n].is_some() {
            return false;
        }
        let position = p.temporary[n].unwrap_or(p.permanent[p.current_player.index()][n]);
        // Subtraction avoids overflow at user-configured u32::MAX heights.
        if p.rules.column_heights[n] - position < count {
            return false;
        }
        if p.temporary[n].is_none() {
            markers += 1;
        }
    }
    markers <= p.rules.markers
}

pub(super) fn actions(p: &CantStopPosition) -> Vec<Action> {
    if p.status != SingleWinnerOutcome::Ongoing {
        return vec![];
    }
    match p.phase {
        Phase::StartTurn => vec![Action::Roll],
        Phase::Decision => vec![Action::Stop, Action::Roll],
        Phase::Allocation => {
            let [a, b, c, d] = p.dice.expect("allocation follows a checked Roll");
            let mut full = Vec::new();
            let mut single = Vec::new();
            // Stable pairing order; do not confuse canonical pair identity with
            // sorting the list of legal choices. Both sums must be used if legal.
            for (first, second) in [(a + b, c + d), (a + c, b + d), (a + d, b + c)] {
                let pair = Allocation::pair(first, second).expect("die sums are columns");
                if legal(p, pair) {
                    let action = Action::Allocate(pair);
                    if !full.contains(&action) {
                        full.push(action);
                    }
                } else {
                    for column in [first, second] {
                        let allocation = Allocation::single(column).expect("die sum is a column");
                        let action = Action::Allocate(allocation);
                        if legal(p, allocation) && !single.contains(&action) {
                            single.push(action);
                        }
                    }
                }
            }
            full.extend(single);
            if full.is_empty() {
                full.push(Action::Bust);
            }
            full
        }
    }
}

impl ForwardModel for CantStopForwardModel {
    type State = CantStopState;
    type Observation = CantStopObservation;
    type Action = Action;
    type Parameters = CantStopParameters;
    type Error = Error;
    fn init_state(
        &self,
        parameters: &CantStopParameters,
        player_count: usize,
    ) -> Result<CantStopState, Error> {
        parameters.rules.validate(player_count)?;
        Ok(CantStopState {
            position: CantStopPosition {
                rules: parameters.rules.clone(),
                permanent: vec![[0; COLUMN_COUNT]; player_count],
                temporary: [None; COLUMN_COUNT],
                owners: [None; COLUMN_COUNT],
                dice: None,
                current_player: PlayerId::new(0),
                phase: Phase::StartTurn,
                counters: TurnCounters::default(),
                status: SingleWinnerOutcome::Ongoing,
            },
            random: ChanceRng::new(parameters.seed),
        })
    }
    fn player_count(&self, state: &CantStopState) -> usize {
        state.position.player_count()
    }
    fn next_player(&self, state: &CantStopState) -> Decision {
        state.position.next_player()
    }
    fn observe(
        &self,
        state: &CantStopState,
        observer: PlayerId,
    ) -> Result<CantStopObservation, Error> {
        validate_player(observer, state.position.player_count())?;
        Ok(CantStopObservation {
            position: state.position.clone(),
            observer,
        })
    }
    fn offered_actions(&self, observation: &CantStopObservation) -> Vec<Action> {
        if observation.position.next_player() == Decision::Player(observation.observer) {
            actions(&observation.position)
        } else {
            vec![]
        }
    }
    fn available_actions(
        &self,
        state: &CantStopState,
        actor: PlayerId,
    ) -> Result<Vec<Action>, Error> {
        let p = &state.position;
        validate_player(actor, p.player_count())?;
        if p.status != SingleWinnerOutcome::Ongoing {
            return Ok(vec![]);
        }
        if p.current_player != actor {
            return Err(Error::WrongPlayer {
                expected: p.current_player,
                actual: actor,
            });
        }
        Ok(actions(p))
    }
    fn next(
        &self,
        state: &mut CantStopState,
        actor: PlayerId,
        action: &Action,
    ) -> Result<(), Error> {
        self.next_recorded(state, actor, action).map(|_| ())
    }
    fn player_result(
        &self,
        state: &CantStopState,
        player: PlayerId,
    ) -> Result<PlayerResult, Error> {
        state.position.player_result(player)
    }
    fn player_score(&self, state: &CantStopState, player: PlayerId) -> Result<f64, Error> {
        state.position.player_score(player)
    }
}
impl SimulationModel for CantStopForwardModel {
    fn simulation_fork(
        &self,
        observation: &CantStopObservation,
        random: &mut SimulationRng,
    ) -> Result<CantStopState, Error> {
        // Observations are typed, immutable facts created by observe(). Validation
        // precedes drawing a fork seed, including for nonacting observers.
        validate_player(observation.observer, observation.position.player_count())?;
        Ok(CantStopState {
            position: observation.position.clone(),
            random: random.fork_chance(),
        })
    }
}
