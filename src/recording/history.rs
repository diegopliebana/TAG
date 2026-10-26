//! Recreate earlier positions by replaying accepted moves from the recording root.
//!
//! A small cache retains some reconstructed states to save repeated work. Cache size
//! is a runtime setting, not part of the recorded rules or decisions.
use super::{ControllerSnapshot, RecordedMove};
use crate::core::{Decision, ForwardModel, PlayerId};
use std::collections::VecDeque;

/// Settings controlling which rebuilt states are retained to save repeated replay work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone/Copy: small settings; PartialEq/Eq: exact limit equality.
pub struct HistoryCachePolicy {
    pub checkpoint_interval: std::num::NonZeroUsize,
    /// Maximum cached states in addition to the root; zero disables caching.
    pub max_states: usize,
}
impl Default for HistoryCachePolicy {
    fn default() -> Self {
        Self {
            checkpoint_interval: std::num::NonZeroUsize::new(16).unwrap(),
            max_states: 4,
        }
    }
}

/// A player's view of an earlier position, identified by the number of recorded moves applied.
#[derive(Debug)] // Debug: inspect historical data when game data supports Debug.
pub struct HistoricalView<O, M> {
    pub position: usize,
    pub observation: O,
    pub actions: Vec<M>,
}

/// Invalid history location or reconstruction data.
#[derive(Debug, PartialEq, Eq)] // PartialEq/Eq: exact contextual equality when the game error supports it.
pub enum HistoryError<G> {
    InvalidPosition { requested: usize, last: usize },
    InvalidActor { position: usize, actor: PlayerId },
    DecisionMismatch { position: usize },
    Game { position: usize, error: G },
}
impl<G: std::fmt::Display> std::fmt::Display for HistoryError<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPosition { requested, last } => {
                write!(f, "history position {requested} exceeds {last}")
            }
            Self::InvalidActor { position, actor } => write!(
                f,
                "history move {position} has invalid player {}",
                actor.index()
            ),
            Self::DecisionMismatch { position } => write!(
                f,
                "history move {position} has inconsistent resulting decision"
            ),
            Self::Game { position, error } => write!(f, "history move {position}: {error}"),
        }
    }
}
impl<G: std::error::Error + 'static> std::error::Error for HistoryError<G> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Game { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// An owned initial state, initial controller snapshots and accepted moves.
/// Runtime reconstruction uses a configurable bounded cache; defaults are four
/// additional states, every 16 moves.
pub struct GameHistory<S, M, C> {
    initial_state: S,
    initial_controllers: Vec<ControllerSnapshot<C>>,
    moves: Vec<RecordedMove<M, C>>,
    cache: VecDeque<(usize, S)>,
    cache_policy: HistoryCachePolicy,
}

impl<S, M, C> GameHistory<S, M, C> {
    /// Consume the initial state and controller snapshots into an empty recording.
    /// Returns an owned history; caller supplies exact initial conditions.
    pub fn new(initial_state: S, initial_controllers: Vec<ControllerSnapshot<C>>) -> Self {
        Self::with_cache_policy(
            initial_state,
            initial_controllers,
            HistoryCachePolicy::default(),
        )
    }
    /// Consume root state/controller data and copied cache settings into an empty history.
    /// Returns owned history. A nonzero interval is enforced by its type; zero
    /// max_states disables caching. Callers choose limits appropriate to state size.
    pub fn with_cache_policy(
        initial_state: S,
        initial_controllers: Vec<ControllerSnapshot<C>>,
        cache_policy: HistoryCachePolicy,
    ) -> Self {
        Self {
            initial_state,
            initial_controllers,
            moves: Vec::new(),
            cache: VecDeque::new(),
            cache_policy,
        }
    }
    /// Read the initial state through a reference; the recording keeps owning it.
    pub fn initial_state(&self) -> &S {
        &self.initial_state
    }
    /// Read the saved initial controllers in player-seat order.
    pub fn initial_controllers(&self) -> &[ControllerSnapshot<C>] {
        &self.initial_controllers
    }
    /// Read accepted move records in order from the recording's starting state.
    pub fn moves(&self) -> &[RecordedMove<M, C>] {
        &self.moves
    }
    /// Consume an accepted live record and append it for later reconstruction.
    /// External file records require separate validation before acceptance.
    pub fn record(&mut self, record: RecordedMove<M, C>) {
        self.moves.push(record);
    }
    /// Consume history and transfer its accepted moves to the caller.
    /// Root and runtime cache storage are released; returned records retain ownership.
    pub fn into_moves(self) -> Vec<RecordedMove<M, C>> {
        self.moves
    }
    /// Copied cache size; the initial state is stored separately.
    pub fn cached_state_count(&self) -> usize {
        self.cache.len()
    }
    /// Return copied limits governing additional reconstructed state snapshots.
    pub fn cache_policy(&self) -> HistoryCachePolicy {
        self.cache_policy
    }

    /// Restore independent game state by applying recorded actions to a snapshot.
    ///
    /// # Arguments
    /// Exclusive self access updates only the bounded runtime cache. Rules are
    /// passed by a shared (read-only) reference; position counts accepted moves from the initial state.
    ///
    /// # Returns
    /// Owned simulation state. Game-state Clone must preserve independent snapshots.
    ///
    /// # Errors
    /// Out-of-range positions, invalid recorded moves or inconsistent decisions.
    pub fn position_at<F>(
        &mut self,
        model: &F,
        position: usize,
    ) -> Result<S, HistoryError<F::Error>>
    where
        F: ForwardModel<State = S, Action = M>,
        S: Clone,
    {
        self.check_position(position)?;
        let nearest = self
            .cache
            .iter()
            .filter(|(index, _)| *index <= position)
            .max_by_key(|(index, _)| *index);
        let (start, mut state) = match nearest {
            Some((index, state)) => (*index, state.clone()),
            None => (0, self.initial_state.clone()),
        };
        for index in start..position {
            let record = &self.moves[index];
            model
                .next(&mut state, record.actor, &record.action)
                .map_err(|error| HistoryError::Game {
                    position: index + 1,
                    error,
                })?;
            if model.next_player(&state) != record.decision {
                return Err(HistoryError::DecisionMismatch {
                    position: index + 1,
                });
            }
            let completed = index + 1;
            if self.cache_policy.max_states > 0
                && completed.is_multiple_of(self.cache_policy.checkpoint_interval.get())
                && !self.cache.iter().any(|(index, _)| *index == completed)
            {
                if self.cache.len() == self.cache_policy.max_states {
                    self.cache.pop_front();
                }
                self.cache.push_back((completed, state.clone()));
            }
        }
        Ok(state)
    }

    /// Obtain a historical observer's owned facts the player is allowed to know and offered actions.
    /// Rules are passed by a shared (read-only) reference; position/observer are copied; cache is exclusively
    /// borrowed. Errors preserve the live game and controller progress.
    pub fn view_at<F>(
        &mut self,
        model: &F,
        position: usize,
        observer: PlayerId,
    ) -> Result<HistoricalView<F::Observation, M>, HistoryError<F::Error>>
    where
        F: ForwardModel<State = S, Action = M>,
        S: Clone,
    {
        let state = self.position_at(model, position)?;
        let observation = model
            .observe(&state, observer)
            .map_err(|error| HistoryError::Game { position, error })?;
        let actions = if model.next_player(&state) == Decision::Player(observer) {
            model
                .available_actions(&state, observer)
                .map_err(|error| HistoryError::Game { position, error })?
        } else {
            vec![]
        };
        Ok(HistoricalView {
            position,
            observation,
            actions,
        })
    }

    /// Restore owned seat snapshots after the requested number of accepted moves.
    /// Shared self borrow; copied position. Each move updates only its acting seat.
    /// Out-of-range positions/seats return contextual history errors.
    pub fn controllers_at<G>(
        &self,
        position: usize,
    ) -> Result<Vec<ControllerSnapshot<C>>, HistoryError<G>>
    where
        C: Clone,
    {
        self.check_position(position)?;
        let mut controllers = self.initial_controllers.iter().collect::<Vec<_>>();
        for (index, record) in self.moves.iter().take(position).enumerate() {
            let seat =
                controllers
                    .get_mut(record.actor.index())
                    .ok_or(HistoryError::InvalidActor {
                        position: index + 1,
                        actor: record.actor,
                    })?;
            *seat = &record.controller_after;
        }
        // Copy each seat's final snapshot once after finding its latest accepted move.
        Ok(controllers.into_iter().cloned().collect())
    }

    fn check_position<G>(&self, position: usize) -> Result<(), HistoryError<G>> {
        if position > self.moves.len() {
            Err(HistoryError::InvalidPosition {
                requested: position,
                last: self.moves.len(),
            })
        } else {
            Ok(())
        }
    }
}
