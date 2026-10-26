//! Count simulation work and enforce decision budgets during search and evaluation.
use super::{Budgets, Error, Result, ValueCompletion};
use crate::core::{ForwardModel, PlayerId};

/// A monotonic clock supplied by the platform. Tests can supply a controlled clock.
pub trait Clock {
    fn now_ms(&self) -> f64;
    fn supports_time_budget(&self) -> bool {
        true
    }
}
/// Native elapsed clock. Wasm callers supply a browser-backed Clock for timed decisions.
pub struct SystemClock {
    #[cfg(not(target_arch = "wasm32"))]
    origin: std::time::Instant,
}
impl Default for SystemClock {
    fn default() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            origin: std::time::Instant::now(),
        }
    }
}
impl Clock for SystemClock {
    fn now_ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.origin.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }
    fn supports_time_budget(&self) -> bool {
        !cfg!(target_arch = "wasm32")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Cancelled,
    Time,
    Iterations,
    ForwardModel,
    Resources,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub completed_iterations: u64,
    pub forward_model_calls: u64,
    pub state_copies: u64,
    pub state_evaluations: u64,
    pub action_evaluations: u64,
}
pub(super) fn increment(value: &mut u64) -> Result<()> {
    *value = value.checked_add(1).ok_or(Error::Overflow)?;
    Ok(())
}

/// Evaluators use this service for every simulated transition and independent copy.
/// The live state is never passed to a transition; callers own the simulation state.
pub struct SimulationContext<'a, F: ForwardModel> {
    pub(super) model: &'a F,
    pub(super) budgets: Budgets,
    pub(super) counters: Counters,
    pub(super) value_completion: ValueCompletion,
    pub(super) max_output_bytes: usize,
    pub(super) policy_node: Option<usize>,
    clock: &'a dyn Clock,
    cancelled: &'a dyn Fn() -> bool,
    started_ms: f64,
    last_ms: std::cell::Cell<f64>,
    clock_invalid: std::cell::Cell<bool>,
}
impl<'a, F: ForwardModel> SimulationContext<'a, F> {
    pub(super) fn new(
        model: &'a F,
        budgets: Budgets,
        clock: &'a dyn Clock,
        cancelled: &'a dyn Fn() -> bool,
    ) -> Result<Self> {
        budgets.validate()?;
        if budgets.time_ms.is_some() && !clock.supports_time_budget() {
            return Err(Error::Invalid(
                "timed Wasm search requires a platform Clock".into(),
            ));
        }
        let started_ms = clock.now_ms();
        if !started_ms.is_finite() {
            return Err(Error::Invalid("clock returned a nonfinite time".into()));
        }
        Ok(Self {
            model,
            budgets,
            counters: Counters::default(),
            value_completion: ValueCompletion::RequireFullVector,
            max_output_bytes: usize::MAX,
            policy_node: None,
            clock,
            cancelled,
            started_ms,
            last_ms: std::cell::Cell::new(started_ms),
            clock_invalid: std::cell::Cell::new(false),
        })
    }
    pub fn model(&self) -> &F {
        self.model
    }
    pub fn counters(&self) -> &Counters {
        &self.counters
    }
    /// Check the numeric size before constructing a value or action-score buffer.
    /// `count` is the number of f64 values. This bounds outputs, not a custom
    /// evaluator's private memory or a game's own state allocation.
    ///
    /// # Arguments
    /// `self` is read-only; `count` is copied and checked before multiplying by eight.
    pub fn require_values(&self, count: usize) -> Result<()> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(Error::Overflow)?;
        if bytes > self.max_output_bytes {
            return Err(Error::ResourceLimit);
        }
        Ok(())
    }
    /// The effective policy is shared with nested successor-state evaluators.
    pub fn value_completion(&self) -> ValueCompletion {
        self.value_completion
    }
    pub fn elapsed_ms(&self) -> f64 {
        let now = self.clock.now_ms();
        if !now.is_finite() || now < self.last_ms.get() {
            self.clock_invalid.set(true);
        } else {
            self.last_ms.set(now);
        }
        self.last_ms.get() - self.started_ms
    }
    /// Check cancellation and elapsed time before or after a potentially expensive operation.
    pub fn check_interrupt(&self) -> Result<()> {
        if (self.cancelled)() {
            return Err(Error::Cancelled);
        }
        let elapsed = self.elapsed_ms();
        if self.clock_invalid.get() {
            return Err(Error::Invalid(
                "clock must remain finite and monotonic".into(),
            ));
        }
        if self.budgets.time_ms.is_some_and(|limit| elapsed >= limit) {
            return Err(Error::BudgetExhausted);
        }
        Ok(())
    }
    pub fn remaining_transitions(&self) -> Option<u64> {
        self.budgets
            .forward_model_calls
            .map(|limit| limit.saturating_sub(self.counters.forward_model_calls))
    }
    /// Check that the remaining forward-model budget can score every candidate
    /// before the evaluator starts simulating successor states.
    ///
    /// # Arguments
    /// `self` is read-only; `count` is the copied number of planned transitions.
    /// Insufficient budget returns an error without consuming any transition.
    pub fn require_transitions(&self, count: u64) -> Result<()> {
        self.check_interrupt()?;
        if self
            .remaining_transitions()
            .is_some_and(|remaining| count > remaining)
        {
            return Err(Error::BudgetExhausted);
        }
        Ok(())
    }
    /// Create an independently owned simulation copy and record its cost.
    ///
    /// # Arguments
    /// `self` is borrowed exclusively for accounting. `state` is read-only and
    /// remains unchanged. Its Clone implementation must preserve independence.
    ///
    /// # Errors
    /// Cancellation, elapsed-time exhaustion or a copy-counter overflow.
    pub fn copy_state(&mut self, state: &F::State) -> Result<F::State>
    where
        F::State: Clone,
    {
        self.check_interrupt()?;
        increment(&mut self.counters.state_copies)?;
        let copy = state.clone();
        self.check_interrupt()?;
        Ok(copy)
    }
    /// Reconstruct one trajectory with fresh search-owned chance and account for it.
    pub fn fork(
        &mut self,
        observation: &F::Observation,
        random: &mut crate::core::random::SimulationRng,
    ) -> Result<F::State>
    where
        F: crate::core::simulation::SimulationModel,
        F::Error: std::fmt::Display,
    {
        self.check_interrupt()?;
        increment(&mut self.counters.state_copies)?;
        let state = self
            .model
            .simulation_fork(observation, random)
            .map_err(|e| Error::Game(e.to_string()))?;
        self.check_interrupt()?;
        Ok(state)
    }
    /// Apply exactly one action to an owned simulation and charge it before execution.
    ///
    /// # Arguments
    /// `self` is borrowed exclusively to charge work. `state` is an exclusively
    /// borrowed simulation, `actor` is its decision maker, and `action` is read-only.
    ///
    /// # Errors
    /// An exhausted limit prevents application. A game failure is reported with
    /// its message; the call still counts as work. Time/cancellation detected after
    /// success leaves that simulation advanced but prevents back-propagation of
    /// the interrupted iteration.
    pub fn apply(&mut self, state: &mut F::State, actor: PlayerId, action: &F::Action) -> Result<()>
    where
        F::Error: std::fmt::Display,
    {
        self.require_transitions(1)?;
        increment(&mut self.counters.forward_model_calls)?;
        self.model
            .next(state, actor, action)
            .map_err(|e| Error::Game(e.to_string()))?;
        self.check_interrupt()
    }
    pub(super) fn reasons(&self) -> Vec<StopReason> {
        let mut reasons = Vec::new();
        if (self.cancelled)() {
            reasons.push(StopReason::Cancelled);
        }
        if self
            .budgets
            .time_ms
            .is_some_and(|limit| self.elapsed_ms() >= limit)
        {
            reasons.push(StopReason::Time);
        }
        if self
            .budgets
            .iterations
            .is_some_and(|limit| self.counters.completed_iterations >= limit)
        {
            reasons.push(StopReason::Iterations);
        }
        if self.remaining_transitions() == Some(0) {
            reasons.push(StopReason::ForwardModel);
        }
        reasons
    }
}
