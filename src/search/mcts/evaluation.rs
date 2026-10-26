//! State values and independently registered native action evaluators.
use super::{
    Error, EvaluatorSpec, Parameter, Parameters, Result, SimulationContext, ValueCompletion,
};
use crate::core::evaluation::Evaluator;
use crate::core::{ForwardModel, PlayerId, PlayerResult};
use std::collections::BTreeMap;
use std::{future::Future, pin::Pin};

/// An evaluation may finish immediately in Rust or wait for an external response.
/// The future borrows the simulation read-only until its result is available.
pub type EvaluationFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + 'a>>;

/// A scalar always identifies its perspective explicitly; a vector follows absolute seat order.
#[derive(Clone, Debug, PartialEq)]
pub enum StateValues {
    Full(Vec<f64>),
    Scalar { player: PlayerId, value: f64 },
}

/// Validate an evaluator output and construct the complete player-value vector
/// used by back-propagation.
///
/// # Arguments
/// `values` transfers ownership of the output. `players` gives the required vector
/// length. `policy` specifies how one explicitly identified scalar may be completed.
///
/// # Errors
/// Invalid player identities, incomplete vectors or nonfinite values are rejected.
pub fn complete_values(
    values: StateValues,
    players: usize,
    policy: ValueCompletion,
) -> Result<Vec<f64>> {
    let vector = match values {
        StateValues::Full(values) if values.len() == players => values,
        StateValues::Full(_) => {
            return Err(Error::Evaluation(
                "state vector must contain every player exactly once".into(),
            ));
        }
        StateValues::Scalar { player, value } => {
            if player.index() >= players || !value.is_finite() {
                return Err(Error::Evaluation(
                    "scalar has invalid player or value".into(),
                ));
            }
            let fill = match policy {
                ValueCompletion::RequireFullVector => {
                    return Err(Error::Evaluation("full player vector required".into()));
                }
                ValueCompletion::FillZeros => 0.0,
                ValueCompletion::Negation => -value,
            };
            let mut values = vec![fill; players];
            values[player.index()] = value;
            values
        }
    };
    if players == 0 || vector.iter().any(|v| !v.is_finite()) {
        return Err(Error::Evaluation(
            "player values must be finite and nonempty".into(),
        ));
    }
    Ok(vector)
}

/// Estimate player values with a native Rust state evaluation function.
/// Any internal forward-model calls must use the supplied simulation context.
pub trait StateEvaluator<F: ForwardModel> {
    /// Read `state` without changing it and return owned values with explicit perspectives.
    /// `context` provides accounting for any internal copies or transitions.
    ///
    /// # Arguments
    /// `self` may update private evaluator state. `state` is a shared simulation
    /// reference; `context` is exclusively borrowed for budget-aware work.
    fn evaluate(
        &mut self,
        state: &F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<StateValues>;
    fn evaluate_async<'a, 'model>(
        &'a mut self,
        state: &'a F::State,
        context: &'a mut SimulationContext<'model, F>,
    ) -> EvaluationFuture<'a, StateValues> {
        Box::pin(std::future::ready(self.evaluate(state, context)))
    }
    /// Supply actual terminal rewards in absolute player order, bypassing estimates.
    /// The default explicitly maps win/draw/loss to +1/0/-1. A game with score-based
    /// or team rewards overrides this method instead of reusing that convention.
    ///
    /// # Arguments
    /// `self` may update private evaluator state. `state` is a shared terminal
    /// simulation reference; `context` supplies read-only rules and work accounting.
    fn terminal(
        &mut self,
        state: &F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>>
    where
        F::Error: std::fmt::Display,
    {
        terminal_values(context.model(), state)
    }
    /// Return owned persistent bytes, or explicitly decline checkpoint support.
    fn checkpoint(&self) -> Result<Vec<u8>> {
        Err(Error::UnsupportedCheckpoint("state evaluator".into()))
    }
    fn restore(&mut self, _bytes: &[u8]) -> Result<()> {
        Err(Error::UnsupportedCheckpoint("state evaluator".into()))
    }
}
/// Compute h(s,a) for each supplied available action a in state s.
/// Scores follow the action-list order and the deciding player's perspective.
pub trait ActionEvaluator<F: ForwardModel> {
    /// Score `actions` in their supplied order for `actor`, reading `state` only.
    /// Return owned finite scores. Use `context` for every simulated transition.
    ///
    /// # Arguments
    /// `self` may update private evaluator state. `state` and `actions` are shared
    /// references. `actor` fixes the scores' perspective. `context` is exclusively
    /// borrowed to account for all internal simulation work.
    fn evaluate(
        &mut self,
        state: &F::State,
        actor: PlayerId,
        actions: &[F::Action],
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>>;
    fn evaluate_async<'a, 'model>(
        &'a mut self,
        state: &'a F::State,
        actor: PlayerId,
        actions: &'a [F::Action],
        context: &'a mut SimulationContext<'model, F>,
    ) -> EvaluationFuture<'a, Vec<f64>> {
        Box::pin(std::future::ready(
            self.evaluate(state, actor, actions, context),
        ))
    }
    fn checkpoint(&self) -> Result<Vec<u8>> {
        Err(Error::UnsupportedCheckpoint("action evaluator".into()))
    }
    fn restore(&mut self, _bytes: &[u8]) -> Result<()> {
        Err(Error::UnsupportedCheckpoint("action evaluator".into()))
    }
}
/// Adapt an existing per-player state evaluator without assuming zero-sum returns.
pub struct PlayerEvaluator<E>(pub E);
impl<F, E> StateEvaluator<F> for PlayerEvaluator<E>
where
    F: ForwardModel,
    E: Evaluator<F::State>,
    E::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        state: &F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<StateValues> {
        let count = context.model().player_count(state);
        context.require_values(count)?;
        let mut values = Vec::with_capacity(count);
        for player in 0..count {
            context.check_interrupt()?;
            values.push(
                self.0
                    .evaluate(state, PlayerId::new(player))
                    .map_err(|e| Error::Evaluation(e.to_string()))?,
            );
        }
        Ok(StateValues::Full(values))
    }
}

type StateFactory<F> = Box<dyn Fn(&Parameters) -> Result<Box<dyn StateEvaluator<F>>>>;
type ActionFactory<F> = Box<dyn Fn(&Parameters) -> Result<Box<dyn ActionEvaluator<F>>>>;
/// Explicit factories keep custom evaluator creation outside the tree algorithm.
pub struct EvaluatorRegistry<F: ForwardModel> {
    states: BTreeMap<String, StateFactory<F>>,
    actions: BTreeMap<String, ActionFactory<F>>,
}
impl<F: ForwardModel> Default for EvaluatorRegistry<F> {
    fn default() -> Self {
        Self {
            states: BTreeMap::new(),
            actions: BTreeMap::new(),
        }
    }
}
impl<F> EvaluatorRegistry<F>
where
    F: ForwardModel + 'static,
    F::State: Clone,
    F::Error: std::fmt::Display,
{
    /// Register a state factory. Duplicate names are rejected rather than replaced.
    ///
    /// # Arguments
    /// `name` is copied into the registry. `factory` is owned by the registry and
    /// receives read-only parameters each time an independent evaluator is built.
    pub fn register_state(
        &mut self,
        name: &str,
        factory: impl Fn(&Parameters) -> Result<Box<dyn StateEvaluator<F>>> + 'static,
    ) -> Result<()> {
        if self.states.contains_key(name) {
            return Err(Error::Invalid(format!("duplicate state evaluator {name}")));
        }
        self.states.insert(name.into(), Box::new(factory));
        Ok(())
    }
    /// Register an independent action-evaluator factory.
    ///
    /// # Arguments
    /// `name` is copied and `factory` is owned by the registry. Each invocation
    /// receives read-only parameters. Reserved or duplicate names are errors.
    pub fn register_action(
        &mut self,
        name: &str,
        factory: impl Fn(&Parameters) -> Result<Box<dyn ActionEvaluator<F>>> + 'static,
    ) -> Result<()> {
        if ["uniform_actions", "successor_state"].contains(&name) || self.actions.contains_key(name)
        {
            return Err(Error::Invalid(format!(
                "duplicate or reserved action evaluator {name}"
            )));
        }
        self.actions.insert(name.into(), Box::new(factory));
        Ok(())
    }
    /// Create an owned state evaluator from a borrowed `spec` and read-only registry.
    /// Unknown names and invalid factory parameters return errors.
    pub fn build_state(&self, spec: &EvaluatorSpec) -> Result<Box<dyn StateEvaluator<F>>> {
        self.states
            .get(&spec.name)
            .ok_or_else(|| Error::Invalid(format!("unknown state evaluator {}", spec.name)))?(
            &spec.parameters,
        )
    }
    /// Create an owned action evaluator from a borrowed `spec` and read-only registry.
    /// Nested successor-state evaluators are built independently for this instance.
    pub fn build_action(&self, spec: &EvaluatorSpec) -> Result<Box<dyn ActionEvaluator<F>>> {
        match spec.name.as_str() {
            "uniform_actions" => {
                empty_parameters(&spec.parameters)?;
                Ok(Box::new(UniformActions))
            }
            "successor_state" => {
                if spec.parameters.len() != 1 {
                    return Err(Error::Invalid(
                        "successor_state requires only state_evaluator".into(),
                    ));
                }
                let nested = nested_spec(spec.parameters.get("state_evaluator"))?;
                Ok(Box::new(SuccessorState {
                    evaluator: self.build_state(&nested)?,
                }))
            }
            _ => {
                self.actions.get(&spec.name).ok_or_else(|| {
                    Error::Invalid(format!("unknown action evaluator {}", spec.name))
                })?(&spec.parameters)
            }
        }
    }
}
fn nested_spec(value: Option<&Parameter>) -> Result<EvaluatorSpec> {
    let Some(Parameter::Object(object)) = value else {
        return Err(Error::Invalid("state_evaluator must be an object".into()));
    };
    let Some(Parameter::Text(name)) = object.get("name") else {
        return Err(Error::Invalid("state_evaluator requires a name".into()));
    };
    if object
        .keys()
        .any(|key| key != "name" && key != "parameters")
    {
        return Err(Error::Invalid("unknown nested evaluator field".into()));
    }
    let parameters = match object.get("parameters") {
        None => Parameters::new(),
        Some(Parameter::Object(parameters)) => parameters.clone(),
        _ => {
            return Err(Error::Invalid(
                "evaluator parameters must be an object".into(),
            ));
        }
    };
    Ok(EvaluatorSpec {
        name: name.clone(),
        parameters,
    })
}
fn empty_parameters(parameters: &Parameters) -> Result<()> {
    if parameters.is_empty() {
        Ok(())
    } else {
        Err(Error::Invalid("evaluator takes no parameters".into()))
    }
}
fn restore_stateless(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        Ok(())
    } else {
        Err(Error::Invalid(
            "stateless evaluator checkpoint must be empty".into(),
        ))
    }
}
struct UniformActions;
impl<F: ForwardModel> ActionEvaluator<F> for UniformActions {
    fn evaluate(
        &mut self,
        _: &F::State,
        _: PlayerId,
        actions: &[F::Action],
        _: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        Ok(vec![0.0; actions.len()])
    }
    fn checkpoint(&self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        restore_stateless(bytes)
    }
}
struct SuccessorState<F: ForwardModel> {
    evaluator: Box<dyn StateEvaluator<F>>,
}
impl<F> ActionEvaluator<F> for SuccessorState<F>
where
    F: ForwardModel,
    F::State: Clone,
    F::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        state: &F::State,
        actor: PlayerId,
        actions: &[F::Action],
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        context.require_transitions(u64::try_from(actions.len()).map_err(|_| Error::Overflow)?)?;
        let mut scores = Vec::with_capacity(actions.len());
        for action in actions {
            let mut successor = context.copy_state(state)?;
            context.apply(&mut successor, actor, action)?;
            let values =
                if context.model().next_player(&successor) == crate::core::Decision::Finished {
                    complete_values(
                        StateValues::Full(self.evaluator.terminal(&successor, context)?),
                        context.model().player_count(&successor),
                        ValueCompletion::RequireFullVector,
                    )?
                } else {
                    super::budgets::increment(&mut context.counters.state_evaluations)?;
                    complete_values(
                        self.evaluator.evaluate(&successor, context)?,
                        context.model().player_count(&successor),
                        context.value_completion(),
                    )?
                };
            scores.push(values[actor.index()]);
        }
        Ok(scores)
    }
    fn checkpoint(&self) -> Result<Vec<u8>> {
        self.evaluator.checkpoint()
    }
    fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        self.evaluator.restore(bytes)
    }
}
pub(super) fn terminal_values<F: ForwardModel>(model: &F, state: &F::State) -> Result<Vec<f64>>
where
    F::Error: std::fmt::Display,
{
    (0..model.player_count(state))
        .map(|player| {
            match model
                .player_result(state, PlayerId::new(player))
                .map_err(|e| Error::Game(e.to_string()))?
            {
                PlayerResult::Win => Ok(1.0),
                PlayerResult::Draw => Ok(0.0),
                PlayerResult::Loss => Ok(-1.0),
                PlayerResult::Ongoing => Err(Error::Evaluation(
                    "terminal reward provider requires settled player outcomes".into(),
                )),
            }
        })
        .collect()
}

// Built-in game evaluators have no persistent state. Custom adapters opt into
// checkpoint support explicitly through their own implementations.
macro_rules! game_registry {
    ($model:ty, $evaluator:expr) => {
        impl NativeEvaluators for $model {
            fn native_evaluators() -> EvaluatorRegistry<Self> {
                EvaluatorRegistry::<Self>::for_game()
            }
        }
        impl EvaluatorRegistry<$model> {
            pub fn for_game() -> Self {
                struct GameEvaluator;
                impl StateEvaluator<$model> for GameEvaluator {
                    fn evaluate(
                        &mut self,
                        state: &<$model as ForwardModel>::State,
                        context: &mut SimulationContext<'_, $model>,
                    ) -> Result<StateValues> {
                        PlayerEvaluator($evaluator).evaluate(state, context)
                    }
                    fn checkpoint(&self) -> Result<Vec<u8>> {
                        Ok(Vec::new())
                    }
                    fn restore(&mut self, bytes: &[u8]) -> Result<()> {
                        restore_stateless(bytes)
                    }
                }
                let mut registry = Self::default();
                registry
                    .register_state("game_heuristic", |parameters| {
                        empty_parameters(parameters)?;
                        Ok(Box::new(GameEvaluator))
                    })
                    .expect("fresh registry has no duplicate names");
                registry
            }
        }
    };
}
/// A game supplies its native heuristic factories to platform-independent search.
/// Factories construct independent evaluators for each agent and role.
pub trait NativeEvaluators: ForwardModel + Sized {
    fn native_evaluators() -> EvaluatorRegistry<Self>;
}
game_registry!(
    crate::games::tictactoe::TicTacToeForwardModel,
    crate::games::tictactoe::TicTacToeEvaluator
);
game_registry!(
    crate::games::connect4::Connect4ForwardModel,
    crate::games::connect4::Connect4Evaluator
);

// Love Letter returns every player's value from their own sampled observation.
struct LoveLetterEvaluator;
impl StateEvaluator<crate::games::loveletter::LoveLetterForwardModel> for LoveLetterEvaluator {
    fn evaluate(
        &mut self,
        state: &crate::games::loveletter::LoveLetterState,
        context: &mut SimulationContext<'_, crate::games::loveletter::LoveLetterForwardModel>,
    ) -> Result<StateValues> {
        PlayerEvaluator(crate::games::loveletter::LoveLetterHeuristic).evaluate(state, context)
    }
    fn checkpoint(&self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        restore_stateless(bytes)
    }
}
impl NativeEvaluators for crate::games::loveletter::LoveLetterForwardModel {
    fn native_evaluators() -> EvaluatorRegistry<Self> {
        let mut registry = EvaluatorRegistry::default();
        registry
            .register_state("game_heuristic", |p| {
                empty_parameters(p)?;
                Ok(Box::new(LoveLetterEvaluator))
            })
            .expect("fresh game heuristic name");
        registry
    }
}
// Can't Stop exposes a baseline and configurable progress estimate independently
// for every state/action role. Configuration is retained in the registry spec.
struct CantStopEvaluator(crate::games::cantstop::CantStopProgress);
impl StateEvaluator<crate::games::cantstop::CantStopForwardModel> for CantStopEvaluator {
    fn evaluate(
        &mut self,
        state: &crate::games::cantstop::CantStopState,
        context: &mut SimulationContext<'_, crate::games::cantstop::CantStopForwardModel>,
    ) -> Result<StateValues> {
        PlayerEvaluator(self.0).evaluate(state, context)
    }
    fn checkpoint(&self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn restore(&mut self, bytes: &[u8]) -> Result<()> {
        restore_stateless(bytes)
    }
}
impl NativeEvaluators for crate::games::cantstop::CantStopForwardModel {
    fn native_evaluators() -> EvaluatorRegistry<Self> {
        EvaluatorRegistry::<Self>::for_game()
    }
}
impl EvaluatorRegistry<crate::games::cantstop::CantStopForwardModel> {
    /// Independent baseline/progress factories for each state or successor-action role.
    pub fn for_game() -> Self {
        let mut registry = Self::default();
        registry
            .register_state("game_heuristic", |p| {
                empty_parameters(p)?;
                Ok(Box::new(CantStopEvaluator(
                    crate::games::cantstop::CantStopProgress::default(),
                )))
            })
            .expect("fresh default name");
        registry
            .register_state("progress", |p| {
                if p.keys().any(|key| key != "alpha" && key != "beta") {
                    return Err(Error::Invalid(
                        "progress accepts only alpha and beta".into(),
                    ));
                }
                let number = |key: &str, default: f64| -> Result<f64> {
                    match p.get(key) {
                        None => Ok(default),
                        Some(Parameter::Number(n)) => Ok(*n),
                        Some(Parameter::Integer(n)) => Ok(*n as f64),
                        _ => Err(Error::Invalid(format!("{key} must be numeric"))),
                    }
                };
                let estimate = crate::games::cantstop::CantStopProgress::new(
                    number("alpha", 0.5)?,
                    number("beta", 0.25)?,
                )
                .map_err(|e| Error::Invalid(e.to_string()))?;
                Ok(Box::new(CantStopEvaluator(estimate)))
            })
            .expect("fresh progress name");
        struct Baseline;
        impl StateEvaluator<crate::games::cantstop::CantStopForwardModel> for Baseline {
            fn evaluate(
                &mut self,
                state: &crate::games::cantstop::CantStopState,
                context: &mut SimulationContext<'_, crate::games::cantstop::CantStopForwardModel>,
            ) -> Result<StateValues> {
                PlayerEvaluator(crate::games::cantstop::CantStopBaseline).evaluate(state, context)
            }
            fn checkpoint(&self) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }
            fn restore(&mut self, bytes: &[u8]) -> Result<()> {
                restore_stateless(bytes)
            }
        }
        registry
            .register_state("baseline", |p| {
                empty_parameters(p)?;
                Ok(Box::new(Baseline))
            })
            .expect("fresh baseline name");
        registry
    }
}
