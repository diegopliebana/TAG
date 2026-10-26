//! The search loop runs MCTS iterations and back-propagates their player values.
mod setup;
mod traversal;
use super::budgets::increment;
use super::selection;
use super::tree::{Edge, Node, PathEntry, PendingExpansion, PlayerStatistics};
use super::*;
use crate::agents::Agent;
use crate::core::checkpoint::Checkpoint;
use crate::core::random::{AgentRng, AgentRngSnapshot};
use crate::core::random::{SimulationRng, SimulationRngSnapshot};
use crate::core::simulation::SimulationModel;
use crate::core::{Decision, PlayerId};

/// Visit count and mean player values for one action available at the root.
/// Mean values follow absolute player order: entry i estimates the value for player i.
#[derive(Clone, Debug, PartialEq)]
pub struct RootActionReport<A> {
    pub action: A,
    pub visits: u64,
    /// Q_i(s,a): value sum divided by visits, or zero when the action is unvisited.
    pub mean_values: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct SearchReport<A> {
    pub selected_index: usize,
    pub root_actions: Vec<RootActionReport<A>>,
    pub counters: Counters,
    pub stop_reasons: Vec<StopReason>,
    pub elapsed_ms: f64,
    /// True when no iteration visited a root action and a random legal move was chosen.
    pub fallback: bool,
    pub nodes: usize,
    pub edges: usize,
    pub effective_settings: Settings,
}
/// Settings, evaluator state and RNG progress saved between decisions.
/// Search builds a new tree for each decision; the snapshot contains no tree.
#[derive(Clone, Debug, PartialEq)]
pub struct MctsSnapshot {
    pub algorithm_version: u32,
    pub settings: Settings,
    pub rng: AgentRngSnapshot,
    pub simulation_rng: SimulationRngSnapshot,
    pub state_evaluator: Vec<u8>,
    /// Slots follow bias, PUCT preferences, expansion, rollout, including absent slots.
    pub action_evaluators: [Option<Vec<u8>>; 4],
}
struct Evaluators<F: crate::core::ForwardModel> {
    state: Box<dyn StateEvaluator<F>>,
    bias: Option<Box<dyn ActionEvaluator<F>>>,
    preferences: Option<Box<dyn ActionEvaluator<F>>>,
    expansion: Option<Box<dyn ActionEvaluator<F>>>,
    rollout: Option<Box<dyn ActionEvaluator<F>>>,
}

/// A native MCTS controller with independent evaluator instances and an owned RNG.
pub struct MctsAgent<F: crate::core::ForwardModel> {
    settings: Settings,
    evaluators: Evaluators<F>,
    rng: AgentRng,
    simulation_rng: SimulationRng,
    last_report: Option<SearchReport<F::Action>>,
}
impl<F> MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// Search one permitted position using default budgets and the native clock.
    ///
    /// # Arguments
    /// `model`, `observation` and `actions` are borrowed read-only. `actor` identifies
    /// the root decision maker. The returned report owns action/statistic copies.
    /// Evaluator and RNG progress remain advanced on failure; the live game is unchanged.
    pub fn search(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<SearchReport<F::Action>> {
        self.search_controlled(
            model,
            actor,
            observation,
            actions,
            &BudgetOverrides::default(),
            &SystemClock::default(),
            &|| false,
        )
    }

    /// Search with per-decision limits, a platform clock and a cancellation probe.
    /// `overrides` preserves omitted defaults; `clock` must be monotonic. Cancellation
    /// returns an error and never chooses a real move. All references last only this call.
    ///
    /// # Arguments
    /// `self` is borrowed exclusively to advance evaluator and RNG state. `model`
    /// supplies read-only rules. `actor` identifies the root player. `observation`
    /// is that player's permitted view, and `actions` is its complete offered list.
    /// `overrides` supplies this decision's limits. `clock` measures elapsed time;
    /// `cancelled` is a read-only probe called at operation boundaries.
    ///
    /// # Errors
    /// Invalid actions/settings, game or evaluator failures, cancellation, counter
    /// overflow and unsupported clock sources return explicit search errors.
    #[allow(clippy::too_many_arguments)]
    pub fn search_controlled(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        overrides: &BudgetOverrides,
        clock: &dyn Clock,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<SearchReport<F::Action>> {
        let mut future = Box::pin(self.search_impl(
            model,
            actor,
            observation,
            actions,
            overrides,
            clock,
            cancelled,
            false,
        ));
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => Err(Error::EvaluationRequired),
        }
    }

    /// Search without blocking on an external evaluator. Poll again after delivering
    /// a response, or to observe cancellation/time expiry while an evaluation waits.
    /// The future retains the simulation and borrows this agent until completion.
    /// It also yields after each completed iteration, allowing a coordinator to
    /// visit other trees even when no external prediction is needed.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_controlled_async(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        overrides: &BudgetOverrides,
        clock: &dyn Clock,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<SearchReport<F::Action>> {
        self.search_impl(
            model,
            actor,
            observation,
            actions,
            overrides,
            clock,
            cancelled,
            true,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn search_impl(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        overrides: &BudgetOverrides,
        clock: &dyn Clock,
        cancelled: &dyn Fn() -> bool,
        cooperative: bool,
    ) -> Result<SearchReport<F::Action>> {
        self.last_report = None;
        let mut effective = self.settings.clone();
        effective.budgets = overrides.apply(&self.settings.budgets)?;
        let mut context =
            SimulationContext::new(model, effective.budgets.clone(), clock, cancelled)?;
        context.value_completion = effective.value_completion;
        context.max_output_bytes = effective.limits.max_request_bytes;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        validate_actions(actions, effective.limits.max_actions_per_decision)?;
        let mut validation_rng = SimulationRng::from_snapshot(&self.simulation_rng.snapshot())
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let root = model
            .simulation_fork(observation, &mut validation_rng)
            .map_err(|e| Error::Game(e.to_string()))?;
        increment(&mut context.counters.state_copies)?;
        if model.next_player(&root) != Decision::Player(actor)
            || actor.index() >= model.player_count(&root)
        {
            return Err(Error::Invalid(
                "root actor must match a nonterminal observation".into(),
            ));
        }
        let legal = model
            .available_actions(&root, actor)
            .map_err(|e| Error::Game(e.to_string()))?;
        if legal.len() != actions.len() || actions.iter().any(|action| !legal.contains(action)) {
            return Err(Error::Invalid(
                "offered actions differ from permitted root legal actions".into(),
            ));
        }
        let players = model.player_count(&root);
        context.require_values(players)?;
        context.require_values(players.checked_mul(actions.len()).ok_or(Error::Overflow)?)?;
        let mut tree = vec![Node::default()];
        let mut edge_count = 0usize;
        let mut forced_stop = None;

        while context.reasons().is_empty() && forced_stop.is_none() {
            let mut simulation = match context.fork(observation, &mut self.simulation_rng) {
                Ok(state) => state,
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            };
            let mut path = Vec::new();
            let mut expansion = None;
            let iteration = self
                .iteration(
                    &mut simulation,
                    &mut tree,
                    &mut path,
                    &mut expansion,
                    edge_count,
                    &mut context,
                )
                .await;
            let values = match iteration {
                Ok(values) => values,
                Err(Error::ResourceLimit) => {
                    forced_stop = Some(StopReason::Resources);
                    match self.evaluate(&simulation, &mut context).await {
                        Ok(values) => values,
                        Err(Error::BudgetExhausted) => break,
                        Err(error) => return Err(error),
                    }
                }
                Err(Error::BudgetExhausted) => {
                    match context.check_interrupt() {
                        Err(Error::BudgetExhausted) => break,
                        Err(error) => return Err(error),
                        Ok(()) => {}
                    }
                    forced_stop = Some(StopReason::ForwardModel);
                    match self.evaluate(&simulation, &mut context).await {
                        Ok(values) => values,
                        Err(Error::BudgetExhausted) => break,
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            };
            match context.check_interrupt() {
                Ok(()) => {}
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            }
            // Only completed evaluations insert their new edge. Timed-out work
            // cannot change visit counts or final move selection.
            if let Some(expansion) = expansion {
                let child = tree.len();
                tree.push(Node::default());
                let record = tree[expansion.node]
                    .decisions
                    .entry(expansion.actor.index())
                    .or_default();
                let edge = record.edges.len();
                record.edges.push(Edge {
                    action: expansion.action,
                    child,
                    visits: 0,
                    value_sums: vec![0.0; players],
                });
                path.push(PathEntry {
                    node: expansion.node,
                    actor: expansion.actor,
                    edge,
                });
                edge_count = edge_count.checked_add(1).ok_or(Error::Overflow)?;
            }
            backpropagate(&mut tree, &path, &values)?;
            increment(&mut context.counters.completed_iterations)?;
            if cooperative {
                YieldIteration(false).await;
            }
        }
        context.check_interrupt().or_else(|error| {
            if error == Error::BudgetExhausted {
                Ok(())
            } else {
                Err(error)
            }
        })?;
        let mut reasons = context.reasons();
        if let Some(reason) = forced_stop
            && !reasons.contains(&reason)
        {
            reasons.push(reason);
        }
        let record = tree[0].decisions.get(&actor.index());
        let root_actions: Vec<_> = actions
            .iter()
            .map(|action| {
                let edge = record
                    .and_then(|record| record.edges.iter().find(|edge| &edge.action == action));
                RootActionReport {
                    action: action.clone(),
                    visits: edge.map_or(0, |edge| edge.visits),
                    mean_values: edge.map_or_else(
                        || vec![0.0; players],
                        |edge| {
                            edge.value_sums
                                .iter()
                                .map(|v| v / edge.visits as f64)
                                .collect()
                        },
                    ),
                }
            })
            .collect();
        let visited: Vec<_> = root_actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| (action.visits > 0).then_some(index))
            .collect();
        let fallback = visited.is_empty();
        let selected_index = if fallback {
            self.rng
                .index(actions.len())
                .expect("validated nonempty root actions")
        } else {
            self.final_action(&root_actions, &visited, actor)?
        };
        let report = SearchReport {
            selected_index,
            root_actions,
            counters: context.counters.clone(),
            stop_reasons: reasons,
            elapsed_ms: context.elapsed_ms(),
            fallback,
            nodes: tree.len(),
            edges: edge_count,
            effective_settings: effective,
        };
        Ok(report)
    }
}

async fn action_scores<F: crate::core::ForwardModel>(
    evaluator: &mut dyn ActionEvaluator<F>,
    state: &F::State,
    actor: PlayerId,
    actions: &[F::Action],
    context: &mut SimulationContext<'_, F>,
) -> Result<Vec<f64>> {
    context.check_interrupt()?;
    context.require_values(actions.len())?;
    increment(&mut context.counters.action_evaluations)?;
    let scores = evaluator
        .evaluate_async(state, actor, actions, context)
        .await?;
    context.check_interrupt()?;
    selection::validate_scores(&scores, actions.len())?;
    Ok(scores)
}
fn validate_actions<A: PartialEq>(actions: &[A], limit: usize) -> Result<()> {
    if actions.is_empty() {
        return Err(Error::Invalid(
            "nonterminal decision requires at least one offered action".into(),
        ));
    }
    if actions.len() > limit {
        return Err(Error::ResourceLimit);
    }
    for (index, action) in actions.iter().enumerate() {
        if actions[..index].contains(action) {
            return Err(Error::Invalid("duplicate action identity".into()));
        }
    }
    Ok(())
}
fn backpropagate<A>(tree: &mut [Node<A>], path: &[PathEntry], values: &[f64]) -> Result<()> {
    // Check every sum and visit count before changing statistics. Each value
    // component belongs to its original player; back-propagation never negates it.
    for entry in path {
        let record = &tree[entry.node].decisions[&entry.actor.index()];
        let edge = &record.edges[entry.edge];
        if record.visits == u64::MAX
            || edge.visits == u64::MAX
            || edge.value_sums.len() != values.len()
        {
            return Err(Error::Overflow);
        }
        if edge
            .value_sums
            .iter()
            .zip(values)
            .any(|(sum, value)| !(sum + value).is_finite())
        {
            return Err(Error::Overflow);
        }
    }
    for entry in path {
        let record: &mut PlayerStatistics<A> = tree[entry.node]
            .decisions
            .get_mut(&entry.actor.index())
            .expect("traversed actor exists");
        increment(&mut record.visits)?;
        let edge = &mut record.edges[entry.edge];
        increment(&mut edge.visits)?;
        for (sum, value) in edge.value_sums.iter_mut().zip(values) {
            *sum += value;
        }
    }
    Ok(())
}
impl<F> Agent<F> for MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    type Error = Error;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize> {
        let report = self.search(model, actor, observation, actions)?;
        let index = report.selected_index;
        self.last_report = Some(report);
        Ok(index)
    }
}

// One poll executes at most one completed iteration. Native synchronous searches
// omit this yield; external coordinators use it to visit other independent trees.
struct YieldIteration(bool);
impl std::future::Future for YieldIteration {
    type Output = ();
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        task: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            task.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}
