//! Joint-action search: every edge ends after all simultaneous choices resolve.
use super::*;
use crate::core::joint::Choice;
use crate::search::mcts::settings::{
    BiasMode, ExpansionOrder, JointOpponentPolicy, RolloutPolicy, TreePolicy,
};

struct JointEdge<A> {
    choices: Vec<Choice<A>>,
    child: usize,
    visits: u64,
    sums: Vec<f64>,
}

struct JointNode<A> {
    visits: u64,
    edges: Vec<JointEdge<A>>,
}

impl<A> Default for JointNode<A> {
    fn default() -> Self {
        Self {
            visits: 0,
            edges: Vec::new(),
        }
    }
}

impl<F> MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// The focal player optimizes its own action; simulated opponents use the
    /// configured policy on their own observations of the sampled state.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn search_joint_impl(
        &mut self,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        players: usize,
        effective: Settings,
        mut context: SimulationContext<'_, F>,
        cooperative: bool,
    ) -> Result<SearchReport<F::Action>> {
        if effective.tree_policy != TreePolicy::Ucb1
            || effective.progressive_bias.mode != BiasMode::None
            || effective.expansion.order != ExpansionOrder::Random
            || effective.rollout.policy != RolloutPolicy::Random
        {
            return Err(Error::Invalid(
                "joint decisions currently support UCB1, random expansion and random rollouts"
                    .into(),
            ));
        }
        let mut tree = vec![JointNode::default()];
        let mut edges = 0usize;
        let mut forced_stop = None;
        while context.reasons().is_empty() && forced_stop.is_none() {
            let mut simulation = match context.fork(observation, &mut self.simulation_rng) {
                Ok(state) => state,
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            };
            let mut path: Vec<(usize, usize)> = Vec::new();
            let mut pending: Option<(usize, Vec<Choice<F::Action>>)> = None;
            let result = self
                .joint_iteration(
                    &mut simulation,
                    &mut tree,
                    &mut path,
                    &mut pending,
                    actor,
                    edges,
                    &mut context,
                )
                .await;
            let values = match result {
                Ok(values) => values,
                Err(Error::ResourceLimit) => {
                    forced_stop = Some(StopReason::Resources);
                    match self.evaluate(&simulation, &mut context).await {
                        Ok(values) => values,
                        Err(Error::BudgetExhausted) => break,
                        Err(error) => return Err(error),
                    }
                }
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            };
            match context.check_interrupt() {
                Ok(()) => {}
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            }
            if let Some((node, choices)) = pending {
                let child = tree.len();
                tree.push(JointNode::default());
                let edge = tree[node].edges.len();
                tree[node].edges.push(JointEdge {
                    choices,
                    child,
                    visits: 0,
                    sums: vec![0.0; players],
                });
                path.push((node, edge));
                edges = edges.checked_add(1).ok_or(Error::Overflow)?;
            }
            for &(node, edge) in &path {
                let parent = &mut tree[node];
                increment(&mut parent.visits)?;
                let branch = &mut parent.edges[edge];
                increment(&mut branch.visits)?;
                for (sum, value) in branch.sums.iter_mut().zip(&values) {
                    *sum += value;
                }
            }
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
        let root_actions: Vec<_> = actions
            .iter()
            .map(|action| {
                let mut visits = 0u64;
                let mut sums = vec![0.0; players];
                for edge in &tree[0].edges {
                    if edge
                        .choices
                        .iter()
                        .any(|(seat, chosen)| *seat == actor && chosen == action)
                    {
                        visits += edge.visits;
                        for (sum, value) in sums.iter_mut().zip(&edge.sums) {
                            *sum += value;
                        }
                    }
                }
                RootActionReport {
                    action: action.clone(),
                    visits,
                    mean_values: if visits == 0 {
                        sums
                    } else {
                        sums.into_iter().map(|sum| sum / visits as f64).collect()
                    },
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
                .expect("validated root actions")
        } else {
            self.final_action(&root_actions, &visited, actor)?
        };
        Ok(SearchReport {
            selected_index,
            root_actions,
            counters: context.counters.clone(),
            stop_reasons: reasons,
            elapsed_ms: context.elapsed_ms(),
            fallback,
            forced: false,
            nodes: tree.len(),
            edges,
            effective_settings: effective,
        })
    }

    fn joint_offers(
        &self,
        state: &F::State,
        context: &SimulationContext<'_, F>,
    ) -> Result<Vec<(PlayerId, Vec<F::Action>)>> {
        let seats = context.model().decision_players(state);
        let mut result = Vec::with_capacity(seats.len());
        for seat in seats {
            let view = context
                .model()
                .observe(state, seat)
                .map_err(|e| Error::Game(e.to_string()))?;
            let actions = context.model().offered_actions(&view);
            validate_actions(&actions, self.settings.limits.max_actions_per_decision)?;
            let legal = context
                .model()
                .available_actions(state, seat)
                .map_err(|e| Error::Game(e.to_string()))?;
            if actions != legal {
                return Err(Error::Game(
                    "observation and state offer different actions".into(),
                ));
            }
            result.push((seat, actions));
        }
        Ok(result)
    }

    fn joint_choices(
        &mut self,
        offers: &[(PlayerId, Vec<F::Action>)],
        node: Option<&JointNode<F::Action>>,
        actor: PlayerId,
        rollout: bool,
    ) -> Result<Vec<Choice<F::Action>>> {
        let mut choices = Vec::with_capacity(offers.len());
        for (seat, actions) in offers {
            let index = if rollout {
                self.rng.index(actions.len()).expect("nonempty actions")
            } else if *seat == actor {
                if let Some(node) = node {
                    let mut scores = Vec::with_capacity(actions.len());
                    for action in actions {
                        let matching: Vec<_> = node
                            .edges
                            .iter()
                            .filter(|edge| {
                                edge.choices.iter().any(|(p, a)| p == seat && a == action)
                            })
                            .collect();
                        let visits: u64 = matching.iter().map(|edge| edge.visits).sum();
                        let value: f64 = matching.iter().map(|edge| edge.sums[actor.index()]).sum();
                        let score = if visits == 0 {
                            f64::INFINITY
                        } else {
                            value / visits as f64
                                + self.settings.exploration_constant
                                    * ((node.visits.max(1) as f64).ln() / visits as f64).sqrt()
                        };
                        scores.push(score);
                    }
                    let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let tied: Vec<_> = scores
                        .iter()
                        .enumerate()
                        .filter_map(|(i, score)| (*score == best).then_some(i))
                        .collect();
                    tied[self.rng.index(tied.len()).expect("nonempty candidate set")]
                } else {
                    self.rng.index(actions.len()).expect("nonempty actions")
                }
            } else {
                match self.settings.joint_opponent_policy {
                    JointOpponentPolicy::UniformRandom => {
                        self.rng.index(actions.len()).expect("nonempty actions")
                    }
                    JointOpponentPolicy::FirstAvailable => 0,
                }
            };
            choices.push((*seat, actions[index].clone()));
        }
        Ok(choices)
    }

    #[allow(clippy::too_many_arguments)]
    async fn joint_iteration(
        &mut self,
        simulation: &mut F::State,
        tree: &mut [JointNode<F::Action>],
        path: &mut Vec<(usize, usize)>,
        pending: &mut Option<(usize, Vec<Choice<F::Action>>)>,
        actor: PlayerId,
        edge_count: usize,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        let mut node = 0;
        let mut depth = 0u64;
        loop {
            context.check_interrupt()?;
            if context.model().decision_players(simulation).is_empty()
                || self.settings.tree_depth.is_some_and(|limit| depth >= limit)
                || context.remaining_transitions() == Some(0)
            {
                return self.evaluate(simulation, context).await;
            }
            if context.model().decision_players(simulation).len() > 1
                && !context.model().joint_actions_public_after(simulation)
            {
                return Err(Error::Invalid(
                    "joint-action tree reached a decision with hidden choices".into(),
                ));
            }
            let offers = self.joint_offers(simulation, context)?;
            let choices = self.joint_choices(&offers, Some(&tree[node]), actor, false)?;
            if let Some(edge) = tree[node]
                .edges
                .iter()
                .position(|edge| edge.choices == choices)
            {
                context.apply_joint(simulation, &choices)?;
                path.push((node, edge));
                node = tree[node].edges[edge].child;
                depth = depth.checked_add(1).ok_or(Error::Overflow)?;
            } else {
                if tree.len() >= self.settings.limits.max_nodes
                    || edge_count >= self.settings.limits.max_edges
                {
                    return Err(Error::ResourceLimit);
                }
                context.apply_joint(simulation, &choices)?;
                *pending = Some((node, choices));
                return self.joint_rollout(simulation, context).await;
            }
        }
    }

    async fn joint_rollout(
        &mut self,
        simulation: &mut F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        let mut transitions = 0u64;
        loop {
            context.check_interrupt()?;
            if context.model().decision_players(simulation).is_empty()
                || self
                    .settings
                    .rollout_length
                    .is_some_and(|limit| transitions >= limit)
                || context.remaining_transitions() == Some(0)
            {
                return self.evaluate(simulation, context).await;
            }
            if context.model().decision_players(simulation).len() > 1
                && !context.model().joint_actions_public_after(simulation)
            {
                return Err(Error::Invalid(
                    "joint-action rollout reached a decision with hidden choices".into(),
                ));
            }
            let offers = self.joint_offers(simulation, context)?;
            let choices = self.joint_choices(&offers, None, PlayerId::new(usize::MAX), true)?;
            context.apply_joint(simulation, &choices)?;
            transitions = transitions.checked_add(1).ok_or(Error::Overflow)?;
        }
    }
}
