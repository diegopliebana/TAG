//! Selection, expansion and rollout operate on one independent simulation.
use super::*;

impl<F> MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// Select a tree path, expand one action when possible, and evaluate the
    /// position reached by the rollout or a depth limit. The caller then
    /// back-propagates the returned player values along `path`.
    pub(super) async fn iteration(
        &mut self,
        simulation: &mut F::State,
        tree: &mut [Node<F::Action>],
        path: &mut Vec<PathEntry>,
        expansion: &mut Option<PendingExpansion<F::Action>>,
        edge_count: usize,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        let mut node = 0;
        let mut depth = 0u64;
        loop {
            // The state may differ between visits to this history in a future
            // stochastic game. Read its actor and legal actions on every visit.
            context.check_interrupt()?;
            let actor = match context.model().next_player(simulation) {
                Decision::Finished => return self.evaluate(simulation, context).await,
                Decision::Player(actor) => actor,
            };
            if actor.index() >= context.model().player_count(simulation) {
                return Err(Error::Game("simulation returned an invalid actor".into()));
            }
            if self.settings.tree_depth.is_some_and(|limit| depth >= limit)
                || context.remaining_transitions() == Some(0)
            {
                return self.evaluate(simulation, context).await;
            }
            let actions = context
                .model()
                .available_actions(simulation, actor)
                .map_err(|e| Error::Game(e.to_string()))?;
            validate_actions(&actions, self.settings.limits.max_actions_per_decision)?;
            let record = tree[node].decisions.entry(actor.index()).or_default();
            let unvisited: Vec<_> = actions
                .iter()
                .enumerate()
                .filter_map(|(index, action)| {
                    (!record.edges.iter().any(|edge| edge.action == *action)).then_some(index)
                })
                .collect();
            if !unvisited.is_empty() {
                // Expansion selects one new action. Insert its edge only after
                // evaluation succeeds, so an interrupted iteration adds no visits.
                if tree.len() >= self.settings.limits.max_nodes
                    || edge_count >= self.settings.limits.max_edges
                {
                    return Err(Error::ResourceLimit);
                }
                let index = if let Some(evaluator) = self.evaluators.expansion.as_mut() {
                    let scores =
                        action_scores(evaluator.as_mut(), simulation, actor, &actions, context)
                            .await?;
                    let candidates: Vec<_> = unvisited.iter().map(|index| scores[*index]).collect();
                    unvisited[selection::best(&candidates, &mut self.rng)?]
                } else {
                    unvisited[self
                        .rng
                        .index(unvisited.len())
                        .expect("nonempty unvisited actions")]
                };
                context.apply(simulation, actor, &actions[index])?;
                *expansion = Some(PendingExpansion {
                    node,
                    actor,
                    action: actions[index].clone(),
                });
                depth = depth.checked_add(1).ok_or(Error::Overflow)?;
                if self.settings.tree_depth.is_some_and(|limit| depth >= limit) {
                    return self.evaluate(simulation, context).await;
                }
                return self.rollout(simulation, context).await;
            }
            let preferences = if self.settings.tree_policy == TreePolicy::Puct {
                match self.evaluators.preferences.as_mut() {
                    None => vec![1.0 / actions.len() as f64; actions.len()],
                    Some(evaluator) => {
                        // Identify the policy's tree node explicitly. A player can
                        // act again below the root, so its identity is insufficient.
                        context.policy_node = Some(node);
                        let result =
                            action_scores(evaluator.as_mut(), simulation, actor, &actions, context)
                                .await;
                        context.policy_node = None;
                        let scores = result?;
                        match self.settings.puct_preferences.source {
                            PreferenceSource::NativeSoftmax => selection::softmax(
                                &scores,
                                self.settings.puct_preferences.temperature.unwrap_or(1.0),
                            )?,
                            PreferenceSource::Probabilities => selection::probabilities(&scores)?,
                            PreferenceSource::Uniform => {
                                unreachable!("validated preference settings")
                            }
                        }
                    }
                }
            } else {
                vec![0.0; actions.len()]
            };
            let heuristics = match self.evaluators.bias.as_mut() {
                Some(evaluator) => {
                    action_scores(evaluator.as_mut(), simulation, actor, &actions, context).await?
                }
                None => vec![0.0; actions.len()],
            };
            let mut scores = Vec::with_capacity(actions.len());
            // Each decision maker uses its own value component. This also works
            // when the same player receives consecutive decisions.
            for (index, action) in actions.iter().enumerate() {
                let edge = record
                    .edges
                    .iter()
                    .find(|edge| edge.action == *action)
                    .expect("all legal actions visited");
                scores.push(selection::tree_score(
                    self.settings.tree_policy,
                    self.settings.exploration_constant,
                    record.visits,
                    edge.visits,
                    edge.value_sums[actor.index()] / edge.visits as f64,
                    preferences[index],
                    (
                        self.settings.progressive_bias.mode,
                        heuristics[index],
                        self.settings.progressive_bias.bias_visits.unwrap_or(1.0),
                    ),
                )?);
            }
            let index = selection::best(&scores, &mut self.rng)?;
            let edge_index = record
                .edges
                .iter()
                .position(|edge| edge.action == actions[index])
                .expect("selected edge exists");
            context.apply(simulation, actor, &actions[index])?;
            path.push(PathEntry {
                node,
                actor,
                edge: edge_index,
            });
            node = record.edges[edge_index].child;
            depth = depth.checked_add(1).ok_or(Error::Overflow)?;
        }
    }
    /// Simulate actions after expansion using the independently configured
    /// rollout policy. Length zero evaluates this expanded leaf immediately.
    async fn rollout(
        &mut self,
        simulation: &mut F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        let mut actions_taken = 0u64;
        loop {
            // Rollouts use a separate policy. Enabling a tree action heuristic
            // never implicitly enables it here.
            context.check_interrupt()?;
            let actor = match context.model().next_player(simulation) {
                Decision::Finished => return self.evaluate(simulation, context).await,
                Decision::Player(actor) => actor,
            };
            if self
                .settings
                .rollout_length
                .is_some_and(|limit| actions_taken >= limit)
                || context.remaining_transitions() == Some(0)
            {
                return self.evaluate(simulation, context).await;
            }
            let actions = context
                .model()
                .available_actions(simulation, actor)
                .map_err(|e| Error::Game(e.to_string()))?;
            validate_actions(&actions, self.settings.limits.max_actions_per_decision)?;
            let index = match self.evaluators.rollout.as_mut() {
                None => self
                    .rng
                    .index(actions.len())
                    .expect("nonempty legal actions"),
                Some(evaluator) => {
                    let scores =
                        action_scores(evaluator.as_mut(), simulation, actor, &actions, context)
                            .await?;
                    match self.settings.rollout.policy {
                        RolloutPolicy::Greedy => selection::best(&scores, &mut self.rng)?,
                        RolloutPolicy::Softmax => selection::sample(
                            &selection::softmax(
                                &scores,
                                self.settings.rollout.temperature.unwrap_or(1.0),
                            )?,
                            &mut self.rng,
                        )?,
                        RolloutPolicy::Random => unreachable!("validated rollout settings"),
                    }
                }
            };
            context.apply(simulation, actor, &actions[index])?;
            actions_taken = actions_taken.checked_add(1).ok_or(Error::Overflow)?;
        }
    }
    pub(super) async fn evaluate(
        &mut self,
        state: &F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        context.check_interrupt()?;
        let count = context.model().player_count(state);
        context.require_values(count)?;
        let values = if context.model().next_player(state) == Decision::Finished {
            complete_values(
                StateValues::Full(self.evaluators.state.terminal(state, context)?),
                count,
                ValueCompletion::RequireFullVector,
            )?
        } else {
            increment(&mut context.counters.state_evaluations)?;
            complete_values(
                self.evaluators.state.evaluate_async(state, context).await?,
                count,
                self.settings.value_completion,
            )?
        };
        context.check_interrupt()?;
        Ok(values)
    }
    pub(super) fn final_action(
        &mut self,
        actions: &[RootActionReport<F::Action>],
        visited: &[usize],
        actor: PlayerId,
    ) -> Result<usize> {
        let scores: Vec<_> = visited
            .iter()
            .map(|index| match self.settings.final_selection.policy {
                FinalPolicy::MostVisits => actions[*index].visits as f64,
                FinalPolicy::HighestMean => actions[*index].mean_values[actor.index()],
                FinalPolicy::VisitSampling => (actions[*index].visits as f64).ln(),
            })
            .collect();
        let chosen = if self.settings.final_selection.policy == FinalPolicy::VisitSampling {
            selection::sample(
                &selection::softmax(
                    &scores,
                    self.settings.final_selection.temperature.unwrap_or(1.0),
                )?,
                &mut self.rng,
            )?
        } else {
            selection::best(&scores, &mut self.rng)?
        };
        Ok(visited[chosen])
    }
}
