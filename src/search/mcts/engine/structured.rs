//! Component-node search. Partial edges retain the pre-action simulation;
//! only a leaf applies a complete move and consumes transition/depth budget.
use super::*;
use crate::core::ActionChoice;

impl<F> MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn search_structured_impl(
        &mut self,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        players: usize,
        effective: Settings,
        mut context: SimulationContext<'_, F>,
        cooperative: bool,
    ) -> Result<SearchReport<F::Action>> {
        let mut tree: Vec<Node<ActionChoice<F::Action>>> = vec![Node::default()];
        let mut prefixes = vec![Vec::<u32>::new()];
        let mut edge_count = 0usize;
        let mut forced_stop = None;
        while context.reasons().is_empty() && forced_stop.is_none() {
            let mut simulation = match context.fork(observation, &mut self.simulation_rng) {
                Ok(state) => state,
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            };
            let mut path = Vec::new();
            let iteration = self
                .structured_iteration(
                    &mut simulation,
                    &mut tree,
                    &mut prefixes,
                    &mut path,
                    &mut edge_count,
                    players,
                    &mut context,
                )
                .await;
            let values = match iteration {
                Ok(values) => values,
                Err(Error::ResourceLimit) => {
                    forced_stop = Some(StopReason::Resources);
                    break;
                }
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
            };
            match context.check_interrupt() {
                Ok(()) => {}
                Err(Error::BudgetExhausted) => break,
                Err(error) => return Err(error),
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
        let root_actions = actions
            .iter()
            .map(|action| {
                let path = context.model().action_path(observation, action);
                let mut node = 0;
                let mut edge = None;
                for component in path {
                    edge = tree[node].decisions.get(&actor.index()).and_then(|record| {
                        record
                            .edges
                            .iter()
                            .find(|edge| edge.action.component == component)
                    });
                    if let Some(found) = edge {
                        node = found.child;
                    } else {
                        break;
                    }
                }
                let complete = edge.filter(|edge| edge.action.action.as_ref() == Some(action));
                RootActionReport {
                    action: action.clone(),
                    visits: complete.map_or(0, |edge| edge.visits),
                    mean_values: complete.map_or_else(
                        || vec![0.0; players],
                        |edge| {
                            if edge.visits == 0 {
                                vec![0.0; players]
                            } else {
                                edge.value_sums
                                    .iter()
                                    .map(|sum| sum / edge.visits as f64)
                                    .collect()
                            }
                        },
                    ),
                }
            })
            .collect::<Vec<_>>();
        let visited = root_actions
            .iter()
            .enumerate()
            .filter_map(|(index, report)| (report.visits > 0).then_some(index))
            .collect::<Vec<_>>();
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
            edges: edge_count,
            effective_settings: effective,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn structured_iteration(
        &mut self,
        simulation: &mut F::State,
        tree: &mut Vec<Node<ActionChoice<F::Action>>>,
        prefixes: &mut Vec<Vec<u32>>,
        path: &mut Vec<PathEntry>,
        edge_count: &mut usize,
        players: usize,
        context: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        let mut node = 0;
        let mut depth = 0u64;
        let mut expanded = false;
        loop {
            context.check_interrupt()?;
            let actor = match context.model().next_player(simulation) {
                Decision::Finished => return self.evaluate(simulation, context).await,
                Decision::Player(actor) => actor,
            };
            if actor.index() >= context.model().player_count(simulation) {
                return Err(Error::Game("simulation returned an invalid actor".into()));
            }
            let prefix = &prefixes[node];
            if prefix.is_empty()
                && (self.settings.tree_depth.is_some_and(|limit| depth >= limit)
                    || context.remaining_transitions() == Some(0))
            {
                return self.evaluate(simulation, context).await;
            }
            let view = context
                .model()
                .observe(simulation, actor)
                .map_err(|error| Error::Game(error.to_string()))?;
            let actions = context
                .model()
                .available_actions(simulation, actor)
                .map_err(|error| Error::Game(error.to_string()))?;
            validate_actions(&actions, self.settings.limits.max_actions_per_decision)?;
            let choices = context
                .model()
                .action_choices(&view, prefix)
                .map_err(|error| Error::Invalid(error.to_string()))?;
            if choices.is_empty() {
                return Err(Error::Invalid("empty component choice list".into()));
            }
            let paths = actions
                .iter()
                .map(|action| context.model().action_path(&view, action))
                .collect::<Vec<_>>();
            let node_limit_reached = tree.len() >= self.settings.limits.max_nodes;
            let record = tree[node].decisions.entry(actor.index()).or_default();
            let unvisited = choices
                .iter()
                .enumerate()
                .filter_map(|(index, choice)| {
                    record
                        .edges
                        .iter()
                        .find(|edge| edge.action.component == choice.component)
                        .is_none_or(|edge| edge.visits == 0)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            let selected = if !unvisited.is_empty() {
                expanded = true;
                if let Some(evaluator) = self.evaluators.expansion.as_mut() {
                    let scores =
                        action_scores(evaluator.as_mut(), simulation, actor, &actions, context)
                            .await?;
                    let aggregated = aggregate(&scores, &paths, prefix, &choices, false)?;
                    let candidates = unvisited
                        .iter()
                        .map(|&index| aggregated[index])
                        .collect::<Vec<_>>();
                    unvisited[selection::best(&candidates, &mut self.rng)?]
                } else {
                    unvisited[self.rng.index(unvisited.len()).expect("nonempty choices")]
                }
            } else {
                let preferences = if self.settings.tree_policy == TreePolicy::Puct {
                    let probabilities = if let Some(evaluator) =
                        self.evaluators.preferences.as_mut()
                    {
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
                            PreferenceSource::Uniform => unreachable!("validated settings"),
                        }
                    } else {
                        vec![1.0 / actions.len() as f64; actions.len()]
                    };
                    aggregate(&probabilities, &paths, prefix, &choices, true)?
                } else {
                    vec![0.0; choices.len()]
                };
                let heuristics = if let Some(evaluator) = self.evaluators.bias.as_mut() {
                    let scores =
                        action_scores(evaluator.as_mut(), simulation, actor, &actions, context)
                            .await?;
                    aggregate(&scores, &paths, prefix, &choices, false)?
                } else {
                    vec![0.0; choices.len()]
                };
                let scores = choices
                    .iter()
                    .enumerate()
                    .map(|(index, choice)| {
                        let edge = record
                            .edges
                            .iter()
                            .find(|edge| edge.action.component == choice.component)
                            .expect("all components visited");
                        selection::tree_score(
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
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                selection::best(&scores, &mut self.rng)?
            };
            let chosen = choices[selected].clone();
            if record
                .edges
                .iter()
                .all(|edge| edge.action.component != chosen.component)
                && (node_limit_reached || *edge_count >= self.settings.limits.max_edges)
            {
                return Err(Error::ResourceLimit);
            }
            let edge_index = if let Some(index) = record
                .edges
                .iter()
                .position(|edge| edge.action.component == chosen.component)
            {
                index
            } else {
                let child = tree.len();
                let mut next_prefix = prefix.clone();
                if chosen.action.is_some() {
                    next_prefix.clear();
                } else {
                    next_prefix.push(chosen.component);
                }
                tree.push(Node::default());
                prefixes.push(next_prefix);
                let record = tree[node]
                    .decisions
                    .get_mut(&actor.index())
                    .expect("actor record");
                let index = record.edges.len();
                record.edges.push(Edge {
                    action: chosen.clone(),
                    child,
                    visits: 0,
                    value_sums: vec![0.0; players],
                });
                *edge_count = edge_count.checked_add(1).ok_or(Error::Overflow)?;
                index
            };
            let child = tree[node].decisions[&actor.index()].edges[edge_index].child;
            path.push(PathEntry {
                node,
                actor,
                edge: edge_index,
            });
            if let Some(action) = chosen.action {
                context.apply(simulation, actor, &action)?;
                depth = depth.checked_add(1).ok_or(Error::Overflow)?;
                if expanded {
                    if self.settings.tree_depth.is_some_and(|limit| depth >= limit) {
                        return self.evaluate(simulation, context).await;
                    }
                    return self.rollout(simulation, context).await;
                }
            }
            node = child;
        }
    }
}

/// Sum policy mass over descendant complete actions, or average heuristic
/// scores over the same descendants. Policy mass is conditional on the prefix.
fn aggregate<A>(
    scores: &[f64],
    paths: &[Vec<u32>],
    prefix: &[u32],
    choices: &[ActionChoice<A>],
    probability: bool,
) -> Result<Vec<f64>> {
    let mut sums = vec![0.0; choices.len()];
    let mut counts = vec![0usize; choices.len()];
    for (score, path) in scores.iter().zip(paths) {
        if path.len() <= prefix.len() || !path.starts_with(prefix) {
            continue;
        }
        let index = choices
            .iter()
            .position(|choice| choice.component == path[prefix.len()])
            .ok_or_else(|| Error::Invalid("action path has no legal component".into()))?;
        sums[index] += score;
        counts[index] += 1;
    }
    if counts.contains(&0) {
        return Err(Error::Invalid("component has no complete action".into()));
    }
    if probability {
        let total: f64 = sums.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::Invalid("invalid component policy mass".into()));
        }
        Ok(sums.into_iter().map(|sum| sum / total).collect())
    } else {
        Ok(sums
            .into_iter()
            .zip(counts)
            .map(|(sum, count)| sum / count as f64)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_action_policy_mass_is_conditioned_at_each_prefix() {
        let paths = vec![vec![1, 0], vec![1, 1], vec![2]];
        let weights = [0.2, 0.3, 0.5];
        let root = [
            ActionChoice::<()> {
                component: 1,
                action: None,
            },
            ActionChoice {
                component: 2,
                action: Some(()),
            },
        ];
        assert_eq!(
            aggregate(&weights, &paths, &[], &root, true).unwrap(),
            vec![0.5, 0.5]
        );
        let children = [
            ActionChoice {
                component: 0,
                action: Some(()),
            },
            ActionChoice {
                component: 1,
                action: Some(()),
            },
        ];
        let conditional = aggregate(&weights, &paths, &[1], &children, true).unwrap();
        assert!((conditional[0] - 0.4).abs() < 1e-12);
        assert!((conditional[1] - 0.6).abs() < 1e-12);
    }
}
