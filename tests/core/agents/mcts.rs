//! MCTS behavior tests include independent arithmetic and controlled simulation work.
use std::cell::Cell;
use std::rc::Rc;
use tag_rust::agents::Agent;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::connect4::{Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{PlaceMark, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::search::mcts::*;

fn settings(iterations: u64) -> Settings {
    Settings {
        seed: Some(42),
        budgets: Budgets {
            iterations: Some(iterations),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Settings::default()
    }
}

#[test]
fn completion_is_explicit_for_all_players_and_rejects_malformed_outputs() {
    let scalar = || StateValues::Scalar {
        player: PlayerId::new(1),
        value: 0.7,
    };
    assert!(complete_values(scalar(), 3, ValueCompletion::RequireFullVector).is_err());
    assert_eq!(
        complete_values(scalar(), 3, ValueCompletion::FillZeros).unwrap(),
        vec![0.0, 0.7, 0.0]
    );
    assert_eq!(
        complete_values(scalar(), 3, ValueCompletion::Negation).unwrap(),
        vec![-0.7, 0.7, -0.7]
    );
    for policy in [
        ValueCompletion::RequireFullVector,
        ValueCompletion::FillZeros,
        ValueCompletion::Negation,
    ] {
        let full = vec![0.8, 0.6, 0.1];
        assert_eq!(
            complete_values(StateValues::Full(full.clone()), 3, policy).unwrap(),
            full
        );
        assert!(complete_values(StateValues::Full(vec![0.1, 0.2]), 3, policy).is_err());
        assert!(complete_values(StateValues::Full(vec![0.1, f64::NAN, 0.2]), 3, policy).is_err());
        assert!(
            complete_values(
                StateValues::Scalar {
                    player: PlayerId::new(3),
                    value: 0.1
                },
                3,
                policy
            )
            .is_err()
        );
    }
}

// This controlled fixture varies the outcome of the same root action. It tests
// history bookkeeping without claiming a playable stochastic game or chance API.
#[derive(Clone, Debug, PartialEq)]
struct Position {
    depth: u64,
    branch: usize,
}
struct Model {
    transitions: Cell<u64>,
    varying: bool,
    terminal_depth: u64,
    opponent_choice: bool,
    fail_transition: bool,
}
impl Default for Model {
    fn default() -> Self {
        Self {
            transitions: Cell::new(0),
            varying: false,
            terminal_depth: 20,
            opponent_choice: false,
            fail_transition: false,
        }
    }
}
impl ForwardModel for Model {
    type State = Position;
    type Observation = Position;
    type Action = usize;
    type Parameters = ();
    type Error = String;
    fn init_state(&self, _: &(), _: usize) -> std::result::Result<Position, String> {
        Ok(Position {
            depth: 0,
            branch: 0,
        })
    }
    fn player_count(&self, _: &Position) -> usize {
        3
    }
    fn next_player(&self, state: &Position) -> Decision {
        if state.depth >= self.terminal_depth {
            Decision::Finished
        } else {
            Decision::Player(PlayerId::new(if self.opponent_choice && state.depth > 0 {
                1
            } else if state.depth == 0 {
                0
            } else {
                state.branch
            }))
        }
    }
    fn observe(&self, state: &Position, _: PlayerId) -> std::result::Result<Position, String> {
        Ok(state.clone())
    }
    fn offered_actions(&self, state: &Position) -> Vec<usize> {
        if self.next_player(state) == Decision::Finished {
            vec![]
        } else if state.depth == 0 {
            vec![0]
        } else if self.opponent_choice {
            vec![10, 11]
        } else {
            vec![10 + state.branch]
        }
    }
    fn available_actions(
        &self,
        state: &Position,
        actor: PlayerId,
    ) -> std::result::Result<Vec<usize>, String> {
        if self.next_player(state) != Decision::Player(actor) {
            return Err("wrong actor".into());
        }
        Ok(self.offered_actions(state))
    }
    fn next(
        &self,
        state: &mut Position,
        actor: PlayerId,
        action: &usize,
    ) -> std::result::Result<(), String> {
        if !self.available_actions(state, actor)?.contains(action) {
            return Err("unavailable action".into());
        }
        self.transitions.set(self.transitions.get() + 1);
        if state.depth == 0 && self.varying {
            state.branch = (self.transitions.get() % 2) as usize + 1;
        }
        if self.opponent_choice && state.depth == 1 {
            state.branch = *action - 10;
        }
        state.depth += 1;
        if self.fail_transition {
            return Err("injected internal failure after mutation".into());
        }
        Ok(())
    }
    fn player_result(
        &self,
        state: &Position,
        player: PlayerId,
    ) -> std::result::Result<PlayerResult, String> {
        Ok(if state.depth < self.terminal_depth {
            PlayerResult::Ongoing
        } else {
            match player.index() {
                0 => PlayerResult::Win,
                1 => PlayerResult::Draw,
                _ => PlayerResult::Loss,
            }
        })
    }
    fn player_score(&self, _: &Position, _: PlayerId) -> std::result::Result<f64, String> {
        Ok(0.0)
    }
}
impl PublicInformationModel for Model {
    fn simulation_state(&self, observation: &Position) -> std::result::Result<Position, String> {
        Ok(observation.clone())
    }
}
struct FixedValues;
impl StateEvaluator<Model> for FixedValues {
    fn evaluate(
        &mut self,
        _: &Position,
        _: &mut SimulationContext<'_, Model>,
    ) -> Result<StateValues> {
        Ok(StateValues::Full(vec![0.8, 0.6, 0.1]))
    }
}
fn registry() -> EvaluatorRegistry<Model> {
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(FixedValues)))
        .unwrap();
    registry
}
fn run(model: &Model, config: Settings) -> SearchReport<usize> {
    let mut agent = MctsAgent::new(config, &registry()).unwrap();
    agent
        .search(
            model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0,
            },
            &[0],
        )
        .unwrap()
}

#[test]
fn zero_rollout_evaluates_expanded_leaf_and_depth_limit_skips_rollouts() {
    let model = Model::default();
    let mut config = settings(1);
    config.rollout_length = Some(0);
    let report = run(&model, config.clone());
    assert_eq!(report.counters.forward_model_calls, 1);
    assert_eq!(report.root_actions[0].visits, 1);
    assert_eq!(report.root_actions[0].mean_values, vec![0.8, 0.6, 0.1]);
    config.tree_depth = Some(1);
    config.rollout_length = None;
    assert_eq!(run(&model, config.clone()).counters.forward_model_calls, 1);
    config.tree_depth = None;
    config.rollout_length = Some(2);
    assert_eq!(run(&model, config).counters.forward_model_calls, 3);
}

#[test]
fn open_loop_replays_actions_and_keeps_live_source_independent() {
    let model = Model::default();
    let mut config = settings(3);
    config.rollout_length = Some(0);
    let report = run(&model, config);
    // Three simulations traverse one, two and three actions from the root.
    assert_eq!(report.counters.forward_model_calls, 6);
    assert_eq!(model.transitions.get(), 6);
    assert_eq!(report.counters.state_copies, 4);
    assert_eq!(report.root_actions[0].visits, 3);
    assert_eq!(report.nodes, 4);
}

#[test]
fn varying_actors_and_legality_do_not_replay_unavailable_actions() {
    let model = Model {
        varying: true,
        ..Model::default()
    };
    let mut config = settings(50);
    config.rollout_length = Some(0);
    let report = run(&model, config);
    assert_eq!(report.counters.completed_iterations, 50);
    assert_eq!(report.counters.forward_model_calls, model.transitions.get());
    assert_eq!(report.root_actions[0].visits, 50);
}

#[test]
fn strict_transition_budget_completes_current_iteration_at_its_current_position() {
    for cap in [0, 1, 2, 5, 20] {
        let model = Model::default();
        let mut config = settings(100);
        config.rollout_length = None;
        config.budgets.forward_model_calls = Some(cap);
        let report = run(&model, config);
        assert_eq!(report.counters.forward_model_calls, cap);
        assert_eq!(model.transitions.get(), cap);
        assert!(report.stop_reasons.contains(&StopReason::ForwardModel));
        assert_eq!(report.fallback, cap == 0);
        assert_eq!(report.counters.completed_iterations, u64::from(cap > 0));
        if cap == 20 {
            assert_eq!(report.root_actions[0].mean_values, vec![1.0, 0.0, -1.0]);
        }
    }
}

#[test]
fn resource_limit_is_reported_without_inventing_root_visits() {
    let mut config = settings(100);
    config.limits.max_nodes = 1;
    let report = run(&Model::default(), config);
    assert!(report.fallback);
    assert_eq!(report.nodes, 1);
    assert_eq!(report.edges, 0);
    assert_eq!(report.stop_reasons, vec![StopReason::Resources]);
}

#[test]
fn terminal_rewards_bypass_an_evaluator_that_would_fail() {
    struct Failing;
    impl StateEvaluator<Model> for Failing {
        fn evaluate(
            &mut self,
            _: &Position,
            _: &mut SimulationContext<'_, Model>,
        ) -> Result<StateValues> {
            panic!("terminal positions must not call state evaluator")
        }
    }
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Failing)))
        .unwrap();
    let model = Model {
        terminal_depth: 1,
        ..Model::default()
    };
    let mut agent = MctsAgent::new(settings(1), &registry).unwrap();
    let report = agent
        .search(
            &model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0,
            },
            &[0],
        )
        .unwrap();
    assert_eq!(report.root_actions[0].mean_values, vec![1.0, 0.0, -1.0]);
    assert_eq!(report.counters.state_evaluations, 0);
}

struct ManualClock(Rc<Cell<f64>>);
impl Clock for ManualClock {
    fn now_ms(&self) -> f64 {
        self.0.get()
    }
}
struct SlowValues(Rc<Cell<f64>>);
impl StateEvaluator<Model> for SlowValues {
    fn evaluate(
        &mut self,
        _: &Position,
        _: &mut SimulationContext<'_, Model>,
    ) -> Result<StateValues> {
        self.0.set(20.0);
        Ok(StateValues::Full(vec![0.8, 0.6, 0.1]))
    }
}
#[test]
fn deadline_discards_unfinished_expansion_even_at_transition_exhaustion() {
    let clock = Rc::new(Cell::new(0.0));
    let captured = Rc::clone(&clock);
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", move |_| {
            Ok(Box::new(SlowValues(Rc::clone(&captured))))
        })
        .unwrap();
    let mut config = settings(100);
    config.rollout_length = Some(0);
    config.budgets.time_ms = Some(10.0);
    config.budgets.forward_model_calls = Some(1);
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let model = Model::default();
    let report = agent
        .search_controlled(
            &model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0,
            },
            &[0],
            &BudgetOverrides::default(),
            &ManualClock(clock),
            &|| false,
        )
        .unwrap();
    assert_eq!(report.counters.completed_iterations, 0);
    assert_eq!(report.counters.forward_model_calls, 1);
    assert_eq!(report.edges, 0);
    assert_eq!(
        report.stop_reasons,
        vec![StopReason::Time, StopReason::ForwardModel]
    );
    assert!(report.fallback);
}

#[test]
fn cancellation_never_returns_a_selected_move() {
    let mut agent = MctsAgent::new(settings(10), &registry()).unwrap();
    let model = Model::default();
    assert_eq!(
        agent.search_controlled(
            &model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0
            },
            &[0],
            &BudgetOverrides::default(),
            &ManualClock(Rc::new(Cell::new(0.0))),
            &|| true
        ),
        Err(Error::Cancelled)
    );
}

fn successor_spec() -> EvaluatorSpec {
    let nested = Parameters::from([("name".into(), Parameter::Text("game_heuristic".into()))]);
    EvaluatorSpec {
        name: "successor_state".into(),
        parameters: Parameters::from([("state_evaluator".into(), Parameter::Object(nested))]),
    }
}
#[test]
fn action_heuristics_charge_their_successor_transitions_and_preflight_whole_lists() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let before = state.clone();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let mut config = settings(1);
    config.expansion = ExpansionSettings {
        order: ExpansionOrder::Heuristic,
        evaluator: Some(successor_spec()),
    };
    config.rollout_length = Some(0);
    config.budgets.forward_model_calls = Some(10);
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    let mut agent = MctsAgent::new(config.clone(), &registry).unwrap();
    let report = agent.search(&model, actor, &observation, &actions).unwrap();
    assert_eq!(report.counters.forward_model_calls, 10); // Nine scored successors and one expansion.
    assert_eq!(report.counters.state_copies, 11); // Root, iteration and nine scores.
    assert!(!report.fallback);
    assert_eq!(state, before);
    config.budgets.forward_model_calls = Some(8);
    let report = MctsAgent::new(config, &registry)
        .unwrap()
        .search(&model, actor, &observation, &actions)
        .unwrap();
    assert_eq!(report.counters.forward_model_calls, 0);
    assert!(report.fallback);
    assert!(report.stop_reasons.contains(&StopReason::ForwardModel));
}

#[test]
fn snapshot_restores_next_search_and_resolves_seed_once() {
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    let mut config = settings(40);
    config.seed = None;
    let calls = Cell::new(0);
    let mut agent = MctsAgent::with_seed_source(config, &registry, || {
        calls.set(calls.get() + 1);
        Ok(987)
    })
    .unwrap();
    assert_eq!(agent.settings().seed, Some(987));
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    agent.search(&model, actor, &observation, &actions).unwrap();
    let checkpoint = agent.snapshot().unwrap();
    let mut restored = MctsAgent::from_snapshot(&checkpoint, &registry).unwrap();
    let original = agent.search(&model, actor, &observation, &actions).unwrap();
    let resumed = restored
        .search(&model, actor, &observation, &actions)
        .unwrap();
    assert_eq!(original.root_actions, resumed.root_actions);
    assert_eq!(original.selected_index, resumed.selected_index);
    assert_eq!(original.counters, resumed.counters);
    assert_eq!(agent.rng().snapshot(), restored.rng().snapshot());
    assert_eq!(calls.get(), 1);
    assert!(
        MctsAgent::new(settings(1), &self::registry())
            .unwrap()
            .snapshot()
            .is_err()
    );
}

#[test]
fn explicit_seed_does_not_call_automatic_source_and_native_source_records_its_seed() {
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    MctsAgent::with_seed_source(settings(1), &registry, || {
        panic!("explicit seed must not call source")
    })
    .unwrap();
    let mut config = settings(1);
    config.seed = None;
    let agent = MctsAgent::new(config, &registry).unwrap();
    assert_eq!(agent.settings().seed, Some(agent.rng().initial_seed()));
}

#[test]
fn native_search_completes_both_games_in_both_seats() {
    fn play<F>(model: F, parameters: F::Parameters, registry: EvaluatorRegistry<F>)
    where
        F: PublicInformationModel + 'static,
        F::State: Clone,
        F::Action: Clone + PartialEq,
        F::Error: std::fmt::Display + std::fmt::Debug,
    {
        for mcts_seat in 0..2 {
            let mut state = model.init_state(&parameters, 2).unwrap();
            let mut agent = MctsAgent::new(settings(100), &registry).unwrap();
            let mut moves = 0;
            while let Decision::Player(actor) = model.next_player(&state) {
                let observation = model.observe(&state, actor).unwrap();
                let actions = model.offered_actions(&observation);
                let index = if actor.index() == mcts_seat {
                    agent.choose(&model, actor, &observation, &actions).unwrap()
                } else {
                    0
                };
                model.next(&mut state, actor, &actions[index]).unwrap();
                moves += 1;
                assert!(moves <= 100);
            }
            assert!(moves > 0);
        }
    }
    play(
        TicTacToeForwardModel,
        TicTacToeParameters::default(),
        EvaluatorRegistry::<TicTacToeForwardModel>::for_game(),
    );
    play(
        Connect4ForwardModel,
        Connect4Parameters::default(),
        EvaluatorRegistry::<Connect4ForwardModel>::for_game(),
    );
}

#[test]
fn native_search_takes_an_immediate_win() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for (i, (x, y)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        model
            .next(&mut state, PlayerId::new(i % 2), &PlaceMark { x, y })
            .unwrap();
    }
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let mut agent = MctsAgent::new(
        settings(500),
        &EvaluatorRegistry::<TicTacToeForwardModel>::for_game(),
    )
    .unwrap();
    let chosen = agent.choose(&model, actor, &observation, &actions).unwrap();
    assert_eq!(actions[chosen], PlaceMark { x: 0, y: 2 });
}

#[test]
fn opponent_selection_maximizes_its_own_value_instead_of_minimizing_root_value() {
    struct ConflictingValues;
    impl StateEvaluator<Model> for ConflictingValues {
        fn evaluate(
            &mut self,
            state: &Position,
            _: &mut SimulationContext<'_, Model>,
        ) -> Result<StateValues> {
            Ok(StateValues::Full(
                if state.depth >= 2 && state.branch == 0 {
                    vec![0.9, 0.8, 0.1]
                } else {
                    vec![0.1, 0.2, 0.7]
                },
            ))
        }
    }
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(ConflictingValues)))
        .unwrap();
    let model = Model {
        opponent_choice: true,
        ..Model::default()
    };
    let mut config = settings(500);
    config.tree_depth = Some(2);
    config.rollout_length = Some(0);
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let report = agent
        .search(
            &model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0,
            },
            &[0],
        )
        .unwrap();
    // Player 1 favors the .8 outcome, which also gives Player 0 .9. A paranoid
    // opponent minimizing Player 0 would favor the other branch instead.
    assert!(report.root_actions[0].mean_values[0] > 0.7);
    assert!(report.root_actions[0].mean_values[1] > 0.6);
}

#[test]
fn every_native_policy_combination_runs_and_restores_its_independent_evaluators() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    for tree in [TreePolicy::Ucb1, TreePolicy::Puct] {
        for bias in [BiasMode::None, BiasMode::Additive, BiasMode::TagBlend] {
            for rollout in [
                RolloutPolicy::Random,
                RolloutPolicy::Greedy,
                RolloutPolicy::Softmax,
            ] {
                for final_policy in [
                    FinalPolicy::MostVisits,
                    FinalPolicy::HighestMean,
                    FinalPolicy::VisitSampling,
                ] {
                    let mut config = settings(30);
                    config.tree_policy = tree;
                    config.progressive_bias.mode = bias;
                    if bias != BiasMode::None {
                        config.progressive_bias.evaluator =
                            Some(EvaluatorSpec::named("uniform_actions"));
                    }
                    config.rollout.policy = rollout;
                    if rollout != RolloutPolicy::Random {
                        config.rollout.evaluator = Some(successor_spec());
                    }
                    config.final_selection.policy = final_policy;
                    let mut agent = MctsAgent::new(config, &registry).unwrap();
                    let checkpoint = agent.snapshot().unwrap();
                    let mut restored = MctsAgent::from_snapshot(&checkpoint, &registry).unwrap();
                    let report = agent.search(&model, actor, &observation, &actions).unwrap();
                    let restored_report = restored
                        .search(&model, actor, &observation, &actions)
                        .unwrap();
                    assert_eq!(report.counters.completed_iterations, 30);
                    assert_eq!(report.root_actions, restored_report.root_actions);
                    assert_eq!(report.selected_index, restored_report.selected_index);
                    assert_eq!(agent.rng().snapshot(), restored.rng().snapshot());
                }
            }
        }
    }
}

#[test]
fn puct_accepts_all_three_sources_and_rejects_zero_probability_mass() {
    struct ConstantScores(f64);
    impl ActionEvaluator<TicTacToeForwardModel> for ConstantScores {
        fn evaluate(
            &mut self,
            _: &tag_rust::games::tictactoe::TicTacToeState,
            _: PlayerId,
            actions: &[PlaceMark],
            _: &mut SimulationContext<'_, TicTacToeForwardModel>,
        ) -> Result<Vec<f64>> {
            Ok(vec![self.0; actions.len()])
        }
    }
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let mut registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    registry
        .register_action("ones", |_| Ok(Box::new(ConstantScores(1.0))))
        .unwrap();
    for source in [
        PreferenceSource::Uniform,
        PreferenceSource::NativeSoftmax,
        PreferenceSource::Probabilities,
    ] {
        let mut config = settings(30);
        config.tree_policy = TreePolicy::Puct;
        config.puct_preferences.source = source;
        if source != PreferenceSource::Uniform {
            config.puct_preferences.evaluator = Some(EvaluatorSpec::named("ones"));
        }
        let report = MctsAgent::new(config, &registry)
            .unwrap()
            .search(&model, actor, &observation, &actions)
            .unwrap();
        assert_eq!(report.counters.completed_iterations, 30);
    }
    let mut config = settings(30);
    config.tree_policy = TreePolicy::Puct;
    config.puct_preferences.source = PreferenceSource::Probabilities;
    config.puct_preferences.evaluator = Some(EvaluatorSpec::named("uniform_actions"));
    assert!(
        MctsAgent::new(config, &registry)
            .unwrap()
            .search(&model, actor, &observation, &actions)
            .is_err()
    );
}

#[test]
fn native_scalar_outputs_use_the_configured_completion_policy() {
    struct Scalar;
    impl StateEvaluator<Model> for Scalar {
        fn evaluate(
            &mut self,
            _: &Position,
            _: &mut SimulationContext<'_, Model>,
        ) -> Result<StateValues> {
            Ok(StateValues::Scalar {
                player: PlayerId::new(1),
                value: 0.7,
            })
        }
    }
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Scalar)))
        .unwrap();
    let model = Model::default();
    for (policy, expected) in [
        (ValueCompletion::FillZeros, vec![0.0, 0.7, 0.0]),
        (ValueCompletion::Negation, vec![-0.7, 0.7, -0.7]),
    ] {
        let mut config = settings(1);
        config.rollout_length = Some(0);
        config.value_completion = policy;
        let report = MctsAgent::new(config, &registry)
            .unwrap()
            .search(
                &model,
                PlayerId::new(0),
                &Position {
                    depth: 0,
                    branch: 0,
                },
                &[0],
            )
            .unwrap();
        assert_eq!(report.root_actions[0].mean_values, expected);
        assert_eq!(report.effective_settings.value_completion, policy);
    }
    let mut config = settings(1);
    config.rollout_length = Some(0);
    assert!(
        MctsAgent::new(config, &registry)
            .unwrap()
            .search(
                &model,
                PlayerId::new(0),
                &Position {
                    depth: 0,
                    branch: 0
                },
                &[0]
            )
            .is_err()
    );
}

#[test]
fn game_specific_terminal_rewards_are_not_replaced_by_win_loss_or_completion() {
    struct ScoreRewards;
    impl StateEvaluator<Model> for ScoreRewards {
        fn evaluate(
            &mut self,
            _: &Position,
            _: &mut SimulationContext<'_, Model>,
        ) -> Result<StateValues> {
            panic!("terminal only fixture")
        }
        fn terminal(
            &mut self,
            _: &Position,
            _: &mut SimulationContext<'_, Model>,
        ) -> Result<Vec<f64>> {
            Ok(vec![5.0, 2.0, 1.0])
        }
    }
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(ScoreRewards)))
        .unwrap();
    let mut config = settings(1);
    config.value_completion = ValueCompletion::Negation;
    let model = Model {
        terminal_depth: 1,
        ..Model::default()
    };
    let report = MctsAgent::new(config, &registry)
        .unwrap()
        .search(
            &model,
            PlayerId::new(0),
            &Position {
                depth: 0,
                branch: 0,
            },
            &[0],
        )
        .unwrap();
    assert_eq!(report.root_actions[0].mean_values, vec![5.0, 2.0, 1.0]);
}

#[test]
fn exhaustion_during_tree_replay_finishes_that_iteration_without_expansion() {
    let model = Model::default();
    let mut config = settings(10);
    config.rollout_length = Some(0);
    config.budgets.forward_model_calls = Some(2);
    let report = run(&model, config);
    assert_eq!(report.counters.completed_iterations, 2);
    assert_eq!(report.counters.forward_model_calls, 2);
    assert_eq!(report.root_actions[0].visits, 2);
    assert_eq!(report.edges, 1);
}

#[test]
fn cancelling_after_simulated_actions_preserves_the_original_position() {
    let model = Model::default();
    let observation = Position {
        depth: 0,
        branch: 0,
    };
    let mut agent = MctsAgent::new(settings(50), &registry()).unwrap();
    let result = agent.search_controlled(
        &model,
        PlayerId::new(0),
        &observation,
        &[0],
        &BudgetOverrides::default(),
        &ManualClock(Rc::new(Cell::new(0.0))),
        &|| model.transitions.get() >= 3,
    );
    assert_eq!(result, Err(Error::Cancelled));
    assert_eq!(model.transitions.get(), 3);
    assert_eq!(
        observation,
        Position {
            depth: 0,
            branch: 0
        }
    );
    assert!(agent.last_report().is_none());
}

#[test]
fn root_validation_and_output_limits_reject_invalid_decisions() {
    let model = Model::default();
    let observation = Position {
        depth: 0,
        branch: 0,
    };
    let mut agent = MctsAgent::new(settings(5), &registry()).unwrap();
    for actions in [vec![], vec![0, 0], vec![1]] {
        assert!(
            agent
                .search(&model, PlayerId::new(0), &observation, &actions)
                .is_err()
        );
    }
    assert!(
        agent
            .search(&model, PlayerId::new(1), &observation, &[0])
            .is_err()
    );
    let mut config = settings(5);
    config.limits.max_request_bytes = 8;
    assert_eq!(
        MctsAgent::new(config, &registry()).unwrap().search(
            &model,
            PlayerId::new(0),
            &observation,
            &[0]
        ),
        Err(Error::ResourceLimit)
    );
    assert_eq!(model.transitions.get(), 0);
}

#[cfg(feature = "mcts-json")]
#[test]
fn checked_in_json_files_load_and_effective_settings_round_trip() {
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    for text in [
        include_str!("../../../examples/mcts/basic.json"),
        include_str!("../../../examples/mcts/puct.json"),
    ] {
        let settings = Settings::from_json(text).unwrap();
        let agent = MctsAgent::with_seed_source(settings, &registry, || Ok(57)).unwrap();
        let restored = Settings::from_json(&agent.settings().to_json().unwrap()).unwrap();
        assert_eq!(&restored, agent.settings());
        assert!(restored.seed.is_some());
    }
}

#[cfg(feature = "mcts-json")]
#[test]
fn json_defaults_nulls_overrides_and_invalid_combinations_are_checked() {
    let minimal = Settings::from_json(r#"{"schema_version":1}"#).unwrap();
    assert_eq!(minimal, Settings::default());
    assert_eq!(
        Settings::from_json(&minimal.to_json().unwrap()).unwrap(),
        minimal
    );
    let configured = Settings::from_json(r#"{"schema_version":1,"seed":null,"tree_depth":null,"rollout_length":0,"budgets":{"time_ms":null,"forward_model_calls":20}}"#).unwrap();
    assert_eq!(configured.tree_depth, None);
    assert_eq!(configured.rollout_length, Some(0));
    assert_eq!(configured.budgets.iterations, None);
    let overrides =
        BudgetOverrides::from_json(r#"{"iterations":null,"forward_model_calls":5}"#).unwrap();
    let merged = overrides.apply(&minimal.budgets).unwrap();
    assert_eq!(merged.iterations, None);
    assert_eq!(merged.forward_model_calls, Some(5));
    assert!(
        BudgetOverrides::from_json(r#"{"iterations":null}"#)
            .unwrap()
            .apply(&minimal.budgets)
            .is_err()
    );
    for text in [
        r#"{}"#,
        r#"{"schema_version":2}"#,
        r#"{"schema_version":1,"unknown":1}"#,
        r#"{"schema_version":1,"tree_depth":0}"#,
        r#"{"schema_version":1,"budgets":{}}"#,
        r#"{"schema_version":1,"rollout":{"temperature":1}}"#,
        r#"{"schema_version":1,"progressive_bias":{"evaluator":null}}"#,
    ] {
        assert!(Settings::from_json(text).is_err(), "accepted {text}");
    }
}

#[cfg(feature = "inference")]
#[test]
fn local_search_drivers_preserve_reports_rng_and_discarded_work() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    use tag_rust::core::random::AgentRng;
    use tag_rust::inference::{LocalMctsSearch, build_local_mcts};

    let clock = ManualClock(Rc::new(Cell::new(0.0)));
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
    let config = settings(40);
    let mut sync_rng = AgentRng::new(987);
    let mut async_rng = AgentRng::new(987);
    let mut reference = build_local_mcts(&config, &sync_rng, None, &registry).unwrap();
    let mut context = Context::from_waker(Waker::noop());
    // Use several decisions to check continuation beyond the initial seed.
    for _ in 0..3 {
        let expected = reference
            .search_controlled(
                &model,
                actor,
                &observation,
                &actions,
                &Default::default(),
                &clock,
                &|| false,
            )
            .unwrap();
        let actual = LocalMctsSearch::new(&config, &mut sync_rng, None, &registry)
            .unwrap()
            .search(&model, actor, &observation, &actions, &clock)
            .unwrap();
        assert_eq!(actual, expected);
        let search = LocalMctsSearch::new(&config, &mut async_rng, None, &registry).unwrap();
        let mut future =
            Box::pin(search.search_async(&model, actor, &observation, &actions, &clock));
        let asynchronous = loop {
            if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
                break result.unwrap();
            }
        };
        drop(future);
        assert_eq!(asynchronous, expected);
        assert_eq!(sync_rng.snapshot(), reference.rng().snapshot());
        assert_eq!(async_rng.snapshot(), sync_rng.snapshot());
    }
    let before = async_rng.snapshot();
    let search = LocalMctsSearch::new(&config, &mut async_rng, None, &registry).unwrap();
    let mut future = Box::pin(search.search_async(&model, actor, &observation, &actions, &clock));
    assert!(future.as_mut().poll(&mut context).is_pending());
    drop(future);
    assert_eq!(async_rng.snapshot(), before);
}

#[test]
fn internal_transition_error_aborts_search_without_reusing_the_failed_simulation() {
    let mut model = Model {
        fail_transition: true,
        ..Model::default()
    };
    let view = model.init_state(&(), 3).unwrap();
    let before = view.clone();
    let mut agent = MctsAgent::new(settings(4), &registry()).unwrap();
    let result = agent.search(&model, PlayerId::new(0), &view, &[0]);
    assert!(
        matches!(result, Err(Error::Game(message)) if message.contains("internal failure after mutation"))
    );
    assert_eq!(model.transitions.get(), 1); // Failure is not retried or scored.
    assert_eq!(view, before); // The real/root observation was never mutated.
    assert!(agent.last_report().is_none());
    model.fail_transition = false;
    assert!(agent.search(&model, PlayerId::new(0), &view, &[0]).is_ok());
}
