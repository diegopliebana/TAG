//! External callers use the same ForwardModel and runner as deterministic games.
use tag_rust::agents::RandomAgent;
use tag_rust::core::{
    Decision, ForwardModel, PlayerId, PlayerResult, checkpoint::Checkpoint, evaluation::Evaluator,
    random::SimulationRng, simulation::SimulationModel,
};
use tag_rust::games::cantstop::*;

#[test]
fn public_api_runner_and_exact_agent_game_continuation() {
    let model = CantStopForwardModel;
    for players in 2..=4 {
        let mut state = model
            .init_state(
                &CantStopParameters {
                    seed: 2026,
                    ..Default::default()
                },
                players,
            )
            .unwrap();
        let mut agents: Vec<_> = (0..players)
            .map(|seat| RandomAgent::new(100 + seat as u64))
            .collect();
        let mut restored_state = CantStopState::from_snapshot(&state.snapshot()).unwrap();
        let mut restored_agents: Vec<_> = agents
            .iter()
            .map(|a| RandomAgent::from_snapshot(&a.snapshot()).unwrap())
            .collect();
        let mut trace = Vec::new();
        tag_rust::runner::run_game_observed(
            &model,
            &mut state,
            &mut agents,
            |_, state, actor, action, _| {
                trace.push((
                    actor,
                    action,
                    state.position().clone(),
                    state.chance_snapshot(),
                ));
            },
        )
        .unwrap();
        let mut restored_trace = Vec::new();
        tag_rust::runner::run_game_observed(
            &model,
            &mut restored_state,
            &mut restored_agents,
            |_, state, actor, action, _| {
                restored_trace.push((
                    actor,
                    action,
                    state.position().clone(),
                    state.chance_snapshot(),
                ));
            },
        )
        .unwrap();
        assert_eq!(trace, restored_trace);
        assert_eq!(state, restored_state);
        assert_eq!(model.next_player(&state), Decision::Finished);
        let outcomes: Vec<_> = (0..players)
            .map(|seat| model.player_result(&state, PlayerId::new(seat)).unwrap())
            .collect();
        assert_eq!(
            outcomes.iter().filter(|&&r| r == PlayerResult::Win).count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|&&r| r == PlayerResult::Loss)
                .count(),
            players - 1
        );
        assert!(
            CantStopProgress::default()
                .evaluate(&state, PlayerId::new(0))
                .unwrap()
                .abs()
                == 1.0
        );
    }
}

#[test]
fn forks_from_nonacting_views_and_terminal_views_preserve_public_facts() {
    let model = CantStopForwardModel;
    let mut state = model.init_state(&CantStopParameters::default(), 2).unwrap();
    let mut random = SimulationRng::new(1);
    for seat in 0..2 {
        let observation = model.observe(&state, PlayerId::new(seat)).unwrap();
        assert_eq!(observation.observer(), PlayerId::new(seat));
        let fork = model.simulation_fork(&observation, &mut random).unwrap();
        assert_eq!(fork.position(), state.position());
    }
    let mut agents = [RandomAgent::new(1), RandomAgent::new(2)];
    tag_rust::runner::run_game(&model, &mut state, &mut agents).unwrap();
    let observation = model.observe(&state, PlayerId::new(1)).unwrap();
    let terminal = model.simulation_fork(&observation, &mut random).unwrap();
    assert_eq!(terminal.position(), state.position());
    assert!(model.offered_actions(&observation).is_empty());
    assert!(
        model
            .available_actions(&terminal, PlayerId::new(1))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn sampled_osla_averages_each_action_instead_of_selecting_the_largest_sample() {
    use tag_rust::agents::{Agent, SampledOslaAgent};
    use tag_rust::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters, TicTacToeState};
    struct Scripted(usize);
    impl Evaluator<TicTacToeState> for Scripted {
        type Error = String;
        fn evaluate(&mut self, _: &TicTacToeState, _: PlayerId) -> Result<f64, String> {
            let value = [1.0, -1.0, 0.6, 0.6, 0.6005, 0.6005]
                .get(self.0)
                .copied()
                .unwrap_or(0.0);
            self.0 += 1;
            Ok(value)
        }
    }
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let mut agent = SampledOslaAgent::new(Scripted(0), 99, std::num::NonZeroU32::new(2).unwrap());
    assert_eq!(
        agent.choose(&model, actor, &observation, &actions).unwrap(),
        2
    );
    assert_eq!(agent.last_transitions(), 18);
}

#[test]
fn sampled_osla_forks_are_diverse_and_restore_both_random_streams() {
    use std::{cell::RefCell, rc::Rc};
    use tag_rust::agents::{Agent, SampledOslaAgent};
    struct Dice(Rc<RefCell<Vec<[u8; 4]>>>);
    impl Evaluator<CantStopState> for Dice {
        type Error = String;
        fn evaluate(&mut self, state: &CantStopState, _: PlayerId) -> Result<f64, String> {
            self.0.borrow_mut().push(state.position().dice().unwrap());
            Ok(0.0)
        }
    }
    let model = CantStopForwardModel;
    let state = model
        .init_state(
            &CantStopParameters {
                seed: 77,
                ..Default::default()
            },
            3,
        )
        .unwrap();
    let before = state.snapshot();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let dice = Rc::new(RefCell::new(Vec::new()));
    let mut agent = SampledOslaAgent::new(
        Dice(dice.clone()),
        42,
        std::num::NonZeroU32::new(32).unwrap(),
    );
    assert_eq!(
        agent.choose(&model, actor, &observation, &actions).unwrap(),
        0
    );
    assert_eq!(agent.last_transitions(), 32);
    assert!(dice.borrow().iter().any(|roll| *roll != dice.borrow()[0]));
    assert_eq!(state, before);
    let mut state = state.clone();
    model
        .next_with_roll(&mut state, actor, [1, 2, 3, 4])
        .unwrap();
    let before = state.snapshot();
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    assert!(actions.len() > 1);
    let mut agent =
        SampledOslaAgent::new(CantStopBaseline, 88, std::num::NonZeroU32::new(8).unwrap());
    agent.choose(&model, actor, &observation, &actions).unwrap();
    let saved = agent.snapshot();
    let mut restored = SampledOslaAgent::<CantStopBaseline>::from_snapshot(&saved).unwrap();
    for _ in 0..4 {
        assert_eq!(
            agent.choose(&model, actor, &observation, &actions).unwrap(),
            restored
                .choose(&model, actor, &observation, &actions)
                .unwrap()
        );
        assert_eq!(agent.snapshot(), restored.snapshot());
    }
    assert_ne!(agent.snapshot().simulation_rng, saved.simulation_rng);
    assert_ne!(agent.snapshot().rng, saved.rng);
    assert_eq!(state, before);
}

#[test]
fn stochastic_mcts_samples_new_dice_and_preserves_player_vectors_and_checkpoints() {
    use std::{cell::RefCell, rc::Rc};
    use tag_rust::search::mcts::*;
    struct Dice(Rc<RefCell<Vec<[u8; 4]>>>);
    impl StateEvaluator<CantStopForwardModel> for Dice {
        fn evaluate(
            &mut self,
            state: &CantStopState,
            _: &mut SimulationContext<'_, CantStopForwardModel>,
        ) -> tag_rust::search::mcts::Result<StateValues> {
            let dice = state.position().dice().unwrap();
            self.0.borrow_mut().push(dice);
            Ok(StateValues::Full(vec![
                dice.iter().map(|&v| f64::from(v)).sum::<f64>() / 24.0,
                0.2,
                0.3,
                0.4,
            ]))
        }
    }
    let model = CantStopForwardModel;
    let state = model
        .init_state(
            &CantStopParameters {
                seed: 55,
                ..Default::default()
            },
            4,
        )
        .unwrap();
    let before = state.snapshot();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let dice = Rc::new(RefCell::new(Vec::new()));
    let mut registry = EvaluatorRegistry::default();
    let shared = dice.clone();
    registry
        .register_state("dice", move |_| Ok(Box::new(Dice(shared.clone()))))
        .unwrap();
    let mut settings = Settings {
        seed: Some(42),
        tree_depth: Some(1),
        rollout_length: Some(0),
        ..Settings::default()
    };
    settings.budgets.iterations = Some(64);
    settings.state_evaluator = EvaluatorSpec::named("dice");
    let mut agent = MctsAgent::new(settings.clone(), &registry).unwrap();
    let report = agent.search(&model, actor, &observation, &actions).unwrap();
    assert_eq!(report.counters.completed_iterations, 64);
    assert_eq!(report.counters.forward_model_calls, 64);
    assert!(dice.borrow().iter().any(|roll| *roll != dice.borrow()[0]));
    for (index, expected) in [(1, 0.2), (2, 0.3), (3, 0.4)] {
        assert!((report.root_actions[0].mean_values[index] - expected).abs() < 1e-12);
    }
    assert_eq!(state, before);
    let registry = CantStopForwardModel::native_evaluators();
    settings.state_evaluator = EvaluatorSpec::named("progress");
    settings.tree_depth = Some(8);
    settings.rollout_length = Some(8);
    let mut agent = MctsAgent::new(settings, &registry).unwrap();
    agent.search(&model, actor, &observation, &actions).unwrap();
    let saved = agent.snapshot().unwrap();
    let mut restored = MctsAgent::from_snapshot(&saved, &registry).unwrap();
    for _ in 0..3 {
        let first = agent.search(&model, actor, &observation, &actions).unwrap();
        let second = restored
            .search(&model, actor, &observation, &actions)
            .unwrap();
        assert_eq!(first.root_actions, second.root_actions);
        assert_eq!(first.counters, second.counters);
        assert_eq!(agent.snapshot().unwrap(), restored.snapshot().unwrap());
    }
    let calls = std::cell::Cell::new(0);
    assert_eq!(
        agent
            .search_controlled(
                &model,
                actor,
                &observation,
                &actions,
                &Default::default(),
                &SystemClock::default(),
                &|| {
                    calls.set(calls.get() + 1);
                    calls.get() > 20
                }
            )
            .unwrap_err(),
        Error::Cancelled
    );
    assert_eq!(state, before);
    let mut invalid = saved;
    invalid.simulation_rng.generator.stream = 0;
    assert!(MctsAgent::from_snapshot(&invalid, &registry).is_err());
}

#[test]
fn stochastic_search_agents_complete_two_three_and_four_player_games() {
    use tag_rust::agents::{Agent, SampledOslaAgent};
    use tag_rust::search::mcts::*;
    let model = CantStopForwardModel;
    for players in 2..=4 {
        let parameters = CantStopParameters {
            seed: 500 + players as u64,
            rules: CantStopRules {
                column_heights: [2; 11],
                columns_to_win: 1,
                ..Default::default()
            },
        };
        for use_mcts in [false, true] {
            let mut state = model.init_state(&parameters, players).unwrap();
            let registry = CantStopForwardModel::native_evaluators();
            let mut search = (0..players)
                .map(|seat| {
                    let mut settings = Settings {
                        seed: Some(100 + seat as u64),
                        rollout_length: Some(12),
                        ..Settings::default()
                    };
                    settings.budgets.iterations = Some(32);
                    settings.budgets.forward_model_calls = Some(200);
                    MctsAgent::new(settings, &registry).unwrap()
                })
                .collect::<Vec<_>>();
            let mut osla = (0..players)
                .map(|seat| {
                    SampledOslaAgent::new(
                        CantStopProgress::default(),
                        100 + seat as u64,
                        std::num::NonZeroU32::new(4).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let mut count = 0;
            while let Decision::Player(actor) = model.next_player(&state) {
                assert!(count < 2000);
                count += 1;
                let observation = model.observe(&state, actor).unwrap();
                let actions = model.offered_actions(&observation);
                let index = if use_mcts {
                    let report = search[actor.index()]
                        .search(&model, actor, &observation, &actions)
                        .unwrap();
                    assert!(report.counters.forward_model_calls <= 200);
                    report.selected_index
                } else {
                    osla[actor.index()]
                        .choose(&model, actor, &observation, &actions)
                        .unwrap()
                };
                model.next(&mut state, actor, &actions[index]).unwrap();
            }
            assert_eq!(
                (0..players)
                    .filter(|&p| model.player_result(&state, PlayerId::new(p)).unwrap()
                        == PlayerResult::Win)
                    .count(),
                1
            );
        }
    }
}

#[test]
fn cantstop_search_registries_validate_independent_evaluator_roles_and_budgets() {
    use tag_rust::search::mcts::*;
    let model = CantStopForwardModel;
    let state = model.init_state(&CantStopParameters::default(), 3).unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    let registry = EvaluatorRegistry::<CantStopForwardModel>::for_game();
    for name in ["baseline", "progress", "game_heuristic"] {
        let mut settings = Settings {
            seed: Some(2026),
            rollout_length: Some(5),
            ..Settings::default()
        };
        settings.budgets.iterations = Some(32);
        settings.budgets.forward_model_calls = Some(7);
        settings.state_evaluator = EvaluatorSpec::named(name);
        settings.rollout.policy = RolloutPolicy::Greedy;
        settings.rollout.evaluator = Some(EvaluatorSpec {
            name: "successor_state".into(),
            parameters: [(
                "state_evaluator".into(),
                Parameter::Object([("name".into(), Parameter::Text("baseline".into()))].into()),
            )]
            .into(),
        });
        let report = MctsAgent::new(settings, &registry)
            .unwrap()
            .search(&model, actor, &observation, &actions)
            .unwrap();
        assert!(report.counters.forward_model_calls <= 7);
        assert_eq!(report.root_actions[0].mean_values.len(), 3);
    }
    for (name, key, value) in [
        ("progress", "alpha", Parameter::Number(-1.0)),
        ("progress", "beta", Parameter::Number(f64::NAN)),
        ("baseline", "alpha", Parameter::Integer(1)),
        ("progress", "unknown", Parameter::Integer(0)),
    ] {
        let mut settings = Settings {
            seed: Some(1),
            ..Settings::default()
        };
        settings.state_evaluator = EvaluatorSpec {
            name: name.into(),
            parameters: [(key.into(), value)].into(),
        };
        assert!(MctsAgent::new(settings, &registry).is_err());
    }
}

#[cfg(feature = "mcts-json")]
#[test]
fn decoded_allocations_preserve_constructor_invariants() {
    for data in [
        serde_json::json!({"columns":[2,0],"length":0}),
        serde_json::json!({"columns":[2,3],"length":3}),
        serde_json::json!({"columns":[3,2],"length":2}),
        serde_json::json!({"columns":[2,1],"length":1}),
        serde_json::json!({"columns":[1,0],"length":1}),
    ] {
        assert!(serde_json::from_value::<Allocation>(data).is_err());
    }
    for action in [
        CantStopAction::Roll,
        CantStopAction::Stop,
        CantStopAction::Bust,
        CantStopAction::Allocate(Allocation::pair(4, 9).unwrap()),
        CantStopAction::Allocate(Allocation::single(7).unwrap()),
    ] {
        assert_eq!(
            serde_json::from_str::<CantStopAction>(&serde_json::to_string(&action).unwrap())
                .unwrap(),
            action
        );
    }
}
