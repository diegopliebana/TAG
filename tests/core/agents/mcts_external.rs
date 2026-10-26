//! External evaluation must preserve native search arithmetic and pending lifecycles.
use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::tictactoe::{TicTacToeForwardModel as Model, TicTacToeParameters};
use tag_rust::search::mcts::*;

fn poll<T>(future: Pin<&mut impl Future<Output = T>>) -> Poll<T> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}
fn settings() -> Settings {
    let mut settings = Settings {
        seed: Some(42),
        rollout_length: Some(0),
        ..Settings::default()
    };
    settings.budgets.iterations = Some(80);
    settings
}

#[test]
fn policy_requests_identify_root_and_deeper_nodes_without_using_player_identity() {
    let model = Model;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    let exchange = ExternalEvaluations::new(actor, ValueObservers::AllPlayers).unwrap();
    let mut registry = EvaluatorRegistry::<Model>::for_game();
    exchange
        .register(&mut registry, "external", "policy")
        .unwrap();
    let mut config = settings();
    config.tree_policy = TreePolicy::Puct;
    config.state_evaluator.name = "external".into();
    config.puct_preferences.source = PreferenceSource::Probabilities;
    config.puct_preferences.evaluator = Some(EvaluatorSpec {
        name: "policy".into(),
        parameters: Parameters::new(),
    });
    config.budgets.iterations = Some(300);
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let clock = SystemClock::default();
    let overrides = BudgetOverrides::default();
    let mut future = Box::pin(agent.search_controlled_async(
        &model,
        actor,
        &observation,
        &actions,
        &overrides,
        &clock,
        &|| false,
    ));
    let (mut root, mut deeper) = (false, false);
    loop {
        if let Poll::Ready(result) = poll(future.as_mut()) {
            result.unwrap();
            break;
        }
        if let Some(request) = exchange.take_request() {
            let response = match request.kind {
                EvaluationKind::Values => {
                    assert_eq!(request.policy_node, None);
                    ExternalResponse::Values(StateValues::Full(vec![0., 0.]))
                }
                EvaluationKind::Policy => {
                    let node = request.policy_node.expect("policy node must be explicit");
                    let moves = request.observations[0].1.move_count();
                    assert_eq!(node == 0, moves == 0);
                    root |= node == 0;
                    deeper |= node != 0;
                    ExternalResponse::Policy(vec![1.; request.actions.len()])
                }
            };
            exchange.respond(request.id, response).unwrap();
        }
    }
    assert!(root && deeper);
}

#[test]
fn external_values_match_native_statistics_without_replaying_transitions() {
    use tag_rust::core::evaluation::Evaluator;
    use tag_rust::games::tictactoe::TicTacToeEvaluator;
    let model = Model;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    let native = MctsAgent::new(settings(), &EvaluatorRegistry::<Model>::for_game())
        .unwrap()
        .search(&model, actor, &observation, &actions)
        .unwrap();
    let exchange = ExternalEvaluations::new(actor, ValueObservers::AllPlayers).unwrap();
    let mut registry = EvaluatorRegistry::<Model>::for_game();
    exchange
        .register(&mut registry, "external_values", "external_policy")
        .unwrap();
    let mut config = settings();
    config.state_evaluator.name = "external_values".into();
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let clock = SystemClock::default();
    let overrides = BudgetOverrides::default();
    let mut future = Box::pin(agent.search_controlled_async(
        &model,
        actor,
        &observation,
        &actions,
        &overrides,
        &clock,
        &|| false,
    ));
    let external = loop {
        if let Poll::Ready(result) = poll(future.as_mut()) {
            break result.unwrap();
        }
        if let Some(request) = exchange.take_request() {
            assert_eq!(request.observations.len(), 2);
            let values = request
                .observations
                .iter()
                .map(|(player, observation)| {
                    let simulation = tag_rust::core::PublicInformationModel::simulation_state(
                        &model,
                        observation,
                    )
                    .unwrap();
                    TicTacToeEvaluator.evaluate(&simulation, *player).unwrap()
                })
                .collect();
            exchange
                .respond(
                    request.id,
                    ExternalResponse::Values(StateValues::Full(values)),
                )
                .unwrap();
        }
    };
    assert_eq!(native.root_actions, external.root_actions);
    assert_eq!(native.selected_index, external.selected_index);
    assert_eq!(native.counters, external.counters);
    assert_eq!(state.move_count(), 0);
}

struct ManualClock(Cell<f64>);
impl Clock for ManualClock {
    fn now_ms(&self) -> f64 {
        self.0.get()
    }
}

#[test]
fn dropping_a_pending_future_invalidates_its_response_and_checkpoint_is_explicitly_unsupported() {
    let model = Model;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    let exchange = ExternalEvaluations::new(actor, ValueObservers::AllPlayers).unwrap();
    let mut registry = EvaluatorRegistry::<Model>::for_game();
    exchange
        .register(&mut registry, "external", "policy")
        .unwrap();
    let mut config = settings();
    config.state_evaluator.name = "external".into();
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let clock = SystemClock::default();
    let overrides = BudgetOverrides::default();
    let mut future = Box::pin(agent.search_controlled_async(
        &model,
        actor,
        &observation,
        &actions,
        &overrides,
        &clock,
        &|| false,
    ));
    assert!(poll(future.as_mut()).is_pending());
    let request = exchange.take_request().unwrap();
    drop(future);
    assert!(!exchange.is_waiting());
    assert!(
        exchange
            .respond(
                request.id,
                ExternalResponse::Values(StateValues::Full(vec![0., 0.]))
            )
            .is_err()
    );
    assert!(matches!(
        agent.snapshot(),
        Err(Error::UnsupportedCheckpoint(_))
    ));
    assert_eq!(
        agent.search(&model, actor, &observation, &actions),
        Err(Error::EvaluationRequired)
    );
    assert!(!exchange.is_waiting());
}

#[test]
fn waiting_timeouts_discard_expansion_and_replies_require_current_identity() {
    let model = Model;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    let exchange = ExternalEvaluations::new(actor, ValueObservers::RootPlayer).unwrap();
    let mut registry = EvaluatorRegistry::<Model>::for_game();
    exchange
        .register(&mut registry, "external", "policy")
        .unwrap();
    let mut config = settings();
    config.state_evaluator.name = "external".into();
    config.budgets.time_ms = Some(10.0);
    let mut agent = MctsAgent::new(config, &registry).unwrap();
    let clock = ManualClock(Cell::new(0.0));
    let overrides = BudgetOverrides::default();
    let mut future = Box::pin(agent.search_controlled_async(
        &model,
        actor,
        &observation,
        &actions,
        &overrides,
        &clock,
        &|| false,
    ));
    assert!(poll(future.as_mut()).is_pending());
    let request = exchange.take_request().unwrap();
    assert_eq!(request.observations.len(), 1);
    let valid = ExternalResponse::Values(StateValues::Full(vec![0.3, 0.8]));
    assert!(
        exchange
            .respond(
                RequestId {
                    search: request.id.search + 1,
                    ..request.id
                },
                valid.clone()
            )
            .is_err()
    );
    assert!(
        exchange
            .respond(
                request.id,
                ExternalResponse::Values(StateValues::Full(vec![f64::NAN, 0.]))
            )
            .is_err()
    );
    assert!(exchange.is_waiting());
    clock.0.set(11.0);
    exchange.respond(request.id, valid.clone()).unwrap();
    assert!(exchange.respond(request.id, valid.clone()).is_err());
    let Poll::Ready(Ok(report)) = poll(future.as_mut()) else {
        panic!("deadline must complete the search");
    };
    assert_eq!(report.counters.completed_iterations, 0);
    assert_eq!(report.edges, 0);
    assert!(report.stop_reasons.contains(&StopReason::Time));
    assert!(exchange.respond(request.id, valid).is_err());
}

#[test]
fn forward_model_exhaustion_completes_one_pending_iteration_and_cancel_invalidates_it() {
    let model = Model;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    for cancel in [false, true] {
        let exchange = ExternalEvaluations::new(actor, ValueObservers::AllPlayers).unwrap();
        let mut registry = EvaluatorRegistry::<Model>::for_game();
        exchange
            .register(&mut registry, "external", "policy")
            .unwrap();
        let mut config = settings();
        config.state_evaluator.name = "external".into();
        config.budgets.forward_model_calls = Some(1);
        let mut agent = MctsAgent::new(config, &registry).unwrap();
        let clock = SystemClock::default();
        let overrides = BudgetOverrides::default();
        let mut future = Box::pin(agent.search_controlled_async(
            &model,
            actor,
            &observation,
            &actions,
            &overrides,
            &clock,
            &|| false,
        ));
        assert!(poll(future.as_mut()).is_pending());
        let request = exchange.take_request().unwrap();
        if cancel {
            exchange.cancel();
            assert!(matches!(
                poll(future.as_mut()),
                Poll::Ready(Err(Error::Cancelled))
            ));
            assert!(
                exchange
                    .respond(
                        request.id,
                        ExternalResponse::Values(StateValues::Full(vec![0., 0.]))
                    )
                    .is_err()
            );
        } else {
            exchange
                .respond(
                    request.id,
                    ExternalResponse::Values(StateValues::Full(vec![0.2, 0.6])),
                )
                .unwrap();
            let report = loop {
                if let Poll::Ready(result) = poll(future.as_mut()) {
                    break result.unwrap();
                }
            };
            assert_eq!(report.counters.completed_iterations, 1);
            assert_eq!(report.counters.forward_model_calls, 1);
            assert_eq!(report.edges, 1);
        }
    }
}
