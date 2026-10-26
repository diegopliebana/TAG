use super::*;
use tag_rust::core::Decision;

#[test]
fn learner_teacher_matches_original_osla_action_and_rng() {
    fn verify<F, E>(
        mut actual: EpisodeGame<F, E>,
        mut expected: EpisodeGame<F, E>,
        mut teacher: Opponent<E>,
    ) where
        F: LearningGame + PublicInformationModel + Clone,
        F::State: Clone,
        F::Action: Clone + PartialEq,
        F::Parameters: Clone,
        F::Error: std::fmt::Display,
        E: Evaluator<F::State>,
        E::Error: std::fmt::Display,
    {
        actual.opponent_turn().unwrap();
        expected.opponent_turn().unwrap();
        while !actual.ended {
            let model = expected.game.session.model();
            let actor = PlayerId::new(expected.learner);
            let view = model
                .observe(expected.game.session.state(), actor)
                .unwrap_or_else(|e| panic!("{e}"));
            let actions = model.offered_actions(&view);
            let index = teacher.choose(model, actor, &view, &actions).unwrap();
            let id = model.action_id(&view, &actions[index]).unwrap();
            actual.validate_teacher().unwrap();
            let observed = actual.step_teacher().unwrap();
            let reference = expected.step(id).unwrap();
            assert_eq!(observed.action, Some(id));
            assert_eq!(observed.teacher_rng.unwrap(), teacher.rng::<F>());
            assert_eq!(observed.observation, reference.observation);
            assert_eq!(observed.rng, reference.rng);
            assert_eq!(observed.reward, reference.reward);
        }
    }
    for name in ["tictactoe", "connect4"] {
        for seat in [0, 1] {
            let plain = config(name, seat, "random");
            let mut taught = plain.clone();
            taught.teacher = true;
            taught.include_intermediate = true;
            match (
                taught.create_seeded(17, Some(29)).unwrap(),
                plain.create(17).unwrap(),
            ) {
                (Runtime::TicTacToe(a), Runtime::TicTacToe(b)) => {
                    let t = native(
                        &taught.teacher_config(),
                        &a.game.description,
                        ttt::TicTacToeEvaluator,
                        29,
                    )
                    .unwrap();
                    verify(a, b, t);
                }
                (Runtime::Connect4(a), Runtime::Connect4(b)) => {
                    let t = native(
                        &taught.teacher_config(),
                        &a.game.description,
                        c4::Connect4Evaluator,
                        29,
                    )
                    .unwrap();
                    verify(a, b, t);
                }
                _ => unreachable!(),
            }
        }
    }
}

fn config(name: &str, learner: usize, opponent: &str) -> Config {
    Config {
        name: name.into(),
        parameters: BTreeMap::new(),
        learner,
        opponent: opponent.into(),
        settings: BTreeMap::new(),
        payloads: vec!["cells".into(), "board".into()],
        horizon: None,
        limits: LearningLimits::default(),
        include_intermediate: false,
        teacher: false,
    }
}
fn compare<F, E>(config: Config, model: F, parameters: F::Parameters, evaluator: E)
where
    F: LearningGame + PublicInformationModel + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Parameters: Clone,
    F::Error: std::fmt::Display,
    E: Evaluator<F::State> + Clone,
    E::Error: std::fmt::Display,
{
    let seed = 731;
    let mut runtime = config.create(seed).unwrap();
    let spec = runtime.spec();
    let mut expected_agent = native(&config, &spec, evaluator, seed).unwrap();
    let mut expected = model
        .init_state(&parameters, 2)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut steps = 0;
    let mut moves = 0;
    let names = ["cells", "board"];
    fn opponent<F, E>(
        model: &F,
        state: &mut F::State,
        agent: &mut Opponent<E>,
        learner: usize,
        moves: &mut u64,
    ) where
        F: LearningGame + PublicInformationModel,
        F::State: Clone,
        F::Action: Clone,
        F::Error: std::fmt::Display,
        E: Evaluator<F::State>,
        E::Error: std::fmt::Display,
    {
        if let Decision::Player(actor) = model.next_player(state)
            && actor.index() != learner
        {
            let view = model
                .observe(state, actor)
                .unwrap_or_else(|e| panic!("{e}"));
            let actions = model.offered_actions(&view);
            let i = agent.choose(model, actor, &view, &actions).unwrap();
            model
                .next(state, actor, &actions[i])
                .unwrap_or_else(|e| panic!("{e}"));
            *moves += 1;
        }
    }
    opponent(
        &model,
        &mut expected,
        &mut expected_agent,
        config.learner,
        &mut moves,
    );
    let mut result = runtime.opening().unwrap();
    loop {
        let view = model
            .observe(&expected, PlayerId::new(config.learner))
            .unwrap_or_else(|e| panic!("{e}"));
        let encoded = model.encode(&view, &names, config.limits).unwrap();
        assert_eq!(result.observation, encoded);
        assert_eq!(result.rng, expected_agent.rng::<F>());
        assert_eq!((result.steps, result.moves), (steps, moves));
        let done = model.next_player(&expected) == Decision::Finished;
        assert_eq!(result.terminated, done);
        if done {
            break;
        }
        let id = encoded.offered_action_ids[0];
        runtime.validate(id).unwrap();
        let action = model.action_from_id(&view, id).unwrap();
        model
            .next(&mut expected, PlayerId::new(config.learner), &action)
            .unwrap_or_else(|e| panic!("{e}"));
        steps += 1;
        moves += 1;
        opponent(
            &model,
            &mut expected,
            &mut expected_agent,
            config.learner,
            &mut moves,
        );
        result = runtime.step(id).unwrap();
    }
    let expected_reward = match result.observation.results[config.learner] {
        ResultTag::Win => 1.,
        ResultTag::Loss => -1.,
        _ => 0.,
    };
    assert_eq!(result.reward, expected_reward);
}
#[test]
fn every_native_policy_matches_rule_transitions_and_rng_for_both_seats() {
    for learner in 0..2 {
        for policy in ["first_available", "random", "osla", "osla_python"] {
            compare(
                config("tictactoe", learner, policy),
                ttt::TicTacToeForwardModel,
                ttt::TicTacToeParameters::default(),
                ttt::TicTacToeEvaluator,
            );
            compare(
                config("connect4", learner, policy),
                c4::Connect4ForwardModel,
                c4::Connect4Parameters::default(),
                c4::Connect4Evaluator,
            );
        }
    }
}
#[test]
fn rejects_unsupported_setup_and_predictable_limits() {
    for policy in ["external", "unknown"] {
        assert!(config("tictactoe", 0, policy).create(0).is_err());
    }
    let mut c = config("tictactoe", 1, "random");
    c.parameters.insert("grid_size".into(), 1);
    assert!(c.create(0).is_err());
    c = config("tictactoe", 0, "random");
    c.settings.insert("seed".into(), 1);
    assert!(c.create(0).is_err());
    c = config("tictactoe", 0, "osla_python");
    c.settings.insert("max_candidates".into(), 8);
    assert!(c.create(0).is_err());
    c = config("tictactoe", 0, "random");
    c.payloads = vec!["image".into()];
    assert!(c.create(0).is_err());
    c = config("tictactoe", 0, "random");
    c.payloads = vec!["cells".into(), "cells".into()];
    assert!(c.create(0).is_err());
}
#[test]
fn validation_does_not_advance_and_terminal_precedes_horizon() {
    let mut c = config("tictactoe", 0, "first_available");
    c.horizon = Some(1);
    let mut r = c.create(9).unwrap();
    let before = r.opening().unwrap();
    assert!(r.validate(99).is_err());
    let after = dispatch!(&r, g, g.output(0)).unwrap();
    assert_eq!(before.observation, after.observation);
    assert_eq!(before.rng, after.rng);
    let out = r.step(4).unwrap();
    assert!(out.truncated);
    assert!(!out.terminated);
    assert!(out.observation.action_mask.iter().any(|v| *v));
    assert!(r.validate(1).is_err());
    c.parameters.insert("grid_size".into(), 1);
    let mut r = c.create(9).unwrap();
    r.opening().unwrap();
    let out = r.step(0).unwrap();
    assert!(out.terminated);
    assert!(!out.truncated);
    assert_eq!(out.reward, 1.);
}
#[test]
fn later_encoding_failure_retains_moves_without_a_rollback_snapshot() {
    let c = config("tictactoe", 0, "first_available");
    let mut r = c.create(2).unwrap();
    r.opening().unwrap();
    let Runtime::TicTacToe(g) = &mut r else {
        unreachable!()
    };
    g.payloads = vec!["injected-output-failure".into()];
    assert!(r.step(4).is_err());
    let Runtime::TicTacToe(g) = &r else {
        unreachable!()
    };
    assert_eq!(g.moves, 2);
    let view = g.game.observe(0, &["cells"]).unwrap();
    assert_eq!(view.progress[0], 2);
}

#[test]
fn failing_opponent_retains_the_learner_move_for_failed_episode_handling() {
    struct FailingEvaluator;
    impl Evaluator<ttt::TicTacToeState> for FailingEvaluator {
        type Error = &'static str;
        fn evaluate(&mut self, _: &ttt::TicTacToeState, _: PlayerId) -> Result<f64, Self::Error> {
            Err("injected evaluator failure")
        }
    }
    let c = config("tictactoe", 0, "osla");
    let Engine::TicTacToe(game) =
        Engine::create("tictactoe", BTreeMap::new(), 2, c.limits).unwrap()
    else {
        unreachable!()
    };
    let opponent = Opponent::Builtin(Box::new(BuiltinAgent::Osla(
        tag_rust::agents::OslaAgent::new(FailingEvaluator, 3),
    )));
    let mut episode = EpisodeGame::new(game, opponent, None, &c).unwrap();
    episode.validate(4).unwrap();
    assert!(
        episode
            .step(4)
            .err()
            .unwrap()
            .contains("injected evaluator failure")
    );
    assert_eq!(episode.moves, 1);
    assert_eq!(
        episode.game.observe(0, &["cells"]).unwrap().tensors[0].values[4],
        1.0
    );
}
