//! Public API proof: existing controllers and runner consume only observations.
use tag_rust::{
    agents::{Agent, RandomAgent},
    core::{Decision, ForwardModel, PlayerId, PlayerResult, checkpoint::Checkpoint},
    games::loveletter::*,
};

#[test]
fn native_runner_and_agent_checkpoints_preserve_complete_multiplayer_games() {
    let model = LoveLetterForwardModel;
    for n in 2..=4 {
        let mut state = model
            .init_state(
                &LoveLetterParameters {
                    seed: 2026,
                    ..Default::default()
                },
                n,
            )
            .unwrap();
        let mut agents: Vec<_> = (0..n).map(|i| RandomAgent::new(i as u64 + 1)).collect();
        let mut restored = LoveLetterState::from_snapshot(&state.snapshot()).unwrap();
        let mut restored_agents: Vec<_> = agents
            .iter()
            .map(|a| RandomAgent::from_snapshot(&a.snapshot()).unwrap())
            .collect();
        // Keep several mid-round checkpoints, then compare every accepted action.
        let mut count = 0;
        while let Decision::Player(actor) = model.next_player(&state) {
            assert!(count < 3000);
            let observation = model.observe(&state, actor).unwrap();
            let actions = model.offered_actions(&observation);
            assert_eq!(actions, model.available_actions(&state, actor).unwrap());
            for seat in 0..n {
                if PlayerId::new(seat) != actor {
                    assert!(
                        model
                            .offered_actions(&model.observe(&state, PlayerId::new(seat)).unwrap())
                            .is_empty()
                    );
                }
            }
            let i = agents[actor.index()]
                .choose(&model, actor, &observation, &actions)
                .unwrap();
            let other_observation = model.observe(&restored, actor).unwrap();
            let other_actions = model.offered_actions(&other_observation);
            let j = restored_agents[actor.index()]
                .choose(&model, actor, &other_observation, &other_actions)
                .unwrap();
            assert_eq!(i, j);
            assert_eq!(observation, other_observation);
            model.next(&mut state, actor, &actions[i]).unwrap();
            model.next(&mut restored, actor, &other_actions[j]).unwrap();
            assert_eq!(state, restored);
            count += 1;
            if count % 7 == 0 {
                restored = LoveLetterState::from_snapshot(&state.snapshot()).unwrap();
                restored_agents = agents
                    .iter()
                    .map(|a| RandomAgent::from_snapshot(&a.snapshot()).unwrap())
                    .collect();
            }
        }
        assert!(
            state
                .public()
                .outcome
                .winners()
                .iter()
                .all(|&p| model.player_result(&state, p).unwrap() == PlayerResult::Win)
        );
        for seat in 0..n {
            assert_ne!(
                model.player_result(&state, PlayerId::new(seat)).unwrap(),
                PlayerResult::Ongoing
            );
        }
        let mut fresh = model
            .init_state(
                &LoveLetterParameters {
                    seed: 2026,
                    ..Default::default()
                },
                n,
            )
            .unwrap();
        let mut fresh_agents: Vec<_> = (0..n).map(|i| RandomAgent::new(i as u64 + 1)).collect();
        tag_rust::runner::run_game(&model, &mut fresh, &mut fresh_agents).unwrap();
        assert_eq!(fresh, state);
    }
}

#[test]
fn sampled_osla_and_existing_mcts_finish_games_and_resume_agent_streams() {
    use tag_rust::agents::{MctsAgent, SampledOslaAgent};
    use tag_rust::search::mcts::{Budgets, NativeEvaluators, Settings};
    let model = LoveLetterForwardModel;
    let registry = LoveLetterForwardModel::native_evaluators();
    for n in 2..=4 {
        let mut state = model
            .init_state(
                &LoveLetterParameters {
                    seed: 15,
                    ..Default::default()
                },
                n,
            )
            .unwrap();
        let mut osla = SampledOslaAgent::new(
            LoveLetterHeuristic,
            200,
            std::num::NonZeroU32::new(2).unwrap(),
        );
        let mut mcts = MctsAgent::new(
            Settings {
                seed: Some(300),
                budgets: Budgets {
                    iterations: Some(8),
                    time_ms: None,
                    forward_model_calls: Some(128),
                },
                rollout_length: Some(3),
                ..Default::default()
            },
            &registry,
        )
        .unwrap();
        let mut turns = 0;
        while let Decision::Player(actor) = model.next_player(&state) {
            assert!(turns < 3000);
            let o = model.observe(&state, actor).unwrap();
            let actions = model.offered_actions(&o);
            let before = state.snapshot();
            let index = if actor.index() % 2 == 0 {
                let mut restored =
                    SampledOslaAgent::<LoveLetterHeuristic>::from_snapshot(&osla.snapshot())
                        .unwrap();
                let i = osla.choose(&model, actor, &o, &actions).unwrap();
                assert_eq!(i, restored.choose(&model, actor, &o, &actions).unwrap());
                assert_eq!(osla.snapshot(), restored.snapshot());
                assert_eq!(osla.last_transitions(), 2 * actions.len() as u64);
                i
            } else {
                let mut restored =
                    MctsAgent::from_snapshot(&mcts.snapshot().unwrap(), &registry).unwrap();
                let i = mcts.choose(&model, actor, &o, &actions).unwrap();
                assert_eq!(i, restored.choose(&model, actor, &o, &actions).unwrap());
                assert_eq!(mcts.snapshot().unwrap(), restored.snapshot().unwrap());
                i
            };
            assert_eq!(state, before);
            model.next(&mut state, actor, &actions[index]).unwrap();
            turns += 1;
        }
        assert!(!state.public().outcome.winners().is_empty());
    }
}
