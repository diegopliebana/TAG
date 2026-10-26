use tag_rust::agents::{Agent, EmptyActionList, RandomAgent};
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::runner::{run_game, run_game_recorded};

#[test]
fn random_receives_normal_state_actions_and_preserves_exact_independent_checkpoints() {
    let model = ttt::TicTacToeForwardModel;
    let state = model
        .init_state(&ttt::TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let observation = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&observation);
    for seed in [0, 7, u64::MAX] {
        let mut source = RandomAgent::new(seed);
        let mut reference = AgentRng::new(seed);
        for count in [0, 1, 9, 3, 7, 2, 9, 1, 0] {
            let before = source.snapshot();
            let result = source.choose(&model, actor, &observation, &actions[..count]);
            assert_eq!(result.ok(), reference.index(count));
            assert_eq!(source.snapshot(), reference.snapshot());
            if count <= 1 {
                assert_eq!(source.snapshot(), before);
            }
        }
        assert_eq!(
            source.choose(&model, actor, &observation, &[]),
            Err(EmptyActionList)
        );
        let snapshot = source.snapshot();
        let mut restored = RandomAgent::from_snapshot(&snapshot).unwrap();
        let mut independent = RandomAgent::from_snapshot(&snapshot).unwrap();
        for _ in 0..50 {
            assert_eq!(
                source.choose(&model, actor, &observation, &actions),
                restored.choose(&model, actor, &observation, &actions)
            );
        }
        assert_eq!(independent.snapshot(), snapshot);
        assert_eq!(
            independent.choose(&model, actor, &observation, &actions),
            RandomAgent::from_snapshot(&snapshot).unwrap().choose(
                &model,
                actor,
                &observation,
                &actions
            )
        );
        assert_eq!(
            model
                .init_state(&ttt::TicTacToeParameters::default(), 2)
                .unwrap(),
            state
        );
    }
}
fn headless<F>(model: F, parameters: F::Parameters)
where
    F: ForwardModel,
    F::State: Clone + PartialEq + std::fmt::Debug,
    F::Action: PartialEq,
    F::Error: std::fmt::Debug,
{
    let initial = model.init_state(&parameters, 2).unwrap();
    let mut direct = initial.clone();
    let mut recorded = initial.clone();
    let mut first = [RandomAgent::new(7), RandomAgent::new(u64::MAX)];
    let mut second = [RandomAgent::new(7), RandomAgent::new(u64::MAX)];
    run_game(&model, &mut direct, &mut first).unwrap();
    let mut replay = initial;
    let mut count = 0;
    run_game_recorded(&model, &mut recorded, &mut second, |r| {
        let observation = model.observe(&replay, r.actor).unwrap();
        let actions = model.offered_actions(&observation);
        // Every accepted action was offered at its original player-visible state.
        assert!(actions.contains(&r.action));
        model.next(&mut replay, r.actor, &r.action).unwrap();
        count += 1;
    })
    .unwrap();
    assert_eq!(model.next_player(&direct), Decision::Finished);
    assert_eq!(direct, recorded);
    assert_eq!(direct, replay);
    assert!(count > 0);
    for seat in 0..2 {
        assert_eq!(first[seat].snapshot(), second[seat].snapshot());
    }
}
#[test]
fn random_headless_and_recorded_matches_agree_for_both_games_and_configured_boards() {
    headless(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters::default(),
    );
    headless(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters {
            grid_size: 5,
            ..Default::default()
        },
    );
    headless(c4::Connect4ForwardModel, c4::Connect4Parameters::default());
    headless(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters {
            width: 4,
            height: 3,
            winning_length: 3,
            ..Default::default()
        },
    );
}
