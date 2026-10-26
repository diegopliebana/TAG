use super::branches::finish;
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;
use tag_rust::agents::{FirstAvailableAgent, OslaAgent};
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{ForwardModel, PlayerId, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::recording::{ControllerSnapshot, GameHistory};
use tag_rust::replay::*;
use tag_rust::runner::run_game_recorded;
use tag_rust::session::{BranchId, BranchSession, BuiltinAgent, Controller, Session};
fn roundtrip<F, E>(model: F, parameters: F::Parameters, evaluator: E)
where
    F: ReplayGame<SavedEvaluator = E> + PublicInformationModel,
    F::State: Clone + PartialEq + Debug,
    F::Action: Clone + PartialEq + Debug + Serialize + DeserializeOwned,
    F::Parameters: Clone + PartialEq + Debug + Serialize + DeserializeOwned,
    F::Error: Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()> + Copy,
    E::Error: Debug,
    E::RestoreError: Debug,
{
    let mut game = BranchSession::new(
        Session::create(
            model.clone(),
            &parameters,
            2,
            vec![
                Controller::osla(evaluator, u64::MAX),
                Controller::osla(evaluator, 8),
            ],
        )
        .unwrap(),
    );
    finish(&mut game);
    let length = game.info(BranchId(0)).unwrap().move_count;
    let child = game
        .fork(BranchId(0), length.min(2), vec![None, None])
        .unwrap();
    finish(&mut game);
    game.fork(
        child,
        0,
        vec![
            Some(Controller::Human),
            Some(Controller::first_available(7)),
        ],
    )
    .unwrap();
    finish(&mut game);
    game.select(child).unwrap();
    let text = export_replay(&game, &parameters, 0, 250).unwrap();
    assert!(text.contains(&u64::MAX.to_string()));
    let mut loaded = import_replay::<F, E>(&text, model.clone(), ReplayLimits::default()).unwrap();
    assert_eq!(loaded.parameters, parameters);
    assert_eq!(loaded.cursor, 0);
    assert_eq!(loaded.delay_ms, 250);
    assert_eq!(loaded.session.active_branch(), child);
    assert_eq!(loaded.session.branch_count(), game.branch_count());
    for id in 0..game.branch_count() {
        let id = BranchId(id);
        assert_eq!(game.info(id).unwrap(), loaded.session.info(id).unwrap());
        assert_eq!(game.moves(id).unwrap(), loaded.session.moves(id).unwrap());
        for position in 0..=game.info(id).unwrap().move_count {
            let original = game.position_at(id, position).unwrap();
            assert_eq!(original, loaded.session.position_at(id, position).unwrap());
            assert_eq!(
                game.controllers_at(id, position).unwrap(),
                loaded.session.controllers_at(id, position).unwrap()
            );
            let text = export_state(&model, &parameters, &original).unwrap();
            let (p, state) = import_state(&text, &model, ReplayLimits::default()).unwrap();
            assert_eq!(p, parameters);
            assert_eq!(state, original);
            let session = Session::from_state(
                model.clone(),
                state,
                vec![
                    Controller::Human,
                    Controller::<BuiltinAgent<E>>::first_available(9),
                ],
            )
            .unwrap();
            let mut resumed = BranchSession::new(session);
            finish(&mut resumed);
            if position == 1 {
                let replay = export_replay(
                    &resumed,
                    &p,
                    resumed.info(BranchId(0)).unwrap().move_count,
                    0,
                )
                .unwrap();
                let decoded =
                    import_replay::<F, E>(&replay, model.clone(), ReplayLimits::default()).unwrap();
                assert_eq!(decoded.session.root(), &original);
                assert_eq!(
                    decoded.session.active().session().state(),
                    resumed.active().session().state()
                );
            }
            let reset = resumed.reset().unwrap();
            assert_eq!(resumed.info(reset).unwrap().move_count, 0);
            assert_eq!(resumed.active().session().state(), &original);
        }
    }
    // Native headless records can use the same export/import path.
    let mut state = model.init_state(&parameters, 2).unwrap();
    let mut agents = [
        BuiltinAgent::Osla(OslaAgent::new(evaluator, 7)),
        BuiltinAgent::FirstAvailable(FirstAvailableAgent::new(8)),
    ];
    let mut history = GameHistory::new(
        state.clone(),
        agents
            .iter()
            .map(|a| ControllerSnapshot::Ai(a.snapshot()))
            .collect(),
    );
    run_game_recorded(&model, &mut state, &mut agents, |record| {
        history.record(record)
    })
    .unwrap();
    let native = BranchSession::<F, BuiltinAgent<E>>::from_history(model.clone(), history).unwrap();
    let text = export_replay(
        &native,
        &parameters,
        native.info(BranchId(0)).unwrap().move_count,
        0,
    )
    .unwrap();
    let loaded = import_replay::<F, E>(&text, model, ReplayLimits::default()).unwrap();
    assert_eq!(loaded.session.active().session().state(), &state);
}
#[test]
fn replay_and_standalone_states_roundtrip_both_games_and_native_recordings() {
    for size in [1, 3, 5] {
        roundtrip(
            TicTacToeForwardModel,
            TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            TicTacToeEvaluator,
        );
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3), (4, 3, 1)] {
        roundtrip(
            Connect4ForwardModel,
            Connect4Parameters {
                width,
                height,
                winning_length,
                ..Default::default()
            },
            Connect4Evaluator,
        );
    }
}
#[test]
fn malformed_replays_versions_actions_progress_and_limits_are_rejected() {
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters::default();
    let mut game = BranchSession::new(
        Session::create(
            model,
            &parameters,
            2,
            vec![
                Controller::osla(TicTacToeEvaluator, 7),
                Controller::osla(TicTacToeEvaluator, 8),
            ],
        )
        .unwrap(),
    );
    finish(&mut game);
    game.fork(BranchId(0), 1, vec![None, None]).unwrap();
    finish(&mut game);
    let original = game.active().session().state().clone();
    let text = export_replay(&game, &parameters, 0, 0).unwrap();
    let file: serde_json::Value = serde_json::from_str(&text).unwrap();
    let bad: Vec<(&str, serde_json::Value)> = vec![
        ("/versions/format", 99.into()),
        ("/versions/rules", 99.into()),
        ("/versions/agents", 99.into()),
        ("/versions/evaluators", 99.into()),
        ("/versions/rng", "other".into()),
        ("/kind", "tag-state".into()),
        ("/branches/1/id", 0.into()),
        ("/branches/1/parent", 1.into()),
        ("/branches/1/parent_position", 999.into()),
        ("/branches/0/moves/0/actor", 1.into()),
        ("/branches/0/moves/0/action/x", 99.into()),
        ("/branches/0/moves/0/next_player", serde_json::Value::Null),
        (
            "/branches/0/moves/0/controller_after/rng/word_position",
            "99".into(),
        ),
        ("/branches/0/controllers/0/rng/version", 99.into()),
        ("/branches/0/controllers/0/rng/initial_seed", "-1".into()),
        ("/selected_branch", 99.into()),
        ("/cursor", 999.into()),
        ("/initial_state/counters/0", 999.into()),
    ];
    for (path, value) in bad {
        let mut changed = file.clone();
        *changed.pointer_mut(path).unwrap() = value;
        assert!(
            import_replay::<_, TicTacToeEvaluator>(
                &changed.to_string(),
                model,
                ReplayLimits::default()
            )
            .is_err(),
            "{path}"
        );
    }
    for limits in [
        ReplayLimits {
            max_bytes: 1,
            ..Default::default()
        },
        ReplayLimits {
            max_depth: 1,
            ..Default::default()
        },
        ReplayLimits {
            max_branches: 1,
            ..Default::default()
        },
        ReplayLimits {
            max_moves: 1,
            ..Default::default()
        },
        ReplayLimits {
            max_cells: 1,
            ..Default::default()
        },
        ReplayLimits {
            max_verification_work: 1,
            ..Default::default()
        },
    ] {
        assert!(import_replay::<_, TicTacToeEvaluator>(&text, model, limits).is_err());
    }
    let mut unknown = file;
    unknown["extra"] = true.into();
    assert!(
        import_replay::<_, TicTacToeEvaluator>(
            &unknown.to_string(),
            model,
            ReplayLimits::default()
        )
        .is_err()
    );
    assert!(import_replay::<_, TicTacToeEvaluator>("{", model, ReplayLimits::default()).is_err());
    assert_eq!(game.active().session().state(), &original);
}
#[test]
fn standalone_state_validation_rejects_gravity_counts_outcomes_and_metadata() {
    let model = Connect4ForwardModel;
    let parameters = Connect4Parameters::default();
    let mut state = model.init_state(&parameters, 2).unwrap();
    model
        .next(
            &mut state,
            PlayerId::new(0),
            &tag_rust::games::connect4::DropPiece { column: 0 },
        )
        .unwrap();
    let text = export_state(&model, &parameters, &state).unwrap();
    let file: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut floating = file.clone();
    floating["state"]["cells"][4] = 0.into();
    floating["state"]["cells"][5] = serde_json::Value::Null;
    assert!(import_state(&floating.to_string(), &model, ReplayLimits::default()).is_err());
    for (path, value) in [
        ("/state/current_player", 0.into()),
        ("/state/cells/5", 1.into()),
        ("/state/counters/1", 99.into()),
        ("/state/outcome/kind", "draw".into()),
        ("/parameters/winning_length", 999.into()),
    ] {
        let mut invalid = file.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        assert!(
            import_state(&invalid.to_string(), &model, ReplayLimits::default()).is_err(),
            "{path}"
        );
    }
    let text = export_state(
        &TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        &TicTacToeForwardModel
            .init_state(&TicTacToeParameters::default(), 2)
            .unwrap(),
    )
    .unwrap();
    assert!(import_state(&text, &model, ReplayLimits::default()).is_err());
    assert!(import_replay::<_, Connect4Evaluator>(&text, model, ReplayLimits::default()).is_err());
}

fn exhaustive_states<F>(
    model: &F,
    parameters: &F::Parameters,
    state: F::State,
    seen: &mut std::collections::HashSet<Vec<Option<usize>>>,
) where
    F: PersistableGame<SavedState = BoardStateData>,
    F::State: Clone + PartialEq + Debug,
    F::Parameters: Clone + Serialize + DeserializeOwned,
    F::Error: Debug,
{
    let data = model.state_data(&state);
    if !seen.insert(data.cells) {
        return;
    }
    let text = export_state(model, parameters, &state).unwrap();
    let (_, restored) = import_state(&text, model, ReplayLimits::default()).unwrap();
    assert_eq!(restored, state);
    if let tag_rust::core::Decision::Player(actor) = model.next_player(&state) {
        for action in model.available_actions(&state, actor).unwrap() {
            let mut next = state.clone();
            model.next(&mut next, actor, &action).unwrap();
            exhaustive_states(model, parameters, next, seen);
        }
    }
}
#[test]
fn every_reachable_small_game_state_passes_independent_state_validation() {
    let mut states = std::collections::HashSet::new();
    let parameters = TicTacToeParameters::default();
    exhaustive_states(
        &TicTacToeForwardModel,
        &parameters,
        TicTacToeForwardModel.init_state(&parameters, 2).unwrap(),
        &mut states,
    );
    assert_eq!(states.len(), 5478);
    let mut states = std::collections::HashSet::new();
    let parameters = Connect4Parameters {
        width: 3,
        height: 3,
        winning_length: 3,
        ..Default::default()
    };
    exhaustive_states(
        &Connect4ForwardModel,
        &parameters,
        Connect4ForwardModel.init_state(&parameters, 2).unwrap(),
        &mut states,
    );
    assert!(states.len() > 500);
}
#[test]
fn incorrect_winning_cells_and_play_after_a_terminal_win_are_rejected() {
    let model = Connect4ForwardModel;
    let parameters = Connect4Parameters::default();
    let mut state = model.init_state(&parameters, 2).unwrap();
    for (turn, column) in [0, 1, 0, 1, 0, 1, 0].into_iter().enumerate() {
        model
            .next(
                &mut state,
                PlayerId::new(turn % 2),
                &tag_rust::games::connect4::DropPiece { column },
            )
            .unwrap();
    }
    let text = export_state(&model, &parameters, &state).unwrap();
    let mut file: serde_json::Value = serde_json::from_str(&text).unwrap();
    file["state"]["winning_cells"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(import_state(&file.to_string(), &model, ReplayLimits::default()).is_err());
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters::default();
    // Both players have winning rows; a legal game would have ended before this board.
    let data = BoardStateData {
        cells: vec![
            Some(0),
            Some(1),
            None,
            Some(0),
            Some(1),
            None,
            Some(0),
            Some(1),
            None,
        ],
        current_player: 1,
        counters: [6, 5, 5, 0],
        outcome: OutcomeData::Won(1),
        winning_cells: vec![],
    };
    assert!(model.restore_state(&parameters, &data).is_err());
}

#[test]
fn exported_parameters_must_match_the_state_rules() {
    let model = Connect4ForwardModel;
    let parameters = Connect4Parameters::default();
    let state = model.init_state(&parameters, 2).unwrap();
    let different = Connect4Parameters {
        winning_length: 3,
        ..parameters
    };
    assert!(export_state(&model, &different, &state).is_err());
    let session = Session::from_state(
        model,
        state,
        vec![Controller::osla(Connect4Evaluator, 7), Controller::Human],
    )
    .unwrap();
    assert!(export_replay(&BranchSession::new(session), &different, 0, 0).is_err());
}

#[test]
fn replay_branch_depth_and_state_work_limits_are_enforced() {
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters::default();
    let mut game = BranchSession::new(
        Session::create(
            model,
            &parameters,
            2,
            vec![Controller::osla(TicTacToeEvaluator, 7), Controller::Human],
        )
        .unwrap(),
    );
    for _ in 0..17 {
        game.fork(game.active_branch(), 0, vec![None, None])
            .unwrap();
    }
    let text = export_replay(&game, &parameters, 0, 0).unwrap();
    assert!(
        import_replay::<_, TicTacToeEvaluator>(
            &text,
            model,
            ReplayLimits {
                max_depth: 16,
                ..Default::default()
            }
        )
        .is_err()
    );
    let text = export_state(&model, &parameters, game.root()).unwrap();
    assert!(
        import_state(
            &text,
            &model,
            ReplayLimits {
                max_verification_work: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

fn check_saved_examples<F, E>(model: F, middle: &str, finished: &str, replay: &str)
where
    F: ReplayGame<SavedEvaluator = E, SavedState = BoardStateData> + PublicInformationModel,
    F::State: Clone + PartialEq + Debug,
    F::Action: Clone + PartialEq + DeserializeOwned,
    F::Parameters: DeserializeOwned,
    F::Error: Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: Debug,
    E::RestoreError: Debug,
{
    let (_, mut state) = import_state(middle, &model, ReplayLimits::default()).unwrap();
    let data = model.state_data(&state);
    assert_eq!(data.counters[0], 3);
    assert_eq!(data.current_player, 1);
    assert_eq!(data.outcome, OutcomeData::Ongoing);
    let mut agents = [FirstAvailableAgent::new(10), FirstAvailableAgent::new(11)];
    tag_rust::runner::run_game(&model, &mut state, &mut agents).unwrap();
    assert_eq!(
        model.next_player(&state),
        tag_rust::core::Decision::Finished
    );
    let (_, state) = import_state(finished, &model, ReplayLimits::default()).unwrap();
    assert_eq!(model.state_data(&state).outcome, OutcomeData::Won(1));
    let mut loaded = import_replay::<F, E>(replay, model, ReplayLimits::default()).unwrap();
    assert_eq!(loaded.cursor, 2);
    assert_eq!(loaded.delay_ms, 250);
    assert_eq!(loaded.session.branch_count(), 2);
    assert_eq!(loaded.session.active_branch(), BranchId(1));
    assert_eq!(loaded.session.active().session().state(), &state);
    assert!(matches!(
        loaded.session.initial_controllers(BranchId(1)).unwrap()[0],
        ControllerSnapshot::Human
    ));
    loaded.session.select(BranchId(0)).unwrap();
    assert_eq!(loaded.session.info(BranchId(0)).unwrap().move_count, 9);
}
#[test]
fn checked_in_save_examples_load_and_continue_with_new_controllers() {
    check_saved_examples::<_, TicTacToeEvaluator>(
        TicTacToeForwardModel,
        include_str!("../../../examples/saves/tictactoe.tag-state.json"),
        include_str!("../../../examples/saves/tictactoe-finished.tag-state.json"),
        include_str!("../../../examples/saves/tictactoe.tag-replay.json"),
    );
    check_saved_examples::<_, Connect4Evaluator>(
        Connect4ForwardModel,
        include_str!("../../../examples/saves/connect4.tag-state.json"),
        include_str!("../../../examples/saves/connect4-finished.tag-state.json"),
        include_str!("../../../examples/saves/connect4.tag-replay.json"),
    );
}
