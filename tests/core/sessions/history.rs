use std::fmt::Debug;
use tag_rust::agents::{FirstAvailableAgent, OslaAgent};
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters, TicTacToeState,
};
use tag_rust::recording::{ControllerSnapshot, GameHistory, HistoryError, RecordedMove};
use tag_rust::runner::run_game_recorded;
use tag_rust::session::{Controller, RecordedSession, RecordingError, Session, SessionError};

fn exact_history<F, E>(make_model: impl Fn() -> F, parameters: &F::Parameters, evaluator: E)
where
    F: PublicInformationModel + Clone,
    F::State: Clone + Debug + PartialEq,
    F::Action: Clone + Debug + PartialEq,
    F::Observation: Debug + PartialEq,
    F::Error: Debug,
    E: Copy + Checkpoint + Evaluator<F::State>,
    E::Snapshot: Debug + PartialEq + Eq,
    E::RestoreError: Debug,
    E::Error: Debug,
{
    for seed in [0, 17, 200] {
        let seats = || {
            vec![
                Controller::osla(evaluator, seed),
                Controller::osla(evaluator, seed + 1),
            ]
        };
        let mut recorded =
            RecordedSession::new(Session::create(make_model(), parameters, 2, seats()).unwrap());
        let mut states = vec![recorded.session().state().clone()];
        let mut checkpoints = vec![
            (0..2)
                .map(|seat| {
                    recorded
                        .session()
                        .controller(PlayerId::new(seat))
                        .unwrap()
                        .snapshot()
                })
                .collect::<Vec<_>>(),
        ];
        while let Decision::Player(_) = recorded.session().next_player() {
            let version = recorded.session().position_version();
            recorded.advance_ai(version).unwrap();
            states.push(recorded.session().state().clone());
            checkpoints.push(
                (0..2)
                    .map(|seat| {
                        recorded
                            .session()
                            .controller(PlayerId::new(seat))
                            .unwrap()
                            .snapshot()
                    })
                    .collect(),
            );
        }
        let original = recorded.session().state().clone();
        let version = recorded.session().position_version();
        let model = make_model();
        for position in (0..states.len()).rev() {
            assert_eq!(recorded.position_at(position).unwrap(), states[position]);
            assert_eq!(
                recorded
                    .history()
                    .controllers_at::<F::Error>(position)
                    .unwrap(),
                checkpoints[position]
            );
            for seat in 0..2 {
                let observer = PlayerId::new(seat);
                let historical = recorded.view_at(position, observer).unwrap();
                assert_eq!(
                    historical.observation,
                    model.observe(&states[position], observer).unwrap()
                );
                assert_eq!(
                    historical.actions,
                    model.offered_actions(&historical.observation)
                );
            }
            let mut restored = RecordedSession::new(recorded.restore_session_at(position).unwrap());
            for (index, expected) in states.iter().enumerate().skip(position + 1) {
                let accepted = restored
                    .advance_ai(restored.session().position_version())
                    .unwrap();
                assert_eq!(
                    accepted.action,
                    recorded.history().moves()[index - 1].action
                );
                assert_eq!(restored.session().state(), expected);
                assert_eq!(
                    restored.history().moves().last().unwrap().controller_after,
                    recorded.history().moves()[index - 1].controller_after
                );
            }
        }
        assert_eq!(recorded.session().state(), &original);
        assert_eq!(recorded.session().position_version(), version);
        let mut agents = vec![
            OslaAgent::new(evaluator, seed),
            OslaAgent::new(evaluator, seed + 1),
        ];
        let mut state = model.init_state(parameters, 2).unwrap();
        let mut history = GameHistory::new(
            state.clone(),
            agents
                .iter()
                .map(|agent| ControllerSnapshot::Ai(agent.snapshot()))
                .collect(),
        );
        run_game_recorded(&model, &mut state, &mut agents, |record| {
            history.record(record)
        })
        .unwrap();
        assert_eq!(&state, &original);
        assert_eq!(history.moves().len(), states.len() - 1); // Only accepted live moves appear despite OSLA simulations.
        for (position, expected) in states.iter().enumerate() {
            assert_eq!(history.position_at(&model, position).unwrap(), *expected);
            assert_eq!(
                history.controllers_at::<F::Error>(position).unwrap(),
                checkpoints[position]
                    .iter()
                    .map(|seat| match seat {
                        ControllerSnapshot::Ai(tag_rust::session::BuiltinAgentSnapshot::Osla(
                            snapshot,
                        )) => ControllerSnapshot::Ai(snapshot.clone()),
                        _ => panic!("OSLA seats"),
                    })
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn every_recorded_position_restores_exact_ai_continuation_in_both_games() {
    for size in [1, 3, 5] {
        exact_history(
            || TicTacToeForwardModel,
            &TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            TicTacToeEvaluator,
        );
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3), (4, 3, 1)] {
        exact_history(
            || Connect4ForwardModel,
            &Connect4Parameters {
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
fn cancelled_stale_and_foreign_ai_proposals_preserve_live_progress() {
    let create = || {
        RecordedSession::new(
            Session::create(
                TicTacToeForwardModel,
                &TicTacToeParameters::default(),
                2,
                vec![
                    Controller::osla(TicTacToeEvaluator, 17),
                    Controller::osla(TicTacToeEvaluator, 18),
                ],
            )
            .unwrap(),
        )
    };
    let mut live = create();
    let mut other = create();
    let initial = live.session().state().clone();
    let checkpoint = live
        .session()
        .controller(PlayerId::new(0))
        .unwrap()
        .snapshot();
    let discarded = live.prepare_ai(0).unwrap();
    let first_action = *discarded.action(); // Copy the tiny coordinate action for comparison.
    drop(discarded);
    assert_eq!(live.session().state(), &initial);
    assert_eq!(
        live.session()
            .controller(PlayerId::new(0))
            .unwrap()
            .snapshot(),
        checkpoint
    );
    let proposed = live.prepare_ai(0).unwrap();
    assert_eq!(*proposed.action(), first_action);
    assert!(matches!(
        other.commit_ai(proposed),
        Err(RecordingError::ForeignProposal)
    ));
    assert_eq!(other.session().state(), &initial);
    let stale = live.prepare_ai(0).unwrap();
    live.advance_ai(0).unwrap();
    let accepted = live.session().state().clone();
    assert!(matches!(
        live.commit_ai(stale),
        Err(RecordingError::Session(
            SessionError::StalePositionVersion { .. }
        ))
    ));
    assert_eq!(live.session().state(), &accepted);
    assert_eq!(live.history().moves().len(), 1);
}

#[test]
fn mixed_human_history_restores_each_seats_identity_and_progress() {
    let mut game = RecordedSession::new(
        Session::create(
            TicTacToeForwardModel,
            &TicTacToeParameters::default(),
            2,
            vec![
                Controller::Human,
                Controller::<tag_rust::session::BuiltinAgent<TicTacToeEvaluator>>::first_available(
                    99,
                ),
            ],
        )
        .unwrap(),
    );
    game.submit_human(0, PlayerId::new(0), PlaceMark { x: 1, y: 1 })
        .unwrap();
    game.advance_ai(1).unwrap();
    assert!(matches!(
        game.history().moves()[0].controller_after,
        ControllerSnapshot::Human
    ));
    let restored = game.restore_session_at(1).unwrap();
    assert!(matches!(
        restored.controller(PlayerId::new(0)),
        Some(Controller::Human)
    ));
    assert!(matches!(
        restored.controller(PlayerId::new(1)),
        Some(Controller::Ai(_))
    ));
    let mut resumed = RecordedSession::new(restored);
    assert_eq!(
        resumed.advance_ai(0).unwrap().action,
        game.history().moves()[1].action
    );
    assert_eq!(resumed.session().state(), game.session().state());
    assert!(matches!(
        game.position_at(3),
        Err(HistoryError::InvalidPosition { .. })
    ));
    assert!(matches!(
        game.restore_session_at(3),
        Err(RecordingError::History(
            HistoryError::InvalidPosition { .. }
        ))
    ));
}

struct FailingEvaluator {
    calls: usize,
}
impl Evaluator<TicTacToeState> for FailingEvaluator {
    type Error = &'static str;
    fn evaluate(&mut self, _: &TicTacToeState, _: PlayerId) -> Result<f64, Self::Error> {
        self.calls += 1;
        Err("blocked")
    }
}
impl Checkpoint for FailingEvaluator {
    type Snapshot = usize;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> usize {
        self.calls
    } // Copied persistent counter.
    fn from_snapshot(snapshot: &usize) -> Result<Self, Self::RestoreError> {
        Ok(Self { calls: *snapshot })
    }
}

#[test]
fn failed_preparation_discards_evaluator_changes_and_leaves_history_empty() {
    let mut game = RecordedSession::new(
        Session::create(
            TicTacToeForwardModel,
            &TicTacToeParameters::default(),
            2,
            vec![
                Controller::osla(FailingEvaluator { calls: 0 }, 7),
                Controller::Human,
            ],
        )
        .unwrap(),
    );
    let initial = game.session().state().clone();
    let before = game
        .session()
        .controller(PlayerId::new(0))
        .unwrap()
        .snapshot();
    assert!(matches!(
        game.advance_ai(0),
        Err(RecordingError::Session(SessionError::Execution(_)))
    ));
    assert_eq!(game.session().state(), &initial);
    assert_eq!(
        game.session()
            .controller(PlayerId::new(0))
            .unwrap()
            .snapshot(),
        before
    );
    assert!(game.history().moves().is_empty());
}

#[test]
fn recording_resumes_from_current_conditions_and_checks_cache_bound() {
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters {
        grid_size: 10,
        ..Default::default()
    };
    let mut state = model.init_state(&parameters, 2).unwrap();
    let mut agents = [FirstAvailableAgent::new(1), FirstAvailableAgent::new(2)];
    let mut history = GameHistory::new(
        state.clone(),
        agents
            .iter()
            .map(|agent| ControllerSnapshot::Ai(agent.snapshot()))
            .collect(),
    );
    run_game_recorded(&model, &mut state, &mut agents, |record| {
        history.record(record)
    })
    .unwrap();
    assert!(history.moves().len() > 64);
    assert_eq!(
        history.position_at(&model, history.moves().len()).unwrap(),
        state
    );
    assert_eq!(history.cached_state_count(), 4);
    assert_eq!(
        history.position_at(&model, 0).unwrap(),
        *history.initial_state()
    );
    assert_eq!(history.cached_state_count(), 4);
    // Own and mutate a recovered position; later reconstruction retains the recording's root.
    let mut snapshot = history.position_at(&model, 0).unwrap();
    model
        .next(&mut snapshot, PlayerId::new(0), &PlaceMark { x: 0, y: 0 })
        .unwrap();
    assert_eq!(history.position_at(&model, 0).unwrap().move_count(), 0);
}

#[test]
fn recording_can_start_from_an_existing_position_and_controller_progress() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    model
        .next(&mut state, PlayerId::new(0), &PlaceMark { x: 1, y: 1 })
        .unwrap();
    let mut agents = [
        OslaAgent::new(TicTacToeEvaluator, 7),
        OslaAgent::new(TicTacToeEvaluator, 8),
    ];
    // Advance the generators before recording to exercise continuation from existing progress.
    use tag_rust::agents::Agent;
    for agent in &mut agents {
        <OslaAgent<TicTacToeEvaluator> as Agent<TicTacToeForwardModel>>::rng_mut(agent).index(23);
    }
    let initial = state.clone();
    let mut history = GameHistory::new(
        initial.clone(),
        agents
            .iter()
            .map(|a| ControllerSnapshot::Ai(a.snapshot()))
            .collect(),
    );
    run_game_recorded(&model, &mut state, &mut agents, |record| {
        history.record(record)
    })
    .unwrap();
    assert_eq!(history.position_at(&model, 0).unwrap(), initial);
    assert_eq!(
        history.position_at(&model, history.moves().len()).unwrap(),
        state
    );
    assert_eq!(history.moves().len() + 1, state.move_count());
    let mut restored_state = initial;
    let mut restored_agents: Vec<OslaAgent<TicTacToeEvaluator>> = history
        .initial_controllers()
        .iter()
        .map(|seat| match seat {
            ControllerSnapshot::Ai(snapshot) => OslaAgent::from_snapshot(snapshot).unwrap(),
            ControllerSnapshot::Human => panic!("AI seats"),
        })
        .collect::<Vec<_>>();
    let mut actions = Vec::new();
    run_game_recorded(
        &model,
        &mut restored_state,
        &mut restored_agents,
        |record| actions.push(record.action),
    )
    .unwrap();
    assert_eq!(restored_state, state);
    assert_eq!(
        actions,
        history.moves().iter().map(|r| r.action).collect::<Vec<_>>()
    );
}

#[test]
fn corrupt_records_return_contextual_reconstruction_errors() {
    let model = TicTacToeForwardModel;
    let root = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let seats = vec![ControllerSnapshot::<()>::Human, ControllerSnapshot::Human];
    let make_record = |actor, decision| RecordedMove {
        actor,
        action: PlaceMark { x: 0, y: 0 },
        decision,
        controller_after: ControllerSnapshot::Human,
    };
    let mut invalid_actor = GameHistory::new(root.clone(), seats.clone());
    invalid_actor.record(make_record(
        PlayerId::new(2),
        Decision::Player(PlayerId::new(1)),
    ));
    assert!(matches!(
        invalid_actor.controllers_at::<()>(1),
        Err(HistoryError::InvalidActor { position: 1, .. })
    ));
    assert!(matches!(
        invalid_actor.position_at(&model, 1),
        Err(HistoryError::Game { position: 1, .. })
    ));
    let mut bad_decision = GameHistory::new(root.clone(), seats.clone());
    bad_decision.record(make_record(PlayerId::new(0), Decision::Finished));
    assert!(matches!(
        bad_decision.position_at(&model, 1),
        Err(HistoryError::DecisionMismatch { position: 1 })
    ));
    let mut occupied_cell = GameHistory::new(root.clone(), seats);
    occupied_cell.record(make_record(
        PlayerId::new(0),
        Decision::Player(PlayerId::new(1)),
    ));
    occupied_cell.record(make_record(
        PlayerId::new(1),
        Decision::Player(PlayerId::new(0)),
    ));
    assert!(matches!(
        occupied_cell.position_at(&model, 2),
        Err(HistoryError::Game { position: 2, .. })
    ));
    assert_eq!(occupied_cell.position_at(&model, 0).unwrap(), root);
}

#[test]
fn headless_failure_keeps_accepted_records_and_live_agent_progress() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let mut agents = [
        tag_rust::session::BuiltinAgent::FirstAvailable(FirstAvailableAgent::new(1)),
        tag_rust::session::BuiltinAgent::Osla(OslaAgent::new(FailingEvaluator { calls: 0 }, 2)),
    ];
    let mut history = GameHistory::new(
        state.clone(),
        agents
            .iter()
            .map(|a| ControllerSnapshot::Ai(a.snapshot()))
            .collect(),
    );
    assert!(
        run_game_recorded(&model, &mut state, &mut agents, |record| history
            .record(record))
        .is_err()
    );
    assert_eq!(history.moves().len(), 1);
    assert_eq!(state.move_count(), 1);
    assert_eq!(history.position_at(&model, 1).unwrap(), state);
    let tag_rust::session::BuiltinAgentSnapshot::Osla(snapshot) = agents[1].snapshot() else {
        panic!("OSLA seat")
    };
    assert_eq!(snapshot.evaluator, 1);
    // The recorded accepted boundary precedes the failed decision's evaluator changes.
    let ControllerSnapshot::Ai(tag_rust::session::BuiltinAgentSnapshot::Osla(accepted)) =
        &history.controllers_at::<()>(1).unwrap()[1]
    else {
        panic!("OSLA seat")
    };
    assert_eq!(accepted.evaluator, 0);
}

struct RestoreFailureEvaluator {
    calls: usize,
    fail_at: usize,
}
impl Evaluator<TicTacToeState> for RestoreFailureEvaluator {
    type Error = std::convert::Infallible;
    fn evaluate(&mut self, _: &TicTacToeState, _: PlayerId) -> Result<f64, Self::Error> {
        self.calls += 1;
        Ok(0.0)
    }
}
impl Checkpoint for RestoreFailureEvaluator {
    type Snapshot = (usize, usize);
    type RestoreError = &'static str;
    fn snapshot(&self) -> Self::Snapshot {
        (self.calls, self.fail_at)
    }
    fn from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        if snapshot.0 == snapshot.1 {
            return Err("restoration rejected");
        }
        Ok(Self {
            calls: snapshot.0,
            fail_at: snapshot.1,
        })
    }
}

#[test]
fn restoration_failure_during_preparation_and_commit_preserves_live_game() {
    for fail_at in [0, 9] {
        let mut game = RecordedSession::new(
            Session::create(
                TicTacToeForwardModel,
                &TicTacToeParameters::default(),
                2,
                vec![
                    Controller::osla(RestoreFailureEvaluator { calls: 0, fail_at }, 7),
                    Controller::Human,
                ],
            )
            .unwrap(),
        );
        let state = game.session().state().clone();
        let before = game
            .session()
            .controller(PlayerId::new(0))
            .unwrap()
            .snapshot();
        let result = match game.prepare_ai(0) {
            Ok(proposal) => {
                assert_eq!(fail_at, 9); // Evaluating nine candidate cells changes only the temporary evaluator.
                game.commit_ai(proposal)
            }
            Err(error) => {
                assert_eq!(fail_at, 0);
                Err(error)
            }
        };
        assert!(matches!(result, Err(RecordingError::Restore(_))));
        assert_eq!(game.session().state(), &state);
        assert_eq!(game.session().position_version(), 0);
        assert_eq!(
            game.session()
                .controller(PlayerId::new(0))
                .unwrap()
                .snapshot(),
            before
        );
        assert!(game.history().moves().is_empty());
    }
}

#[test]
fn controller_restoration_copies_only_the_final_snapshot_of_each_seat() {
    struct CountedSnapshot {
        copies: std::rc::Rc<std::cell::Cell<usize>>,
        value: usize,
    }
    impl Clone for CountedSnapshot {
        fn clone(&self) -> Self {
            self.copies.set(self.copies.get() + 1);
            Self {
                copies: self.copies.clone(),
                value: self.value,
            }
        }
    }
    let copies = std::rc::Rc::new(std::cell::Cell::new(0));
    let snapshot = |value| {
        ControllerSnapshot::Ai(CountedSnapshot {
            copies: copies.clone(),
            value,
        })
    };
    let mut history = GameHistory::new((), vec![snapshot(0), snapshot(0), snapshot(0)]);
    for index in 0..100 {
        history.record(RecordedMove {
            actor: PlayerId::new(index % 2),
            action: (),
            decision: Decision::Finished,
            controller_after: snapshot(index + 1),
        });
    }
    let restored = history.controllers_at::<()>(100).unwrap();
    assert_eq!(copies.get(), 3);
    for (seat, value) in restored.iter().zip([99, 100, 0]) {
        let ControllerSnapshot::Ai(snapshot) = seat else {
            panic!("AI seat")
        };
        assert_eq!(snapshot.value, value);
    }
}

#[test]
fn configurable_cache_limits_preserve_reconstructed_positions() {
    use tag_rust::recording::HistoryCachePolicy;
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters {
        grid_size: 10,
        ..Default::default()
    };
    let root = model.init_state(&parameters, 2).unwrap();
    let mut state = root.clone();
    let mut agents = [FirstAvailableAgent::new(0), FirstAvailableAgent::new(1)];
    let seats = agents
        .iter()
        .map(|a| ControllerSnapshot::Ai(a.snapshot()))
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    run_game_recorded(&model, &mut state, &mut agents, |r| records.push(r)).unwrap();
    for max_states in [0, 2, 64] {
        let policy = HistoryCachePolicy {
            checkpoint_interval: std::num::NonZeroUsize::new(1).unwrap(),
            max_states,
        };
        let mut history = GameHistory::with_cache_policy(root.clone(), seats.clone(), policy);
        for record in records.iter().cloned() {
            history.record(record);
        }
        assert_eq!(history.position_at(&model, records.len()).unwrap(), state);
        assert_eq!(history.cached_state_count(), max_states.min(records.len()));
        for position in (0..records.len()).rev() {
            let restored = history.position_at(&model, position).unwrap();
            assert_eq!(restored.move_count(), position);
            assert!(history.cached_state_count() <= max_states);
        }
        assert_eq!(history.position_at(&model, 0).unwrap(), root);
    }
    let session = Session::create(
        model,
        &parameters,
        2,
        vec![
            Controller::Ai(FirstAvailableAgent::new(0)),
            Controller::Human,
        ],
    )
    .unwrap();
    let mut recording = RecordedSession::with_cache_policy(
        session,
        HistoryCachePolicy {
            max_states: 0,
            ..Default::default()
        },
    );
    recording.advance_ai(0).unwrap();
    assert_eq!(recording.position_at(1).unwrap().move_count(), 1);
    assert_eq!(recording.history().cached_state_count(), 0);
}
