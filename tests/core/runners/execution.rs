use tag_rust::agents::{Agent, EmptyActionList, FirstAvailableAgent};
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::tictactoe::{Cell, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::runner::{RunnerError, run_game};

#[test]
fn first_available_agents_finish_tictactoe_with_expected_outcome() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let mut agents = [FirstAvailableAgent::new(100), FirstAvailableAgent::new(200)];
    assert_eq!(run_game(&model, &mut state, &mut agents), Ok(()));
    assert_eq!(model.next_player(&state), Decision::Finished);
    assert_eq!(state.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
    assert_eq!(state.move_count(), 7);
    assert_eq!(state.completed_turn_count(), 6);
    for (x, y) in [(0, 2), (1, 1), (2, 0)] {
        assert_eq!(state.cell(x, y), Ok(Cell::X));
    }
    assert_eq!(model.player_score(&state, PlayerId::new(0)), Ok(1.0));
    assert_eq!(model.player_score(&state, PlayerId::new(1)), Ok(-1.0));
    let finished = state.clone(); // Independent terminal snapshot.
    assert_eq!(
        run_game(&model, &mut state, &mut [] as &mut [FirstAvailableAgent]),
        Ok(())
    );
    assert_eq!(state, finished); // A finished game returns with its final state intact.
}

struct FixedSelection {
    rng: AgentRng,
    index: usize,
    calls: usize,
}

impl<F: ForwardModel> Agent<F> for FixedSelection {
    type Error = EmptyActionList;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,                     // Exclusive borrow; records how often this controller acts.
        _model: &F,                    // Shared rules borrow.
        _actor: PlayerId,              // Copied seat.
        _observation: &F::Observation, // Shared player view.
        actions: &[F::Action],         // Shared available choices.
    ) -> Result<usize, EmptyActionList> {
        if actions.is_empty() {
            return Err(EmptyActionList);
        }
        // Copied choice; controller remains usable.
        self.calls += 1;
        Ok(self.index)
    }
}

#[test]
fn invalid_choices_preserve_the_game_and_ongoing_result() {
    let model = TicTacToeForwardModel;
    for index in [9, usize::MAX] {
        let mut state = model
            .init_state(&TicTacToeParameters::default(), 2)
            .unwrap();
        let original = state.clone();
        let mut agents = [FixedSelection {
            rng: AgentRng::new(100),
            index,
            calls: 0,
        }];
        let expected = RunnerError::InvalidSelection {
            player: PlayerId::new(0),
            index,
            action_count: 9,
        };
        assert_eq!(run_game(&model, &mut state, &mut agents), Err(expected));
        assert_eq!(agents[0].calls, 1);
        assert_eq!(state, original);
        assert_eq!(
            model.player_result(&state, PlayerId::new(0)),
            Ok(PlayerResult::Ongoing)
        );
    }
}

#[test]
fn missing_controller_preserves_already_accepted_moves() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    assert_eq!(
        run_game(&model, &mut state, &mut [] as &mut [FirstAvailableAgent]),
        Err(RunnerError::MissingController {
            player: PlayerId::new(0)
        })
    );
    assert_eq!(state.move_count(), 0);
    let mut agents = [FirstAvailableAgent::new(100)];
    assert_eq!(
        run_game(&model, &mut state, &mut agents),
        Err(RunnerError::MissingController {
            player: PlayerId::new(1)
        })
    );
    assert_eq!(state.cell(0, 0), Ok(Cell::X));
    assert_eq!(state.move_count(), 1);
    assert_eq!(state.current_player(), PlayerId::new(1));
    let mut agents = [FirstAvailableAgent::new(100), FirstAvailableAgent::new(200)];
    assert_eq!(run_game(&model, &mut state, &mut agents), Ok(())); // Resume the supplied game.
    assert_eq!(state.move_count(), 7);
}

// Tiny three-player test model checks generic execution.
// Action ownership is moved or borrowed throughout this test model.
#[derive(PartialEq)]
struct TestAction;
struct TestModel {
    empty_actions: bool,
    fail_observe: bool,
    fail_next: bool,
}

#[derive(Debug, PartialEq, Eq)] // Equality: compare failures in tests; Eq: reflexive variant equality.
// Test failures are returned by ownership.
enum TestFailure {
    Observation,
    Transition,
}

impl ForwardModel for TestModel {
    type State = usize;
    type Action = TestAction;
    type Observation = PlayerId;
    type Parameters = ();
    type Error = TestFailure;

    fn init_state(&self, _parameters: &(), _player_count: usize) -> Result<usize, TestFailure> {
        // Shared model/parameters borrows; copied count; returns owned initial progress.
        Ok(0)
    }
    fn player_count(&self, _state: &usize) -> usize {
        // Shared borrows; copied count out.
        3
    }
    fn next_player(&self, state: &usize) -> Decision {
        // Shared borrows; returns copied decision, with a deliberately non-cyclic seat order.
        match *state {
            0 => Decision::Player(PlayerId::new(2)),
            1 => Decision::Player(PlayerId::new(0)),
            2 => Decision::Player(PlayerId::new(1)),
            _ => Decision::Finished,
        }
    }
    fn observe(&self, _state: &usize, observer: PlayerId) -> Result<PlayerId, TestFailure> {
        // Shared borrows and copied observer; returns owned view or failure.
        if self.fail_observe {
            Err(TestFailure::Observation)
        } else {
            Ok(observer)
        }
    }
    fn offered_actions(&self, _observation: &PlayerId) -> Vec<TestAction> {
        panic!("live runner must generate choices from state")
    }
    fn available_actions(
        &self,
        _state: &usize,
        _actor: PlayerId,
    ) -> Result<Vec<TestAction>, TestFailure> {
        if self.empty_actions {
            Ok(vec![])
        } else {
            Ok(vec![TestAction])
        }
    }
    fn next(
        &self,
        state: &mut usize,
        actor: PlayerId,
        _action: &TestAction,
    ) -> Result<(), TestFailure> {
        // Shared model/action, exclusive state borrow, copied actor; owned failure or success.
        assert_eq!(self.next_player(state), Decision::Player(actor));
        if self.fail_next {
            *state += 1;
            return Err(TestFailure::Transition);
        }
        *state += 1;
        Ok(())
    }
    fn player_result(
        &self,
        _state: &usize,
        _player: PlayerId,
    ) -> Result<PlayerResult, TestFailure> {
        // Shared borrows and copied seat; returns the ongoing result.
        Ok(PlayerResult::Ongoing)
    }
    fn player_score(&self, _state: &usize, player: PlayerId) -> Result<f64, TestFailure> {
        // Shared borrows and copied seat; distinct positive scores avoid zero-sum assumptions.
        Ok(player.index() as f64 + 1.0)
    }
}

#[test]
fn generic_runner_follows_three_player_decisions_and_accepts_nonclone_actions() {
    let model = TestModel {
        empty_actions: false,
        fail_observe: false,
        fail_next: false,
    };
    let mut state = 0;
    let mut agents = [
        FixedSelection {
            rng: AgentRng::new(100),
            index: 0,
            calls: 0,
        },
        FixedSelection {
            rng: AgentRng::new(100),
            index: 0,
            calls: 0,
        },
        FixedSelection {
            rng: AgentRng::new(100),
            index: 0,
            calls: 0,
        },
    ];
    assert_eq!(run_game(&model, &mut state, &mut agents), Ok(()));
    assert_eq!(state, 3);
    for agent in agents {
        assert_eq!(agent.calls, 1);
    }
}

#[test]
fn empty_offers_and_game_errors_stop_before_agent_or_state_changes() {
    for (empty_actions, fail_observe, fail_next, expected, calls) in [
        (
            true,
            false,
            false,
            RunnerError::NoAvailableActions {
                player: PlayerId::new(2),
            },
            0,
        ),
        (
            false,
            true,
            false,
            RunnerError::Game(TestFailure::Observation),
            0,
        ),
        (
            false,
            false,
            true,
            RunnerError::Game(TestFailure::Transition),
            1,
        ),
    ] {
        let model = TestModel {
            empty_actions,
            fail_observe,
            fail_next,
        };
        let mut state = 0;
        let mut agents = [
            FixedSelection {
                rng: AgentRng::new(100),
                index: 0,
                calls: 0,
            },
            FixedSelection {
                rng: AgentRng::new(100),
                index: 0,
                calls: 0,
            },
            FixedSelection {
                rng: AgentRng::new(100),
                index: 0,
                calls: 0,
            },
        ];
        assert_eq!(run_game(&model, &mut state, &mut agents), Err(expected));
        assert_eq!(state, usize::from(fail_next));
        assert_eq!(agents[2].calls, calls);
    }
}

#[test]
fn baseline_rejects_an_empty_action_list() {
    let mut agent = FirstAvailableAgent::new(100);
    assert_eq!(
        <FirstAvailableAgent as Agent<TestModel>>::choose(
            &mut agent,
            &TestModel {
                empty_actions: false,
                fail_observe: false,
                fail_next: false
            },
            PlayerId::new(0),
            &PlayerId::new(0),
            &[]
        ),
        Err(EmptyActionList)
    );
}

#[test]
fn human_submission_uses_state_actions_without_constructing_an_observation() {
    use tag_rust::session::{Controller, Session};
    let mut session = Session::<TestModel, FirstAvailableAgent>::create(
        TestModel {
            empty_actions: false,
            fail_observe: true,
            fail_next: false,
        },
        &(),
        3,
        vec![Controller::Human, Controller::Human, Controller::Human],
    )
    .unwrap();
    // This game deliberately fails every observation request. A validated human
    // action still works because submission does not ask an agent or render a view.
    session
        .submit_human(0, PlayerId::new(2), TestAction)
        .unwrap();
    assert_eq!(*session.state(), 1);
    assert!(session.decision_view(PlayerId::new(0)).is_err());
}

#[test]
fn random_and_first_available_receive_the_same_observation_contract() {
    use tag_rust::agents::RandomAgent;
    fn check<A: Agent<TestModel>>(mut agents: [A; 3]) {
        let model = TestModel {
            empty_actions: false,
            fail_observe: true,
            fail_next: false,
        };
        let mut state = 0;
        assert!(matches!(
            run_game(&model, &mut state, &mut agents),
            Err(RunnerError::Game(TestFailure::Observation))
        ));
        assert_eq!(state, 0);
    }
    check(std::array::from_fn(|i| FirstAvailableAgent::new(i as u64)));
    check(std::array::from_fn(|i| RandomAgent::new(i as u64)));
}

#[test]
fn internal_transition_failure_stops_human_and_ai_sessions_after_partial_mutation() {
    use tag_rust::session::{Controller, Session, SessionError};
    for ai in [false, true] {
        let controllers = (0..3)
            .map(|i| {
                if ai {
                    Controller::Ai(FirstAvailableAgent::new(i))
                } else {
                    Controller::Human
                }
            })
            .collect();
        let mut session = Session::create(
            TestModel {
                empty_actions: false,
                fail_observe: false,
                fail_next: true,
            },
            &(),
            3,
            controllers,
        )
        .unwrap();
        let result = if ai {
            session.advance_ai(0)
        } else {
            session.submit_human(0, PlayerId::new(2), TestAction)
        };
        assert!(matches!(
            result,
            Err(SessionError::Execution(RunnerError::Game(
                TestFailure::Transition
            )))
        ));
        assert!(session.is_stopped());
        assert_eq!(*session.state(), 1); // The failed transition was not rolled back.
        assert_eq!(session.position_version(), 0); // No successful move was accepted.
        assert!(matches!(
            session.advance_ai(0),
            Err(SessionError::GameStopped)
        ));
        assert!(matches!(
            session.submit_human(0, PlayerId::new(0), TestAction),
            Err(SessionError::GameStopped)
        ));
        assert_eq!(*session.state(), 1); // Neither retry applied anything.
    }
}
