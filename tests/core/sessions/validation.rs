use tag_rust::agents::OslaError;
use tag_rust::agents::{Agent, EmptyActionList, FirstAvailableAgent};
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId};
use tag_rust::games::tictactoe::TicTacToeState;
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeError, TicTacToeForwardModel, TicTacToeParameters,
};
use tag_rust::runner::RunnerError;
use tag_rust::session::{BuiltinAgentError, Controller, Session, SessionError};

type HumanSession = Session<TicTacToeForwardModel, FirstAvailableAgent>;

fn humans() -> HumanSession {
    Session::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![Controller::Human, Controller::Human],
    )
    .unwrap()
}

#[test]
fn creation_checks_controllers_and_game_parameters() {
    let missing = HumanSession::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![],
    );
    assert!(matches!(
        missing,
        Err(SessionError::ControllerCount {
            expected: 2,
            actual: 0
        })
    ));
    let invalid = HumanSession::create(
        TicTacToeForwardModel,
        &TicTacToeParameters {
            grid_size: 0,
            ..Default::default()
        },
        2,
        vec![Controller::Human, Controller::Human],
    );
    assert!(matches!(
        invalid,
        Err(SessionError::Execution(RunnerError::Game(_)))
    ));
}

#[test]
fn rejected_human_requests_preserve_state_and_position_version() {
    let mut session = humans();
    let initial = session.state().clone();
    assert!(matches!(
        session.submit_human(1, PlayerId::new(0), PlaceMark { x: 0, y: 0 }),
        Err(SessionError::StalePositionVersion {
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        session.submit_human(0, PlayerId::new(1), PlaceMark { x: 0, y: 0 }),
        Err(SessionError::WrongActor { .. })
    ));
    assert!(matches!(
        session.submit_human(0, PlayerId::new(0), PlaceMark { x: 10, y: 0 }),
        Err(SessionError::ActionNotOffered { .. })
    ));
    assert!(matches!(
        session.advance_ai(0),
        Err(SessionError::HumanRequired { .. })
    ));
    assert_eq!(session.state(), &initial);
    assert_eq!(session.position_version(), 0);
    session
        .submit_human(0, PlayerId::new(0), PlaceMark { x: 0, y: 0 })
        .unwrap();
    let accepted = session.state().clone();
    assert!(matches!(
        session.submit_human(1, PlayerId::new(1), PlaceMark { x: 0, y: 0 }),
        Err(SessionError::ActionNotOffered { .. })
    ));
    assert!(matches!(
        session.submit_human(0, PlayerId::new(1), PlaceMark { x: 0, y: 1 }),
        Err(SessionError::StalePositionVersion { .. })
    ));
    assert_eq!(session.state(), &accepted);
    assert_eq!(session.position_version(), 1);
}

#[test]
fn views_are_independent_and_terminal_requests_preserve_completion() {
    let mut session = humans();
    let original = session.decision_view(PlayerId::new(0)).unwrap();
    assert_eq!(original.actions.len(), 9);
    assert!(
        session
            .decision_view(PlayerId::new(1))
            .unwrap()
            .actions
            .is_empty()
    );
    assert!(matches!(
        session.decision_view(PlayerId::new(2)),
        Err(TicTacToeError::InvalidPlayer { .. })
    ));
    while let Decision::Player(actor) = session.next_player() {
        let view = session.decision_view(actor).unwrap();
        session
            .submit_human(
                view.position_version,
                actor,
                view.actions.into_iter().next().unwrap(),
            )
            .unwrap();
    }
    assert_eq!(original.observation.move_count(), 0);
    assert_eq!(original.actions.len(), 9);
    let finished = session.state().clone();
    let position_version = session.position_version();
    assert!(matches!(
        session.advance_ai(position_version),
        Err(SessionError::GameFinished)
    ));
    assert!(matches!(
        session.submit_human(position_version, PlayerId::new(0), PlaceMark { x: 2, y: 2 }),
        Err(SessionError::GameFinished)
    ));
    assert_eq!(session.state(), &finished);
    assert_eq!(session.position_version(), position_version);
}

struct FaultAgent {
    rng: AgentRng,
    calls: usize,
    invalid: bool,
}
impl<F: ForwardModel> Agent<F> for FaultAgent {
    type Error = EmptyActionList;
    fn rng(&self) -> &AgentRng {
        &self.rng
    } // Shared RNG borrow.
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    } // Exclusive RNG borrow.
    fn choose(
        &mut self,
        _model: &F,
        _actor: PlayerId,
        _view: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, EmptyActionList> {
        self.calls += 1;
        if self.invalid {
            return Ok(actions.len());
        }
        if self.calls == 1 {
            return Err(EmptyActionList);
        }
        Ok(0)
    }
}

#[test]
fn invalid_requests_skip_ai_and_agent_failures_allow_resumption() {
    let mut session = Session::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![
            Controller::Ai(FaultAgent {
                rng: AgentRng::new(17),
                calls: 0,
                invalid: false,
            }),
            Controller::Human,
        ],
    )
    .unwrap();
    let original = session.state().clone();
    assert!(matches!(
        session.submit_human(0, PlayerId::new(0), PlaceMark { x: 0, y: 0 }),
        Err(SessionError::AiRequired { .. })
    ));
    assert!(matches!(
        session.advance_ai(99),
        Err(SessionError::StalePositionVersion { .. })
    ));
    let Some(Controller::Ai(agent)) = session.controller(PlayerId::new(0)) else {
        panic!()
    };
    assert_eq!(agent.calls, 0);
    assert!(matches!(
        session.advance_ai(0),
        Err(SessionError::Execution(RunnerError::Agent { .. }))
    ));
    assert_eq!(session.state(), &original);
    assert_eq!(session.position_version(), 0);
    session.advance_ai(0).unwrap();
    assert_eq!(session.position_version(), 1);
    assert_eq!(session.next_player(), Decision::Player(PlayerId::new(1)));
    let Some(Controller::Ai(agent)) = session.controller(PlayerId::new(0)) else {
        panic!()
    };
    assert_eq!(agent.calls, 2);
}

#[test]
fn invalid_ai_selection_preserves_game_and_reports_the_offered_count() {
    let mut session = Session::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![
            Controller::Ai(FaultAgent {
                rng: AgentRng::new(17),
                calls: 0,
                invalid: true,
            }),
            Controller::Human,
        ],
    )
    .unwrap();
    let original = session.state().clone();
    assert!(
        matches!(session.advance_ai(0), Err(SessionError::Execution(RunnerError::InvalidSelection {
        player, index: 9, action_count: 9,
    })) if player == PlayerId::new(0))
    );
    assert_eq!(session.state(), &original);
    assert_eq!(session.position_version(), 0);
}

struct FailingEvaluator;
impl Evaluator<TicTacToeState> for FailingEvaluator {
    type Error = &'static str;
    fn evaluate(&mut self, _: &TicTacToeState, _: PlayerId) -> Result<f64, Self::Error> {
        Err("evaluation unavailable")
    }
}

#[test]
fn mixed_builtin_controller_preserves_osla_failure_context() {
    let mut session = Session::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![Controller::osla(FailingEvaluator, 7), Controller::Human],
    )
    .unwrap();
    let original = session.state().clone();
    assert!(
        matches!(session.advance_ai(0), Err(SessionError::Execution(RunnerError::Agent {
        player, error: BuiltinAgentError::Osla(OslaError::Evaluation { candidate: 0, error: "evaluation unavailable" }),
    })) if player == PlayerId::new(0))
    );
    assert_eq!(session.state(), &original);
    assert_eq!(session.position_version(), 0);
}

struct ThreeSeatModel;
#[derive(Debug, PartialEq, Eq)] // PartialEq/Eq: exact progress equality; Debug: inspect failed assertions.
struct State {
    moves: usize,
}
#[derive(Debug, PartialEq, Eq)] // PartialEq/Eq: compare action content; Debug: inspect the owned move.
struct Action(Box<usize>);

impl ForwardModel for ThreeSeatModel {
    type State = State;
    type Action = Action;
    type Observation = usize;
    type Parameters = ();
    type Error = &'static str;
    fn init_state(&self, _: &(), count: usize) -> Result<State, Self::Error> {
        if count == 3 {
            Ok(State { moves: 0 })
        } else {
            Err("three seats required")
        }
    }
    fn player_count(&self, _: &State) -> usize {
        3
    }
    fn next_player(&self, state: &State) -> Decision {
        if state.moves == 6 {
            Decision::Finished
        } else {
            Decision::Player(PlayerId::new(state.moves % 3))
        }
    }
    fn observe(&self, state: &State, observer: PlayerId) -> Result<usize, Self::Error> {
        if observer.index() >= 3 {
            Err("unknown observer")
        } else if self.next_player(state) == Decision::Player(observer) {
            Ok(state.moves)
        } else {
            Ok(usize::MAX)
        }
    }
    fn offered_actions(&self, view: &usize) -> Vec<Action> {
        if *view == usize::MAX {
            vec![]
        } else {
            vec![Action(Box::new(*view))]
        }
    }
    fn available_actions(
        &self,
        state: &State,
        actor: PlayerId,
    ) -> Result<Vec<Action>, Self::Error> {
        Ok(self.offered_actions(&self.observe(state, actor)?))
    }
    fn next(&self, state: &mut State, actor: PlayerId, action: &Action) -> Result<(), Self::Error> {
        if self.next_player(state) != Decision::Player(actor) || *action.0 != state.moves {
            return Err("invalid move");
        }
        state.moves += 1;
        Ok(())
    }
    fn player_result(
        &self,
        state: &State,
        _: PlayerId,
    ) -> Result<tag_rust::core::PlayerResult, Self::Error> {
        Ok(if state.moves == 6 {
            tag_rust::core::PlayerResult::Draw
        } else {
            tag_rust::core::PlayerResult::Ongoing
        })
    }
    fn player_score(&self, _: &State, _: PlayerId) -> Result<f64, Self::Error> {
        Ok(0.0)
    }
}

#[test]
fn generic_session_supports_three_seats_with_owned_nonclone_actions_and_state() {
    let mut session = Session::create(
        ThreeSeatModel,
        &(),
        3,
        vec![
            Controller::Human,
            Controller::Ai(FirstAvailableAgent::new(2)),
            Controller::Human,
        ],
    )
    .unwrap();
    for count in 0..6 {
        let actor = PlayerId::new(count % 3);
        let record = if actor.index() == 1 {
            session.advance_ai(count as u64).unwrap()
        } else {
            session
                .submit_human(count as u64, actor, Action(Box::new(count)))
                .unwrap()
        };
        assert_eq!(record.actor, actor);
        assert_eq!(*record.action.0, count);
        assert_eq!(session.state().moves, count + 1);
    }
    assert_eq!(session.next_player(), Decision::Finished);
}
