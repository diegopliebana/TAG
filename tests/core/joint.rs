//! A small simultaneous game exercises the shared decision boundary.
use tag_rust::agents::{Agent, RandomAgent, SampledOslaAgent};
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::joint::{self, JointError};
use tag_rust::core::random::SimulationRng;
use tag_rust::core::simulation::SimulationModel;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::runner::{RunnerError, run_game, run_joint_game, run_joint_game_recorded};
use tag_rust::search::mcts::{
    Budgets, EvaluatorRegistry, MctsAgent, Result as SearchResult, Settings, SimulationContext,
    StateEvaluator, StateValues,
};
use tag_rust::session::{Controller, JointSession, Session, SessionError};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Position {
    secret: [usize; 2],
    chance: usize,
    choices: Option<[usize; 2]>,
    public_choices: bool,
    remaining: u8,
    fail_after_resolution: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct View {
    seat: PlayerId,
    own_secret: usize,
    remaining: u8,
    last_choices: Option<[usize; 2]>,
    public_choices: bool,
    finished: bool,
}

#[derive(Clone)]
struct Model;

impl ForwardModel for Model {
    type State = Position;
    type Action = usize;
    type Observation = View;
    type Parameters = ();
    type Error = String;

    fn init_state(&self, _: &(), players: usize) -> Result<Position, String> {
        if players != 2 {
            return Err("two players required".into());
        }
        Ok(Position {
            secret: [1, 0],
            chance: 1,
            choices: None,
            public_choices: true,
            remaining: 1,
            fail_after_resolution: false,
        })
    }
    fn player_count(&self, _: &Position) -> usize {
        2
    }
    fn next_player(&self, state: &Position) -> Decision {
        if state.remaining == 0 {
            Decision::Finished
        } else {
            Decision::Player(PlayerId::new(0))
        }
    }
    fn decision_players(&self, state: &Position) -> Vec<PlayerId> {
        if state.remaining == 0 {
            vec![]
        } else {
            vec![PlayerId::new(0), PlayerId::new(1)]
        }
    }
    fn joint_actions_public_after(&self, state: &Position) -> bool {
        state.public_choices
    }
    fn observe(&self, state: &Position, seat: PlayerId) -> Result<View, String> {
        if seat.index() >= 2 {
            return Err("invalid seat".into());
        }
        Ok(View {
            seat,
            own_secret: state.secret[seat.index()],
            remaining: state.remaining,
            last_choices: state.public_choices.then_some(state.choices).flatten(),
            public_choices: state.public_choices,
            finished: state.remaining == 0,
        })
    }
    fn offered_actions(&self, view: &View) -> Vec<usize> {
        if view.finished { vec![] } else { vec![0, 1] }
    }
    fn available_actions(&self, state: &Position, seat: PlayerId) -> Result<Vec<usize>, String> {
        Ok(self.offered_actions(&self.observe(state, seat)?))
    }
    fn next(&self, _: &mut Position, _: PlayerId, _: &usize) -> Result<(), String> {
        Err("a complete decision is required".into())
    }
    fn resolve_decision(
        &self,
        state: &mut Position,
        choices: &[(PlayerId, usize)],
    ) -> Result<(), JointError<String>> {
        joint::validate(self, state, choices)?;
        state.choices = Some([choices[0].1, choices[1].1]);
        if state.fail_after_resolution {
            return Err(JointError::Rule("injected rule failure".into()));
        }
        state.remaining -= 1;
        Ok(())
    }
    fn player_result(&self, state: &Position, seat: PlayerId) -> Result<PlayerResult, String> {
        if state.remaining > 0 {
            return Ok(PlayerResult::Ongoing);
        }
        let Some(choices) = state.choices else {
            return Ok(PlayerResult::Ongoing);
        };
        let a = (choices[0] == state.secret[0]) as usize + state.chance;
        let b = (choices[1] == state.secret[1]) as usize;
        Ok(if a == b {
            PlayerResult::Draw
        } else if (a > b) == (seat.index() == 0) {
            PlayerResult::Win
        } else {
            PlayerResult::Loss
        })
    }
    fn player_score(&self, state: &Position, seat: PlayerId) -> Result<f64, String> {
        Ok(match self.player_result(state, seat)? {
            PlayerResult::Win => 1.0,
            PlayerResult::Loss => -1.0,
            _ => 0.0,
        })
    }
}

impl SimulationModel for Model {
    fn simulation_fork(
        &self,
        observation: &View,
        random: &mut SimulationRng,
    ) -> Result<Position, String> {
        if observation.seat.index() >= 2 || observation.finished {
            return Err("invalid observation".into());
        }
        let mut chance = random.fork_chance();
        let mut secret = [0, 0];
        secret[observation.seat.index()] = observation.own_secret;
        secret[1 - observation.seat.index()] = chance.index(2).unwrap();
        Ok(Position {
            secret,
            chance: chance.index(2).unwrap(),
            choices: observation.last_choices,
            public_choices: observation.public_choices,
            remaining: observation.remaining,
            fail_after_resolution: false,
        })
    }
}

struct Values;
impl StateEvaluator<Model> for Values {
    fn evaluate(
        &mut self,
        state: &Position,
        _: &mut SimulationContext<'_, Model>,
    ) -> SearchResult<StateValues> {
        if state.choices.is_none() {
            return Err(tag_rust::search::mcts::Error::Evaluation(
                "partial decision reached evaluator".into(),
            ));
        }
        Ok(StateValues::Full(vec![0.0, 0.0]))
    }
}

struct FinalOnly;
impl Evaluator<Position> for FinalOnly {
    type Error = String;
    fn evaluate(&mut self, state: &Position, player: PlayerId) -> Result<f64, String> {
        if state.choices.is_none() {
            return Err("partial decision reached evaluator".into());
        }
        Model.player_score(state, player)
    }
}

#[test]
fn joint_validation_and_pending_submissions_preserve_predecision_views() {
    let model = Model;
    let state = model.init_state(&(), 2).unwrap();
    let original = state.clone();
    let mut invalid = state.clone();
    assert!(matches!(
        joint::apply(&model, &mut invalid, &[(PlayerId::new(1), 0)]),
        Err(JointError::Invalid(_))
    ));
    assert_eq!(invalid, original);

    let mut legacy = Session::<_, RandomAgent>::from_state(
        model.clone(),
        original.clone(),
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    assert!(matches!(
        legacy.submit_human(0, PlayerId::new(0), 0),
        Err(SessionError::SimultaneousRequiresJointSession)
    ));
    assert_eq!(legacy.state(), &original);
    assert!(matches!(
        joint::apply(
            &model,
            &mut invalid,
            &[(PlayerId::new(1), 0), (PlayerId::new(0), 1)]
        ),
        Err(JointError::Invalid(_))
    ));
    assert_eq!(invalid, original);

    let mut session = JointSession::<_, RandomAgent>::from_state(
        model.clone(),
        state,
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    let initial_snapshot = session.snapshot().unwrap();
    let original_view = session.decision_view(PlayerId::new(0)).unwrap();
    let before = original_view.observation;
    assert!(
        session
            .submit_human(0, PlayerId::new(1), 0)
            .unwrap()
            .is_none()
    );
    assert!(session.snapshot().is_err());
    assert_eq!(session.state(), &original);
    let after_other_choice = session.decision_view(PlayerId::new(0)).unwrap();
    assert_eq!(after_other_choice.observation, before);
    assert_eq!(after_other_choice.actions, original_view.actions);
    assert_eq!(session.waiting_seats(), vec![PlayerId::new(0)]);
    assert!(session.submit_human(0, PlayerId::new(0), 9).is_err());
    assert_eq!(session.waiting_seats(), vec![PlayerId::new(0)]);
    let accepted = session
        .submit_human(0, PlayerId::new(0), 1)
        .unwrap()
        .unwrap();
    assert_eq!(
        accepted.choices,
        vec![(PlayerId::new(0), 1), (PlayerId::new(1), 0)]
    );
    assert_eq!(accepted.position_version, 1);
    assert_eq!(session.state().choices, Some([1, 0]));
    assert_eq!(
        session
            .decision_view(PlayerId::new(0))
            .unwrap()
            .observation
            .last_choices,
        Some([1, 0])
    );
    assert_eq!(
        session
            .decision_view(PlayerId::new(1))
            .unwrap()
            .observation
            .last_choices,
        Some([1, 0])
    );
    let finished_snapshot = session.snapshot().unwrap();
    let restored =
        JointSession::<_, RandomAgent>::from_snapshot(model.clone(), &finished_snapshot).unwrap();
    assert_eq!(restored.state(), session.state());
    assert_eq!(initial_snapshot.position_version, 0);

    let mut retry = JointSession::<_, RandomAgent>::from_state(
        model,
        original.clone(),
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    retry.submit_human(0, PlayerId::new(0), 0).unwrap();
    retry.cancel().unwrap();
    assert_eq!(
        retry.waiting_seats(),
        vec![PlayerId::new(0), PlayerId::new(1)]
    );
    assert!(retry.submit_human(0, PlayerId::new(1), 0).is_err());
    assert_eq!(retry.state(), &original);
    retry.reset(original.clone()).unwrap();
    assert_eq!(retry.position_version(), 2);
    assert_eq!(retry.waiting_seats().len(), 2);

    let mut reverse = JointSession::<_, RandomAgent>::from_state(
        Model,
        original.clone(),
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    let player_one = reverse.decision_view(PlayerId::new(1)).unwrap();
    reverse.submit_human(0, PlayerId::new(0), 1).unwrap();
    let player_one_after = reverse.decision_view(PlayerId::new(1)).unwrap();
    assert_eq!(player_one_after.observation, player_one.observation);
    assert_eq!(player_one_after.actions, player_one.actions);
    let accepted_reverse = reverse
        .submit_human(0, PlayerId::new(1), 0)
        .unwrap()
        .unwrap();
    assert_eq!(accepted_reverse.choices, accepted.choices);
    assert_eq!(reverse.state(), session.state());
}

#[test]
fn runner_records_one_complete_joint_transition() {
    let model = Model;
    let mut state = model.init_state(&(), 2).unwrap();
    let mut agents = [RandomAgent::new(3), RandomAgent::new(4)];
    assert!(matches!(
        run_game(&model, &mut state, &mut agents),
        Err(RunnerError::SimultaneousRequiresJointRunner)
    ));
    let mut records = Vec::new();
    run_joint_game_recorded(&model, &mut state, &mut agents, |record| {
        records.push(record)
    })
    .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].choices.len(), 2);
    assert!(records[0].next_seats.is_empty());
    assert_eq!(records[0].controllers_after.len(), 2);
    assert_eq!(records[0].state_after, state);
    let mut second = model.init_state(&(), 2).unwrap();
    let mut agents2 = [RandomAgent::new(3), RandomAgent::new(4)];
    let mut calls = 0;
    run_joint_game(&model, &mut second, &mut agents2, |_, position, choices| {
        calls += 1;
        assert_eq!(position.choices, Some([choices[0].1, choices[1].1]));
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(second, state);
}

#[test]
fn interactive_ai_choice_waits_for_the_other_actor() {
    let model = Model;
    let state = model.init_state(&(), 2).unwrap();
    let mut session = JointSession::<_, RandomAgent>::from_state(
        model,
        state.clone(),
        vec![Controller::Ai(RandomAgent::new(8)), Controller::Human],
    )
    .unwrap();
    let before = session.decision_view(PlayerId::new(1)).unwrap();
    assert!(session.advance_ai(0).unwrap().is_none());
    assert_eq!(session.state(), &state);
    let after = session.decision_view(PlayerId::new(1)).unwrap();
    assert_eq!(before.observation, after.observation);
    assert_eq!(before.actions, after.actions);
    assert!(
        session
            .submit_human(0, PlayerId::new(1), 1)
            .unwrap()
            .is_some()
    );
}

#[test]
fn internal_rule_failure_stops_interactive_execution() {
    let model = Model;
    let mut state = model.init_state(&(), 2).unwrap();
    state.fail_after_resolution = true;
    let mut session = JointSession::<_, RandomAgent>::from_state(
        model,
        state,
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    assert!(
        session
            .submit_human(0, PlayerId::new(0), 1)
            .unwrap()
            .is_none()
    );
    assert!(session.submit_human(0, PlayerId::new(1), 0).is_err());
    assert!(session.is_stopped());
    assert!(session.waiting_seats().is_empty());
    assert_eq!(session.state().choices, Some([1, 0]));
    assert!(session.submit_human(0, PlayerId::new(0), 0).is_err());
    assert!(session.snapshot().is_err());
    session.reset(Model.init_state(&(), 2).unwrap()).unwrap();
    assert!(!session.is_stopped());
    assert_eq!(session.position_version(), 1);
}

#[test]
fn mcts_edges_and_budget_end_at_joint_boundaries() {
    let model = Model;
    let state = model.init_state(&(), 2).unwrap();
    let observation = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&observation);
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Values)))
        .unwrap();
    let settings = Settings {
        seed: Some(77),
        rollout_length: Some(0),
        budgets: Budgets {
            iterations: Some(25),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Settings::default()
    };
    let mut search = MctsAgent::new(settings.clone(), &registry).unwrap();
    let report = search
        .search(&model, PlayerId::new(0), &observation, &actions)
        .unwrap();
    assert_eq!(report.counters.completed_iterations, 25);
    assert_eq!(report.counters.forward_model_calls, 25);
    assert_eq!(report.counters.state_evaluations, 0);
    assert_eq!(
        report.root_actions.iter().map(|a| a.visits).sum::<u64>(),
        25
    );
    assert_eq!(report.edges, 4); // Two choices per seat produce four joint edges.
    assert!(
        report
            .root_actions
            .iter()
            .all(|action| action.mean_values.len() == 2)
    );
    let hidden_alternative = Position {
        secret: [1, 1],
        ..state
    };
    let same_view = model
        .observe(&hidden_alternative, PlayerId::new(0))
        .unwrap();
    assert_eq!(same_view, observation);
    let mut second_search = MctsAgent::new(settings, &registry).unwrap();
    let second = second_search
        .search(&model, PlayerId::new(0), &same_view, &actions)
        .unwrap();
    assert_eq!(second.selected_index, report.selected_index);
    assert_eq!(second.root_actions, report.root_actions);
    assert_eq!(second.counters, report.counters);
}

#[test]
fn mcts_evaluation_and_rollout_follow_complete_two_round_decisions() {
    let model = Model;
    let mut state = model.init_state(&(), 2).unwrap();
    state.remaining = 2;
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&view);
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Values)))
        .unwrap();

    let direct = Settings {
        seed: Some(33),
        rollout_length: Some(0),
        budgets: Budgets {
            iterations: Some(1),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Settings::default()
    };
    let report = MctsAgent::new(direct.clone(), &registry)
        .unwrap()
        .search(&model, PlayerId::new(0), &view, &actions)
        .unwrap();
    assert_eq!(report.counters.forward_model_calls, 1);
    assert_eq!(report.counters.state_evaluations, 1);

    let mut rolled = direct;
    rolled.rollout_length = Some(1);
    let report = MctsAgent::new(rolled, &registry)
        .unwrap()
        .search(&model, PlayerId::new(0), &view, &actions)
        .unwrap();
    assert_eq!(report.counters.forward_model_calls, 2);
    assert_eq!(report.counters.state_evaluations, 0);
    assert_eq!(report.counters.completed_iterations, 1);
}

#[test]
fn joint_tree_rejects_choices_that_remain_hidden_after_resolution() {
    let model = Model;
    let mut state = model.init_state(&(), 2).unwrap();
    state.public_choices = false;
    state.remaining = 2;
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&view);
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Values)))
        .unwrap();
    let settings = Settings {
        seed: Some(11),
        ..Settings::default()
    };
    let mut search = MctsAgent::new(settings, &registry).unwrap();
    let error = search
        .search(&model, PlayerId::new(0), &view, &actions)
        .unwrap_err();
    assert!(matches!(error, tag_rust::search::mcts::Error::Invalid(_)));
}

#[test]
fn sampled_osla_evaluates_complete_joint_successors() {
    let model = Model;
    let state = model.init_state(&(), 2).unwrap();
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&view);
    let mut agent = SampledOslaAgent::new(FinalOnly, 19, std::num::NonZeroU32::new(3).unwrap());
    let selected = agent
        .choose(&model, PlayerId::new(0), &view, &actions)
        .unwrap();
    assert!(selected < actions.len());
    assert_eq!(agent.last_transitions(), 6);
}
