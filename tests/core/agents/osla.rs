//! OSLA contract checks use a tiny model with non-Clone actions and visible work counts.
use std::cell::Cell;
use std::rc::Rc;
use tag_rust::agents::{Agent, OslaAgent, OslaError};
use tag_rust::core::evaluation::{EVALUATION_TIE_TOLERANCE, Evaluator};
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};

#[derive(Debug, PartialEq, Eq)] // No Clone/Copy: model errors are moved; equality compares test failures.
enum Failure {
    Reconstruction,
    Transition,
    Evaluation,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Failure {}

// OSLA borrows each action.
struct Move {
    index: usize,
}
struct State {
    candidate: Option<usize>,
    copies: Rc<Cell<usize>>,
}
impl Clone for State {
    fn clone(&self) -> Self {
        // Count branch snapshots; each copy owns its own candidate field.
        self.copies.set(self.copies.get() + 1);
        Self {
            candidate: self.candidate,
            copies: Rc::clone(&self.copies),
        }
    }
}
#[derive(Default)] // Counters and flags start at zero/false.
struct Model {
    reconstructions: Cell<usize>,
    transitions: Cell<usize>,
    reconstruction_failure: bool,
    failed_move: Option<usize>,
}
impl ForwardModel for Model {
    type State = State;
    type Observation = State;
    type Action = Move;
    type Parameters = ();
    type Error = Failure;
    fn init_state(&self, _: &(), _: usize) -> Result<State, Failure> {
        Ok(State {
            candidate: None,
            copies: Rc::new(Cell::new(0)),
        })
    }
    fn player_count(&self, _: &State) -> usize {
        3
    }
    fn next_player(&self, _: &State) -> Decision {
        Decision::Player(PlayerId::new(2))
    }
    fn observe(&self, state: &State, _: PlayerId) -> Result<State, Failure> {
        Ok(state.clone())
    }
    fn offered_actions(&self, _: &State) -> Vec<Move> {
        Vec::new()
    }
    fn available_actions(&self, _: &State, _: PlayerId) -> Result<Vec<Move>, Failure> {
        Ok(Vec::new())
    }
    fn next(&self, state: &mut State, actor: PlayerId, action: &Move) -> Result<(), Failure> {
        assert_eq!(actor, PlayerId::new(1));
        assert!(
            state.candidate.is_none(),
            "each candidate must start from the base position"
        );
        self.transitions.set(self.transitions.get() + 1);
        state.candidate = Some(action.index);
        if self.failed_move == Some(action.index) {
            return Err(Failure::Transition);
        }
        Ok(())
    }
    fn player_result(&self, _: &State, _: PlayerId) -> Result<PlayerResult, Failure> {
        Ok(PlayerResult::Ongoing)
    }
    fn player_score(&self, _: &State, _: PlayerId) -> Result<f64, Failure> {
        Ok(0.0)
    }
}
impl PublicInformationModel for Model {
    fn simulation_state(&self, observation: &State) -> Result<State, Failure> {
        self.reconstructions.set(self.reconstructions.get() + 1);
        if self.reconstruction_failure {
            return Err(Failure::Reconstruction);
        }
        // Share only test instrumentation; simulation fields remain independent.
        Ok(State {
            candidate: observation.candidate,
            copies: Rc::clone(&observation.copies),
        })
    }
}
struct Values {
    values: Vec<f64>,
    calls: Rc<Cell<usize>>,
    failed_candidate: Option<usize>,
}
impl Evaluator<State> for Values {
    type Error = Failure;
    fn evaluate(&mut self, state: &State, player: PlayerId) -> Result<f64, Failure> {
        assert_eq!(
            player,
            PlayerId::new(1),
            "perspective stays with the original actor"
        );
        self.calls.set(self.calls.get() + 1);
        let candidate = state.candidate.unwrap();
        if self.failed_candidate == Some(candidate) {
            return Err(Failure::Evaluation);
        }
        Ok(self.values[candidate])
    }
}
fn setup(
    values: Vec<f64>,
    seed: u64,
) -> (Model, State, Vec<Move>, OslaAgent<Values>, Rc<Cell<usize>>) {
    let model = Model::default();
    let observation = model.init_state(&(), 3).unwrap();
    let actions = (0..values.len())
        .map(|index| {
            // Give each candidate its own action identity.
            Move { index }
        })
        .collect();
    let calls = Rc::new(Cell::new(0));
    let agent = OslaAgent::new(
        Values {
            values,
            calls: Rc::clone(&calls),
            failed_candidate: None,
        },
        seed,
    );
    (model, observation, actions, agent, calls)
}
fn choose(
    model: &Model,
    view: &State,
    actions: &[Move],
    agent: &mut OslaAgent<Values>,
) -> Result<usize, OslaError<Failure, Failure>> {
    agent.choose(model, PlayerId::new(1), view, actions)
}
fn next_random(agent: &mut OslaAgent<Values>) -> Option<usize> {
    <OslaAgent<Values> as Agent<Model>>::rng_mut(agent).index(1000)
}

#[test]
fn independent_branches_are_evaluated_once_in_original_perspective() {
    let (model, view, actions, mut agent, calls) = setup(vec![-3.0, -1.0, -2.0], 10);
    assert_eq!(choose(&model, &view, &actions, &mut agent).unwrap(), 1);
    assert_eq!(model.reconstructions.get(), 1);
    assert_eq!(model.transitions.get(), 3);
    assert_eq!(calls.get(), 3);
    assert_eq!(view.copies.get(), 3);
    assert!(view.candidate.is_none());
    // A unique best move retains the RNG position.
    assert_eq!(next_random(&mut agent), AgentRng::new(10).index(1000));
}

#[test]
fn tolerance_is_inclusive_and_compared_to_true_maximum_without_chaining() {
    let values = vec![
        0.0,
        -EVALUATION_TIE_TOLERANCE,
        -EVALUATION_TIE_TOLERANCE - 1e-10,
        -0.0018,
    ];
    let (model, view, actions, mut agent, _) = setup(values, 42);
    let mut reference = AgentRng::new(42);
    let mut seen = [false; 2];
    for _ in 0..128 {
        let expected = reference.index(2).unwrap();
        let chosen = choose(&model, &view, &actions, &mut agent).unwrap();
        assert_eq!(chosen, expected);
        seen[chosen] = true;
    }
    assert!(seen.into_iter().all(|visited| {
        // Both boundary-eligible moves must occur in this fixed seeded sequence.
        visited
    }));
    let (model, view, actions, mut agent, _) = setup(vec![0.4982, 0.4991, 0.5000], 100);
    for _ in 0..64 {
        assert_ne!(choose(&model, &view, &actions, &mut agent).unwrap(), 0);
    }
}

#[test]
fn equal_seeds_replay_and_other_agents_do_not_advance_each_others_streams() {
    let (model, view, actions, mut first, _) = setup(vec![5.0; 3], 20);
    let (_, _, _, mut second, _) = setup(vec![5.0; 3], 20);
    let (_, _, _, mut unrelated, _) = setup(vec![5.0; 3], 21);
    let mut difference = false;
    for _ in 0..64 {
        let a = choose(&model, &view, &actions, &mut first).unwrap();
        assert_eq!(a, choose(&model, &view, &actions, &mut second).unwrap());
        difference |= a != choose(&model, &view, &actions, &mut unrelated).unwrap();
        for _ in 0..3 {
            let _ = next_random(&mut unrelated);
        }
    }
    assert!(difference);
}

#[test]
fn empty_input_is_an_error_and_single_choice_is_evaluated() {
    for values in [vec![], vec![0.5]] {
        let count = values.len();
        let (model, view, actions, mut agent, calls) = setup(values, 15);
        let result = choose(&model, &view, &actions, &mut agent);
        if count == 0 {
            let error = result.unwrap_err();
            assert!(matches!(error, OslaError::EmptyActions));
            assert!(std::error::Error::source(&error).is_none());
            assert_eq!(
                error.to_string(),
                "action selection requires a nonempty action list"
            );
        } else {
            assert_eq!(result.unwrap(), 0);
        }
        assert_eq!(model.reconstructions.get(), count);
        assert_eq!(model.transitions.get(), count);
        assert_eq!(calls.get(), count);
        assert_eq!(next_random(&mut agent), AgentRng::new(15).index(1000));
    }
}

#[test]
fn reconstruction_and_transition_errors_report_stage_and_preserve_rng() {
    let (mut model, view, actions, mut agent, calls) = setup(vec![0.0; 3], 3);
    model.reconstruction_failure = true;
    let error = choose(&model, &view, &actions, &mut agent).unwrap_err();
    assert!(matches!(
        error,
        OslaError::Reconstruction(Failure::Reconstruction)
    ));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(model.transitions.get(), 0);
    assert_eq!(calls.get(), 0);
    assert_eq!(next_random(&mut agent), AgentRng::new(3).index(1000));

    let (mut model, view, actions, mut agent, calls) = setup(vec![0.0; 3], 4);
    model.failed_move = Some(1);
    assert!(matches!(
        choose(&model, &view, &actions, &mut agent),
        Err(OslaError::Transition {
            candidate: 1,
            error: Failure::Transition
        })
    ));
    assert_eq!(model.transitions.get(), 2);
    assert_eq!(calls.get(), 1);
    assert!(view.candidate.is_none());
    assert_eq!(next_random(&mut agent), AgentRng::new(4).index(1000));
}

#[test]
fn evaluator_failure_keeps_prior_evaluator_progress_without_selection_draw() {
    let (model, view, actions, _, _) = setup(vec![0.0; 3], 5);
    let calls = Rc::new(Cell::new(0));
    let mut agent = OslaAgent::new(
        Values {
            values: vec![0.0; 3],
            calls: Rc::clone(&calls),
            failed_candidate: Some(1),
        },
        5,
    );
    let error = choose(&model, &view, &actions, &mut agent).unwrap_err();
    assert!(matches!(
        error,
        OslaError::Evaluation {
            candidate: 1,
            error: Failure::Evaluation
        }
    ));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(calls.get(), 2);
    assert_eq!(model.transitions.get(), 2);
    assert!(view.candidate.is_none());
    assert_eq!(next_random(&mut agent), AgentRng::new(5).index(1000));
}

#[test]
fn nan_and_infinities_fail_instead_of_entering_tie_selection() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (model, view, actions, mut agent, calls) = setup(vec![0.0, value, 1.0], 6);
        let error = choose(&model, &view, &actions, &mut agent).unwrap_err();
        assert!(matches!(
            error,
            OslaError::NonfiniteValue { candidate: 1, .. }
        ));
        assert!(std::error::Error::source(&error).is_none());
        assert_eq!(calls.get(), 2);
        assert_eq!(model.transitions.get(), 2);
        assert_eq!(next_random(&mut agent), AgentRng::new(6).index(1000));
    }
}

#[test]
fn finite_extreme_scales_select_correct_maximum() {
    let (model, view, actions, mut agent, _) = setup(vec![-f64::MAX, f64::MAX, 0.0], 1);
    assert_eq!(choose(&model, &view, &actions, &mut agent).unwrap(), 1);
}

#[test]
fn actual_game_observation_and_live_state_are_unchanged_by_planning() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let view = model.observe(&state, PlayerId::new(0)).unwrap();
    let original_state = state.clone();
    let original_view = view.clone();
    let actions = model.offered_actions(&view);
    let mut agent = OslaAgent::new(TicTacToeEvaluator, 7);
    let index = agent
        .choose(&model, PlayerId::new(0), &view, &actions)
        .unwrap();
    assert!(index < actions.len());
    assert_eq!(state, original_state);
    assert_eq!(view, original_view);
    // Simulated rule validation rejects a wrong actor.
    assert!(matches!(
        agent.choose(&model, PlayerId::new(1), &view, &actions),
        Err(OslaError::Transition { candidate: 0, .. })
    ));
}
