//! Seeded OSLA matches use the same runner and test controller wrapper in both games.
use std::fmt::Debug;
use tag_rust::agents::{Agent, FirstAvailableAgent, OslaAgent, OslaError};
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::connect4::{
    Connect4Evaluator, Connect4ForwardModel, Connect4Parameters, DropPiece,
};
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters,
};
use tag_rust::runner::{RunnerError, run_game};

enum Policy<E> {
    Baseline(FirstAvailableAgent),
    Osla(OslaAgent<E>),
}
struct RecordingAgent<E> {
    policy: Policy<E>,
    choices: Vec<usize>,
}
impl<F, E> Agent<F> for RecordingAgent<E>
where
    F: PublicInformationModel,
    F::State: Clone,
    E: Evaluator<F::State>,
{
    type Error = OslaError<F::Error, E::Error>;
    fn rng(&self) -> &AgentRng {
        match &self.policy {
            Policy::Baseline(agent) => <FirstAvailableAgent as Agent<F>>::rng(agent),
            Policy::Osla(agent) => <OslaAgent<E> as Agent<F>>::rng(agent),
        }
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        match &mut self.policy {
            Policy::Baseline(agent) => <FirstAvailableAgent as Agent<F>>::rng_mut(agent),
            Policy::Osla(agent) => <OslaAgent<E> as Agent<F>>::rng_mut(agent),
        }
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        view: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        // Borrow the selected policy and record its successful offered-action index.
        let choice = match &mut self.policy {
            Policy::Baseline(agent) => agent.choose(model, actor, view, actions).map_err(|_| {
                // Preserve the empty-input failure in the test wrapper.
                OslaError::EmptyActions
            })?,
            Policy::Osla(agent) => agent.choose(model, actor, view, actions)?,
        };
        self.choices.push(choice);
        Ok(choice)
    }
}
fn controller<E>(evaluator: E, seed: u64, baseline: bool) -> RecordingAgent<E> {
    // Consume the evaluator; return an owned, explicitly seeded test controller.
    RecordingAgent {
        policy: if baseline {
            Policy::Baseline(FirstAvailableAgent::new(seed))
        } else {
            Policy::Osla(OslaAgent::new(evaluator, seed))
        },
        choices: Vec::new(),
    }
}
fn complete<F, E>(
    model: &F,
    parameters: &F::Parameters,
    evaluator: impl Fn() -> E,
    seeds: [u64; 2],
    baseline: Option<usize>,
) -> (F::State, [Vec<usize>; 2])
where
    F: PublicInformationModel,
    F::State: Clone + Debug + PartialEq,
    F::Error: Debug,
    E: Evaluator<F::State>,
    E::Error: Debug,
{
    // Borrow rules/configuration; construct each seat's evaluator and RNG independently.
    let mut state = model.init_state(parameters, 2).unwrap();
    let mut agents = [
        controller(evaluator(), seeds[0], baseline == Some(0)),
        controller(evaluator(), seeds[1], baseline == Some(1)),
    ];
    run_game(model, &mut state, &mut agents).unwrap();
    assert_eq!(model.next_player(&state), Decision::Finished);
    for seat in 0..2 {
        assert_ne!(
            model.player_result(&state, PlayerId::new(seat)).unwrap(),
            PlayerResult::Ongoing
        );
        assert!(
            model
                .offered_actions(&model.observe(&state, PlayerId::new(seat)).unwrap())
                .is_empty()
        );
    }
    let saved = state.clone();
    let lengths = [agents[0].choices.len(), agents[1].choices.len()];
    run_game(model, &mut state, &mut agents).unwrap();
    assert_eq!(state, saved);
    assert_eq!([agents[0].choices.len(), agents[1].choices.len()], lengths);
    let [first, second] = agents;
    (state, [first.choices, second.choices])
}

#[test]
fn seeded_full_matches_replay_in_both_default_and_configured_games() {
    for seeds in [[0, 1], [100, 200], [u64::MAX, 42]] {
        for size in [1, 3, 4] {
            let params = TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            };
            let first = complete(
                &TicTacToeForwardModel,
                &params,
                || TicTacToeEvaluator,
                seeds,
                None,
            );
            let replay = complete(
                &TicTacToeForwardModel,
                &params,
                || TicTacToeEvaluator,
                seeds,
                None,
            );
            assert_eq!(first, replay);
            assert!(first.0.move_count() <= size * size);
            assert_eq!(
                first.0.move_count(),
                first.1.iter().map(Vec::len).sum::<usize>()
            );
        }
        for (width, height, target) in [(7, 6, 4), (1, 1, 1), (1, 6, 4), (4, 3, 3), (8, 4, 5)] {
            let params = Connect4Parameters {
                width,
                height,
                winning_length: target,
                ..Default::default()
            };
            let first = complete(
                &Connect4ForwardModel,
                &params,
                || Connect4Evaluator,
                seeds,
                None,
            );
            let replay = complete(
                &Connect4ForwardModel,
                &params,
                || Connect4Evaluator,
                seeds,
                None,
            );
            assert_eq!(first, replay);
            assert!(first.0.move_count() <= width * height);
            assert_eq!(
                first.0.move_count(),
                first.1.iter().map(Vec::len).sum::<usize>()
            );
        }
    }
}

#[test]
fn baseline_and_osla_can_share_the_runner_in_either_seat() {
    for baseline in [0, 1] {
        complete(
            &TicTacToeForwardModel,
            &TicTacToeParameters::default(),
            || TicTacToeEvaluator,
            [10, 20],
            Some(baseline),
        );
        complete(
            &Connect4ForwardModel,
            &Connect4Parameters::default(),
            || Connect4Evaluator,
            [10, 20],
            Some(baseline),
        );
    }
}

#[test]
fn runner_applies_the_immediate_win_even_when_it_is_later_in_the_offer() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for (x, y) in [(2, 0), (0, 0), (2, 1), (0, 1)] {
        let actor = state.current_player();
        model.next(&mut state, actor, &PlaceMark { x, y }).unwrap();
    }
    let moves = model.available_actions(&state, PlayerId::new(0)).unwrap();
    assert_ne!(moves[0], PlaceMark { x: 2, y: 2 });
    let mut agents = [
        OslaAgent::new(TicTacToeEvaluator, 100),
        OslaAgent::new(TicTacToeEvaluator, 200),
    ];
    run_game(&model, &mut state, &mut agents).unwrap();
    assert_eq!(state.move_count(), 5);
    assert_eq!(
        state.player_result(PlayerId::new(0)).unwrap(),
        PlayerResult::Win
    );

    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    for column in [3, 0, 3, 0, 3, 1] {
        let actor = state.current_player();
        model
            .next(&mut state, actor, &DropPiece { column })
            .unwrap();
    }
    assert_eq!(
        model.available_actions(&state, PlayerId::new(0)).unwrap()[0].column,
        0
    );
    let mut agents = [
        OslaAgent::new(Connect4Evaluator, 100),
        OslaAgent::new(Connect4Evaluator, 200),
    ];
    run_game(&model, &mut state, &mut agents).unwrap();
    assert_eq!(state.move_count(), 7);
    assert_eq!(
        state.player_result(PlayerId::new(0)).unwrap(),
        PlayerResult::Win
    );
    assert_eq!(state.winning_cells(), &[(3, 5), (3, 4), (3, 3), (3, 2)]);
}

#[test]
fn tied_winning_moves_use_the_acting_agents_seed_in_actual_matches() {
    let model = TicTacToeForwardModel;
    let mut base = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for (x, y) in [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)] {
        let actor = base.current_player();
        model.next(&mut base, actor, &PlaceMark { x, y }).unwrap();
    }
    let mut seen = [false; 2];
    for seed in 0..64 {
        let expected = AgentRng::new(seed).index(2).unwrap();
        let mut state = base.clone(); // Independently owned match start.
        let mut agents = [
            OslaAgent::new(TicTacToeEvaluator, seed),
            OslaAgent::new(TicTacToeEvaluator, 200),
        ];
        run_game(&model, &mut state, &mut agents).unwrap();
        let x = [0, 2][expected];
        assert_eq!(
            state.cell(x, 2).unwrap(),
            tag_rust::games::tictactoe::Cell::X
        );
        assert_eq!(state.move_count(), 7);
        assert_eq!(
            state.player_result(PlayerId::new(0)).unwrap(),
            PlayerResult::Win
        );
        seen[expected] = true;
    }
    assert_eq!(seen, [true, true]);

    let model = Connect4ForwardModel;
    let parameters = Connect4Parameters {
        width: 5,
        height: 1,
        winning_length: 1,
        ..Default::default()
    };
    let mut seen = [false; 5];
    for seed in 0..64 {
        let column = AgentRng::new(seed).index(5).unwrap();
        let mut state = model.init_state(&parameters, 2).unwrap();
        let mut agents = [
            OslaAgent::new(Connect4Evaluator, seed),
            OslaAgent::new(Connect4Evaluator, 200),
        ];
        run_game(&model, &mut state, &mut agents).unwrap();
        assert_eq!(state.winning_cells(), &[(column, 0)]);
        assert_eq!(state.move_count(), 1);
        seen[column] = true;
    }
    assert_eq!(seen, [true; 5]);
}

struct FailOnce<E> {
    evaluator: E,
    fail: bool,
}
#[derive(Debug)] // Owned failure details move through OSLA and the runner.
enum EvaluationFailure<E> {
    Requested,
    Game(E),
}
impl<S, E: Evaluator<S>> Evaluator<S> for FailOnce<E> {
    type Error = EvaluationFailure<E::Error>;
    fn evaluate(&mut self, state: &S, player: PlayerId) -> Result<f64, Self::Error> {
        if self.fail {
            self.fail = false;
            return Err(EvaluationFailure::Requested);
        }
        self.evaluator
            .evaluate(state, player)
            .map_err(EvaluationFailure::Game)
    }
}
fn resume<F, E>(model: &F, parameters: &F::Parameters, evaluator: impl Fn() -> E)
where
    F: PublicInformationModel,
    F::State: Clone + Debug + PartialEq,
    F::Error: Debug,
    E: Evaluator<F::State>,
    E::Error: Debug,
{
    let mut state = model.init_state(parameters, 2).unwrap();
    let initial_view = model.observe(&state, PlayerId::new(0)).unwrap();
    let mut agents = [
        OslaAgent::new(
            FailOnce {
                evaluator: evaluator(),
                fail: false,
            },
            100,
        ),
        OslaAgent::new(
            FailOnce {
                evaluator: evaluator(),
                fail: true,
            },
            200,
        ),
    ];
    let mut reference = model.init_state(parameters, 2).unwrap();
    let mut expected_agents = [
        OslaAgent::new(evaluator(), 100),
        OslaAgent::new(evaluator(), 200),
    ];
    let offers = model.offered_actions(&initial_view);
    let index = expected_agents[0]
        .choose(model, PlayerId::new(0), &initial_view, &offers)
        .unwrap();
    model
        .next(&mut reference, PlayerId::new(0), &offers[index])
        .unwrap();
    assert!(
        matches!(run_game(model, &mut state, &mut agents), Err(RunnerError::Agent {
        player, error: OslaError::Evaluation { candidate: 0, error: EvaluationFailure::Requested }
    }) if player == PlayerId::new(1))
    );
    assert_eq!(state, reference);
    assert_eq!(
        model.player_result(&state, PlayerId::new(1)).unwrap(),
        PlayerResult::Ongoing
    );
    run_game(model, &mut state, &mut agents).unwrap();
    run_game(model, &mut reference, &mut expected_agents).unwrap();
    assert_eq!(state, reference);
    for seat in 0..2 {
        let actual = <OslaAgent<FailOnce<E>> as Agent<F>>::rng_mut(&mut agents[seat]).index(1000);
        let expected = <OslaAgent<E> as Agent<F>>::rng_mut(&mut expected_agents[seat]).index(1000);
        assert_eq!(actual, expected);
    }
}

#[test]
fn osla_evaluator_failure_preserves_the_current_decision_then_resumes_both_games() {
    resume(
        &TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        || TicTacToeEvaluator,
    );
    resume(
        &Connect4ForwardModel,
        &Connect4Parameters::default(),
        || Connect4Evaluator,
    );
}
