use std::fmt::Debug;
use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::session::{BranchError, BranchId, BranchSession, BuiltinAgent, Controller, Session};

// Clone shares the transition counter while copying the stateless rules wrapper.
#[derive(Clone)]
struct CountedModel(std::rc::Rc<std::cell::Cell<usize>>);
impl ForwardModel for CountedModel {
    type State = tag_rust::games::tictactoe::TicTacToeState;
    type Action = tag_rust::games::tictactoe::PlaceMark;
    type Observation = tag_rust::games::tictactoe::TicTacToeObservation;
    type Parameters = TicTacToeParameters;
    type Error = tag_rust::games::tictactoe::TicTacToeError;
    fn init_state(&self, p: &Self::Parameters, n: usize) -> Result<Self::State, Self::Error> {
        TicTacToeForwardModel.init_state(p, n)
    }
    fn player_count(&self, s: &Self::State) -> usize {
        TicTacToeForwardModel.player_count(s)
    }
    fn next_player(&self, s: &Self::State) -> Decision {
        TicTacToeForwardModel.next_player(s)
    }
    fn observe(&self, s: &Self::State, p: PlayerId) -> Result<Self::Observation, Self::Error> {
        TicTacToeForwardModel.observe(s, p)
    }
    fn offered_actions(&self, o: &Self::Observation) -> Vec<Self::Action> {
        TicTacToeForwardModel.offered_actions(o)
    }
    fn available_actions(
        &self,
        s: &Self::State,
        p: PlayerId,
    ) -> Result<Vec<Self::Action>, Self::Error> {
        TicTacToeForwardModel.available_actions(s, p)
    }
    fn next(&self, s: &mut Self::State, p: PlayerId, a: &Self::Action) -> Result<(), Self::Error> {
        self.0.set(self.0.get() + 1);
        TicTacToeForwardModel.next(s, p, a)
    }
    fn player_result(&self, s: &Self::State, p: PlayerId) -> Result<PlayerResult, Self::Error> {
        TicTacToeForwardModel.player_result(s, p)
    }
    fn player_score(&self, s: &Self::State, p: PlayerId) -> Result<f64, Self::Error> {
        TicTacToeForwardModel.player_score(s, p)
    }
}

#[test]
fn nested_branch_selection_replays_the_inherited_prefix_once_and_preserves_cache_limits() {
    use tag_rust::agents::FirstAvailableAgent;
    use tag_rust::recording::{ControllerSnapshot, GameHistory, HistoryCachePolicy};
    let transitions = std::rc::Rc::new(std::cell::Cell::new(0));
    let model = CountedModel(transitions.clone());
    let root = model
        .init_state(
            &TicTacToeParameters {
                grid_size: 8,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    let history = GameHistory::with_cache_policy(
        root,
        vec![ControllerSnapshot::Human, ControllerSnapshot::Human],
        HistoryCachePolicy {
            max_states: 0,
            ..Default::default()
        },
    );
    let mut game = BranchSession::<_, FirstAvailableAgent>::from_history(model, history).unwrap();
    let advance = |game: &mut BranchSession<CountedModel, FirstAvailableAgent>, moves| {
        for _ in 0..moves {
            let Decision::Player(actor) = game.active().session().next_player() else {
                panic!("workload must remain ongoing")
            };
            let input = game.active().session().decision_view(actor).unwrap();
            game.submit_human(input.position_version, actor, input.actions[0])
                .unwrap();
        }
    };
    advance(&mut game, 40);
    let child = game.fork(BranchId(0), 20, vec![None, None]).unwrap();
    advance(&mut game, 10);
    let nested = game.fork(child, 25, vec![None, None]).unwrap();
    advance(&mut game, 3);
    let expected = game.active().session().state().clone();
    let expected_controllers = game.controllers_at(nested, 28).unwrap();
    game.select(BranchId(0)).unwrap();
    transitions.set(0);
    game.select(nested).unwrap();
    assert_eq!(transitions.get(), 28);
    assert_eq!(game.active().session().state(), &expected);
    assert_eq!(
        game.controllers_at(nested, 28).unwrap(),
        expected_controllers
    );
    assert_eq!(game.active().history().cache_policy().max_states, 0);
    game.position_at(nested, 27).unwrap();
    assert_eq!(game.active().history().cached_state_count(), 0);
}

pub(super) fn finish<F, E>(game: &mut BranchSession<F, BuiltinAgent<E>>)
where
    F: PublicInformationModel + Clone,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: Debug,
    E: Evaluator<F::State> + Checkpoint<Snapshot = ()>,
    E::Error: Debug,
    E::RestoreError: Debug,
{
    while let Decision::Player(actor) = game.active().session().next_player() {
        let version = game.active().session().position_version();
        if matches!(
            game.active().session().controller(actor),
            Some(Controller::Human)
        ) {
            let view = game.active().session().decision_view(actor).unwrap();
            game.submit_human(version, actor, view.actions.into_iter().next().unwrap())
                .unwrap();
        } else {
            game.advance_ai(version).unwrap();
        }
    }
}
fn branch_cases<F, E>(model: F, parameters: F::Parameters, evaluator: E)
where
    F: PublicInformationModel + Clone,
    F::State: Clone + PartialEq + Debug,
    F::Action: Clone + PartialEq + Debug,
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
                Controller::osla(evaluator, 7),
                Controller::osla(evaluator, 8),
            ],
        )
        .unwrap(),
    );
    finish(&mut game);
    let root_tip = game.active().session().state().clone();
    let records = game.moves(BranchId(0)).unwrap().to_vec();
    let position = records.len().min(2);
    let state = game.position_at(BranchId(0), position).unwrap();
    let controllers = game.controllers_at(BranchId(0), position).unwrap();
    let child = game.fork(BranchId(0), position, vec![None, None]).unwrap();
    assert_eq!(game.active().session().state(), &state);
    assert_eq!(game.initial_controllers(child).unwrap(), controllers);
    assert_eq!(
        game.info(child).unwrap().parent,
        Some((BranchId(0), position))
    );
    assert!(game.moves(child).unwrap().is_empty());
    for expected in records.iter().skip(position) {
        let accepted = game
            .advance_ai(game.active().session().position_version())
            .unwrap();
        assert_eq!(accepted.action, expected.action);
    }
    assert_eq!(game.active().session().state(), &root_tip);
    assert_eq!(game.moves(BranchId(0)).unwrap(), records);
    // Nested forks may start in their parent's inherited prefix.
    let nested = game
        .fork(child, 0, vec![Some(Controller::Human), None])
        .unwrap();
    assert_eq!(game.active().session().state(), game.root());
    finish(&mut game);
    let nested_tip = game.active().session().state().clone();
    game.select(BranchId(0)).unwrap();
    assert_eq!(game.active().session().state(), &root_tip);
    game.select(nested).unwrap();
    assert_eq!(game.active().session().state(), &nested_tip);
    let before = game.active().session().state().clone();
    let count = game.branch_count();
    assert!(matches!(
        game.fork(BranchId(99), 0, vec![None, None]),
        Err(BranchError::UnknownBranch(_))
    ));
    assert!(
        game.fork(BranchId(0), records.len() + 1, vec![None, None])
            .is_err()
    );
    assert!(game.fork(BranchId(0), 0, vec![None]).is_err());
    assert!(game.select(BranchId(99)).is_err());
    assert_eq!(game.branch_count(), count);
    assert_eq!(game.active().session().state(), &before);
    let reset = game.reset().unwrap();
    assert_ne!(reset, BranchId(0));
    assert_eq!(game.active().session().state(), game.root());
    finish(&mut game);
    assert_eq!(game.active().session().state(), &root_tip);
    game.select(child).unwrap();
    for p in 0..=records.len() {
        assert_eq!(
            game.position_at(child, p).unwrap(),
            game.position_at(BranchId(0), p).unwrap()
        );
    }
    let terminal = game
        .fork(BranchId(0), records.len(), vec![None, None])
        .unwrap();
    assert_eq!(game.info(terminal).unwrap().move_count, records.len());
    assert!(game.advance_ai(0).is_err());
    assert!(game.moves(terminal).unwrap().is_empty());
}
#[test]
fn independent_nested_branches_preserve_history_seats_and_exact_ai_progress() {
    for size in [1, 3, 5] {
        branch_cases(
            TicTacToeForwardModel,
            TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            TicTacToeEvaluator,
        );
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3), (4, 3, 1)] {
        branch_cases(
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
fn switching_branches_invalidates_pending_proposals() {
    let mut game = BranchSession::new(
        Session::create(
            TicTacToeForwardModel,
            &TicTacToeParameters::default(),
            2,
            vec![
                Controller::osla(TicTacToeEvaluator, 7),
                Controller::osla(TicTacToeEvaluator, 8),
            ],
        )
        .unwrap(),
    );
    let pending = game.prepare_ai(0).unwrap();
    game.fork(BranchId(0), 0, vec![None, None]).unwrap();
    assert!(game.commit_ai(pending).is_err());
    assert_eq!(game.active().session().position_version(), 1);
    assert!(game.moves(game.active_branch()).unwrap().is_empty());
    let next = game.prepare_ai(1).unwrap();
    game.commit_ai(next).unwrap();
    assert_eq!(game.active().session().position_version(), 2);
}

#[test]
fn delayed_human_input_is_rejected_after_forking_and_selecting_branches() {
    let mut game = BranchSession::<_, tag_rust::agents::FirstAvailableAgent>::new(
        Session::create(
            TicTacToeForwardModel,
            &TicTacToeParameters::default(),
            2,
            vec![Controller::Human, Controller::Human],
        )
        .unwrap(),
    );
    let action = tag_rust::games::tictactoe::PlaceMark { x: 1, y: 1 };
    let branch = game.fork(BranchId(0), 0, vec![None, None]).unwrap();
    assert!(
        game.submit_human(0, tag_rust::core::PlayerId::new(0), action)
            .is_err()
    );
    game.submit_human(1, tag_rust::core::PlayerId::new(0), action)
        .unwrap();
    let old = game.active().session().position_version();
    game.select(BranchId(0)).unwrap();
    assert!(
        game.submit_human(old, tag_rust::core::PlayerId::new(0), action)
            .is_err()
    );
    assert!(game.moves(BranchId(0)).unwrap().is_empty());
    game.select(branch).unwrap();
    assert_eq!(game.active().session().state().move_count(), 1);
}
