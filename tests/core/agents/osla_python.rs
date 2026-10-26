//! Test batched lookahead against the existing policy and exercise suspended execution.
use tag_rust::agents::*;
use tag_rust::core::{
    Decision, ForwardModel, PlayerId, PublicInformationModel, checkpoint::Checkpoint,
    evaluation::Evaluator,
};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{LearningGame, LearningLimits};
use tag_rust::session::{EvaluationSession, EvaluationStatus};

fn config(rows: usize) -> OslaPythonConfig {
    OslaPythonConfig {
        evaluator: EvaluatorIdentity {
            name: "test_zero".into(),
            version: 1,
            configuration: "{}".into(),
        },
        payloads: vec!["cells".into(), "board".into()],
        limits: EvaluationLimits {
            max_rows: rows,
            ..Default::default()
        },
    }
}

/// Deliberately supplies learning observations without opting into reconstruction.
#[derive(Clone)]
struct ObservationOnly;
impl ForwardModel for ObservationOnly {
    type State = ttt::TicTacToeState;
    type Action = ttt::PlaceMark;
    type Observation = ttt::TicTacToeObservation;
    type Parameters = ttt::TicTacToeParameters;
    type Error = ttt::TicTacToeError;
    fn init_state(&self, p: &Self::Parameters, players: usize) -> Result<Self::State, Self::Error> {
        ttt::TicTacToeForwardModel.init_state(p, players)
    }
    fn player_count(&self, s: &Self::State) -> usize {
        ttt::TicTacToeForwardModel.player_count(s)
    }
    fn next_player(&self, s: &Self::State) -> Decision {
        ttt::TicTacToeForwardModel.next_player(s)
    }
    fn observe(&self, s: &Self::State, p: PlayerId) -> Result<Self::Observation, Self::Error> {
        ttt::TicTacToeForwardModel.observe(s, p)
    }
    fn offered_actions(&self, o: &Self::Observation) -> Vec<Self::Action> {
        ttt::TicTacToeForwardModel.offered_actions(o)
    }
    fn available_actions(
        &self,
        s: &Self::State,
        p: PlayerId,
    ) -> Result<Vec<Self::Action>, Self::Error> {
        ttt::TicTacToeForwardModel.available_actions(s, p)
    }
    fn next(&self, s: &mut Self::State, p: PlayerId, a: &Self::Action) -> Result<(), Self::Error> {
        ttt::TicTacToeForwardModel.next(s, p, a)
    }
    fn player_result(
        &self,
        s: &Self::State,
        p: PlayerId,
    ) -> Result<tag_rust::core::PlayerResult, Self::Error> {
        ttt::TicTacToeForwardModel.player_result(s, p)
    }
    fn player_score(&self, s: &Self::State, p: PlayerId) -> Result<f64, Self::Error> {
        ttt::TicTacToeForwardModel.player_score(s, p)
    }
}
impl LearningGame for ObservationOnly {
    fn learning_spec(
        &self,
        p: &Self::Parameters,
        limits: LearningLimits,
    ) -> Result<tag_rust::learning::LearningSpec, tag_rust::learning::LearningError> {
        ttt::TicTacToeForwardModel.learning_spec(p, limits)
    }
    fn encode(
        &self,
        o: &Self::Observation,
        names: &[&str],
        limits: LearningLimits,
    ) -> Result<tag_rust::learning::EncodedObservation, tag_rust::learning::LearningError> {
        ttt::TicTacToeForwardModel.encode(o, names, limits)
    }
    fn action_id(
        &self,
        o: &Self::Observation,
        a: &Self::Action,
    ) -> Result<usize, tag_rust::learning::LearningError> {
        ttt::TicTacToeForwardModel.action_id(o, a)
    }
    fn action_from_id(
        &self,
        o: &Self::Observation,
        id: usize,
    ) -> Result<Self::Action, tag_rust::learning::LearningError> {
        ttt::TicTacToeForwardModel.action_from_id(o, id)
    }
}

#[test]
fn ordinary_observation_copy_reset_and_human_input_do_not_require_reconstruction() {
    let model = ObservationOnly;
    let mut session = EvaluationSession::from_state(
        model.clone(),
        model.init_state(&Default::default(), 2).unwrap(),
    )
    .unwrap();
    let observation = session
        .model()
        .observe(session.state(), PlayerId::new(0))
        .unwrap();
    assert_eq!(
        session
            .model()
            .encode(&observation, &["cells"], LearningLimits::default())
            .unwrap()
            .tensors[0]
            .values
            .len(),
        9
    );
    session
        .apply_human(PlayerId::new(0), ttt::PlaceMark { x: 0, y: 0 })
        .unwrap();
    let copy = session.snapshot().unwrap();
    session
        .reset_state(model.init_state(&Default::default(), 2).unwrap())
        .unwrap();
    assert_eq!(session.state().move_count(), 0);
    assert_eq!(copy.state().move_count(), 1);
}
struct ZeroBatch;
impl BatchEvaluator for ZeroBatch {
    fn evaluate(&mut self, batch: &CandidateBatch) -> Result<Vec<f64>, OslaPythonError> {
        Ok(vec![0.; batch.action_ids.len()])
    }
}
struct Zero;
impl<S> Evaluator<S> for Zero {
    type Error = std::convert::Infallible;
    fn evaluate(&mut self, _: &S, _: PlayerId) -> Result<f64, Self::Error> {
        Ok(0.)
    }
}

fn parity<F>(model: F, parameters: F::Parameters)
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug + std::fmt::Display,
{
    let mut old = [OslaAgent::new(Zero, 7), OslaAgent::new(Zero, 8)];
    let mut new = [
        OslaPythonAgent::new(ZeroBatch, config(1), 7).unwrap(),
        OslaPythonAgent::new(ZeroBatch, config(3), 8).unwrap(),
    ];
    let mut state = model.init_state(&parameters, 2).unwrap();
    while let Decision::Player(actor) = model.next_player(&state) {
        let view = model.observe(&state, actor).unwrap();
        let actions = model.offered_actions(&view);
        let a = old[actor.index()]
            .choose(&model, actor, &view, &actions)
            .unwrap();
        let b = new[actor.index()]
            .choose(&model, actor, &view, &actions)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(
            Agent::<F>::rng(&old[actor.index()]).snapshot(),
            Agent::<F>::rng(&new[actor.index()]).snapshot()
        );
        model.next(&mut state, actor, &actions[a]).unwrap();
    }
}

#[test]
fn native_batch_backends_match_osla_seeded_choices_and_rng_on_both_games() {
    parity(ttt::TicTacToeForwardModel, Default::default());
    parity(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters {
            grid_size: 5,
            ..Default::default()
        },
    );
    parity(c4::Connect4ForwardModel, Default::default());
    parity(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters {
            width: 4,
            height: 3,
            winning_length: 3,
            ..Default::default()
        },
    );
}

fn open_lines_parity<F, E>(model: F, parameters: F::Parameters, evaluator: E)
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug + std::fmt::Display,
    E: Evaluator<F::State> + Clone,
    E::Error: std::fmt::Debug,
{
    let spec = model
        .learning_spec(&parameters, LearningLimits::default())
        .unwrap();
    for rows in [1, 3, 1024] {
        let backend = OpenLinesBatchEvaluator::new(&spec).unwrap();
        assert_eq!(
            OpenLinesBatchEvaluator::from_snapshot(&backend.snapshot()).unwrap(),
            backend
        );
        let mut settings = config(rows);
        settings.payloads = vec!["cells".into()];
        let mut original = [
            OslaAgent::new(evaluator.clone(), 7),
            OslaAgent::new(evaluator.clone(), 8),
        ];
        let mut batched = [
            OslaPythonAgent::new(backend.clone(), settings.clone(), 7).unwrap(),
            OslaPythonAgent::new(backend, settings, 8).unwrap(),
        ];
        let mut state = model.init_state(&parameters, 2).unwrap();
        let mut scoring = evaluator.clone();
        while let Decision::Player(actor) = model.next_player(&state) {
            let view = model.observe(&state, actor).unwrap();
            let actions = model.offered_actions(&view);
            let work = batched[actor.index()]
                .prepare(&model, actor, &view, &actions)
                .unwrap();
            let mut backend = OpenLinesBatchEvaluator::new(&spec).unwrap();
            let mut offset = 0;
            while offset < actions.len() {
                let batch = work.batch(&model, offset).unwrap();
                let values = backend.evaluate(&batch).unwrap();
                for (i, &value) in values.iter().enumerate() {
                    let mut candidate = state.clone();
                    model
                        .next(&mut candidate, actor, &actions[offset + i])
                        .unwrap();
                    assert_eq!(value, scoring.evaluate(&candidate, actor).unwrap());
                }
                offset += values.len();
            }
            let a = original[actor.index()]
                .choose(&model, actor, &view, &actions)
                .unwrap();
            let b = batched[actor.index()]
                .choose(&model, actor, &view, &actions)
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(
                Agent::<F>::rng(&original[actor.index()]).snapshot(),
                Agent::<F>::rng(&batched[actor.index()]).snapshot()
            );
            model.next(&mut state, actor, &actions[a]).unwrap();
        }
    }
}

#[test]
fn open_line_batches_match_native_scores_choices_and_rng() {
    for size in [1, 3, 5] {
        open_lines_parity(
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            ttt::TicTacToeEvaluator,
        );
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3), (3, 2, 1)] {
        open_lines_parity(
            c4::Connect4ForwardModel,
            c4::Connect4Parameters {
                width,
                height,
                winning_length,
                ..Default::default()
            },
            c4::Connect4Evaluator,
        );
    }
}

#[test]
fn open_line_backend_checks_descriptions_batches_and_terminal_perspective() {
    let model = ttt::TicTacToeForwardModel;
    let mut spec = model
        .learning_spec(&Default::default(), LearningLimits::default())
        .unwrap();
    let state = model.init_state(&Default::default(), 2).unwrap();
    let actor = PlayerId::new(0);
    let view = model.observe(&state, actor).unwrap();
    let agent = OslaPythonAgent::new(ZeroBatch, config(3), 7).unwrap();
    let mut batch = agent
        .prepare(&model, actor, &view, &model.offered_actions(&view))
        .unwrap()
        .batch(&model, 0)
        .unwrap();
    let mut backend = OpenLinesBatchEvaluator::new(&spec).unwrap();
    batch.results = vec![1, 3, 2, 2, 3, 1];
    assert_eq!(backend.evaluate(&batch).unwrap(), vec![1.0, 0.0, -1.0]);
    batch.observer = PlayerId::new(1);
    assert_eq!(backend.evaluate(&batch).unwrap(), vec![-1.0, 0.0, 1.0]);
    batch
        .payloads
        .iter_mut()
        .find(|p| p.name == "cells")
        .unwrap()
        .values[0] = f32::NAN;
    assert!(backend.evaluate(&batch).is_err());
    spec.heuristic.as_mut().unwrap().indices[0] = 99;
    assert!(OpenLinesBatchEvaluator::new(&spec).is_err());
    spec.heuristic.as_mut().unwrap().indices[0] = 0;
    spec.heuristic.as_mut().unwrap().version = 99;
    assert!(OpenLinesBatchEvaluator::new(&spec).is_err());
}

#[test]
fn candidates_use_independent_states_original_observer_and_bounded_global_selection() {
    let model = ttt::TicTacToeForwardModel;
    let state = model.init_state(&Default::default(), 2).unwrap();
    let mut session = EvaluationSession::from_state(model, state).unwrap();
    let mut settings = config(9);
    settings.limits.learning.max_bytes = 400; // Only two 194-byte rows fit.
    session
        .set_controller(
            PlayerId::new(0),
            Some(OslaPythonAgent::new(ExternalEvaluation, settings, 42).unwrap()),
        )
        .unwrap();
    let initial = session.checkpoint(PlayerId::new(0)).unwrap();
    let mut request = session.prepare().unwrap();
    let mut ids = Vec::new();
    loop {
        assert!(request.batch.action_ids.len() <= 2);
        assert_eq!(request.batch.observer, PlayerId::new(0));
        assert_eq!(request.batch.total, 9);
        assert!(request.batch.next_players.iter().all(|&p| p == 1));
        for (row, &id) in request.batch.action_ids.iter().enumerate() {
            let cells = &request.batch.payloads[0].values[row * 9..(row + 1) * 9];
            assert_eq!(cells.iter().filter(|&&v| v != 0.).count(), 1);
            assert_eq!(cells[id], 1.);
            assert_eq!(&request.batch.progress[row * 4..row * 4 + 4], &[1, 1, 1, 0]);
        }
        ids.extend_from_slice(&request.batch.action_ids);
        let values = request
            .batch
            .action_ids
            .iter()
            .map(|&id| id as f64)
            .collect::<Vec<_>>();
        let result = session
            .finish(
                &request.key,
                &request.evaluator,
                &request.batch.action_ids,
                &values,
            )
            .unwrap();
        assert_eq!(session.checkpoint(PlayerId::new(0)).unwrap(), initial);
        assert_eq!(session.state().counters().move_count(), 0);
        match result {
            EvaluationStatus::Request(next) => request = *next,
            EvaluationStatus::Ready(proposal) => {
                assert_eq!(proposal.action_id, 8);
                session.commit(&proposal.key).unwrap();
                break;
            }
        }
    }
    assert_eq!(ids, (0..9).collect::<Vec<_>>());
    assert_eq!(session.state().counters().move_count(), 1);
    // A single best action retains the generator position even after commitment.
    assert_eq!(session.checkpoint(PlayerId::new(0)).unwrap(), initial);
}

fn session() -> EvaluationSession<ttt::TicTacToeForwardModel> {
    let model = ttt::TicTacToeForwardModel;
    let state = model.init_state(&Default::default(), 2).unwrap();
    let mut session = EvaluationSession::from_state(model, state).unwrap();
    session
        .set_controller(
            PlayerId::new(0),
            Some(OslaPythonAgent::new(ExternalEvaluation, config(9), 19).unwrap()),
        )
        .unwrap();
    session
}

#[test]
fn invalid_responses_cancellation_duplicate_and_cross_session_handles_preserve_progress() {
    let mut game = session();
    let before = game.checkpoint(PlayerId::new(0)).unwrap();
    let request = game.prepare().unwrap();
    assert!(matches!(game.prepare(), Err(OslaPythonError::Busy)));
    assert!(matches!(game.snapshot(), Err(OslaPythonError::Busy)));
    let mut other = session();
    let foreign = other.prepare().unwrap();
    assert!(matches!(
        other.finish(
            &request.key,
            &request.evaluator,
            &request.batch.action_ids,
            &[0.; 9]
        ),
        Err(OslaPythonError::Stale)
    ));
    assert!(other.cancel(&foreign.key).is_ok());
    let mut wrong_evaluator = request.evaluator.clone();
    wrong_evaluator.configuration = "{\"weight\":2}".into();
    assert!(
        game.finish(
            &request.key,
            &wrong_evaluator,
            &request.batch.action_ids,
            &[0.; 9]
        )
        .is_err()
    );
    assert!(
        game.finish(&request.key, &request.evaluator, &[8; 9], &[0.; 9])
            .is_err()
    );
    for values in [vec![0.; 8], vec![f64::NAN; 9], vec![f64::INFINITY; 9]] {
        assert!(
            game.finish(
                &request.key,
                &request.evaluator,
                &request.batch.action_ids,
                &values
            )
            .is_err()
        );
        assert_eq!(game.checkpoint(PlayerId::new(0)).unwrap(), before);
    }
    let EvaluationStatus::Ready(proposal) = game
        .finish(
            &request.key,
            &request.evaluator,
            &request.batch.action_ids,
            &[0.; 9],
        )
        .unwrap()
    else {
        panic!()
    };
    assert!(
        game.finish(
            &request.key,
            &request.evaluator,
            &request.batch.action_ids,
            &[0.; 9]
        )
        .is_err()
    );
    let chosen = proposal.action_id;
    game.cancel(&proposal.key).unwrap();
    assert_eq!(game.checkpoint(PlayerId::new(0)).unwrap(), before);
    assert!(game.commit(&proposal.key).is_err());
    let next = game.prepare().unwrap();
    let EvaluationStatus::Ready(proposal) = game
        .finish(&next.key, &next.evaluator, &next.batch.action_ids, &[0.; 9])
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(proposal.action_id, chosen);
    game.commit(&proposal.key).unwrap();
    assert_ne!(game.checkpoint(PlayerId::new(0)).unwrap(), before);
    assert!(game.commit(&proposal.key).is_err());
}

#[test]
fn settled_copies_reset_and_controller_changes_manage_checkpoints_and_pending_work() {
    let mut game = session();
    let initial = game.checkpoint(PlayerId::new(0)).unwrap().unwrap();
    let mut copied = game.snapshot().unwrap();
    let a = game.prepare().unwrap();
    let b = copied.prepare().unwrap();
    let EvaluationStatus::Ready(pa) = game
        .finish(&a.key, &a.evaluator, &a.batch.action_ids, &[0.; 9])
        .unwrap()
    else {
        panic!()
    };
    let EvaluationStatus::Ready(pb) = copied
        .finish(&b.key, &b.evaluator, &b.batch.action_ids, &[0.; 9])
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(pa.action_id, pb.action_id);
    game.commit(&pa.key).unwrap();
    assert_eq!(copied.state().counters().move_count(), 0);
    game.reset_state(
        ttt::TicTacToeForwardModel
            .init_state(&Default::default(), 2)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        game.checkpoint(PlayerId::new(0)).unwrap(),
        Some(initial.clone())
    );
    copied.set_controller(PlayerId::new(0), None).unwrap();
    assert!(copied.commit(&pb.key).is_err());
    assert!(copied.prepare().is_err());
    let mut bad = initial.clone();
    bad.policy_version = 2;
    assert!(OslaPythonAgent::<ExternalEvaluation>::from_snapshot(&bad).is_err());
    bad = initial;
    bad.rng.word_position = 1u128 << 68;
    assert!(OslaPythonAgent::<ExternalEvaluation>::from_snapshot(&bad).is_err());
}

#[test]
fn external_agents_require_values_and_selection_uses_true_maximum_with_atomic_validation() {
    let model = ttt::TicTacToeForwardModel;
    let state = model.init_state(&Default::default(), 2).unwrap();
    let observation = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&observation);
    let mut agent = OslaPythonAgent::new(ExternalEvaluation, config(2), u64::MAX).unwrap();
    let before = agent.snapshot();
    assert_eq!(
        agent.choose(&model, PlayerId::new(0), &observation, &actions),
        Err(OslaPythonError::ExternalEvaluationRequired)
    );
    assert!(agent.select(2, &[0., f64::NAN]).is_err());
    assert_eq!(agent.snapshot(), before);
    for _ in 0..100 {
        assert_ne!(agent.select(3, &[0., 0.0009, 0.0018]).unwrap(), 0);
    }
    let restored = OslaPythonAgent::<ExternalEvaluation>::from_snapshot(&agent.snapshot()).unwrap();
    assert_eq!(agent.snapshot(), restored.snapshot());
    let mut limited = config(1);
    limited.limits.learning = LearningLimits {
        max_cells: 8,
        ..Default::default()
    };
    let agent = OslaPythonAgent::new(ExternalEvaluation, limited, 0).unwrap();
    assert!(matches!(
        agent.prepare(&model, PlayerId::new(0), &observation, &actions),
        Err(OslaPythonError::ResourceLimit)
    ));
}
