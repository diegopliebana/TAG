//! Learning tests compare encoded data and the cell groups used by the heuristic with actual rules.
use tag_rust::core::{Decision, ForwardModel, PlayerId, evaluation::Evaluator};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::*;
const LIMITS: LearningLimits = LearningLimits {
    max_cells: 4096,
    max_bytes: 64 * 1024 * 1024,
};

#[test]
fn general_descriptions_need_no_board_or_line_heuristic() {
    let identity = EncodingIdentity {
        game: "synthetic-card-fixture",
        rules_version: 3,
        encoding_version: 2,
        configuration: EncodingConfiguration::Named(
            [("deck".to_owned(), "example-v1".to_owned())].into(),
        ),
        player_count: 4,
    };
    let spec = LearningSpec {
        identity: identity.clone(),
        payloads: vec![PayloadSpec::Features(TensorSpec {
            name: "own_hand",
            dtype: DType::Float32,
            shape: vec![8, 3],
            axes: vec!["slot", "feature"],
            components: vec!["rank", "suit", "valid"],
            bounds: None,
        })],
        metadata: vec![],
        actions: ActionSpec {
            size: 0,
            mapping: ActionMapping::Candidates,
        },
        heuristic: None,
    };
    assert_eq!(spec.identity.player_count, 4);
    assert_eq!(spec.identity.width(), 0);
    assert!(spec.heuristic.is_none());
    assert!(spec.actions.coordinates(&identity, 0).is_err());
    let indexed = ActionSpec {
        size: 50,
        mapping: ActionMapping::Indexed,
    };
    assert!(indexed.id(&identity, 0, 0).is_err());
    let mut copied = spec.clone();
    if let EncodingConfiguration::Named(settings) = &mut copied.identity.configuration {
        settings.insert("deck".into(), "different".into());
    }
    assert_ne!(copied.identity, identity);
}

fn verify<F>(model: &F, parameters: &F::Parameters)
where
    F: LearningGame,
    F::Error: std::fmt::Debug,
    F::Action: PartialEq + std::fmt::Debug,
{
    let spec = model.learning_spec(parameters, LIMITS).unwrap();
    let mut state = model.init_state(parameters, 2).unwrap();
    loop {
        for seat in 0..2 {
            let obs = model.observe(&state, PlayerId::new(seat)).unwrap();
            let encoded = model.encode(&obs, &["cells", "board"], LIMITS).unwrap();
            assert_eq!(encoded.identity, spec.identity);
            let offered = model.offered_actions(&obs);
            assert_eq!(
                encoded.action_mask.iter().filter(|&&v| v).count(),
                offered.len()
            );
            for (action, &id) in offered.iter().zip(&encoded.offered_action_ids) {
                assert!(encoded.action_mask[id]);
                assert_eq!(model.action_id(&obs, action).unwrap(), id);
                assert_eq!(model.action_from_id(&obs, id).unwrap(), *action);
            }
            for id in 0..spec.actions.size {
                assert_eq!(
                    model
                        .action_id(&obs, &model.action_from_id(&obs, id).unwrap())
                        .unwrap(),
                    id
                );
            }
            assert!(model.action_from_id(&obs, spec.actions.size).is_err());
            let cells = &encoded.tensors[0].values;
            let board = &encoded.tensors[1].values;
            let n = cells.len();
            let w = spec.identity.width();
            let h = spec.identity.height();
            for x in 0..w {
                for y in 0..h {
                    let v = cells[x * h + y];
                    let grid = y * w + x;
                    assert_eq!(board[grid], f32::from(v == 1.0));
                    assert_eq!(board[n + grid], f32::from(v == -1.0));
                    assert_eq!(board[2 * n + grid], f32::from(v == 0.0));
                }
            }
        }
        match model.next_player(&state) {
            Decision::Finished => break,
            Decision::Player(actor) => {
                let obs = model.observe(&state, actor).unwrap();
                let action = model.offered_actions(&obs).remove(0);
                model.next(&mut state, actor, &action).unwrap();
            }
        }
    }
}
#[test]
fn mapping_masks_and_grid_layout_follow_both_games_and_configurations() {
    verify(
        &ttt::TicTacToeForwardModel,
        &ttt::TicTacToeParameters::default(),
    );
    verify(
        &ttt::TicTacToeForwardModel,
        &ttt::TicTacToeParameters {
            grid_size: 5,
            ..Default::default()
        },
    );
    verify(
        &c4::Connect4ForwardModel,
        &c4::Connect4Parameters::default(),
    );
    verify(
        &c4::Connect4ForwardModel,
        &c4::Connect4Parameters {
            width: 4,
            height: 3,
            winning_length: 3,
            ..Default::default()
        },
    );
}
#[test]
fn encoding_is_owned_relative_and_source_observation_stays_unchanged() {
    let model = ttt::TicTacToeForwardModel;
    let mut state = model.init_state(&Default::default(), 2).unwrap();
    model
        .next(&mut state, PlayerId::new(0), &ttt::PlaceMark { x: 1, y: 2 })
        .unwrap();
    let a = model.observe(&state, PlayerId::new(0)).unwrap();
    let original = a.clone();
    let b = model.observe(&state, PlayerId::new(1)).unwrap();
    let mut ea = model.encode(&a, &["cells", "board"], LIMITS).unwrap();
    let eb = model.encode(&b, &["cells"], LIMITS).unwrap();
    assert_eq!(ea.tensors[0].values[5], 1.0);
    assert_eq!(eb.tensors[0].values[5], -1.0);
    assert!(ea.action_mask.iter().all(|v| !v));
    assert_eq!(eb.offered_action_ids.len(), 8);
    assert_eq!(ea.progress, [1, 1, 1, 0]);
    ea.tensors[0].values.fill(42.0);
    ea.action_mask.fill(true);
    model
        .next(&mut state, PlayerId::new(1), &ttt::PlaceMark { x: 0, y: 0 })
        .unwrap();
    assert_eq!(a, original);
    assert_eq!(
        model.encode(&a, &["cells"], LIMITS).unwrap().tensors[0].values[5],
        1.0
    );
    assert_eq!(eb.tensors[0].values[0], 0.0);
}
#[test]
fn payload_and_resource_errors_do_not_mutate_observations() {
    let m = ttt::TicTacToeForwardModel;
    let p = ttt::TicTacToeParameters::default();
    let state = m.init_state(&p, 2).unwrap();
    let o = m.observe(&state, PlayerId::new(0)).unwrap();
    assert!(matches!(
        m.encode(&o, &["image"], LIMITS),
        Err(LearningError::UnsupportedPayload(_))
    ));
    assert!(m.encode(&o, &["cells", "cells"], LIMITS).is_err());
    assert!(
        m.encode(
            &o,
            &["board"],
            LearningLimits {
                max_bytes: 1,
                ..LIMITS
            }
        )
        .is_err()
    );
    assert!(
        m.encode(
            &o,
            &[],
            LearningLimits {
                max_cells: 8,
                ..LIMITS
            }
        )
        .is_err()
    );
    assert!(
        m.learning_spec(
            &p,
            LearningLimits {
                max_bytes: 1,
                ..LIMITS
            }
        )
        .is_err()
    );
    assert!(
        m.learning_spec(
            &ttt::TicTacToeParameters {
                grid_size: usize::MAX,
                ..p
            },
            LIMITS
        )
        .is_err()
    );
    assert!(
        m.learning_spec(&ttt::TicTacToeParameters { grid_size: 0, ..p }, LIMITS)
            .is_err()
    );
    assert_eq!(
        m.encode(&o, &[], LIMITS).unwrap().offered_action_ids,
        (0..9).collect::<Vec<_>>()
    );
}
fn score(spec: &LearningSpec, encoded: &EncodedObservation, seat: usize) -> f64 {
    match encoded.results[seat] {
        ResultTag::Win => return 1.0,
        ResultTag::Loss => return -1.0,
        ResultTag::Draw => return 0.0,
        ResultTag::Ongoing => {}
    }
    let cells = &encoded.tensors[0].values;
    let mut sum = 0.0;
    let mut count = 0;
    for line in spec
        .heuristic
        .as_ref()
        .unwrap()
        .indices
        .chunks(spec.heuristic.as_ref().unwrap().window_length)
    {
        let friendly = line.iter().filter(|&&i| cells[i as usize] == 1.0).count();
        let other = line.iter().filter(|&&i| cells[i as usize] == -1.0).count();
        sum += if friendly > 0 && other > 0 {
            0.0
        } else if friendly > 0 {
            (friendly as f64 / line.len() as f64).powi(2)
        } else {
            -(other as f64 / line.len() as f64).powi(2)
        };
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        0.9 * sum / count as f64
    }
}
fn heuristic_parity<F, E>(m: F, p: F::Parameters, mut evaluator: E)
where
    F: LearningGame,
    F::Error: std::fmt::Debug,
    E: Evaluator<F::State>,
    E::Error: std::fmt::Debug,
{
    let spec = m.learning_spec(&p, LIMITS).unwrap();
    let mut state = m.init_state(&p, 2).unwrap();
    loop {
        for seat in 0..2 {
            let obs = m.observe(&state, PlayerId::new(seat)).unwrap();
            let encoded = m.encode(&obs, &["cells"], LIMITS).unwrap();
            assert_eq!(
                score(&spec, &encoded, seat),
                evaluator.evaluate(&state, PlayerId::new(seat)).unwrap()
            );
        }
        match m.next_player(&state) {
            Decision::Finished => break,
            Decision::Player(actor) => {
                let o = m.observe(&state, actor).unwrap();
                let actions = m.offered_actions(&o);
                m.next(&mut state, actor, actions.last().unwrap()).unwrap();
            }
        }
    }
}
#[test]
fn encoded_windows_and_results_reproduce_native_evaluators() {
    heuristic_parity(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters::default(),
        ttt::TicTacToeEvaluator,
    );
    heuristic_parity(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters::default(),
        c4::Connect4Evaluator,
    );
    heuristic_parity(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters {
            width: 4,
            height: 3,
            winning_length: 3,
            ..Default::default()
        },
        c4::Connect4Evaluator,
    );
    heuristic_parity(
        c4::Connect4ForwardModel,
        c4::Connect4Parameters {
            width: 2,
            height: 2,
            winning_length: 1,
            ..Default::default()
        },
        c4::Connect4Evaluator,
    );
}
#[test]
fn windows_are_ordered_complete_and_target_one_is_not_duplicated() {
    let s = ttt::TicTacToeForwardModel
        .learning_spec(&Default::default(), LIMITS)
        .unwrap();
    assert_eq!(s.heuristic.as_ref().unwrap().indices.len(), 8 * 3);
    assert!(
        s.heuristic
            .as_ref()
            .unwrap()
            .indices
            .chunks(3)
            .any(|line| line == [0, 4, 8])
    );
    let s = c4::Connect4ForwardModel
        .learning_spec(
            &c4::Connect4Parameters {
                width: 2,
                height: 2,
                winning_length: 1,
                ..Default::default()
            },
            LIMITS,
        )
        .unwrap();
    assert_eq!(s.heuristic.as_ref().unwrap().indices, vec![0, 1, 2, 3]);
}
#[test]
fn full_columns_keep_stable_ids_and_terminal_draws_have_no_input() {
    let m = c4::Connect4ForwardModel;
    let mut state = m.init_state(&Default::default(), 2).unwrap();
    for turn in 0..6 {
        m.next(
            &mut state,
            PlayerId::new(turn % 2),
            &c4::DropPiece { column: 0 },
        )
        .unwrap();
    }
    let o = m.observe(&state, PlayerId::new(0)).unwrap();
    let encoded = m.encode(&o, &[], LIMITS).unwrap();
    assert!(!encoded.action_mask[0]);
    assert_eq!(encoded.offered_action_ids, vec![1, 2, 3, 4, 5, 6]);
    let unavailable = m.action_from_id(&o, 0).unwrap();
    let before = state.clone();
    assert!(m.next(&mut state, PlayerId::new(0), &unavailable).is_err());
    assert_eq!(state, before);
    let m = ttt::TicTacToeForwardModel;
    let mut state = m.init_state(&Default::default(), 2).unwrap();
    for (turn, (x, y)) in [
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 2),
        (0, 2),
        (2, 2),
    ]
    .into_iter()
    .enumerate()
    {
        m.next(
            &mut state,
            PlayerId::new(turn % 2),
            &ttt::PlaceMark { x, y },
        )
        .unwrap();
    }
    for seat in 0..2 {
        let o = m.observe(&state, PlayerId::new(seat)).unwrap();
        let encoded = m.encode(&o, &[], LIMITS).unwrap();
        assert_eq!(encoded.results, vec![ResultTag::Draw, ResultTag::Draw]);
        assert!(encoded.action_mask.iter().all(|v| !v));
        assert!(encoded.offered_action_ids.is_empty());
        assert_eq!(encoded.next_player, None);
        assert_eq!(encoded.progress[0], 9);
    }
}
#[test]
fn descriptor_metadata_and_malformed_mapping_are_explicit() {
    let m = c4::Connect4ForwardModel;
    let mut spec = m.learning_spec(&Default::default(), LIMITS).unwrap();
    assert_eq!(spec.payloads[1].tensor().unwrap().shape, vec![3, 6, 7]);
    assert_eq!(spec.metadata[0].dtype, DType::Int64);
    assert_eq!(spec.metadata[1].dtype, DType::Int8);
    assert_eq!(spec.heuristic.as_ref().unwrap().indices.len(), 69 * 4);
    spec.actions.size = usize::MAX;
    assert!(spec.actions.coordinates(&spec.identity, 0).is_err());
    assert!(spec.actions.id(&spec.identity, 0, 0).is_err());
    spec.heuristic.as_mut().unwrap().indices.fill(-1);
    assert!(
        m.learning_spec(&Default::default(), LIMITS)
            .unwrap()
            .heuristic
            .as_ref()
            .unwrap()
            .indices
            .iter()
            .all(|i| *i >= 0)
    );
}
