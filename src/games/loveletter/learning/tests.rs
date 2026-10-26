use super::*;
use crate::core::{
    random::{AgentRng, SimulationRng},
    simulation::SimulationModel,
};
fn names() -> Vec<&'static str> {
    layouts(2, LearningLimits::default())
        .unwrap()
        .iter()
        .map(|s| s.name)
        .collect()
}
fn encoded(s: &LoveLetterState, seat: usize) -> EncodedObservation {
    LoveLetterForwardModel
        .encode(
            &LoveLetterForwardModel
                .observe(s, PlayerId::new(seat))
                .unwrap(),
            &names(),
            LearningLimits::default(),
        )
        .unwrap()
}
#[test]
fn all_action_ids_specs_masks_and_shapes_have_fixed_meanings() {
    let model = LoveLetterForwardModel;
    for n in 2..=4 {
        let p = LoveLetterParameters::default();
        let state = model.init_state(&p, n).unwrap();
        let o = model.observe(&state, PlayerId::new(0)).unwrap();
        let spec = model
            .learning_spec_for_players(&p, n, LearningLimits::default())
            .unwrap();
        assert_eq!(spec.actions.size, 52);
        assert_eq!(spec.identity.player_count, n);
        assert!(spec.heuristic.is_none());
        let mut unique = std::collections::HashSet::new();
        for id in 0..52 {
            let a = model.action_from_id(&o, id).unwrap();
            assert_eq!(model.action_id(&o, &a).unwrap(), id);
            assert!(unique.insert(a));
        }
        assert_eq!(
            model.action_from_id(&o, 8).unwrap(),
            LoveLetterAction::Target {
                card: Card::Priest,
                target: PlayerId::new(0)
            }
        );
        assert_eq!(
            model.action_from_id(&o, 51).unwrap(),
            LoveLetterAction::Guess {
                card: Card::Princess,
                target: PlayerId::new(3)
            }
        );
        assert!(model.action_from_id(&o, 52).is_err());
        assert!(model.action_from_id(&o, usize::MAX).is_err());
        assert!(
            model
                .action_id(
                    &o,
                    &LoveLetterAction::Guess {
                        card: Card::Guard,
                        target: PlayerId::new(0)
                    }
                )
                .is_err()
        );
        for seat in 0..n {
            let e = encoded(&state, seat);
            assert_eq!(e.identity, spec.identity);
            for (tensor, desc) in e.tensors.iter().zip(&spec.payloads) {
                let desc = desc.tensor().unwrap();
                assert_eq!(tensor.shape, desc.shape);
                assert_eq!(tensor.values.len(), tensor.shape.iter().product::<usize>());
            }
            let expected =
                model.offered_actions(&model.observe(&state, PlayerId::new(seat)).unwrap());
            assert_eq!(
                e.offered_action_ids
                    .iter()
                    .map(|&id| model.action_from_id(&o, id).unwrap())
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(e.action_mask.iter().filter(|&&v| v).count(), expected.len());
            let hands = &e.tensors[0].values;
            for other in 0..n {
                if other != seat {
                    assert_eq!(hands[other * 2], -1.);
                    assert_eq!(hands[other * 2 + 1], if other == 0 { -1. } else { 0. });
                }
            }
        }
    }
}
#[test]
fn complete_games_preserve_history_privacy_terminal_results_and_source_independence() {
    let model = LoveLetterForwardModel;
    let mut random = AgentRng::new(99);
    let mut sampling = SimulationRng::new(123);
    let mut private_types = std::collections::HashSet::new();
    for n in 2..=4 {
        for seed in [0, 42, 999] {
            let mut state = model
                .init_state(
                    &LoveLetterParameters {
                        seed,
                        ..Default::default()
                    },
                    n,
                )
                .unwrap();
            let mut moves = 0;
            loop {
                let before = state.clone();
                for seat in 0..n {
                    let o = model.observe(&state, PlayerId::new(seat)).unwrap();
                    let e = encoded(&state, seat);
                    for event in &o.private {
                        private_types.insert(match event.information {
                            PrivateInformation::Received { .. } => 1,
                            PrivateInformation::Priest { .. } => 2,
                            PrivateInformation::Comparison { .. } => 3,
                            PrivateInformation::Exchange { .. } => 4,
                        });
                    }
                    assert_eq!(
                        e.tensors[5]
                            .values
                            .chunks(PUBLIC_WIDTH)
                            .filter(|r| r[0] == 1.)
                            .count(),
                        o.public.events.len()
                    );
                    assert_eq!(
                        e.tensors[6]
                            .values
                            .chunks(PRIVATE_WIDTH)
                            .filter(|r| r[0] == 1.)
                            .count(),
                        o.private.len()
                    );
                    assert_eq!(
                        e.tensors[8]
                            .values
                            .chunks(PRIVATE_WIDTH)
                            .filter(|r| r[0] == 1.)
                            .count(),
                        o.last_round_private.len()
                    );
                    if moves % 7 == 0 {
                        let alternative = model.simulation_fork(&o, &mut sampling).unwrap();
                        assert_eq!(e, encoded(&alternative, seat));
                    }
                    if model.next_player(&state) == Decision::Finished {
                        assert!(e.offered_action_ids.is_empty());
                        assert!(!e.action_mask.iter().any(|&v| v));
                        for s in 0..n {
                            assert_eq!(
                                e.results[s],
                                if state.public.outcome.winners().contains(&PlayerId::new(s)) {
                                    ResultTag::Win
                                } else {
                                    ResultTag::Loss
                                }
                            );
                        }
                    }
                }
                assert_eq!(state, before);
                let Decision::Player(actor) = model.next_player(&state) else {
                    break;
                };
                let actions = model.available_actions(&state, actor).unwrap();
                model
                    .next(
                        &mut state,
                        actor,
                        &actions[random.index(actions.len()).unwrap()],
                    )
                    .unwrap();
                moves += 1;
                assert!(moves < 3000);
            }
        }
    }
    assert_eq!(private_types.len(), 4);
}
#[test]
fn exact_numbers_versions_limits_and_owned_buffers_are_checked() {
    let model = LoveLetterForwardModel;
    let mut p = LoveLetterParameters::default();
    let mut state = model.init_state(&p, 3).unwrap();
    let spec = model
        .learning_spec_for_players(&p, 3, LearningLimits::default())
        .unwrap();
    p.seed = u64::MAX;
    p.first_player = PlayerId::new(2);
    assert_eq!(
        spec,
        model
            .learning_spec_for_players(&p, 3, LearningLimits::default())
            .unwrap()
    );
    p.rules.winning_tokens[1] = u32::MAX;
    assert_ne!(
        spec.identity,
        model
            .learning_spec_for_players(&p, 3, LearningLimits::default())
            .unwrap()
            .identity
    );
    state.public.rules = p.rules.clone();
    state.public.tokens[0] = (1 << 24) + 1;
    state.public.round = (1 << 24) + 1;
    let e = encoded(&state, 0);
    assert_eq!(&e.tensors[1].values[2..4], &[256., 1.]);
    assert_eq!(&e.tensors[4].values[2..6], &[1., 256., 0., 0.]);
    let o = model.observe(&state, PlayerId::new(0)).unwrap();
    for requested in [vec!["hands", "hands"], vec!["no_such_array"]] {
        assert!(
            model
                .encode(&o, &requested, LearningLimits::default())
                .is_err()
        );
    }
    assert!(
        model
            .encode(
                &o,
                &names(),
                LearningLimits {
                    max_bytes: 1024,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(
        model
            .encode(
                &o,
                &[],
                LearningLimits {
                    max_bytes: 1024,
                    ..Default::default()
                }
            )
            .is_ok()
    );
    assert!(
        model
            .learning_spec(
                &p,
                LearningLimits {
                    max_cells: 15,
                    ..Default::default()
                }
            )
            .is_err()
    );
    let mut changed = e;
    changed.tensors[0].values[0] = 999.;
    assert_ne!(changed, encoded(&state, 0));
    let mut too_long = o;
    too_long.private.resize(
        EVENTS + 1,
        PrivateEvent {
            event: 0,
            information: PrivateInformation::Received {
                card: Card::Guard,
                source: DrawSource::Deal,
            },
        },
    );
    assert!(
        model
            .encode(&too_long, &["private_history"], LearningLimits::default())
            .is_err()
    );
}

#[test]
fn priest_memory_stays_at_its_event_after_a_prince_replacement() {
    let prefix = [
        Card::Guard,
        Card::Priest,
        Card::Prince,
        Card::Handmaid,
        Card::Guard,
        Card::Guard,
        Card::Baron,
        Card::Guard,
    ];
    let mut remaining = Card::DECK.to_vec();
    for card in prefix {
        let i = remaining.iter().position(|&c| c == card).unwrap();
        remaining.remove(i);
    }
    let deck: [Card; 16] = prefix
        .into_iter()
        .chain(remaining)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let model = LoveLetterForwardModel;
    let p0 = PlayerId::new(0);
    let p1 = PlayerId::new(1);
    let mut state = model
        .init_with_deck(&LoveLetterParameters::default(), 3, &deck)
        .unwrap();
    model
        .next(
            &mut state,
            p0,
            &LoveLetterAction::Target {
                card: Card::Priest,
                target: p1,
            },
        )
        .unwrap();
    model
        .next(
            &mut state,
            p1,
            &LoveLetterAction::Target {
                card: Card::Prince,
                target: p1,
            },
        )
        .unwrap();
    assert_eq!(state.hands[1], [Card::Baron]);
    let e = encoded(&state, 0);
    assert_eq!(&e.tensors[0].values[2..4], &[-1., 0.]);
    let priest = e.tensors[6]
        .values
        .chunks(PRIVATE_WIDTH)
        .find(|r| r[2] == 2.)
        .unwrap();
    assert_eq!(priest[3], 2.);
    assert_eq!(priest[4], 5.);
    let outsider = encoded(&state, 2);
    assert!(
        !outsider.tensors[6]
            .values
            .chunks(PRIVATE_WIDTH)
            .any(|r| r[2] == 2.)
    );
    let prince_discard = e.tensors[5]
        .values
        .chunks(PUBLIC_WIDTH)
        .find(|r| r[1] == 4. && r[2] == 2. && r[4] == 5.)
        .unwrap();
    assert_eq!(prince_discard[0], 1.);
}

#[test]
fn exchange_and_comparison_values_stay_with_their_audiences_and_perspectives() {
    let prefix = [
        Card::Guard,
        Card::King,
        Card::Guard,
        Card::Handmaid,
        Card::Priest,
        Card::Baron,
    ];
    let mut remaining = Card::DECK.to_vec();
    for card in prefix {
        let i = remaining.iter().position(|&c| c == card).unwrap();
        remaining.remove(i);
    }
    let deck: [Card; 16] = prefix
        .into_iter()
        .chain(remaining)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let model = LoveLetterForwardModel;
    let p0 = PlayerId::new(0);
    let p1 = PlayerId::new(1);
    let mut state = model
        .init_with_deck(&LoveLetterParameters::default(), 3, &deck)
        .unwrap();
    model
        .next(
            &mut state,
            p0,
            &LoveLetterAction::Target {
                card: Card::King,
                target: p1,
            },
        )
        .unwrap();
    model
        .next(
            &mut state,
            p1,
            &LoveLetterAction::Target {
                card: Card::Baron,
                target: p0,
            },
        )
        .unwrap();
    let a = encoded(&state, 0);
    let b = encoded(&state, 1);
    let outsider = encoded(&state, 2);
    let exchange_a = a.tensors[6]
        .values
        .chunks(PRIVATE_WIDTH)
        .find(|r| r[2] == 4.)
        .unwrap();
    let exchange_b = b.tensors[6]
        .values
        .chunks(PRIVATE_WIDTH)
        .find(|r| r[2] == 4.)
        .unwrap();
    assert_eq!(&exchange_a[3..6], &[2., 2., 1.]); // Other seat 1; gave Priest, received Guard.
    assert_eq!(&exchange_b[3..6], &[1., 1., 2.]); // Other seat 0; gave Guard, received Priest.
    let comparison_a = a.tensors[6]
        .values
        .chunks(PRIVATE_WIDTH)
        .find(|r| r[2] == 3.)
        .unwrap();
    let comparison_b = b.tensors[6]
        .values
        .chunks(PRIVATE_WIDTH)
        .find(|r| r[2] == 3.)
        .unwrap();
    assert_eq!(&comparison_a[3..6], &[2., 1., 2.]);
    assert_eq!(&comparison_b[3..6], &[1., 2., 1.]);
    assert!(
        !outsider.tensors[6]
            .values
            .chunks(PRIVATE_WIDTH)
            .any(|r| r[2] == 3. || r[2] == 4.)
    );
    let public = outsider.tensors[5]
        .values
        .chunks(PUBLIC_WIDTH)
        .find(|r| r[1] == 6.)
        .unwrap();
    assert_eq!(&public[2..8], &[2., 1., 0., 0., 0., 1.]); // No private ranks in public comparison.
    assert_eq!(&a.tensors[0].values[..2], &[0., 0.]);
    assert!(a.results.iter().all(|r| *r == ResultTag::Ongoing));
    assert_eq!(a.tensors.last().unwrap().values, [0.; 3]); // Elimination is round-local.
}
