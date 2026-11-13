use super::*;
use crate::agents::{Agent, RandomAgent, StructuredRandomAgent};
use crate::core::random::SimulationRng;
use crate::learning::{LearningGame, LearningLimits};
use crate::runner::run_game;

fn ordered_deck(prefix: &[u8]) -> Vec<u8> {
    let mut remaining = (2..=99).collect::<Vec<_>>();
    for card in prefix {
        cards::take_one(&mut remaining, card).expect("fixture card exists once");
    }
    let mut result = prefix.to_vec();
    result.extend(remaining);
    result
}

fn player(index: usize) -> PlayerId {
    PlayerId::new(index)
}

fn speak_all(model: &TheGameForwardModel, state: &mut State) {
    while let Phase::Communicate { speaker } = state.public.phase {
        model
            .next(state, speaker, &Action::Speak(Message::NoRequest))
            .unwrap();
    }
}

fn inventory(state: &State) -> Vec<u8> {
    let mut cards = state.deck.iter().copied().collect::<Vec<_>>();
    cards.extend(state.hands.iter().flatten().copied());
    cards.extend(state.public.placements.iter().map(|item| item.card));
    cards.sort_unstable();
    cards
}

#[test]
fn learning_ids_and_masks_follow_observed_actions() {
    let model = TheGameForwardModel;
    let state = model
        .init_state(
            &Parameters {
                seed: 42,
                ..Default::default()
            },
            3,
        )
        .unwrap();
    let limits = LearningLimits::default();
    let spec = model
        .learning_spec_for_players(&Parameters::default(), 3, limits)
        .unwrap();
    assert_eq!(spec.actions.size, 406);
    for seat in 0..3 {
        let view = model.observe(&state, player(seat)).unwrap();
        let encoded = model
            .encode(&view, &["hand", "placements", "messages"], limits)
            .unwrap();
        let expected = model
            .offered_actions(&view)
            .iter()
            .map(|action| model.action_id(&view, action).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(encoded.offered_action_ids, expected);
        for &id in &expected {
            assert!(encoded.action_mask[id]);
            let action = model.action_from_id(&view, id).unwrap();
            assert!(model.offered_actions(&view).contains(&action));
        }
        assert_eq!(
            encoded.tensors[0]
                .values
                .iter()
                .filter(|&&v| v == 1.)
                .count(),
            view.hand.len()
        );
    }
}

#[test]
fn simulation_fork_preserves_view_inventory_and_rng_isolation() {
    let model = TheGameForwardModel;
    let mut state = model
        .init_state(
            &Parameters {
                seed: 71,
                ..Default::default()
            },
            3,
        )
        .unwrap();
    speak_all(&model, &mut state);
    let actor = state.public.active_player;
    let first = model
        .available_actions(&state, actor)
        .unwrap()
        .into_iter()
        .find(|a| matches!(a, Action::Play { .. }))
        .unwrap();
    model.next(&mut state, actor, &first).unwrap();
    let original = state.clone();
    let view = model.observe(&state, player(1)).unwrap();
    let mut random = SimulationRng::new(27);
    let saved = random.snapshot();
    let sample = model.simulation_fork(&view, &mut random).unwrap();
    assert_eq!(model.observe(&sample, player(1)).unwrap(), view);
    assert_eq!(inventory(&sample), (2..=99).collect::<Vec<_>>());
    assert_eq!(state, original);
    assert_ne!(random.snapshot(), saved);
    let mut replay = SimulationRng::from_snapshot(&saved).unwrap();
    assert_eq!(model.simulation_fork(&view, &mut replay).unwrap(), sample);
    let mut bad = view.clone();
    bad.hand.push(2);
    let before = random.snapshot();
    assert_eq!(
        model.simulation_fork(&bad, &mut random),
        Err(Error::InvalidObservation)
    );
    assert_eq!(random.snapshot(), before);
}

#[test]
fn structured_mcts_search_is_seed_reproducible_for_team_game() {
    use crate::search::mcts::{Budgets, MctsAgent, NativeEvaluators, Settings};
    let model = TheGameForwardModel;
    let state = model
        .init_state(
            &Parameters {
                seed: 19,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    let actor = match model.next_player(&state) {
        Decision::Player(actor) => actor,
        _ => panic!("expected player"),
    };
    let view = model.observe(&state, actor).unwrap();
    let actions = model.available_actions(&state, actor).unwrap();
    let settings = Settings {
        seed: Some(77),
        structured_actions: true,
        budgets: Budgets {
            iterations: Some(16),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Default::default()
    };
    let registry = TheGameForwardModel::native_evaluators();
    let mut first = MctsAgent::new(settings.clone(), &registry).unwrap();
    let mut replay = MctsAgent::new(settings, &registry).unwrap();
    let one = first.search(&model, actor, &view, &actions).unwrap();
    let two = replay.search(&model, actor, &view, &actions).unwrap();
    assert_eq!(one.selected_index, two.selected_index);
    assert_eq!(one.root_actions, two.root_actions);
    assert_eq!(one.counters, two.counters);
    assert_eq!(first.snapshot().unwrap(), replay.snapshot().unwrap());
}

#[test]
fn component_paths_match_complete_actions_without_changing_state() {
    let model = TheGameForwardModel;
    let deck = ordered_deck(&[99, 2, 88, 3, 77, 4, 66, 5]);
    let state = model
        .init_with_deck(&Parameters::default(), 1, &deck)
        .unwrap();
    let before = state.clone();
    let view = model.observe(&state, player(0)).unwrap();
    let offered = model.offered_actions(&view);
    let root = model.action_choices(&view, &[]).unwrap();
    assert!(root.iter().any(|choice| choice.action.is_none()));
    let mut recovered = Vec::new();
    for choice in root {
        if let Some(action) = choice.action {
            recovered.push(action);
        } else {
            for leaf in model.action_choices(&view, &[choice.component]).unwrap() {
                recovered.push(leaf.action.unwrap());
            }
        }
    }
    assert_eq!(recovered, offered);
    assert!(model.action_choices(&view, &[999]).is_err());
    assert_eq!(state, before);
    let mut agent = StructuredRandomAgent::new(7);
    let selected = agent.choose(&model, player(0), &view, &offered).unwrap();
    assert!(selected < offered.len());
}

#[test]
fn setup_and_random_matches_preserve_cards_and_team_results() {
    let model = TheGameForwardModel;
    let expected = (2..=99).collect::<Vec<_>>();
    for players in 1..=5 {
        let parameters = Parameters {
            seed: 13,
            first_player: player(players - 1),
            communication: CommunicationMode::LimitedChoices,
        };
        let mut state = model.init_state(&parameters, players).unwrap();
        let hand_size = starting_hand_size(players);
        assert_eq!(state.public.hand_counts, vec![hand_size; players]);
        assert_eq!(state.public.draw_count, 98 - players * hand_size);
        assert_eq!(state.public.active_player, parameters.first_player);
        assert_eq!(inventory(&state), expected);
        let mut agents = (0..players)
            .map(|seat| RandomAgent::new(100 + seat as u64))
            .collect::<Vec<_>>();
        run_game(&model, &mut state, &mut agents).unwrap();
        assert_eq!(inventory(&state), expected);
        assert_eq!(model.next_player(&state), Decision::Finished);
        assert_eq!(state.public.placements.len() + state.cards_remaining(), 98);
        let reported = model.game_result(&state).unwrap();
        assert_eq!(reported.players.len(), players);
        assert_eq!(
            reported.score.unwrap().values,
            vec![state.public.placements.len() as f64]
        );
        for seat in 0..players {
            assert_eq!(
                model.player_result(&state, player(seat)).unwrap(),
                model.player_result(&state, player(0)).unwrap()
            );
            assert_eq!(
                model.player_score(&state, player(seat)).unwrap(),
                state.public.placements.len() as f64
            );
        }
    }
}

#[test]
fn controlled_deck_and_seeded_setup_are_exact() {
    let model = TheGameForwardModel;
    let parameters = Parameters::default();
    let deck = ordered_deck(&[99, 2, 88, 3, 77, 4, 66, 5]);
    let state = model.init_with_deck(&parameters, 1, &deck).unwrap();
    assert_eq!(state.privileged_hand(player(0)).unwrap(), &deck[..8]);
    assert_eq!(state.privileged_deck().front(), Some(&deck[8]));
    assert_eq!(inventory(&state), (2..=99).collect::<Vec<_>>());
    let mut invalid = deck.clone();
    invalid[0] = invalid[1];
    assert_eq!(
        model.init_with_deck(&parameters, 1, &invalid),
        Err(Error::InvalidDeck)
    );
    assert_eq!(
        model.init_state(&parameters, 2).unwrap(),
        model.init_state(&parameters, 2).unwrap()
    );
    let ordinary = model.init_state(&parameters, 1).unwrap();
    assert_eq!(state.chance_snapshot(), ordinary.chance_snapshot());
}

#[test]
fn finishing_a_turn_refills_only_to_the_starting_hand_size() {
    let model = TheGameForwardModel;
    let deck = ordered_deck(&[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let mut state = model
        .init_with_deck(&Parameters::default(), 1, &deck)
        .unwrap();
    for card in [2, 3] {
        model
            .next(
                &mut state,
                player(0),
                &Action::Play {
                    card,
                    pile: Pile::AscendingOne,
                },
            )
            .unwrap();
    }
    assert_eq!(state.public.hand_counts, [6]);
    assert_eq!(state.public.draw_count, 90);
    model.next(&mut state, player(0), &Action::EndTurn).unwrap();
    assert_eq!(
        state.privileged_hand(player(0)).unwrap(),
        &[4, 5, 6, 7, 8, 9, 10, 11]
    );
    assert_eq!(state.public.draw_count, 88);
    assert_eq!(state.public.turns_completed, 1);
    assert_eq!(inventory(&state), (2..=99).collect::<Vec<_>>());
}

#[test]
fn messages_are_hidden_until_every_seat_has_spoken() {
    let model = TheGameForwardModel;
    let mut state = model.init_state(&Parameters::default(), 3).unwrap();
    let before = model.observe(&state, player(2)).unwrap();
    assert_eq!(
        state.public.phase,
        Phase::Communicate { speaker: player(0) }
    );
    assert_eq!(
        model.available_actions(&state, player(0)).unwrap().len(),
        Message::COUNT
    );
    model
        .next(
            &mut state,
            player(0),
            &Action::Speak(Message::LeavePile(Pile::AscendingOne)),
        )
        .unwrap();
    assert!(state.public.messages.is_empty());
    assert_eq!(
        state.pending_messages,
        [Message::LeavePile(Pile::AscendingOne)]
    );
    let after = model.observe(&state, player(2)).unwrap();
    assert_eq!(before.hand, after.hand);
    assert_eq!(before.public.pile_tops, after.public.pile_tops);
    assert_eq!(before.public.messages, after.public.messages);
    model
        .next(
            &mut state,
            player(1),
            &Action::Speak(Message::SmallJump(Pile::DescendingTwo)),
        )
        .unwrap();
    model
        .next(&mut state, player(2), &Action::Speak(Message::NoRequest))
        .unwrap();
    assert_eq!(state.public.phase, Phase::Play);
    assert_eq!(state.public.messages.len(), 1);
    assert_eq!(
        state.public.messages[0].messages,
        [
            Message::LeavePile(Pile::AscendingOne),
            Message::SmallJump(Pile::DescendingTwo),
            Message::NoRequest
        ]
    );
    assert!(state.pending_messages.is_empty());
}

#[test]
fn ordered_placements_and_reverse_moves_follow_pile_tops() {
    let model = TheGameForwardModel;
    let deck = ordered_deck(&[47, 37, 75, 65, 8, 9, 10, 11]);
    let mut state = model
        .init_with_deck(&Parameters::default(), 1, &deck)
        .unwrap();
    state.public.pile_tops = [46, 1, 65, 100];
    // These pile tops are a local transition fixture; card conservation is
    // tested separately with complete games above.
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 47,
                pile: Pile::AscendingOne,
            },
        )
        .unwrap();
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 37,
                pile: Pile::AscendingOne,
            },
        )
        .unwrap();
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 75,
                pile: Pile::DescendingOne,
            },
        )
        .unwrap();
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 65,
                pile: Pile::DescendingOne,
            },
        )
        .unwrap();
    assert_eq!(state.public.pile_tops, [37, 1, 65, 100]);
    assert_eq!(state.public.played_this_turn, 4);
    assert!(
        model
            .available_actions(&state, player(0))
            .unwrap()
            .contains(&Action::EndTurn)
    );
}

#[test]
fn bad_first_placement_can_lose_before_second_card() {
    let model = TheGameForwardModel;
    let mut state = model
        .init_state(
            &Parameters {
                communication: CommunicationMode::Silent,
                ..Parameters::default()
            },
            1,
        )
        .unwrap();
    state.hands[0] = vec![9, 10];
    state.deck = VecDeque::from([11]);
    state.public.pile_tops = [1, 100, 8, 8];
    state.sync_counts();
    state.start_turn();
    assert_eq!(state.public.outcome, TeamOutcome::Ongoing);
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 10,
                pile: Pile::AscendingOne,
            },
        )
        .unwrap();
    assert_eq!(state.public.outcome, TeamOutcome::Lost);
    assert_eq!(
        model.player_result(&state, player(0)).unwrap(),
        PlayerResult::Loss
    );
    assert_eq!(model.next_player(&state), Decision::Finished);
}

#[test]
fn exhausted_draw_pile_reduces_minimum_and_skips_empty_hands() {
    let model = TheGameForwardModel;
    let mut state = model
        .init_state(
            &Parameters {
                communication: CommunicationMode::Silent,
                ..Parameters::default()
            },
            2,
        )
        .unwrap();
    state.hands = vec![vec![2], vec![3]];
    state.deck.clear();
    state.public.active_player = player(0);
    state.public.pile_tops = [1, 1, 100, 100];
    state.sync_counts();
    state.start_turn();
    model
        .next(
            &mut state,
            player(0),
            &Action::Play {
                card: 2,
                pile: Pile::AscendingOne,
            },
        )
        .unwrap();
    assert!(
        model
            .available_actions(&state, player(0))
            .unwrap()
            .contains(&Action::EndTurn)
    );
    model.next(&mut state, player(0), &Action::EndTurn).unwrap();
    assert_eq!(state.public.active_player, player(1));
    assert_eq!(state.public.hand_counts, [0, 1]);
    model
        .next(
            &mut state,
            player(1),
            &Action::Play {
                card: 3,
                pile: Pile::AscendingOne,
            },
        )
        .unwrap();
    assert_eq!(state.public.outcome, TeamOutcome::Won);
    for seat in 0..2 {
        assert_eq!(
            model.player_result(&state, player(seat)).unwrap(),
            PlayerResult::Win
        );
    }
}

#[test]
fn observations_and_rejected_actions_do_not_expose_or_change_hidden_data() {
    let model = TheGameForwardModel;
    let mut a = model.init_state(&Parameters::default(), 3).unwrap();
    let mut b = a.clone();
    std::mem::swap(&mut b.hands[1][0], &mut b.deck[0]);
    assert_eq!(
        model.observe(&a, player(0)).unwrap(),
        model.observe(&b, player(0)).unwrap()
    );
    let original = a.snapshot();
    assert_eq!(
        model.next(
            &mut a,
            player(0),
            &Action::Play {
                card: 2,
                pile: Pile::AscendingOne
            }
        ),
        Err(Error::IllegalAction)
    );
    assert_eq!(a, original);
    speak_all(&model, &mut a);
    assert_eq!(original.public.pile_tops, a.public.pile_tops);
    assert_eq!(
        original.privileged_hand(player(0)).unwrap(),
        a.privileged_hand(player(0)).unwrap()
    );
    assert_ne!(a.public.messages, original.public.messages);
}
