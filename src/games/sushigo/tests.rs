use super::*;
use crate::agents::RandomAgent;
use crate::runner::run_joint_game;
use crate::search::mcts::{
    Budgets, EvaluatorRegistry, MctsAgent, Result as SearchResult, Settings, SimulationContext,
    StateEvaluator, StateValues,
};

struct PublicBoundaryValues;
impl StateEvaluator<SushiGoForwardModel> for PublicBoundaryValues {
    fn evaluate(
        &mut self,
        state: &State,
        _: &mut SimulationContext<'_, SushiGoForwardModel>,
    ) -> SearchResult<StateValues> {
        if state.public.last_reveal.len() != state.hands.len() {
            return Err(crate::search::mcts::Error::Evaluation(
                "partial joint action reached evaluator".into(),
            ));
        }
        Ok(StateValues::Full(
            state
                .public
                .scores
                .iter()
                .map(|&score| f64::from(score))
                .collect(),
        ))
    }
}

fn ordered_deck(prefix: &[Card]) -> Vec<Card> {
    let mut rest = Card::deck();
    for &card in prefix {
        cards::take_one(&mut rest, &card).expect("available fixture card");
    }
    let mut result = prefix.to_vec();
    result.extend(rest);
    result
}

#[test]
fn card_value_action_ids_are_unique_and_stable() {
    let mut ids = Vec::new();
    for first in Card::ALL {
        ids.push(Action::Pick(first).id());
        for second in Card::ALL {
            ids.push(Action::Chopsticks { first, second }.id());
        }
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..Action::ACTION_COUNT).collect::<Vec<_>>());
    assert_eq!(Action::Pick(Card::Tempura).id(), 0);
    assert_eq!(Action::Pick(Card::Pudding).id(), 11);
    assert_eq!(
        Action::Chopsticks {
            first: Card::Wasabi,
            second: Card::SquidNigiri
        }
        .id(),
        126
    );
}

fn joint(state: &mut State, choices: &[Action]) {
    let model = SushiGoForwardModel;
    joint::apply(
        &model,
        state,
        &choices
            .iter()
            .copied()
            .enumerate()
            .map(|(i, action)| (PlayerId::new(i), action))
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

fn inventory(state: &State) -> Vec<Card> {
    let mut cards = state.deck.iter().copied().collect::<Vec<_>>();
    cards.extend(state.hands.iter().flatten().copied());
    cards.extend(state.public.played.iter().flatten().map(|entry| entry.card));
    cards.extend(state.public.discard.iter().copied());
    for &count in &state.public.pudding {
        cards.extend(std::iter::repeat_n(Card::Pudding, count));
    }
    cards.sort();
    cards
}

#[test]
fn deck_deal_and_random_complete_games_preserve_cards() {
    let model = SushiGoForwardModel;
    let mut expected = Card::deck();
    expected.sort();
    for n in 2..=5 {
        let mut state = model.init_state(&Parameters { seed: 17 }, n).unwrap();
        assert_eq!(state.public.hand_size, 12 - n);
        let mut agents = (0..n)
            .map(|seat| RandomAgent::new(seat as u64 + 31))
            .collect::<Vec<_>>();
        let mut simulation_rng = SimulationRng::new(100 + n as u64);
        run_joint_game(&model, &mut state, &mut agents, |model, state, choices| {
            assert_eq!(choices.len(), n);
            assert_eq!(inventory(state), expected);
            if !state.public.finished {
                for seat in 0..n {
                    let observation = model.observe(state, PlayerId::new(seat)).unwrap();
                    let fork = model
                        .simulation_fork(&observation, &mut simulation_rng)
                        .unwrap();
                    assert_eq!(
                        model.observe(&fork, PlayerId::new(seat)).unwrap(),
                        observation
                    );
                    assert_eq!(inventory(&fork), expected);
                }
            }
        })
        .unwrap();
        assert_eq!(state.public.round_scores.len(), 3);
        assert!(state.public.finished);
        assert_eq!(inventory(&state), expected);
        assert_eq!(model.decision_players(&state), []);
        assert!(state.public.scores.iter().all(|score| *score >= -6));
    }
}

#[test]
fn final_card_passes_to_recipient_without_another_choice() {
    let model = SushiGoForwardModel;
    // Seat 0 starts with nine Tempura and one Sashimi; seat 1 with nine
    // Dumpling and one Squid. Repeated picks leave the last distinct card.
    let prefix = [vec![Card::Tempura; 9], vec![Card::Dumpling; 9]];
    let mut cards = prefix[0].clone();
    cards.push(Card::Sashimi);
    cards.extend(prefix[1].iter().copied());
    cards.push(Card::SquidNigiri);
    let deck = ordered_deck(&cards);
    let mut state = model
        .init_with_deck(&Parameters { seed: 1 }, 2, &deck)
        .unwrap();
    for _ in 0..9 {
        let a = model.available_actions(&state, PlayerId::new(0)).unwrap();
        let b = model.available_actions(&state, PlayerId::new(1)).unwrap();
        let first = if a.contains(&Action::Pick(Card::Tempura)) {
            Card::Tempura
        } else {
            Card::Dumpling
        };
        let second = if b.contains(&Action::Pick(Card::Dumpling)) {
            Card::Dumpling
        } else {
            Card::Tempura
        };
        joint(&mut state, &[Action::Pick(first), Action::Pick(second)]);
    }
    assert_eq!(state.public.round, 2);
    assert_eq!(state.public.round_scores.len(), 1);
    assert_eq!(
        state
            .public
            .discard
            .iter()
            .filter(|&&c| c == Card::SquidNigiri)
            .count(),
        1
    );
    assert_eq!(
        state
            .public
            .discard
            .iter()
            .filter(|&&c| c == Card::Sashimi)
            .count(),
        1
    );
}

#[test]
fn controlled_three_player_pass_matches_public_tag_joint_fixture() {
    let model = SushiGoForwardModel;
    let mut prefix = vec![Card::Tempura; 9];
    prefix.extend(std::iter::repeat_n(Card::Sashimi, 9));
    prefix.extend(std::iter::repeat_n(Card::Dumpling, 9));
    let deck = ordered_deck(&prefix);
    let mut state = model
        .init_with_deck(&Parameters { seed: 1 }, 3, &deck)
        .unwrap();
    joint(
        &mut state,
        &[
            Action::Pick(Card::Tempura),
            Action::Pick(Card::Sashimi),
            Action::Pick(Card::Dumpling),
        ],
    );
    assert_eq!(state.hands[0], vec![Card::Sashimi; 8]);
    assert_eq!(state.hands[1], vec![Card::Dumpling; 8]);
    assert_eq!(state.hands[2], vec![Card::Tempura; 8]);
    assert_eq!(state.public.last_reveal.len(), 3);
}

#[test]
fn controlled_java_execution_matches_reveal_pass_and_card_points() {
    // Public TAG, seed 12345, first card in each nine-card hand. The card
    // sequence is copied from its printed fixture, not generated by Rust's RNG.
    let hands = [
        vec![
            Card::SquidNigiri,
            Card::Dumpling,
            Card::Maki3,
            Card::Maki1,
            Card::Tempura,
            Card::SalmonNigiri,
            Card::Dumpling,
            Card::SalmonNigiri,
            Card::SquidNigiri,
        ],
        vec![
            Card::Maki2,
            Card::EggNigiri,
            Card::Wasabi,
            Card::Wasabi,
            Card::Tempura,
            Card::Pudding,
            Card::Chopsticks,
            Card::Maki2,
            Card::Maki1,
        ],
        vec![
            Card::SalmonNigiri,
            Card::Maki2,
            Card::Dumpling,
            Card::EggNigiri,
            Card::Pudding,
            Card::Maki3,
            Card::Maki2,
            Card::Dumpling,
            Card::Tempura,
        ],
    ];
    let prefix = hands.iter().flatten().copied().collect::<Vec<_>>();
    let mut state = SushiGoForwardModel
        .init_with_deck(&Parameters { seed: 1 }, 3, &ordered_deck(&prefix))
        .unwrap();
    assert_eq!(state.hands, hands);
    joint(
        &mut state,
        &[
            Action::Pick(Card::SquidNigiri),
            Action::Pick(Card::Maki2),
            Action::Pick(Card::SalmonNigiri),
        ],
    );
    for seat in 0..3 {
        assert_eq!(state.hands[seat], hands[(seat + 1) % 3][1..]);
    }
    assert_eq!(
        state
            .public
            .played
            .iter()
            .map(|cards| cards[0].card)
            .collect::<Vec<_>>(),
        [Card::SquidNigiri, Card::Maki2, Card::SalmonNigiri]
    );
    assert_eq!(
        state
            .public
            .played
            .iter()
            .map(|cards| score_cards(cards))
            .collect::<Vec<_>>(),
        [3, 0, 2]
    );
}

#[test]
fn final_card_is_scored_by_receiving_seat() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 8 }, 2).unwrap();
    state.hands[0] = vec![Card::Tempura, Card::Sashimi];
    state.hands[1] = vec![Card::Dumpling, Card::SquidNigiri];
    state.public.hand_size = 2;
    state.known_by[0][0] = Some(state.hands[0].clone());
    state.known_by[1][1] = Some(state.hands[1].clone());
    joint(
        &mut state,
        &[Action::Pick(Card::Tempura), Action::Pick(Card::Dumpling)],
    );
    // Seat 0 receives Squid; seat 1 receives Sashimi.
    assert_eq!(state.public.round_scores[0], [3, 1]);
}

#[test]
fn chopsticks_order_controls_same_turn_wasabi_pairing() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 7 }, 2).unwrap();
    state.hands[0] = vec![Card::Wasabi, Card::SquidNigiri, Card::Tempura];
    state.hands[1] = vec![Card::Dumpling, Card::Sashimi, Card::Tempura];
    state.public.hand_size = 3;
    state.public.played[0].push(PlayedCard {
        card: Card::Chopsticks,
        on_wasabi: false,
    });
    state.known_by[0][state.public.hand_ids[0]] = Some(state.hands[0].clone());
    state.known_by[1][state.public.hand_ids[1]] = Some(state.hands[1].clone());
    let mut reverse = state.clone();
    joint(
        &mut state,
        &[
            Action::Chopsticks {
                first: Card::Wasabi,
                second: Card::SquidNigiri,
            },
            Action::Pick(Card::Dumpling),
        ],
    );
    joint(
        &mut reverse,
        &[
            Action::Chopsticks {
                first: Card::SquidNigiri,
                second: Card::Wasabi,
            },
            Action::Pick(Card::Dumpling),
        ],
    );
    assert!(
        state.public.played[0]
            .iter()
            .any(|c| c.card == Card::SquidNigiri && c.on_wasabi)
    );
    assert!(
        reverse.public.played[0]
            .iter()
            .any(|c| c.card == Card::SquidNigiri && !c.on_wasabi)
    );
    assert_eq!(score_cards(&state.public.played[0]), 9);
    assert_eq!(score_cards(&reverse.public.played[0]), 3);
    assert_eq!(
        state.public.last_reveal[0].1,
        Action::Chopsticks {
            first: Card::Wasabi,
            second: Card::SquidNigiri
        }
    );
    assert!(state.hands[1].contains(&Card::Chopsticks));
}

#[test]
fn score_sets_maki_pudding_and_shared_winners() {
    let cards = [
        Card::Tempura,
        Card::Tempura,
        Card::Sashimi,
        Card::Sashimi,
        Card::Sashimi,
        Card::Dumpling,
        Card::Dumpling,
        Card::Dumpling,
        Card::EggNigiri,
    ];
    let played = cards.map(|card| PlayedCard {
        card,
        on_wasabi: false,
    });
    assert_eq!(score_cards(&played), 5 + 10 + 6 + 1);
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 1 }, 3).unwrap();
    state.public.played = vec![
        vec![PlayedCard {
            card: Card::Maki3,
            on_wasabi: false,
        }],
        vec![PlayedCard {
            card: Card::Maki2,
            on_wasabi: false,
        }],
        vec![PlayedCard {
            card: Card::Maki2,
            on_wasabi: false,
        }],
    ];
    state.public.round = 3;
    state.public.pudding = vec![2, 1, 1];
    end_round(&mut state);
    assert_eq!(state.public.round_scores[0], [6, 1, 1]);
    assert_eq!(state.public.scores, [12, -2, -2]);
    state.public.scores = vec![10, 10, 5];
    state.public.pudding = vec![2, 2, 0];
    assert_eq!(
        model.player_result(&state, PlayerId::new(0)).unwrap(),
        PlayerResult::Win
    );
    assert_eq!(
        model.player_result(&state, PlayerId::new(1)).unwrap(),
        PlayerResult::Win
    );
    assert_eq!(
        model.player_result(&state, PlayerId::new(2)).unwrap(),
        PlayerResult::Loss
    );
}

#[test]
fn every_card_score_and_set_boundary_matches_the_rulebook() {
    let score = |card: Card, count: usize| {
        score_cards(&vec![
            PlayedCard {
                card,
                on_wasabi: false
            };
            count
        ])
    };
    for (count, expected) in [0, 0, 5, 5, 10].into_iter().enumerate() {
        assert_eq!(score(Card::Tempura, count), expected);
    }
    for (count, expected) in [0, 0, 0, 10, 10, 10, 20].into_iter().enumerate() {
        assert_eq!(score(Card::Sashimi, count), expected);
    }
    for (count, expected) in [0, 1, 3, 6, 10, 15, 15].into_iter().enumerate() {
        assert_eq!(score(Card::Dumpling, count), expected);
    }
    for (card, base) in [
        (Card::EggNigiri, 1),
        (Card::SalmonNigiri, 2),
        (Card::SquidNigiri, 3),
    ] {
        assert_eq!(score(card, 1), base);
        assert_eq!(
            score_cards(&[PlayedCard {
                card,
                on_wasabi: true,
            }]),
            3 * base
        );
    }
    for card in [
        Card::Maki1,
        Card::Maki2,
        Card::Maki3,
        Card::Wasabi,
        Card::Chopsticks,
        Card::Pudding,
    ] {
        assert_eq!(score(card, 1), 0, "{card:?} has no direct round points");
    }
    let mut public = SushiGoForwardModel
        .init_state(&Parameters { seed: 1 }, 2)
        .unwrap()
        .public;
    for card in [
        Card::Wasabi,
        Card::EggNigiri,
        Card::SalmonNigiri,
        Card::Wasabi,
        Card::SquidNigiri,
        Card::EggNigiri,
    ] {
        play(&mut public, 0, card);
    }
    assert_eq!(score_cards(&public.played[0]), 3 + 2 + 9 + 1);
    assert_eq!(
        public.played[0]
            .iter()
            .filter(|entry| entry.on_wasabi)
            .map(|entry| entry.card)
            .collect::<Vec<_>>(),
        [Card::EggNigiri, Card::SquidNigiri]
    );
    play(&mut public, 0, Card::Pudding);
    assert_eq!(public.pudding[0], 1);
    assert_eq!(score_cards(&public.played[0]), 15);
}

#[test]
fn maki_and_pudding_ties_cover_all_award_boundaries() {
    let run = |maki: &[usize], pudding: &[usize], players: usize| {
        let mut state = SushiGoForwardModel
            .init_state(&Parameters { seed: 2 }, players)
            .unwrap();
        state.public.round = 3;
        state.public.played = maki
            .iter()
            .map(|&rolls| {
                vec![
                    PlayedCard {
                        card: Card::Maki1,
                        on_wasabi: false,
                    };
                    rolls
                ]
            })
            .collect();
        state.public.pudding = pudding.to_vec();
        end_round(&mut state);
        (state.public.round_scores[0].clone(), state.public.scores)
    };
    assert_eq!(run(&[0, 0, 0], &[0, 0, 0], 3), (vec![0; 3], vec![0; 3]));
    assert_eq!(run(&[3, 2], &[0, 0], 2), (vec![6, 3], vec![6, 3]));
    assert_eq!(
        run(&[3, 3, 2, 1, 0], &[0; 5], 5),
        (vec![3, 3, 0, 0, 0], vec![3, 3, 0, 0, 0])
    );
    assert_eq!(
        run(&[3, 2, 2, 1], &[0; 4], 4),
        (vec![6, 1, 1, 0], vec![6, 1, 1, 0])
    );
    assert_eq!(
        run(&[5, 4, 4, 4, 4], &[0; 5], 5),
        (vec![6, 0, 0, 0, 0], vec![6, 0, 0, 0, 0])
    );
    assert_eq!(
        run(&[0, 0, 0, 0], &[3, 3, 1, 1], 4),
        (vec![0; 4], vec![3, 3, -3, -3])
    );
    assert_eq!(
        run(&[0, 0, 0], &[3, 2, 2], 3),
        (vec![0; 3], vec![6, -3, -3])
    );
    assert_eq!(run(&[0, 0], &[3, 1], 2), (vec![0; 2], vec![6, 0]));
}

#[test]
fn all_legal_chopsticks_orders_are_public_at_joint_boundaries() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 12 }, 3).unwrap();
    state.hands[0] = vec![Card::Tempura, Card::Tempura, Card::Sashimi, Card::Dumpling];
    state.hands[1] = vec![
        Card::Wasabi,
        Card::EggNigiri,
        Card::SalmonNigiri,
        Card::Pudding,
    ];
    state.hands[2] = vec![Card::Maki1, Card::Maki2, Card::Maki3, Card::Chopsticks];
    state.public.hand_size = 4;
    for seat in 0..3 {
        state.public.played[seat].push(PlayedCard {
            card: Card::Chopsticks,
            on_wasabi: false,
        });
        state.known_by[seat][seat] = Some(state.hands[seat].clone());
    }
    for focal in 0..3 {
        let before = model.observe(&state, PlayerId::new(focal)).unwrap();
        let legal = model.offered_actions(&before);
        let mut seen = std::collections::HashSet::new();
        for action in legal
            .iter()
            .copied()
            .filter(|action| matches!(action, Action::Chopsticks { .. }))
        {
            let mut next = state.clone();
            let mut choices = (0..3)
                .map(|seat| Action::Pick(state.hands[seat][0]))
                .collect::<Vec<_>>();
            choices[focal] = action;
            joint(&mut next, &choices);
            assert_eq!(
                next.public.last_reveal[focal],
                (PlayerId::new(focal), action)
            );
            assert!(seen.insert(next.public.last_reveal[focal].1.id()));
            for observer in 0..3 {
                let observation = model.observe(&next, PlayerId::new(observer)).unwrap();
                assert_eq!(observation.public.last_reveal, next.public.last_reveal);
                assert_eq!(observation.public.played, next.public.played);
            }
        }
        assert!(seen.len() >= 7);
    }
    let mut forward = state.clone();
    let mut reverse = state.clone();
    joint(
        &mut forward,
        &[
            Action::Chopsticks {
                first: Card::Sashimi,
                second: Card::Dumpling,
            },
            Action::Pick(Card::Wasabi),
            Action::Pick(Card::Maki1),
        ],
    );
    joint(
        &mut reverse,
        &[
            Action::Chopsticks {
                first: Card::Dumpling,
                second: Card::Sashimi,
            },
            Action::Pick(Card::Wasabi),
            Action::Pick(Card::Maki1),
        ],
    );
    assert_eq!(
        score_cards(&forward.public.played[0]),
        score_cards(&reverse.public.played[0])
    );
    assert_ne!(forward.public.last_reveal, reverse.public.last_reveal);
    for observer in 0..3 {
        assert_ne!(
            model.observe(&forward, PlayerId::new(observer)).unwrap(),
            model.observe(&reverse, PlayerId::new(observer)).unwrap()
        );
    }
}

#[test]
fn maki_first_tie_removes_second_and_two_player_pudding_has_no_penalty() {
    let model = SushiGoForwardModel;
    let mut tied = model.init_state(&Parameters { seed: 2 }, 3).unwrap();
    tied.public.round = 3;
    tied.public.played = vec![
        vec![PlayedCard {
            card: Card::Maki3,
            on_wasabi: false,
        }],
        vec![PlayedCard {
            card: Card::Maki3,
            on_wasabi: false,
        }],
        vec![PlayedCard {
            card: Card::Maki2,
            on_wasabi: false,
        }],
    ];
    end_round(&mut tied);
    assert_eq!(tied.public.round_scores[0], [3, 3, 0]);

    let mut two = model.init_state(&Parameters { seed: 2 }, 2).unwrap();
    two.public.round = 3;
    two.public.pudding = vec![3, 1];
    end_round(&mut two);
    assert_eq!(two.public.scores, [6, 0]);
}

#[test]
fn tied_total_points_use_pudding_then_share_victory() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 2 }, 3).unwrap();
    state.public.finished = true;
    state.public.scores = vec![20, 20, 18];
    state.public.pudding = vec![1, 2, 3];
    assert_eq!(
        model.player_result(&state, PlayerId::new(0)).unwrap(),
        PlayerResult::Loss
    );
    assert_eq!(
        model.player_result(&state, PlayerId::new(1)).unwrap(),
        PlayerResult::Win
    );
    assert_eq!(
        model.player_result(&state, PlayerId::new(2)).unwrap(),
        PlayerResult::Loss
    );
    state.public.pudding[0] = 2;
    assert_eq!(
        model.player_result(&state, PlayerId::new(0)).unwrap(),
        PlayerResult::Win
    );
    assert_eq!(
        model.player_result(&state, PlayerId::new(1)).unwrap(),
        PlayerResult::Win
    );
}

#[test]
fn observations_and_forks_preserve_known_hands_without_real_hidden_cards() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 5 }, 3).unwrap();
    let viewer = PlayerId::new(0);
    let before = model.observe(&state, viewer).unwrap();
    let mut changed = state.clone();
    changed.hands.swap(1, 2);
    assert_eq!(model.observe(&changed, viewer).unwrap(), before);
    let mut simulation_rng = SimulationRng::new(19);
    let fork = model.simulation_fork(&before, &mut simulation_rng).unwrap();
    assert_eq!(model.observe(&fork, viewer).unwrap(), before);
    assert_eq!(
        model.available_actions(&state, viewer).unwrap(),
        model.offered_actions(&before)
    );
    let choices = (0..3)
        .map(|p| Action::Pick(state.hands[p][0]))
        .collect::<Vec<_>>();
    joint(&mut state, &choices);
    let after = model.observe(&state, viewer).unwrap();
    let known = after
        .known_hands
        .iter()
        .filter(|entry| entry.is_some())
        .count();
    assert!(known >= 2);
    let fork = model.simulation_fork(&after, &mut simulation_rng).unwrap();
    assert_eq!(model.observe(&fork, viewer).unwrap(), after);
    assert_ne!(fork.chance_snapshot(), state.chance_snapshot());
}

#[test]
fn passing_knowledge_covers_only_hands_the_observer_has_held() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 5 }, 3).unwrap();
    let observer = PlayerId::new(0);
    let initial = model.observe(&state, observer).unwrap();
    assert!(initial.known_hands[0].is_some());
    assert!(initial.known_hands[1].is_none());
    assert!(initial.known_hands[2].is_none());

    for completed in 1..=2 {
        let choices = (0..3)
            .map(|seat| Action::Pick(state.hands[seat][0]))
            .collect::<Vec<_>>();
        joint(&mut state, &choices);
        let seen = model.observe(&state, observer).unwrap();
        assert_eq!(
            seen.known_hands
                .iter()
                .filter(|hand| hand.is_some())
                .count(),
            completed + 1
        );
        for (seat, &id) in state.public.hand_ids.iter().enumerate() {
            if let Some(known) = &seen.known_hands[id] {
                assert_eq!(known, &state.hands[seat]);
            }
        }
        if completed == 1 {
            assert!(seen.known_hands[2].is_none());
        }
    }
}

#[test]
fn hidden_worlds_preserve_observations_choices_and_snapshot_independence() {
    let model = SushiGoForwardModel;
    let state = model.init_state(&Parameters { seed: 5 }, 3).unwrap();
    let viewer = PlayerId::new(0);
    let before = model.observe(&state, viewer).unwrap();
    let mut first = model
        .simulation_fork(&before, &mut SimulationRng::new(19))
        .unwrap();
    let mut second = model
        .simulation_fork(&before, &mut SimulationRng::new(29))
        .unwrap();
    assert_ne!(first, second);
    for world in [&first, &second] {
        assert_eq!(model.observe(world, viewer).unwrap(), before);
        assert_eq!(
            model.offered_actions(&before),
            model.available_actions(world, viewer).unwrap()
        );
    }
    let focal = model.offered_actions(&before)[0];
    for world in [&mut first, &mut second] {
        let choices = [
            focal,
            Action::Pick(world.hands[1][0]),
            Action::Pick(world.hands[2][0]),
        ];
        joint(world, &choices);
        let observed = model.observe(world, viewer).unwrap();
        assert_eq!(observed.public.last_reveal.len(), 3);
        assert_eq!(observed.public.last_reveal[0].1, focal);
        assert_eq!(
            model.offered_actions(&observed),
            model.available_actions(world, viewer).unwrap()
        );
    }
    let checkpoint = first.snapshot();
    let restored = State::from_snapshot(&checkpoint).unwrap();
    assert_eq!(first, restored);
    let original_rng = restored.chance_snapshot();
    let action = model.available_actions(&first, viewer).unwrap()[0];
    let choices = [
        action,
        Action::Pick(first.hands[1][0]),
        Action::Pick(first.hands[2][0]),
    ];
    joint(&mut first, &choices);
    assert_eq!(restored, checkpoint);
    assert_eq!(restored.chance_snapshot(), original_rng);
    assert_ne!(first, restored);
}

#[test]
fn mcts_sushi_edges_end_after_complete_public_reveals() {
    let model = SushiGoForwardModel;
    let state = model.init_state(&Parameters { seed: 3 }, 2).unwrap();
    let observation = model.observe(&state, PlayerId::new(0)).unwrap();
    let actions = model.offered_actions(&observation);
    let mut registry = EvaluatorRegistry::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(PublicBoundaryValues)))
        .unwrap();
    let settings = Settings {
        seed: Some(14),
        rollout_length: Some(0),
        budgets: Budgets {
            iterations: Some(12),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Settings::default()
    };
    let report = MctsAgent::new(settings, &registry)
        .unwrap()
        .search(&model, PlayerId::new(0), &observation, &actions)
        .unwrap();
    assert_eq!(report.counters.completed_iterations, 12);
    assert_eq!(
        report
            .root_actions
            .iter()
            .map(|action| action.visits)
            .sum::<u64>(),
        12
    );
    assert!(state.public.last_reveal.is_empty());
}

#[test]
fn incomplete_choices_and_invalid_observations_preserve_state_and_rng() {
    let model = SushiGoForwardModel;
    let mut state = model.init_state(&Parameters { seed: 9 }, 3).unwrap();
    let original = state.clone();
    let first = Action::Pick(state.hands[0][0]);
    assert!(joint::apply(&model, &mut state, &[(PlayerId::new(0), first)]).is_err());
    assert_eq!(state, original);
    let mut observation = model.observe(&state, PlayerId::new(0)).unwrap();
    observation.hand.push(Card::Tempura);
    let mut random = SimulationRng::new(21);
    let checkpoint = random.snapshot();
    assert_eq!(
        model.simulation_fork(&observation, &mut random),
        Err(Error::InvalidObservation)
    );
    assert_eq!(random.snapshot(), checkpoint);
    let mut clone = state.clone();
    let choices = (0..3)
        .map(|p| Action::Pick(clone.hands[p][0]))
        .collect::<Vec<_>>();
    joint(&mut clone, &choices);
    assert_eq!(state, original);
    assert_ne!(clone, state);
}
