//! Independent expected effects, conservation checks and paired hidden-world tests.
use super::*;
use crate::core::{
    Decision, ForwardModel, PlayerId, PlayerResult,
    checkpoint::Checkpoint,
    random::{AgentRng, ChanceRng},
};
const P0: PlayerId = PlayerId::new(0);
const P1: PlayerId = PlayerId::new(1);
const P2: PlayerId = PlayerId::new(2);
const P3: PlayerId = PlayerId::new(3);
const MODEL: LoveLetterForwardModel = LoveLetterForwardModel;

fn parameters(target: u32) -> LoveLetterParameters {
    LoveLetterParameters {
        rules: LoveLetterRules {
            winning_tokens: [target; 3],
        },
        seed: 42,
        ..Default::default()
    }
}
fn remove(cards: &mut Vec<Card>, card: Card) -> Card {
    let i = cards
        .iter()
        .position(|&c| c == card)
        .expect("fixture has enough copies");
    cards.remove(i)
}
fn dealt(hands: &[Card], drawn: Card, target: u32) -> LoveLetterState {
    let mut remaining = Card::DECK.to_vec();
    for &card in hands {
        remove(&mut remaining, card);
    }
    remove(&mut remaining, drawn);
    let removed = remaining.remove(0);
    let mut deck = vec![removed];
    if hands.len() == 2 {
        for _ in 0..3 {
            deck.push(remaining.remove(0));
        }
    }
    deck.extend_from_slice(hands);
    deck.push(drawn);
    deck.extend(remaining);
    MODEL
        .init_with_deck(&parameters(target), hands.len(), &deck.try_into().unwrap())
        .unwrap()
}
fn play(s: &mut LoveLetterState, a: LoveLetterAction) {
    MODEL.next(s, s.public.current_player, &a).unwrap();
    invariants(s);
}
fn target(card: Card, p: PlayerId) -> LoveLetterAction {
    LoveLetterAction::Target { card, target: p }
}
fn offers(s: &LoveLetterState) -> Vec<LoveLetterAction> {
    let from_observation =
        MODEL.offered_actions(&MODEL.observe(s, s.public.current_player).unwrap());
    let from_state = MODEL.available_actions(s, s.public.current_player).unwrap();
    assert_eq!(from_state, from_observation);
    from_state
}
fn invariants(s: &LoveLetterState) {
    let mut cards: Vec<_> = s
        .hands
        .iter()
        .flatten()
        .copied()
        .chain(s.pile.iter().copied())
        .chain(s.removed)
        .chain(s.public.reserve.iter().copied())
        .chain(s.public.discards.iter().flatten().copied())
        .collect();
    cards.sort();
    let mut expected = Card::DECK;
    expected.sort();
    assert_eq!(cards, expected);
    assert_eq!(s.public.draw_count, s.pile.len());
    assert_eq!(s.public.removed_count, usize::from(s.removed.is_some()));
    assert_eq!(
        s.public.counters.move_count(),
        s.public.counters.completed_turn_count()
    );
    if MODEL.next_player(s) != Decision::Finished {
        assert_eq!(s.public.counters.round_count() + 1, s.public.round);
        assert!(!s.public.protected[s.public.current_player.index()]);
        for (seat, hand) in s.hands.iter().enumerate() {
            assert_eq!(
                hand.len(),
                if !s.public.active[seat] {
                    0
                } else if seat == s.public.current_player.index() {
                    2
                } else {
                    1
                }
            );
        }
    } else {
        assert_eq!(s.public.counters.round_count(), s.public.round);
    }
    for seat in 0..s.public.player_count() {
        assert!(
            s.private[seat]
                .iter()
                .all(|e| e.event < s.public.events.len())
        );
        if let Some(round) = &s.public.last_round {
            assert!(
                s.completed_private[seat]
                    .iter()
                    .all(|e| e.event < round.events.len())
            );
        }
    }
}

#[test]
fn selected_classic_deck_setup_domains_and_controlled_chance() {
    assert_eq!(Card::ALL.map(Card::rank), [1, 2, 3, 4, 5, 7, 8, 9]);
    assert_eq!(LoveLetterRules::default().winning_tokens, [6, 5, 4]);
    for n in 2..=4 {
        for first in 0..n {
            let p = LoveLetterParameters {
                seed: u64::MAX,
                first_player: PlayerId::new(first),
                ..Default::default()
            };
            let s = MODEL.init_state(&p, n).unwrap();
            invariants(&s);
            assert_eq!(s.public.reserve.len(), if n == 2 { 3 } else { 0 });
            assert_eq!(s.pile.len(), 16 - 1 - if n == 2 { 3 } else { 0 } - n - 1);
            assert_eq!(s.hands[first].len(), 2);
            assert_eq!(s.public.current_player, PlayerId::new(first));
            let controlled = MODEL.init_with_deck(&p, n, &Card::DECK).unwrap();
            assert_eq!(controlled.chance_snapshot(), s.chance_snapshot());
            assert_eq!(controlled.privileged_removed_card(), Some(Card::Guard));
        }
    }
    for n in [0, 1, 5] {
        assert!(MODEL.init_state(&Default::default(), n).is_err());
    }
    assert!(MODEL.init_state(&parameters(0), 2).is_err());
    assert!(
        MODEL
            .init_state(
                &LoveLetterParameters {
                    first_player: P3,
                    ..Default::default()
                },
                3
            )
            .is_err()
    );
    assert!(
        MODEL
            .init_with_deck(&Default::default(), 2, &[Card::Guard; 16])
            .is_err()
    );
}

#[test]
fn actions_are_complete_portable_and_do_not_prune_guard_guesses() {
    let s = dealt(&[Card::Guard, Card::Princess, Card::King], Card::Guard, 1);
    let actions = offers(&s);
    assert_eq!(actions.len(), 14);
    for p in [P1, P2] {
        for card in Card::ALL {
            if card != Card::Guard {
                assert!(actions.contains(&LoveLetterAction::Guess { target: p, card }));
            }
        }
    }
    assert!(!actions.contains(&LoveLetterAction::Guess {
        target: P1,
        card: Card::Guard
    }));
    assert!(
        MODEL
            .offered_actions(&MODEL.observe(&s, P1).unwrap())
            .is_empty()
    );
    let a = LoveLetterAction::Guess {
        target: P2,
        card: Card::Priest,
    };
    let mut copied = s.clone();
    MODEL.next(&mut copied, P0, &a).unwrap();
    assert_eq!(
        a,
        LoveLetterAction::Guess {
            target: P2,
            card: Card::Priest
        }
    );
}

#[test]
fn countess_is_forced_only_with_prince_or_king_and_never_reveals_the_other_card() {
    for card in [Card::Prince, Card::King] {
        let mut s = dealt(&[Card::Countess, Card::Guard, Card::Guard], card, 1);
        assert_eq!(offers(&s), vec![LoveLetterAction::Play(Card::Countess)]);
        play(&mut s, LoveLetterAction::Play(Card::Countess));
        assert_eq!(s.hands[0], vec![card]);
        assert!(
            !s.public.events.iter().any(
                |e| matches!(e,PublicEvent::Discarded{player,card:c} if *player==P0&&*c==card)
            )
        );
    }
    let s = dealt(
        &[Card::Countess, Card::Guard, Card::Guard],
        Card::Princess,
        1,
    );
    assert_eq!(
        offers(&s),
        vec![
            LoveLetterAction::Play(Card::Countess),
            LoveLetterAction::Play(Card::Princess)
        ]
    );
}

#[test]
fn guard_hits_misses_and_round_elimination_are_distinct_from_game_results() {
    let mut s = dealt(&[Card::Guard, Card::King, Card::Handmaid], Card::Priest, 2);
    play(
        &mut s,
        LoveLetterAction::Guess {
            target: P1,
            card: Card::King,
        },
    );
    assert!(!s.public.active[1]);
    assert!(s.hands[1].is_empty());
    assert_eq!(s.public.discards[1], vec![Card::King]);
    assert_eq!(s.public.current_player, P2);
    assert_eq!(MODEL.player_result(&s, P1).unwrap(), PlayerResult::Ongoing);
    let mut miss = dealt(&[Card::Guard, Card::King, Card::Handmaid], Card::Priest, 2);
    play(
        &mut miss,
        LoveLetterAction::Guess {
            target: P1,
            card: Card::Princess,
        },
    );
    assert!(miss.public.active.iter().all(|&v| v));
    assert!(miss.public.events.contains(&PublicEvent::GuardResult {
        actor: P0,
        target: P1,
        guess: Card::Princess,
        correct: false
    }));
}

#[test]
fn priest_and_baron_private_experiences_have_exact_audiences() {
    let mut s = dealt(
        &[Card::Priest, Card::King, Card::Handmaid],
        Card::Countess,
        2,
    );
    play(&mut s, target(Card::Priest, P1));
    let observation = MODEL.observe(&s, P0).unwrap();
    assert!(observation.private_events().iter().any(|e| e.information
        == PrivateInformation::Priest {
            target: P1,
            card: Card::King
        }));
    for seat in [P1, P2] {
        assert!(
            !MODEL
                .observe(&s, seat)
                .unwrap()
                .private_events()
                .iter()
                .any(|e| matches!(e.information, PrivateInformation::Priest { .. }))
        );
    }
    // Knowledge remains an event at that time, not a claim about the current hand.
    assert_eq!(s.hands[1].len(), 2);
    for (own, other, loser) in [
        (Card::King, Card::Prince, Some(P1)),
        (Card::Priest, Card::King, Some(P0)),
        (Card::Guard, Card::Guard, None),
    ] {
        let mut s = dealt(&[Card::Baron, other, Card::Handmaid], own, 2);
        play(&mut s, target(Card::Baron, P1));
        assert!(s.public.events.contains(&PublicEvent::Compared {
            actor: P0,
            target: P1,
            eliminated: loser
        }));
        for (viewer, opponent, a, b) in [(P0, P1, own, other), (P1, P0, other, own)] {
            assert!(
                MODEL
                    .observe(&s, viewer)
                    .unwrap()
                    .private_events()
                    .iter()
                    .any(|e| e.information
                        == PrivateInformation::Comparison {
                            opponent,
                            own_card: a,
                            other_card: b
                        })
            );
        }
        assert!(
            !MODEL
                .observe(&s, P2)
                .unwrap()
                .private_events()
                .iter()
                .any(|e| matches!(e.information, PrivateInformation::Comparison { .. }))
        );
    }
}

#[test]
fn king_swaps_hands_and_each_participant_remembers_the_exchange() {
    let mut s = dealt(
        &[Card::King, Card::Princess, Card::Handmaid],
        Card::Priest,
        2,
    );
    play(&mut s, target(Card::King, P1));
    assert_eq!(s.hands[0], vec![Card::Princess]);
    assert_eq!(s.hands[1][0], Card::Priest);
    for (viewer, other, given, received) in [
        (P0, P1, Card::Priest, Card::Princess),
        (P1, P0, Card::Princess, Card::Priest),
    ] {
        assert!(
            MODEL
                .observe(&s, viewer)
                .unwrap()
                .private_events()
                .iter()
                .any(|e| e.information
                    == PrivateInformation::Exchange {
                        other,
                        given,
                        received
                    })
        );
    }
    assert!(
        !MODEL
            .observe(&s, P2)
            .unwrap()
            .private_events()
            .iter()
            .any(|e| matches!(e.information, PrivateInformation::Exchange { .. }))
    );
}

#[test]
fn protection_expires_at_next_turn_and_no_target_plays_are_explicit() {
    let mut s = dealt(
        &[Card::Handmaid, Card::Priest, Card::Countess],
        Card::King,
        3,
    );
    play(&mut s, LoveLetterAction::Play(Card::Handmaid));
    assert!(s.public.protected[0]);
    assert!(!offers(&s).iter().any(|a|matches!(a,LoveLetterAction::Target{target,..}|LoveLetterAction::Guess{target,..} if *target==P0)));
    play(&mut s, target(Card::Priest, P2));
    play(&mut s, LoveLetterAction::Play(Card::Countess));
    assert_eq!(s.public.current_player, P0);
    assert!(!s.public.protected[0]);
    for card in [Card::Guard, Card::Priest, Card::Baron, Card::King] {
        let mut s = dealt(&[card, Card::Guard, Card::Guard], Card::Handmaid, 3);
        s.public.protected[1] = true;
        s.public.protected[2] = true;
        assert!(offers(&s).contains(&LoveLetterAction::Play(card)));
        play(&mut s, LoveLetterAction::Play(card));
    }
    let mut s = dealt(&[Card::Prince, Card::Guard, Card::Guard], Card::Handmaid, 3);
    s.public.protected[1] = true;
    s.public.protected[2] = true;
    assert!(offers(&s).contains(&target(Card::Prince, P0)));
    assert!(!offers(&s).contains(&LoveLetterAction::Play(Card::Prince)));
    play(&mut s, target(Card::Prince, P0));
    assert_eq!(s.public.discards[0], vec![Card::Prince, Card::Handmaid]);
}

#[test]
fn prince_replacement_self_target_and_princess_discard() {
    let mut s = dealt(&[Card::Prince, Card::King, Card::Handmaid], Card::Priest, 3);
    let replacement = s.pile[0];
    play(&mut s, target(Card::Prince, P1));
    assert_eq!(s.public.discards[1], vec![Card::King]);
    assert_eq!(s.hands[1][0], replacement);
    let mut s = dealt(
        &[Card::Prince, Card::Princess, Card::Handmaid],
        Card::Priest,
        3,
    );
    let draws = s.pile.len();
    play(&mut s, target(Card::Prince, P1));
    assert!(!s.public.active[1]);
    assert_eq!(s.pile.len(), draws - 1); // next actor's draw only
    assert!(s.hands[1].is_empty());
    let mut s = dealt(
        &[Card::Princess, Card::King, Card::Handmaid],
        Card::Priest,
        3,
    );
    play(&mut s, LoveLetterAction::Play(Card::Princess));
    assert!(!s.public.active[0]);
    assert_eq!(s.public.discards[0], vec![Card::Princess, Card::Priest]);
}

/// Move unneeded remaining cards to publicly discarded piles to create a late-round
/// fixture while retaining the exact multiset. These fixtures test final-turn rules.
fn empty_pile(s: &mut LoveLetterState) {
    while let Some(c) = s.pile.pop_front() {
        s.public.discards[0].push(c);
    }
    s.public.draw_count = 0;
}
#[test]
fn removed_card_is_consumed_exactly_once_and_not_revealed_until_rules_allow_it() {
    let mut s = dealt(&[Card::Prince, Card::King, Card::Handmaid], Card::Priest, 1);
    empty_pile(&mut s);
    let removed = s.removed.unwrap();
    play(&mut s, target(Card::Prince, P1));
    assert_eq!(s.removed, None);
    assert_eq!(s.public.removed_count, 0);
    assert_eq!(s.hands[1], vec![removed]);
    assert!(
        s.public
            .last_round
            .as_ref()
            .unwrap()
            .summary
            .revealed_hands
            .contains(&(P1, removed))
    );
    let private = MODEL.observe(&s, P1).unwrap();
    assert!(private.private_events().iter().any(|e| e.information
        == PrivateInformation::Received {
            card: removed,
            source: DrawSource::Removed
        }));
}

#[test]
fn shared_round_and_game_wins_ignore_discard_totals_and_do_not_create_a_draw() {
    let mut s = dealt(
        &[Card::Countess, Card::Prince, Card::Priest],
        Card::Prince,
        1,
    );
    // Countess+Prince is forced. Both remaining Princes tie despite different discards.
    empty_pile(&mut s);
    let chance_before = s.chance_snapshot();
    play(&mut s, LoveLetterAction::Play(Card::Countess));
    assert_eq!(s.chance_snapshot(), chance_before); // terminal play starts no new round
    assert_eq!(s.public.outcome, LoveLetterOutcome::Won(vec![P0, P1]));
    assert_eq!(s.public.tokens, vec![1, 1, 0]);
    assert_eq!(s.public.counters.move_count(), 1);
    assert_eq!(s.public.counters.round_count(), 1);
    assert_eq!(s.public.counters.turn_count_in_round(), 0);
    for p in [P0, P1] {
        assert_eq!(MODEL.player_result(&s, p).unwrap(), PlayerResult::Win);
    }
    assert_eq!(MODEL.player_result(&s, P2).unwrap(), PlayerResult::Loss);
    assert_eq!(MODEL.next_player(&s), Decision::Finished);
    assert!(offers(&s).is_empty());
    assert!(
        s.public
            .last_round
            .as_ref()
            .unwrap()
            .summary
            .revealed_hands
            .contains(&(P0, Card::Prince))
    );
}

#[test]
fn last_survivor_wins_without_exposing_their_hand_and_everybody_returns_next_round() {
    let mut terminal = dealt(&[Card::Guard, Card::Princess], Card::King, 1);
    play(
        &mut terminal,
        LoveLetterAction::Guess {
            target: P1,
            card: Card::Princess,
        },
    );
    let summary = &terminal.public.last_round.as_ref().unwrap().summary;
    assert_eq!(summary.reason, RoundEnd::LastPlayer);
    assert!(summary.revealed_hands.is_empty());
    assert_eq!(terminal.public.outcome, LoveLetterOutcome::Won(vec![P0]));
    let mut ongoing = dealt(&[Card::Guard, Card::Princess], Card::King, 2);
    play(
        &mut ongoing,
        LoveLetterAction::Guess {
            target: P1,
            card: Card::Princess,
        },
    );
    assert_eq!(ongoing.public.round, 2);
    assert_eq!(ongoing.public.current_player, P0);
    assert!(ongoing.public.active.iter().all(|&v| v));
    assert!(ongoing.public.protected.iter().all(|&v| !v));
    assert_eq!(ongoing.public.tokens, vec![1, 0]);
    assert!(ongoing.public.discards.iter().all(Vec::is_empty));
    assert!(
        MODEL
            .observe(&ongoing, P0)
            .unwrap()
            .last_round_private_events()
            .iter()
            .any(|e| matches!(
                e.information,
                PrivateInformation::Received {
                    card: Card::King,
                    ..
                }
            ))
    );
}

#[test]
fn tied_round_starters_use_reproducible_chance_and_only_tied_winners() {
    let mut seen = std::collections::HashSet::new();
    for seed in 0..40 {
        let mut s = dealt(
            &[Card::Countess, Card::Prince, Card::Priest],
            Card::Prince,
            2,
        );
        empty_pile(&mut s);
        s.random = ChanceRng::new(seed);
        let mut repeated = s.clone();
        play(&mut s, LoveLetterAction::Play(Card::Countess));
        play(&mut repeated, LoveLetterAction::Play(Card::Countess));
        assert_eq!(s, repeated);
        assert!([P0, P1].contains(&s.public.current_player));
        assert_eq!(
            s.public.last_round.as_ref().unwrap().summary.winners,
            vec![P0, P1]
        );
        seen.insert(s.public.current_player);
    }
    assert_eq!(seen.len(), 2);
}

#[test]
fn inaccessible_worlds_rng_and_other_private_events_do_not_change_permitted_inputs() {
    let s = dealt(
        &[Card::Guard, Card::Priest, Card::King, Card::Handmaid],
        Card::Countess,
        3,
    );
    let mut deck = vec![s.removed.unwrap()];
    for hand in &s.hands {
        deck.push(hand[0]);
    }
    deck.push(s.hands[0][1]);
    deck.extend(s.pile.iter().copied());
    deck.swap(2, 3);
    deck.swap(0, 4);
    deck[6..].reverse();
    // Initialize the counterpart independently: every player's own initial
    // private experiences must also agree with their hypothetical hand.
    let twin = MODEL
        .init_with_deck(
            &LoveLetterParameters {
                seed: u64::MAX,
                ..parameters(3)
            },
            4,
            &deck.try_into().unwrap(),
        )
        .unwrap();
    invariants(&s);
    invariants(&twin);
    assert_ne!(s, twin);
    let before = s.chance_snapshot();
    let a = MODEL.observe(&s, P0).unwrap();
    let b = MODEL.observe(&twin, P0).unwrap();
    assert_eq!(a, b);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(MODEL.offered_actions(&a), MODEL.offered_actions(&b));
    assert_eq!(
        MODEL.available_actions(&s, P0).unwrap(),
        MODEL.available_actions(&twin, P0).unwrap()
    );
    assert_eq!(s.chance_snapshot(), before);
    let mut agent1 = crate::agents::RandomAgent::new(123);
    let mut agent2 = crate::agents::RandomAgent::new(123);
    use crate::agents::Agent;
    assert_eq!(
        agent1
            .choose(&MODEL, P0, &a, &MODEL.offered_actions(&a))
            .unwrap(),
        agent2
            .choose(&MODEL, P0, &b, &MODEL.offered_actions(&b))
            .unwrap()
    );
    // A non-participant sees the same Priest action and public transcript,
    // although the privately revealed value is different in the two worlds.
    let mut a = dealt(
        &[Card::Priest, Card::King, Card::Handmaid, Card::Countess],
        Card::Guard,
        3,
    );
    let mut b = dealt(
        &[Card::Priest, Card::Countess, Card::Handmaid, Card::King],
        Card::Guard,
        3,
    );
    play(&mut a, target(Card::Priest, P1));
    play(&mut b, target(Card::Priest, P1));
    assert_eq!(
        MODEL.observe(&a, P2).unwrap(),
        MODEL.observe(&b, P2).unwrap()
    );
    assert_ne!(
        MODEL.observe(&a, P0).unwrap(),
        MODEL.observe(&b, P0).unwrap()
    );
}

#[test]
fn invalid_actions_and_observers_leave_state_and_rng_unchanged() {
    let s = dealt(&[Card::Priest, Card::Princess, Card::Guard], Card::Guard, 2);
    for (actor, action) in [
        (P1, target(Card::Priest, P0)),
        (P0, target(Card::Priest, P0)),
        (
            P0,
            LoveLetterAction::Guess {
                target: P1,
                card: Card::Guard,
            },
        ),
        (P0, LoveLetterAction::Play(Card::King)),
        (P3, LoveLetterAction::Play(Card::Priest)),
    ] {
        let mut copy = s.clone();
        assert!(MODEL.next(&mut copy, actor, &action).is_err());
        assert_eq!(copy, s);
    }
    assert!(MODEL.observe(&s, P3).is_err());
    let mut terminal = dealt(&[Card::Guard, Card::Princess], Card::King, 1);
    play(
        &mut terminal,
        LoveLetterAction::Guess {
            target: P1,
            card: Card::Princess,
        },
    );
    let before = terminal.clone();
    assert_eq!(
        MODEL.next(&mut terminal, P0, &LoveLetterAction::Play(Card::King)),
        Err(LoveLetterError::Finished)
    );
    assert_eq!(terminal, before);
}

#[test]
fn internal_overflow_is_reported_without_rolling_back_legal_action_effects() {
    // Sibling tests may construct private overflow fixtures; external callers cannot.
    let mut overflow = dealt(&[Card::Guard, Card::Princess], Card::King, 2);
    // This deliberately corrupted fixture triggers an internal failure after
    // resolving a legal Guard action. The failed state must be discarded.
    overflow.public.tokens[0] = u32::MAX;
    let before = overflow.clone();
    assert_eq!(
        MODEL.next(
            &mut overflow,
            P0,
            &LoveLetterAction::Guess {
                target: P1,
                card: Card::Princess
            }
        ),
        Err(LoveLetterError::CounterOverflow)
    );
    assert_ne!(overflow, before);
    assert!(
        LoveLetterError::CounterOverflow
            .to_string()
            .contains("counter overflow")
    );
    #[cfg(feature = "replay")]
    {
        let mut overflow = dealt(&[Card::Priest, Card::Princess, Card::Guard], Card::Guard, 2);
        overflow.public.counters =
            crate::core::progress::TurnCounters::from_counts([usize::MAX, usize::MAX, 0, 0])
                .unwrap();
        let before = overflow.clone();
        assert_eq!(
            MODEL.next(&mut overflow, P0, &offers(&before)[0]),
            Err(LoveLetterError::CounterOverflow)
        );
        assert_eq!(overflow, before);
    }
}

#[test]
fn public_effects_do_not_reveal_forced_countess_or_comparison_and_exchange_values() {
    let mut a = dealt(&[Card::Countess, Card::Guard, Card::Priest], Card::King, 3);
    let mut b = dealt(
        &[Card::Countess, Card::Guard, Card::Priest],
        Card::Prince,
        3,
    );
    play(&mut a, LoveLetterAction::Play(Card::Countess));
    play(&mut b, LoveLetterAction::Play(Card::Countess));
    assert_eq!(
        MODEL.observe(&a, P2).unwrap(),
        MODEL.observe(&b, P2).unwrap()
    );
    let mut voluntary = dealt(
        &[Card::Countess, Card::Guard, Card::Priest],
        Card::Handmaid,
        3,
    );
    play(&mut voluntary, LoveLetterAction::Play(Card::Countess));
    assert_eq!(
        MODEL.observe(&a, P2).unwrap(),
        MODEL.observe(&voluntary, P2).unwrap()
    );
    let mut a = dealt(
        &[Card::Baron, Card::Priest, Card::Countess],
        Card::Handmaid,
        3,
    );
    let mut b = dealt(&[Card::Baron, Card::Priest, Card::Countess], Card::King, 3);
    play(&mut a, target(Card::Baron, P1));
    play(&mut b, target(Card::Baron, P1));
    assert_eq!(
        MODEL.observe(&a, P2).unwrap(),
        MODEL.observe(&b, P2).unwrap()
    );
    assert_ne!(
        MODEL.observe(&a, P1).unwrap(),
        MODEL.observe(&b, P1).unwrap()
    );
    let mut a = dealt(&[Card::King, Card::Princess, Card::Guard], Card::Priest, 3);
    let mut b = dealt(
        &[Card::King, Card::Countess, Card::Guard],
        Card::Handmaid,
        3,
    );
    play(&mut a, target(Card::King, P1));
    play(&mut b, target(Card::King, P1));
    assert_eq!(
        MODEL.observe(&a, P2).unwrap(),
        MODEL.observe(&b, P2).unwrap()
    );
    let mut a = dealt(
        &[Card::Guard, Card::Prince, Card::Handmaid],
        Card::Priest,
        3,
    );
    let mut b = dealt(&[Card::Guard, Card::King, Card::Handmaid], Card::Priest, 3);
    for s in [&mut a, &mut b] {
        play(
            s,
            LoveLetterAction::Guess {
                target: P1,
                card: Card::Princess,
            },
        );
    }
    assert_eq!(
        MODEL.observe(&a, P0).unwrap(),
        MODEL.observe(&b, P0).unwrap()
    );
}

#[test]
fn inspection_history_preserves_experiences_across_exchanges_and_replacement() {
    let mut s = dealt(
        &[Card::Priest, Card::King, Card::Princess],
        Card::Handmaid,
        3,
    );
    play(&mut s, target(Card::Priest, P2));
    play(&mut s, target(Card::King, P2));
    assert_eq!(s.hands[1], vec![Card::Princess]);
    let view = MODEL.observe(&s, P0).unwrap();
    assert!(view.private_events().iter().any(|e| e.information
        == PrivateInformation::Priest {
            target: P2,
            card: Card::Princess
        }));
    assert!(view.public().events.contains(&PublicEvent::Exchanged {
        actor: P1,
        target: P2
    }));
    assert!(
        !view
            .private_events()
            .iter()
            .any(|e| matches!(e.information, PrivateInformation::Exchange { .. }))
    );
    let mut s = dealt(
        &[Card::Priest, Card::Prince, Card::Handmaid],
        Card::Countess,
        3,
    );
    play(&mut s, target(Card::Priest, P1));
    play(&mut s, target(Card::Prince, P1));
    let view = MODEL.observe(&s, P0).unwrap();
    assert!(view.private_events().iter().any(|e| e.information
        == PrivateInformation::Priest {
            target: P1,
            card: Card::Prince
        }));
    assert!(view.public().events.contains(&PublicEvent::Discarded {
        player: P1,
        card: Card::Prince
    }));
    assert!(!view.private_events().iter().any(|e| matches!(
        e.information,
        PrivateInformation::Received {
            card: Card::Guard,
            ..
        }
    )));
    // Own the returned buffers: even editing a local view cannot change the state.
    let before = s.clone();
    let mut local = view;
    local.hand.clear();
    local.private.clear();
    local.public.tokens[0] = 99;
    assert_eq!(s, before);
}

#[test]
fn every_offered_initial_transition_preserves_the_classic_card_multiset() {
    let mut transitions = 0;
    for a in Card::ALL {
        for b in Card::ALL {
            for c in Card::ALL {
                for d in Card::ALL {
                    let mut available = Card::DECK.to_vec();
                    let mut valid = true;
                    for card in [a, b, c, d] {
                        if let Some(i) = available.iter().position(|&v| v == card) {
                            available.remove(i);
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let s = dealt(&[a, b, c], d, 1);
                    let choices = offers(&s);
                    assert!(!choices.is_empty());
                    for action in choices {
                        let mut copy = s.clone();
                        play(&mut copy, action);
                        transitions += 1;
                    }
                }
            }
        }
    }
    assert!(transitions > 10_000);
}

#[test]
fn exact_snapshots_complete_seeded_games_and_preserve_future_round_chance() {
    for n in 2..=4 {
        for seed in [0, 42, u64::MAX] {
            let p = LoveLetterParameters {
                seed,
                ..Default::default()
            };
            let mut s = MODEL.init_state(&p, n).unwrap();
            let mut rng = AgentRng::new(seed);
            let mut moves = 0;
            while let Decision::Player(actor) = MODEL.next_player(&s) {
                assert!(moves < 3000);
                invariants(&s);
                let before = s.clone();
                let snapshot = s.snapshot();
                let mut copy = LoveLetterState::from_snapshot(&snapshot).unwrap();
                let a = MODEL.available_actions(&s, actor).unwrap();
                let action = a[rng.index(a.len()).unwrap()];
                MODEL.next(&mut copy, actor, &action).unwrap();
                assert_eq!(s, before);
                MODEL.next(&mut s, actor, &action).unwrap();
                assert_eq!(s, copy);
                moves += 1;
            }
            invariants(&s);
            assert!(!s.public.outcome.winners().is_empty());
            assert!(s.public.counters.round_count() >= p.rules.winning_tokens[n - 2] as usize);
        }
    }
}
