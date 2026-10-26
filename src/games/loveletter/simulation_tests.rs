use super::*;
use crate::core::{
    Decision, ForwardModel, PlayerId,
    checkpoint::Checkpoint,
    random::{AgentRng, SimulationRng},
    simulation::SimulationModel,
};

fn check_cards(s: &LoveLetterState) {
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
    let mut deck = Card::DECK;
    deck.sort();
    assert_eq!(cards, deck);
}
#[test]
fn sampled_rounds_replay_all_observer_knowledge_and_resume_exactly() {
    let model = LoveLetterForwardModel;
    let mut sampling = SimulationRng::new(700);
    for n in 2..=4 {
        for seed in [0, 42, 777] {
            let mut state = model
                .init_state(
                    &LoveLetterParameters {
                        seed,
                        ..Default::default()
                    },
                    n,
                )
                .unwrap();
            let mut choices = AgentRng::new(seed + 100);
            let mut turns = 0;
            loop {
                let saved = state.clone();
                for seat in 0..n {
                    let observation = model.observe(&state, PlayerId::new(seat)).unwrap();
                    let mut restored = SimulationRng::from_snapshot(&sampling.snapshot()).unwrap();
                    let fork = model
                        .simulation_fork(&observation, &mut sampling)
                        .unwrap_or_else(|e| {
                            panic!("n={n} seed={seed} turn={turns} observer={seat}: {e}")
                        });
                    let again = model.simulation_fork(&observation, &mut restored).unwrap();
                    assert_eq!(fork, again);
                    assert_eq!(sampling.snapshot(), restored.snapshot());
                    assert_eq!(
                        model.observe(&fork, PlayerId::new(seat)).unwrap(),
                        observation
                    );
                    check_cards(&fork);
                    // A fork is itself a valid source of observations for other seats.
                    if turns % 13 == 0 {
                        let other = PlayerId::new((seat + 1) % n);
                        let other_o = model.observe(&fork, other).unwrap();
                        let nested = model.simulation_fork(&other_o, &mut sampling).unwrap();
                        assert_eq!(model.observe(&nested, other).unwrap(), other_o);
                    }
                }
                assert_eq!(state, saved);
                let Decision::Player(actor) = model.next_player(&state) else {
                    break;
                };
                let actions = model.offered_actions(&model.observe(&state, actor).unwrap());
                model
                    .next(
                        &mut state,
                        actor,
                        &actions[choices.index(actions.len()).unwrap()],
                    )
                    .unwrap();
                turns += 1;
                assert!(turns < 3000);
            }
        }
    }
}
#[test]
fn sampling_failure_is_atomic_and_unknown_removed_cards_vary() {
    let model = LoveLetterForwardModel;
    let state = model
        .init_state(&LoveLetterParameters::default(), 3)
        .unwrap();
    let o = model.observe(&state, PlayerId::new(0)).unwrap();
    let mut rng = SimulationRng::new(20);
    let before = rng.snapshot();
    assert_eq!(
        model.simulation_fork_with_budget(&o, &mut rng, 0),
        Err(LoveLetterError::SamplingBudgetExceeded)
    );
    assert_eq!(before, rng.snapshot());
    let mut invalid = o.clone();
    invalid.hand = vec![Card::Princess, Card::Princess];
    assert!(model.simulation_fork(&invalid, &mut rng).is_err());
    assert_eq!(before, rng.snapshot());
    let mut removed = std::collections::HashSet::new();
    let mut worlds = std::collections::HashSet::new();
    for _ in 0..40 {
        let fork = model.simulation_fork(&o, &mut rng).unwrap();
        removed.insert(fork.removed);
        worlds.insert(fork.hands.clone());
        assert_ne!(fork.chance_snapshot(), state.chance_snapshot());
    }
    assert!(removed.len() > 1);
    assert!(worlds.len() > 1);
}

#[test]
fn initial_hidden_rank_pairs_match_an_independent_multiset_oracle() {
    // Both our cards and the three reserve cards are known. At this initial
    // decision every pair left in the multiset is possible for (removed,opponent).
    let prefix = [
        Card::Guard,
        Card::King,
        Card::Countess,
        Card::Princess,
        Card::Guard,
        Card::Baron,
        Card::Priest,
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
    let state = model
        .init_with_deck(&LoveLetterParameters::default(), 2, &deck)
        .unwrap();
    let o = model.observe(&state, PlayerId::new(0)).unwrap();
    let mut unknown = Card::DECK.to_vec();
    for card in o.hand.iter().chain(&o.public.reserve) {
        let i = unknown.iter().position(|c| c == card).unwrap();
        unknown.remove(i);
    }
    let mut expected = std::collections::HashSet::new();
    for (i, &a) in unknown.iter().enumerate() {
        for (j, &b) in unknown.iter().enumerate() {
            if i != j {
                expected.insert((a, b));
            }
        }
    }
    assert_eq!(expected.len(), 24);
    let mut actual = std::collections::HashSet::new();
    let mut random = SimulationRng::new(81);
    for _ in 0..1024 {
        let fork = model.simulation_fork(&o, &mut random).unwrap();
        let pair = (fork.removed.unwrap(), fork.hands[1][0]);
        assert!(expected.contains(&pair));
        actual.insert(pair);
    }
    assert_eq!(actual, expected);
}

#[test]
fn search_inputs_and_choices_are_independent_of_the_actual_hidden_world() {
    use crate::agents::{Agent, MctsAgent, SampledOslaAgent};
    use crate::core::evaluation::Evaluator;
    use crate::search::mcts::{Budgets, NativeEvaluators, Settings};
    use std::{cell::RefCell, rc::Rc};
    struct Capture(Rc<RefCell<Vec<LoveLetterState>>>);
    impl Evaluator<LoveLetterState> for Capture {
        type Error = LoveLetterError;
        fn evaluate(&mut self, s: &LoveLetterState, p: PlayerId) -> Result<f64, Self::Error> {
            self.0.borrow_mut().push(s.clone());
            LoveLetterHeuristic.evaluate(s, p)
        }
    }
    let model = LoveLetterForwardModel;
    let real = model
        .init_state(&LoveLetterParameters::default(), 3)
        .unwrap();
    let actor = PlayerId::new(0);
    let o = model.observe(&real, actor).unwrap();
    let alternative = model
        .simulation_fork(&o, &mut SimulationRng::new(99))
        .unwrap();
    assert_ne!(real, alternative);
    let other_o = model.observe(&alternative, actor).unwrap();
    assert_eq!(o, other_o);
    let actions = model.offered_actions(&o);
    let left = Rc::new(RefCell::new(Vec::new()));
    let right = Rc::new(RefCell::new(Vec::new()));
    let mut a = SampledOslaAgent::new(
        Capture(left.clone()),
        42,
        std::num::NonZeroU32::new(4).unwrap(),
    );
    let mut b = SampledOslaAgent::new(
        Capture(right.clone()),
        42,
        std::num::NonZeroU32::new(4).unwrap(),
    );
    assert_eq!(
        a.choose(&model, actor, &o, &actions).unwrap(),
        b.choose(&model, actor, &other_o, &actions).unwrap()
    );
    assert!(!left.borrow().is_empty());
    assert_eq!(*left.borrow(), *right.borrow());
    let registry = LoveLetterForwardModel::native_evaluators();
    let settings = Settings {
        seed: Some(55),
        budgets: Budgets {
            iterations: Some(32),
            time_ms: None,
            forward_model_calls: None,
        },
        ..Default::default()
    };
    let mut a = MctsAgent::new(settings.clone(), &registry).unwrap();
    let mut b = MctsAgent::new(settings, &registry).unwrap();
    let first = a.search(&model, actor, &o, &actions).unwrap();
    let second = b.search(&model, actor, &other_o, &actions).unwrap();
    assert_eq!(first.selected_index, second.selected_index);
    assert_eq!(first.root_actions, second.root_actions);
    assert_eq!(first.counters, second.counters);
    assert_eq!(a.snapshot().unwrap(), b.snapshot().unwrap());
}

#[test]
fn heuristic_uses_only_own_information_and_keeps_shared_terminal_values() {
    use crate::core::{PlayerResult, evaluation::Evaluator};
    let model = LoveLetterForwardModel;
    let mut state = model
        .init_state(&LoveLetterParameters::default(), 3)
        .unwrap();
    let p = PlayerId::new(0);
    let value = LoveLetterHeuristic.evaluate(&state, p).unwrap();
    state.hands[1] = vec![Card::Princess];
    state.hands[2] = vec![Card::King];
    assert_eq!(value, LoveLetterHeuristic.evaluate(&state, p).unwrap());
    state.public.active[0] = false;
    state.hands[0].clear();
    state.public.tokens[0] = 2;
    assert!((LoveLetterHeuristic.evaluate(&state, p).unwrap() - 0.28).abs() < 1e-12);
    state.public.outcome = LoveLetterOutcome::Won(vec![p, PlayerId::new(1)]);
    for seat in 0..3 {
        let p = PlayerId::new(seat);
        assert_eq!(
            state.public.player_result(p).unwrap(),
            if seat < 2 {
                PlayerResult::Win
            } else {
                PlayerResult::Loss
            }
        );
        assert_eq!(
            LoveLetterHeuristic.evaluate(&state, p).unwrap(),
            if seat < 2 { 1.0 } else { -1.0 }
        );
    }
}
