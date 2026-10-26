//! Bounded randomized constraint search over the current round's shuffled deck.
//! A slot denotes a position in that shuffle, never an actual game's card ID.
use super::*;
use crate::core::{
    ForwardModel,
    checkpoint::Checkpoint,
    random::{ChanceRng, SimulationRng},
    simulation::SimulationModel,
};
use std::cmp::Ordering;

/// Maximum constraint-search nodes per fork (including event and deck branches).
pub const DEFAULT_SAMPLING_BUDGET: usize = 100_000;
const ALL: u16 = 0b11_1011_1110; // ranks 1,2,3,4,5,7,8,9
fn bit(c: Card) -> u16 {
    1 << c.rank()
}

#[derive(Clone)]
struct Constraints {
    domains: [u16; 16],
    hands: Vec<Vec<usize>>,
    next: usize,
    comparisons: Vec<(usize, usize, Ordering)>,
}
impl Constraints {
    fn constrain(&mut self, slot: usize, allowed: u16) {
        self.domains[slot] &= allowed;
    }
    fn held(&self, player: crate::core::PlayerId) -> Option<usize> {
        match self.hands.get(player.index())?.as_slice() {
            [slot] => Some(*slot),
            _ => None,
        }
    }
    fn propagate(&mut self) -> bool {
        loop {
            let before = self.domains;
            for &(a, b, order) in &self.comparisons {
                let mut da = 0;
                let mut db = 0;
                for x in Card::ALL {
                    for y in Card::ALL {
                        if self.domains[a] & bit(x) != 0
                            && self.domains[b] & bit(y) != 0
                            && x.cmp(&y) == order
                        {
                            da |= bit(x);
                            db |= bit(y);
                        }
                    }
                }
                self.domains[a] &= da;
                self.domains[b] &= db;
            }
            for card in Card::ALL {
                let b = bit(card);
                let count = Card::DECK.iter().filter(|&&c| c == card).count();
                let fixed = self.domains.iter().filter(|&&d| d == b).count();
                let possible = self.domains.iter().filter(|&&d| d & b != 0).count();
                if fixed > count || possible < count {
                    return false;
                }
                if fixed == count {
                    for d in &mut self.domains {
                        if *d != b {
                            *d &= !b;
                        }
                    }
                } else if possible == count {
                    for d in &mut self.domains {
                        if *d & b != 0 {
                            *d = b;
                        }
                    }
                }
            }
            if self.domains.contains(&0) {
                return false;
            }
            if self.domains == before {
                return true;
            }
        }
    }
}
struct Search<'a> {
    observation: &'a LoveLetterObservation,
    random: ChanceRng,
    remaining: usize,
    exhausted: bool,
}
impl Search<'_> {
    fn node(&mut self) -> bool {
        if self.remaining == 0 {
            self.exhausted = true;
            false
        } else {
            self.remaining -= 1;
            true
        }
    }
    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.random.index(i + 1).expect("nonempty choices"));
        }
    }
    fn events(&mut self, mut c: Constraints, index: usize) -> Option<LoveLetterState> {
        if !self.node() || !c.propagate() {
            return None;
        }
        let o = self.observation;
        let Some(event) = o.public.events.get(index) else {
            let hand = c.hands[o.observer.index()].clone();
            if hand.len() != o.hand.len() {
                return None;
            }
            for (&slot, &card) in hand.iter().zip(&o.hand) {
                c.constrain(slot, bit(card));
            }
            return self.assign(c);
        };
        match event {
            PublicEvent::Received { player, source } => {
                let slot = if *source == DrawSource::Removed {
                    0
                } else {
                    let slot = c.next;
                    c.next += 1;
                    slot
                };
                if slot >= 16 {
                    return None;
                }
                c.hands.get_mut(player.index())?.push(slot);
            }
            PublicEvent::Discarded { player, card } => {
                let mut choices: Vec<_> = (0..c.hands.get(player.index())?.len()).collect();
                self.shuffle(&mut choices);
                for choice in choices {
                    let mut branch = c.clone();
                    let slot = branch.hands[player.index()].remove(choice);
                    branch.constrain(slot, bit(*card));
                    // The first discard follows Played. Prince/King cannot be
                    // played while retaining Countess, even without a target.
                    if matches!(o.public.events.get(index.wrapping_sub(1)), Some(PublicEvent::Played {action,..}) if matches!(action.played_card(), Card::Prince | Card::King))
                    {
                        for slot in branch.hands[player.index()].clone() {
                            branch.constrain(slot, ALL & !bit(Card::Countess));
                        }
                    }
                    if let Some(state) = self.after_event(branch, index) {
                        return Some(state);
                    }
                    if self.exhausted {
                        return None;
                    }
                }
                return None;
            }
            PublicEvent::GuardResult {
                target,
                guess,
                correct,
                ..
            } => {
                c.constrain(
                    c.held(*target)?,
                    if *correct {
                        bit(*guess)
                    } else {
                        ALL & !bit(*guess)
                    },
                );
            }
            PublicEvent::Compared {
                actor,
                target,
                eliminated,
            } => {
                let order = match eliminated {
                    None => Ordering::Equal,
                    Some(p) if p == actor => Ordering::Less,
                    _ => Ordering::Greater,
                };
                c.comparisons
                    .push((c.held(*actor)?, c.held(*target)?, order));
            }
            PublicEvent::Exchanged { actor, target } => {
                c.hands.swap(actor.index(), target.index());
            }
            PublicEvent::RoundEnded { revealed_hands, .. } => {
                for &(p, card) in revealed_hands {
                    c.constrain(c.held(p)?, bit(card));
                }
            }
            _ => {}
        }
        self.after_event(c, index)
    }
    fn after_event(&mut self, mut c: Constraints, index: usize) -> Option<LoveLetterState> {
        let o = self.observation;
        for e in o.private.iter().filter(|e| e.event == index) {
            match e.information {
                PrivateInformation::Received { card, .. } => {
                    c.constrain(*c.hands[o.observer.index()].last()?, bit(card))
                }
                PrivateInformation::Priest { target, card } => {
                    c.constrain(c.held(target)?, bit(card))
                }
                PrivateInformation::Comparison {
                    opponent,
                    own_card,
                    other_card,
                } => {
                    c.constrain(c.held(o.observer)?, bit(own_card));
                    c.constrain(c.held(opponent)?, bit(other_card));
                }
                PrivateInformation::Exchange {
                    other,
                    given,
                    received,
                } => {
                    c.constrain(c.held(o.observer)?, bit(received));
                    c.constrain(c.held(other)?, bit(given));
                }
            }
        }
        self.events(c, index + 1)
    }
    fn assign(&mut self, mut c: Constraints) -> Option<LoveLetterState> {
        if !self.node() || !c.propagate() {
            return None;
        }
        if let Some(slot) = (0..16)
            .filter(|&i| c.domains[i].count_ones() > 1)
            .min_by_key(|&i| c.domains[i].count_ones())
        {
            let mut choices: Vec<_> = Card::ALL
                .into_iter()
                .filter(|&v| c.domains[slot] & bit(v) != 0)
                .collect();
            self.shuffle(&mut choices);
            for card in choices {
                let mut branch = c.clone();
                branch.domains[slot] = bit(card);
                if let Some(state) = self.assign(branch) {
                    return Some(state);
                }
                if self.exhausted {
                    return None;
                }
            }
            None
        } else {
            let deck = c.domains.map(|d| {
                Card::ALL
                    .into_iter()
                    .find(|&v| d == bit(v))
                    .expect("singleton")
            });
            replay(self.observation, &deck)
        }
    }
}

/// Verify every candidate with the actual rules, and reconstruct all players'
/// hypothetical current-round private experiences through normal event delivery.
fn replay(o: &LoveLetterObservation, deck: &[Card; 16]) -> Option<LoveLetterState> {
    let PublicEvent::RoundStarted { starter, .. } = *o.public.events.first()? else {
        return None;
    };
    let model = LoveLetterForwardModel;
    let mut state = model
        .init_with_deck(
            &LoveLetterParameters {
                rules: o.public.rules.clone(),
                seed: 0,
                first_player: starter,
            },
            o.public.player_count(),
            deck,
        )
        .ok()?;
    state.public.round = o.public.round;
    state.public.events[0] = o.public.events[0].clone();
    state.public.tokens = o.public.tokens.clone();
    if let Some(PublicEvent::RoundEnded { winners, .. }) = o
        .public
        .events
        .iter()
        .find(|e| matches!(e, PublicEvent::RoundEnded { .. }))
    {
        for p in winners {
            state.public.tokens[p.index()] = state.public.tokens[p.index()].checked_sub(1)?;
        }
    } else {
        state.public.last_round = o.public.last_round.clone();
    }
    state.public.last_actor = o.public.last_actor;
    for event in &o.public.events {
        if let PublicEvent::Played { actor, action } = event {
            model.next(&mut state, *actor, action).ok()?;
        }
    }
    // Absolute counters and previous rounds do not constrain this independent
    // shuffle. Their public values and the observer's retained memory are kept.
    state.public.counters = o.public.counters;
    state.completed_private[o.observer.index()] = o.last_round_private.clone();
    if model.observe(&state, o.observer).ok()?.eq(o) {
        Some(state)
    } else {
        None
    }
}

impl LoveLetterForwardModel {
    /// Construct a hypothetical state with an explicit constraint-search budget.
    /// First-feasible randomized search is not a uniform posterior sampler.
    /// Failure leaves the caller's random stream unchanged; no real state is read.
    pub fn simulation_fork_with_budget(
        &self,
        o: &LoveLetterObservation,
        random: &mut SimulationRng,
        budget: usize,
    ) -> Result<LoveLetterState, LoveLetterError> {
        let n = o.public.player_count();
        if !(2..=4).contains(&n) {
            return Err(LoveLetterError::UnsupportedPlayerCount(n));
        }
        o.public.validate_player(o.observer)?;
        let mut source =
            SimulationRng::from_snapshot(&random.snapshot()).expect("valid simulation stream");
        let sampling = source.fork_chance();
        let future = source.fork_chance();
        let mut c = Constraints {
            domains: [ALL; 16],
            hands: vec![vec![]; n],
            next: 1 + o.public.reserve.len(),
            comparisons: vec![],
        };
        if o.public.reserve.len() != if n == 2 { 3 } else { 0 } {
            return Err(LoveLetterError::InvalidState("invalid public reserve"));
        }
        for (i, &card) in o.public.reserve.iter().enumerate() {
            c.constrain(i + 1, bit(card));
        }
        let mut search = Search {
            observation: o,
            random: sampling,
            remaining: budget,
            exhausted: false,
        };
        let mut state = search.events(c, 0).ok_or(if search.exhausted {
            LoveLetterError::SamplingBudgetExceeded
        } else {
            LoveLetterError::InvalidState("no hidden world matches the observation")
        })?;
        state.random = future;
        *random = source;
        Ok(state)
    }
}
impl SimulationModel for LoveLetterForwardModel {
    fn simulation_fork(
        &self,
        o: &LoveLetterObservation,
        random: &mut SimulationRng,
    ) -> Result<LoveLetterState, LoveLetterError> {
        self.simulation_fork_with_budget(o, random, DEFAULT_SAMPLING_BUDGET)
    }
}
