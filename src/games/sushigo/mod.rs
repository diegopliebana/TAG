//! Original 108-card Sushi Go! rules. Complete decisions resolve together.
//! Rule behaviour follows Gamewright's rulebook and the designer's Chopsticks
//! clarification; public TAG's `games/sushigo` is a comparison source.
use crate::core::cards;
use crate::core::checkpoint::Checkpoint;
use crate::core::joint::{self, Choice, JointError};
use crate::core::random::{ChanceRng, ChanceRngSnapshot, SimulationRng};
use crate::core::simulation::SimulationModel;
use crate::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use std::collections::VecDeque;
use std::fmt;

/// The twelve distinct card values. Maki values include their visible roll count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Card {
    Tempura = 0,
    Sashimi = 1,
    Dumpling = 2,
    Maki1 = 3,
    Maki2 = 4,
    Maki3 = 5,
    SquidNigiri = 6,
    SalmonNigiri = 7,
    EggNigiri = 8,
    Wasabi = 9,
    Chopsticks = 10,
    Pudding = 11,
}

impl Card {
    pub const ALL: [Self; 12] = [
        Self::Tempura,
        Self::Sashimi,
        Self::Dumpling,
        Self::Maki1,
        Self::Maki2,
        Self::Maki3,
        Self::SquidNigiri,
        Self::SalmonNigiri,
        Self::EggNigiri,
        Self::Wasabi,
        Self::Chopsticks,
        Self::Pudding,
    ];
    pub const COUNTS: [usize; 12] = [14, 14, 14, 6, 12, 8, 5, 10, 5, 6, 4, 10];
    /// Stable card-value ID, independent of a hidden hand's card positions.
    pub const fn id(self) -> usize {
        self as usize
    }
    fn deck() -> Vec<Self> {
        Self::ALL
            .into_iter()
            .zip(Self::COUNTS)
            .flat_map(|(card, count)| std::iter::repeat_n(card, count))
            .collect()
    }
    fn nigiri_points(self) -> Option<i32> {
        match self {
            Self::EggNigiri => Some(1),
            Self::SalmonNigiri => Some(2),
            Self::SquidNigiri => Some(3),
            _ => None,
        }
    }
    fn maki_rolls(self) -> i32 {
        match self {
            Self::Maki1 => 1,
            Self::Maki2 => 2,
            Self::Maki3 => 3,
            _ => 0,
        }
    }
}

/// A card-value choice. Two Chopsticks cards are ordered as they are played.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Action {
    Pick(Card),
    Chopsticks { first: Card, second: Card },
}

impl Action {
    /// Stable ID: 0–11 for one card, 12–155 for ordered Chopsticks pairs.
    pub const fn id(self) -> usize {
        match self {
            Self::Pick(card) => card.id(),
            Self::Chopsticks { first, second } => 12 + 12 * first.id() + second.id(),
        }
    }
    pub const ACTION_COUNT: usize = 156;
    fn cards(self) -> [Option<Card>; 2] {
        match self {
            Self::Pick(card) => [Some(card), None],
            Self::Chopsticks { first, second } => [Some(first), Some(second)],
        }
    }
}

/// Low-level setup receives a resolved integer seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameters {
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    UnsupportedPlayers(usize),
    InvalidPlayer(PlayerId),
    InvalidDeck,
    IllegalAction,
    Finished,
    SimultaneousOnly,
    InvalidObservation,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// A visible card retains its Wasabi placement until the round is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayedCard {
    pub card: Card,
    pub on_wasabi: bool,
}

/// Facts visible to every seat. The latest reveal records ordered Chopsticks picks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicState {
    pub round: usize,
    pub scores: Vec<i32>,
    pub round_scores: Vec<Vec<i32>>,
    pub pudding: Vec<usize>,
    pub played: Vec<Vec<PlayedCard>>,
    pub discard: Vec<Card>,
    pub draw_count: usize,
    pub hand_size: usize,
    pub hand_ids: Vec<usize>,
    pub last_reveal: Vec<Choice<Action>>,
    pub finished: bool,
}

/// Complete state. `known_by[p][id]` is an exact multiset of a hand seat `p` saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub public: PublicState,
    hands: Vec<Vec<Card>>,
    deck: VecDeque<Card>,
    known_by: Vec<Vec<Option<Vec<Card>>>>,
    random: ChanceRng,
}
impl State {
    pub fn privileged_hand(&self, seat: PlayerId) -> Result<&[Card], Error> {
        self.hands
            .get(seat.index())
            .map(Vec::as_slice)
            .ok_or(Error::InvalidPlayer(seat))
    }
    pub fn privileged_draw_pile(&self) -> &VecDeque<Card> {
        &self.deck
    }
    pub fn chance_snapshot(&self) -> ChanceRngSnapshot {
        self.random.snapshot()
    }
    /// Canonical complete-state facts for versioned replay verification.
    /// This contains hidden hands and the chance stream; never send it in a player view.
    pub fn verification_facts(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}",
            self.public,
            self.hands,
            self.deck,
            self.known_by,
            self.random.snapshot()
        )
    }
}
impl Checkpoint for State {
    type Snapshot = Self;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self {
        self.clone()
    }
    fn from_snapshot(snapshot: &Self) -> Result<Self, Self::RestoreError> {
        Ok(snapshot.clone())
    }
}

/// A seat's owned hand, public facts and remembered complete passing hands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub observer: PlayerId,
    pub public: PublicState,
    pub hand: Vec<Card>,
    pub known_hands: Vec<Option<Vec<Card>>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SushiGoForwardModel;

mod evaluation;
mod learning;
pub use evaluation::SushiGoHeuristic;

impl SushiGoForwardModel {
    /// Set the 108-card draw order for controlled rule fixtures.
    pub fn init_with_deck(
        &self,
        parameters: &Parameters,
        players: usize,
        top_first: &[Card],
    ) -> Result<State, Error> {
        if !cards::same_multiset(&Card::deck(), top_first) {
            return Err(Error::InvalidDeck);
        }
        self.initialize(parameters, players, Some(top_first))
    }
    fn initialize(
        &self,
        parameters: &Parameters,
        players: usize,
        supplied: Option<&[Card]>,
    ) -> Result<State, Error> {
        if !(2..=5).contains(&players) {
            return Err(Error::UnsupportedPlayers(players));
        }
        let mut random = ChanceRng::new(parameters.seed);
        let cards = cards::prepare_deck(&Card::deck(), supplied, &mut random)
            .map_err(|_| Error::InvalidDeck)?;
        let public = PublicState {
            round: 1,
            scores: vec![0; players],
            round_scores: Vec::new(),
            pudding: vec![0; players],
            played: vec![Vec::new(); players],
            discard: Vec::new(),
            draw_count: 108,
            hand_size: 12 - players,
            hand_ids: (0..players).collect(),
            last_reveal: Vec::new(),
            finished: false,
        };
        let mut state = State {
            public,
            hands: vec![Vec::new(); players],
            deck: VecDeque::from(cards),
            known_by: vec![vec![None; players]; players],
            random,
        };
        deal(&mut state);
        Ok(state)
    }
}

fn deal(state: &mut State) {
    let n = state.hands.len();
    state.public.hand_size = 12 - n;
    state.public.hand_ids = (0..n).collect();
    state.known_by = vec![vec![None; n]; n];
    for seat in 0..n {
        for _ in 0..state.public.hand_size {
            state.hands[seat]
                .push(cards::draw(&mut state.deck).expect("108 cards cover all three deals"));
        }
        state.known_by[seat][seat] = Some(state.hands[seat].clone());
    }
    state.public.draw_count = state.deck.len();
}

fn actions(hand: &[Card], played: &[PlayedCard]) -> Vec<Action> {
    let mut offered = Vec::new();
    let has_chopsticks = played.iter().any(|card| card.card == Card::Chopsticks);
    for first in Card::ALL {
        if !hand.contains(&first) {
            continue;
        }
        offered.push(Action::Pick(first));
        if has_chopsticks && hand.len() >= 2 {
            for second in Card::ALL {
                if second == first && hand.iter().filter(|&&card| card == first).count() < 2 {
                    continue;
                }
                if hand.contains(&second) {
                    offered.push(Action::Chopsticks { first, second });
                }
            }
        }
    }
    offered
}

impl ForwardModel for SushiGoForwardModel {
    type State = State;
    type Action = Action;
    type Observation = Observation;
    type Parameters = Parameters;
    type Error = Error;
    fn init_state(&self, parameters: &Parameters, players: usize) -> Result<State, Error> {
        self.initialize(parameters, players, None)
    }
    fn player_count(&self, state: &State) -> usize {
        state.hands.len()
    }
    fn next_player(&self, state: &State) -> Decision {
        if state.public.finished {
            Decision::Finished
        } else {
            Decision::Player(PlayerId::new(0))
        }
    }
    fn decision_players(&self, state: &State) -> Vec<PlayerId> {
        if state.public.finished {
            Vec::new()
        } else {
            (0..state.hands.len()).map(PlayerId::new).collect()
        }
    }
    fn joint_actions_public_after(&self, _state: &State) -> bool {
        true
    }
    fn observe(&self, state: &State, observer: PlayerId) -> Result<Observation, Error> {
        let hand = state
            .hands
            .get(observer.index())
            .ok_or(Error::InvalidPlayer(observer))?
            .clone();
        Ok(Observation {
            observer,
            public: state.public.clone(),
            hand,
            known_hands: state.known_by[observer.index()].clone(),
        })
    }
    fn offered_actions(&self, observation: &Observation) -> Vec<Action> {
        if observation.public.finished
            || observation.observer.index() >= observation.public.played.len()
        {
            return Vec::new();
        }
        actions(
            &observation.hand,
            &observation.public.played[observation.observer.index()],
        )
    }
    fn available_actions(&self, state: &State, actor: PlayerId) -> Result<Vec<Action>, Error> {
        let hand = state
            .hands
            .get(actor.index())
            .ok_or(Error::InvalidPlayer(actor))?;
        if state.public.finished {
            return Ok(Vec::new());
        }
        Ok(actions(hand, &state.public.played[actor.index()]))
    }
    fn next(&self, _state: &mut State, _actor: PlayerId, _action: &Action) -> Result<(), Error> {
        Err(Error::SimultaneousOnly)
    }
    fn resolve_decision(
        &self,
        state: &mut State,
        choices: &[Choice<Action>],
    ) -> Result<(), JointError<Error>> {
        joint::validate(self, state, choices)?;
        let n = state.hands.len();
        for &(seat, action) in choices {
            let p = seat.index();
            let id = state.public.hand_ids[p];
            for card in action.cards().into_iter().flatten() {
                cards::take_one(&mut state.hands[p], &card).expect("validated action owns card");
                for observer in 0..n {
                    if let Some(known) = &mut state.known_by[observer][id] {
                        cards::take_one(known, &card).expect("known hand contains revealed card");
                    }
                }
                play(&mut state.public, p, card);
            }
            if matches!(action, Action::Chopsticks { .. }) {
                let chopsticks = state.public.played[p]
                    .iter()
                    .position(|c| c.card == Card::Chopsticks)
                    .expect("validated chopsticks");
                state.public.played[p].remove(chopsticks);
                state.hands[p].push(Card::Chopsticks);
                for observer in 0..n {
                    if let Some(known) = &mut state.known_by[observer][id] {
                        known.push(Card::Chopsticks);
                    }
                }
            }
        }
        state.public.last_reveal = choices.to_vec();
        pass_hands(state);
        if state.hands[0].len() == 1 {
            // This hand was just passed; its final card belongs to the recipient.
            for p in 0..n {
                let card = state.hands[p].pop().expect("one final card");
                play(&mut state.public, p, card);
            }
            end_round(state);
        }
        state.public.hand_size = state.hands[0].len();
        Ok(())
    }
    fn player_result(&self, state: &State, seat: PlayerId) -> Result<PlayerResult, Error> {
        let p = seat.index();
        if p >= state.hands.len() {
            return Err(Error::InvalidPlayer(seat));
        }
        if !state.public.finished {
            return Ok(PlayerResult::Ongoing);
        }
        let best = *state.public.scores.iter().max().expect("players");
        let best_pudding = state
            .public
            .scores
            .iter()
            .enumerate()
            .filter(|&(_, score)| *score == best)
            .map(|(i, _)| state.public.pudding[i])
            .max()
            .expect("winner");
        Ok(
            if state.public.scores[p] == best && state.public.pudding[p] == best_pudding {
                PlayerResult::Win
            } else {
                PlayerResult::Loss
            },
        )
    }
    fn player_score(&self, state: &State, seat: PlayerId) -> Result<f64, Error> {
        state
            .public
            .scores
            .get(seat.index())
            .map(|&score| f64::from(score))
            .ok_or(Error::InvalidPlayer(seat))
    }
}

fn pass_hands(state: &mut State) {
    state.hands.rotate_left(1);
    state.public.hand_ids.rotate_left(1);
    for p in 0..state.hands.len() {
        let id = state.public.hand_ids[p];
        state.known_by[p][id] = Some(state.hands[p].clone());
    }
}

fn play(public: &mut PublicState, seat: usize, card: Card) {
    if card == Card::Pudding {
        public.pudding[seat] += 1;
        return;
    }
    let available_wasabi = public.played[seat]
        .iter()
        .filter(|entry| entry.card == Card::Wasabi)
        .count();
    let used_wasabi = public.played[seat]
        .iter()
        .filter(|entry| entry.on_wasabi)
        .count();
    public.played[seat].push(PlayedCard {
        card,
        on_wasabi: card.nigiri_points().is_some() && available_wasabi > used_wasabi,
    });
}

fn end_round(state: &mut State) {
    let n = state.hands.len();
    let mut round = (0..n)
        .map(|p| score_cards(&state.public.played[p]))
        .collect::<Vec<_>>();
    let maki = state
        .public
        .played
        .iter()
        .map(|cards| cards.iter().map(|c| c.card.maki_rolls()).sum::<i32>())
        .collect::<Vec<_>>();
    let max = *maki.iter().max().expect("players");
    if max > 0 {
        let first = (0..n).filter(|&p| maki[p] == max).collect::<Vec<_>>();
        let award = 6 / first.len() as i32;
        for &p in &first {
            round[p] += award;
        }
        if first.len() == 1 {
            let second_max = maki
                .iter()
                .copied()
                .filter(|&count| count < max)
                .max()
                .unwrap_or(0);
            if second_max > 0 {
                let second = (0..n)
                    .filter(|&p| maki[p] == second_max)
                    .collect::<Vec<_>>();
                let award = 3 / second.len() as i32;
                for p in second {
                    round[p] += award;
                }
            }
        }
    }
    for (p, &score) in round.iter().enumerate() {
        state.public.scores[p] += score;
    }
    state.public.round_scores.push(round);
    for played in &mut state.public.played {
        state
            .public
            .discard
            .extend(played.drain(..).map(|entry| entry.card));
    }
    if state.public.round == 3 {
        let most = *state.public.pudding.iter().max().expect("players");
        let least = *state.public.pudding.iter().min().expect("players");
        if most != least {
            let winners = state
                .public
                .pudding
                .iter()
                .filter(|&&count| count == most)
                .count();
            for p in 0..n {
                if state.public.pudding[p] == most {
                    state.public.scores[p] += 6 / winners as i32;
                }
            }
            if n > 2 {
                let losers = state
                    .public
                    .pudding
                    .iter()
                    .filter(|&&count| count == least)
                    .count();
                for p in 0..n {
                    if state.public.pudding[p] == least {
                        state.public.scores[p] -= 6 / losers as i32;
                    }
                }
            }
        }
        state.public.finished = true;
        state.public.hand_size = 0;
        state.known_by = vec![vec![Some(Vec::new()); n]; n];
    } else {
        state.public.round += 1;
        deal(state);
    }
}

fn score_cards(played: &[PlayedCard]) -> i32 {
    let count = |card| played.iter().filter(|entry| entry.card == card).count();
    let dumpling = [0, 1, 3, 6, 10, 15][count(Card::Dumpling).min(5)];
    let nigiri: i32 = played
        .iter()
        .filter_map(|entry| {
            entry
                .card
                .nigiri_points()
                .map(|points| if entry.on_wasabi { 3 * points } else { points })
        })
        .sum();
    (count(Card::Tempura) / 2) as i32 * 5
        + (count(Card::Sashimi) / 3) as i32 * 10
        + dumpling
        + nigiri
}

impl SimulationModel for SushiGoForwardModel {
    fn simulation_fork(
        &self,
        observation: &Observation,
        random: &mut SimulationRng,
    ) -> Result<State, Error> {
        let n = observation.public.scores.len();
        if !(2..=5).contains(&n)
            || observation.observer.index() >= n
            || observation.public.hand_ids.len() != n
            || observation.public.played.len() != n
            || observation.public.pudding.len() != n
            || observation.known_hands.len() != n
            || !(1..=3).contains(&observation.public.round)
            || observation.public.hand_size > 12 - n
            || (observation.public.hand_size == 0) != observation.public.finished
            || observation.hand.len() != observation.public.hand_size
            || observation.public.hand_ids.iter().any(|&id| id >= n)
            || (0..n).any(|id| {
                observation
                    .public
                    .hand_ids
                    .iter()
                    .filter(|&&candidate| candidate == id)
                    .count()
                    != 1
            })
            || observation.known_hands[observation.public.hand_ids[observation.observer.index()]]
                .as_ref()
                != Some(&observation.hand)
        {
            return Err(Error::InvalidObservation);
        }
        let mut pool = Card::deck();
        let visible = observation
            .public
            .discard
            .iter()
            .copied()
            .chain(
                observation
                    .public
                    .played
                    .iter()
                    .flatten()
                    .map(|entry| entry.card),
            )
            .chain(
                observation
                    .public
                    .pudding
                    .iter()
                    .flat_map(|&count| std::iter::repeat_n(Card::Pudding, count)),
            );
        cards::remove_all(&mut pool, visible).map_err(|_| Error::InvalidObservation)?;
        let mut hands = vec![Vec::new(); n];
        for (seat, hand) in hands.iter_mut().enumerate() {
            let id = observation.public.hand_ids[seat];
            if let Some(known) = &observation.known_hands[id] {
                if known.len() != observation.public.hand_size {
                    return Err(Error::InvalidObservation);
                }
                cards::remove_all(&mut pool, known.iter().copied())
                    .map_err(|_| Error::InvalidObservation)?;
                *hand = known.clone();
            }
        }
        let unknown = hands.iter().filter(|hand| hand.is_empty()).count();
        if pool.len() != unknown * observation.public.hand_size + observation.public.draw_count {
            return Err(Error::InvalidObservation);
        }
        let mut future = random.fork_chance();
        cards::shuffle(&mut pool, &mut future);
        let mut deck = VecDeque::from(pool);
        for hand in &mut hands {
            if hand.is_empty() {
                for _ in 0..observation.public.hand_size {
                    hand.push(cards::draw(&mut deck).expect("validated supply"));
                }
            }
        }
        let mut known_by = vec![vec![None; n]; n];
        // A seat remembers each passing hand it has held this round. Because
        // every intervening pick and Chopsticks return is public, that seat can
        // still infer the exact remaining multiset in a sampled world.
        let passes = (12 - n).saturating_sub(observation.public.hand_size);
        for (owner, hand) in hands.iter().enumerate() {
            let id = observation.public.hand_ids[owner];
            for (seat, knowledge) in known_by.iter_mut().enumerate() {
                if (0..=passes).any(|step| (id + n - step % n) % n == seat) {
                    knowledge[id] = Some(hand.clone());
                }
            }
        }
        known_by[observation.observer.index()] = observation.known_hands.clone();
        Ok(State {
            public: observation.public.clone(),
            hands,
            deck,
            known_by,
            random: future,
        })
    }
}

#[cfg(test)]
mod tests;
