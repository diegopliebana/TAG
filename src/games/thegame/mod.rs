//! The Game: cooperative card placement under limited communication.
//! Rules: Steffen Benndorf, Nürnberger-Spielkarten-Verlag, English rulebook
//! <https://www.nsv.de/wp-content/uploads/2024/04/TheGame_GB.pdf>.
use crate::core::cards;
use crate::core::checkpoint::Checkpoint;
use crate::core::random::{ChanceRng, ChanceRngSnapshot, SimulationRng};
use crate::core::simulation::SimulationModel;
use crate::core::{Decision, ForwardModel, OutcomeScore, PlayerId, PlayerResult, ScoreScope};
use std::collections::VecDeque;
use std::fmt;

pub const CARD_COUNT: usize = 98;
pub const PILE_COUNT: usize = 4;

/// The four piles have fixed identities: two rise from 1 and two fall from 100.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum Pile {
    AscendingOne,
    AscendingTwo,
    DescendingOne,
    DescendingTwo,
}

impl Pile {
    pub const ALL: [Self; PILE_COUNT] = [
        Self::AscendingOne,
        Self::AscendingTwo,
        Self::DescendingOne,
        Self::DescendingTwo,
    ];
    pub const fn index(self) -> usize {
        self as usize
    }
    pub const fn starts_at(self) -> u8 {
        if self.index() < 2 { 1 } else { 100 }
    }
    pub const fn permits(self, top: u8, card: u8) -> bool {
        if self.index() < 2 {
            card > top || card + 10 == top
        } else {
            card < top || card == top + 10
        }
    }
}

/// Selected, fixed-choice public communication. No message carries a card value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum Message {
    NoRequest,
    LeavePile(Pile),
    SmallJump(Pile),
    CanUsePile(Pile),
}

impl Message {
    pub const COUNT: usize = 13;
}

/// The limited-choice mode is the selected rule variant. Silence is useful for
/// controlled communication experiments and for games without human discussion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum CommunicationMode {
    #[default]
    LimitedChoices,
    Silent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    Speak(Message),
    Play { card: u8, pile: Pile },
    EndTurn,
}

/// Low-level setup accepts resolved seeds. The first seat is a rulebook choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameters {
    pub seed: u64,
    pub first_player: PlayerId,
    pub communication: CommunicationMode,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            seed: 0,
            first_player: PlayerId::new(0),
            communication: CommunicationMode::LimitedChoices,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum Phase {
    Communicate { speaker: PlayerId },
    Play,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub enum TeamOutcome {
    Ongoing,
    Won,
    Lost,
}

/// One public communication round, revealed after all seats have spoken.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageRound {
    pub active_player: PlayerId,
    pub messages: Vec<Message>,
}

/// One public card placement; earlier cards were visible when they were laid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct Placement {
    pub player: PlayerId,
    pub pile: Pile,
    pub card: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct PublicState {
    pub first_player: PlayerId,
    pub active_player: PlayerId,
    pub phase: Phase,
    pub outcome: TeamOutcome,
    pub pile_tops: [u8; PILE_COUNT],
    pub placements: Vec<Placement>,
    pub messages: Vec<MessageRound>,
    pub hand_counts: Vec<usize>,
    pub draw_count: usize,
    pub played_this_turn: usize,
    pub turns_completed: usize,
    pub communication: CommunicationMode,
}

/// Full simulation state. Pending messages and hidden cards stay out of views.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub public: PublicState,
    hands: Vec<Vec<u8>>,
    deck: VecDeque<u8>,
    pending_messages: Vec<Message>,
    random: ChanceRng,
}

impl State {
    pub fn privileged_hand(&self, player: PlayerId) -> Result<&[u8], Error> {
        self.hands
            .get(player.index())
            .map(Vec::as_slice)
            .ok_or(Error::InvalidPlayer(player))
    }
    pub fn privileged_deck(&self) -> &VecDeque<u8> {
        &self.deck
    }
    pub fn chance_snapshot(&self) -> ChanceRngSnapshot {
        self.random.snapshot()
    }
    pub fn cards_remaining(&self) -> usize {
        self.deck.len() + self.hands.iter().map(Vec::len).sum::<usize>()
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

/// One player's independently owned information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub observer: PlayerId,
    pub public: PublicState,
    pub hand: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    UnsupportedPlayers(usize),
    InvalidPlayer(PlayerId),
    InvalidFirstPlayer(PlayerId),
    InvalidDeck,
    InvalidObservation,
    SamplingBudgetExceeded,
    WrongPlayer {
        expected: PlayerId,
        actual: PlayerId,
    },
    IllegalAction,
    Finished,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Default)]
pub struct TheGameForwardModel;

impl TheGameForwardModel {
    /// Supply all 98 cards in draw order for controlled rule fixtures. The
    /// normal shuffle still advances the saved chance stream.
    pub fn init_with_deck(
        &self,
        parameters: &Parameters,
        players: usize,
        top_first: &[u8],
    ) -> Result<State, Error> {
        self.initialize(parameters, players, Some(top_first))
    }

    fn initialize(
        &self,
        parameters: &Parameters,
        players: usize,
        supplied: Option<&[u8]>,
    ) -> Result<State, Error> {
        if !(1..=5).contains(&players) {
            return Err(Error::UnsupportedPlayers(players));
        }
        if parameters.first_player.index() >= players {
            return Err(Error::InvalidFirstPlayer(parameters.first_player));
        }
        let mut random = ChanceRng::new(parameters.seed);
        let standard = (2..=99).collect::<Vec<_>>();
        let deck = cards::prepare_deck(&standard, supplied, &mut random)
            .map_err(|_| Error::InvalidDeck)?;
        let hand_size = starting_hand_size(players);
        let mut state = State {
            public: PublicState {
                first_player: parameters.first_player,
                active_player: parameters.first_player,
                phase: Phase::Play,
                outcome: TeamOutcome::Ongoing,
                pile_tops: [1, 1, 100, 100],
                placements: Vec::new(),
                messages: Vec::new(),
                hand_counts: vec![0; players],
                draw_count: 0,
                played_this_turn: 0,
                turns_completed: 0,
                communication: parameters.communication,
            },
            hands: vec![Vec::with_capacity(hand_size); players],
            deck: VecDeque::from(deck),
            pending_messages: Vec::new(),
            random,
        };
        for hand in &mut state.hands {
            for _ in 0..hand_size {
                hand.push(cards::draw(&mut state.deck).expect("98 cards cover initial deal"));
            }
        }
        state.sync_counts();
        state.start_turn();
        Ok(state)
    }
}

impl State {
    fn sync_counts(&mut self) {
        self.public.hand_counts = self.hands.iter().map(Vec::len).collect();
        self.public.draw_count = self.deck.len();
    }
    fn start_turn(&mut self) {
        self.public.played_this_turn = 0;
        self.pending_messages.clear();
        if !can_finish_minimum(
            &self.hands[self.public.active_player.index()],
            self.public.pile_tops,
            minimum(self),
        ) {
            self.public.outcome = TeamOutcome::Lost;
            self.public.phase = Phase::Finished;
        } else if self.hands.len() > 1
            && self.public.communication == CommunicationMode::LimitedChoices
        {
            self.public.phase = Phase::Communicate {
                speaker: PlayerId::new(0),
            };
        } else {
            self.public.phase = Phase::Play;
        }
    }
}

fn starting_hand_size(players: usize) -> usize {
    match players {
        1 => 8,
        2 => 7,
        _ => 6,
    }
}

fn minimum(state: &State) -> usize {
    if state.deck.is_empty() { 1 } else { 2 }
}

/// Check whether some ordering can reach the turn minimum from this position.
/// A poor first placement can still end the game even if another was better.
fn can_finish_minimum(hand: &[u8], tops: [u8; PILE_COUNT], needed: usize) -> bool {
    if needed == 0 {
        return true;
    }
    if hand.len() < needed {
        return false;
    }
    for (index, &card) in hand.iter().enumerate() {
        for pile in Pile::ALL {
            if pile.permits(tops[pile.index()], card) {
                let mut next_hand = hand.to_vec();
                next_hand.remove(index);
                let mut next_tops = tops;
                next_tops[pile.index()] = card;
                if can_finish_minimum(&next_hand, next_tops, needed - 1) {
                    return true;
                }
            }
        }
    }
    false
}

fn actions(public: &PublicState, hand: &[u8], actor: PlayerId) -> Vec<Action> {
    if public.outcome != TeamOutcome::Ongoing {
        return Vec::new();
    }
    match public.phase {
        Phase::Communicate { speaker } if speaker == actor => {
            let mut result = vec![Action::Speak(Message::NoRequest)];
            for pile in Pile::ALL {
                result.push(Action::Speak(Message::LeavePile(pile)));
                result.push(Action::Speak(Message::SmallJump(pile)));
                result.push(Action::Speak(Message::CanUsePile(pile)));
            }
            result
        }
        Phase::Play if public.active_player == actor => {
            let mut result = Vec::new();
            for &card in hand {
                for pile in Pile::ALL {
                    if pile.permits(public.pile_tops[pile.index()], card) {
                        result.push(Action::Play { card, pile });
                    }
                }
            }
            if public.played_this_turn >= if public.draw_count == 0 { 1 } else { 2 } {
                result.push(Action::EndTurn);
            }
            result
        }
        _ => Vec::new(),
    }
}

impl ForwardModel for TheGameForwardModel {
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
        match state.public.phase {
            Phase::Communicate { speaker } => Decision::Player(speaker),
            Phase::Play => Decision::Player(state.public.active_player),
            Phase::Finished => Decision::Finished,
        }
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
        })
    }
    fn offered_actions(&self, observation: &Observation) -> Vec<Action> {
        actions(&observation.public, &observation.hand, observation.observer)
    }
    fn action_path(&self, _observation: &Observation, action: &Action) -> Vec<u32> {
        match *action {
            Action::Speak(message) => {
                let code = match message {
                    Message::NoRequest => 0,
                    Message::LeavePile(pile) => 1 + 3 * pile.index() as u32,
                    Message::SmallJump(pile) => 2 + 3 * pile.index() as u32,
                    Message::CanUsePile(pile) => 3 + 3 * pile.index() as u32,
                };
                vec![code]
            }
            Action::Play { card, pile } => vec![100 + u32::from(card), pile.index() as u32],
            Action::EndTurn => vec![200],
        }
    }
    fn available_actions(&self, state: &State, actor: PlayerId) -> Result<Vec<Action>, Error> {
        if actor.index() >= state.hands.len() {
            return Err(Error::InvalidPlayer(actor));
        }
        if state.public.outcome != TeamOutcome::Ongoing {
            return Ok(Vec::new());
        }
        if self.next_player(state) != Decision::Player(actor) {
            return Err(Error::WrongPlayer {
                expected: match self.next_player(state) {
                    Decision::Player(p) => p,
                    Decision::Finished => unreachable!(),
                },
                actual: actor,
            });
        }
        Ok(actions(&state.public, &state.hands[actor.index()], actor))
    }
    fn next(&self, state: &mut State, actor: PlayerId, action: &Action) -> Result<(), Error> {
        if state.public.outcome != TeamOutcome::Ongoing {
            return Err(Error::Finished);
        }
        if !self.available_actions(state, actor)?.contains(action) {
            return Err(Error::IllegalAction);
        }
        match *action {
            Action::Speak(message) => {
                state.pending_messages.push(message);
                if state.pending_messages.len() == state.hands.len() {
                    state.public.messages.push(MessageRound {
                        active_player: state.public.active_player,
                        messages: std::mem::take(&mut state.pending_messages),
                    });
                    state.public.phase = Phase::Play;
                } else {
                    state.public.phase = Phase::Communicate {
                        speaker: PlayerId::new(state.pending_messages.len()),
                    };
                }
            }
            Action::Play { card, pile } => {
                cards::take_one(&mut state.hands[actor.index()], &card)
                    .expect("validated action owns card");
                state.public.pile_tops[pile.index()] = card;
                state.public.placements.push(Placement {
                    player: actor,
                    pile,
                    card,
                });
                state.public.played_this_turn += 1;
                state.sync_counts();
                if state.cards_remaining() == 0 {
                    state.public.outcome = TeamOutcome::Won;
                    state.public.phase = Phase::Finished;
                } else {
                    let required = minimum(state).saturating_sub(state.public.played_this_turn);
                    if !can_finish_minimum(
                        &state.hands[actor.index()],
                        state.public.pile_tops,
                        required,
                    ) {
                        state.public.outcome = TeamOutcome::Lost;
                        state.public.phase = Phase::Finished;
                    }
                }
            }
            Action::EndTurn => {
                let hand_size = starting_hand_size(state.hands.len());
                let hand = &mut state.hands[actor.index()];
                while hand.len() < hand_size {
                    let Some(card) = cards::draw(&mut state.deck) else {
                        break;
                    };
                    hand.push(card);
                }
                state.public.turns_completed += 1;
                state.sync_counts();
                let n = state.hands.len();
                let next = (1..=n)
                    .map(|offset| (actor.index() + offset) % n)
                    .find(|&seat| !state.hands[seat].is_empty());
                if let Some(seat) = next {
                    state.public.active_player = PlayerId::new(seat);
                    state.start_turn();
                } else {
                    state.public.outcome = TeamOutcome::Won;
                    state.public.phase = Phase::Finished;
                }
            }
        }
        Ok(())
    }
    fn player_result(&self, state: &State, player: PlayerId) -> Result<PlayerResult, Error> {
        if player.index() >= state.hands.len() {
            return Err(Error::InvalidPlayer(player));
        }
        Ok(match state.public.outcome {
            TeamOutcome::Ongoing => PlayerResult::Ongoing,
            TeamOutcome::Won => PlayerResult::Win,
            TeamOutcome::Lost => PlayerResult::Loss,
        })
    }
    fn player_score(&self, state: &State, player: PlayerId) -> Result<f64, Error> {
        if player.index() >= state.hands.len() {
            return Err(Error::InvalidPlayer(player));
        }
        Ok((CARD_COUNT - state.cards_remaining()) as f64)
    }
    fn outcome_score(&self, state: &State) -> Result<Option<OutcomeScore>, Error> {
        Ok(Some(OutcomeScore {
            scope: ScoreScope::Team,
            values: vec![state.public.placements.len() as f64],
            unit: "cards_placed",
            maximum: Some(CARD_COUNT as f64),
        }))
    }
}

/// Uniformly assign cards that this observer has never seen to other hands and
/// the draw pile. Messages submitted in the current round are still hidden;
/// their sampled value is silence, which does not affect placement legality.
impl SimulationModel for TheGameForwardModel {
    fn simulation_fork(
        &self,
        observation: &Observation,
        random: &mut SimulationRng,
    ) -> Result<State, Error> {
        let p = &observation.public;
        let n = p.hand_counts.len();
        if !(1..=5).contains(&n)
            || observation.observer.index() >= n
            || p.first_player.index() >= n
            || p.active_player.index() >= n
            || observation.hand.len() != p.hand_counts[observation.observer.index()]
            || p.hand_counts
                .iter()
                .any(|&count| count > starting_hand_size(n))
            || p.placements.len() > CARD_COUNT
            || (p.outcome == TeamOutcome::Won) != (p.placements.len() == CARD_COUNT)
            || p.messages
                .iter()
                .any(|round| round.messages.len() != n || round.active_player.index() >= n)
            || p.hand_counts.iter().sum::<usize>() + p.draw_count + p.placements.len() != CARD_COUNT
            || matches!(p.phase, Phase::Communicate { speaker } if speaker.index() >= n || p.communication != CommunicationMode::LimitedChoices || n == 1)
            || (p.phase == Phase::Finished) != (p.outcome != TeamOutcome::Ongoing)
        {
            return Err(Error::InvalidObservation);
        }
        let mut tops = [1, 1, 100, 100];
        let mut pool: Vec<u8> = (2..=99).collect();
        for placement in &p.placements {
            let index = placement.pile.index();
            if placement.player.index() >= n
                || !placement.pile.permits(tops[index], placement.card)
                || cards::take_one(&mut pool, &placement.card).is_none()
            {
                return Err(Error::InvalidObservation);
            }
            tops[index] = placement.card;
        }
        if tops != p.pile_tops
            || cards::remove_all(&mut pool, observation.hand.iter().copied()).is_err()
            || pool.len()
                != p.draw_count + p.hand_counts.iter().sum::<usize>() - observation.hand.len()
        {
            return Err(Error::InvalidObservation);
        }
        let mut source = SimulationRng::from_snapshot(&random.snapshot())
            .expect("existing simulation stream is valid");
        for _ in 0..4096 {
            let mut future = source.fork_chance();
            let mut shuffled = pool.clone();
            cards::shuffle(&mut shuffled, &mut future);
            let mut remaining = VecDeque::from(shuffled);
            let mut hands = vec![Vec::new(); n];
            hands[observation.observer.index()] = observation.hand.clone();
            for (seat, hand) in hands.iter_mut().enumerate() {
                if seat != observation.observer.index() {
                    for _ in 0..p.hand_counts[seat] {
                        hand.push(cards::draw(&mut remaining).expect("validated card inventory"));
                    }
                }
            }
            let needed: usize = if p.draw_count == 0 { 1 } else { 2 };
            if p.outcome == TeamOutcome::Ongoing
                && !can_finish_minimum(
                    &hands[p.active_player.index()],
                    p.pile_tops,
                    needed.saturating_sub(p.played_this_turn),
                )
            {
                continue;
            }
            let pending = match p.phase {
                Phase::Communicate { speaker } => vec![Message::NoRequest; speaker.index()],
                _ => Vec::new(),
            };
            let state = State {
                public: p.clone(),
                hands,
                deck: remaining,
                pending_messages: pending,
                random: future,
            };
            *random = source;
            return Ok(state);
        }
        Err(Error::SamplingBudgetExceeded)
    }
}

mod evaluation;
mod learning;
pub use evaluation::TheGameHeuristic;

#[cfg(feature = "replay")]
mod persistence;
#[cfg(test)]
mod tests;
