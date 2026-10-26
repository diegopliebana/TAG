//! Classic rules; shuffled decks and tied starters use real-game chance.
use super::*;
use crate::core::{
    Decision, ForwardModel, PlayerId, PlayerResult, progress::TurnCounters, random::ChanceRng,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct LoveLetterForwardModel;

impl LoveLetterForwardModel {
    /// Controlled initial shuffle for rule fixtures and recorded-event comparisons.
    /// `top_first` is borrowed and must contain the standard multiset. The real
    /// generator consumes a normal shuffle before these supplied outcomes replace
    /// its order, preserving subsequent generator advancement.
    pub fn init_with_deck(
        &self,
        p: &LoveLetterParameters,
        players: usize,
        top_first: &[Card; 16],
    ) -> Result<LoveLetterState, LoveLetterError> {
        let mut expected = Card::DECK;
        expected.sort();
        let mut actual = *top_first;
        actual.sort();
        if actual != expected {
            return Err(LoveLetterError::InvalidDeck);
        }
        self.initialize(p, players, Some(top_first))
    }
    fn initialize(
        &self,
        p: &LoveLetterParameters,
        n: usize,
        supplied: Option<&[Card; 16]>,
    ) -> Result<LoveLetterState, LoveLetterError> {
        if !(2..=4).contains(&n) {
            return Err(LoveLetterError::UnsupportedPlayerCount(n));
        }
        if p.rules.winning_tokens.contains(&0) {
            return Err(LoveLetterError::InvalidParameters(
                "token targets must be positive",
            ));
        }
        if p.first_player.index() >= n {
            return Err(LoveLetterError::InvalidPlayer(p.first_player));
        }
        let public = LoveLetterPublicState {
            rules: p.rules.clone(),
            current_player: p.first_player,
            last_actor: None,
            round: 1,
            tokens: vec![0; n],
            active: vec![true; n],
            protected: vec![false; n],
            reserve: vec![],
            discards: vec![vec![]; n],
            draw_count: 0,
            removed_count: 0,
            counters: TurnCounters::default(),
            outcome: LoveLetterOutcome::Ongoing,
            events: vec![],
            last_round: None,
        };
        let mut state = LoveLetterState {
            public,
            hands: vec![vec![]; n],
            pile: Default::default(),
            removed: None,
            random: ChanceRng::new(p.seed),
            private: vec![vec![]; n],
            completed_private: vec![vec![]; n],
        };
        start_round(&mut state, p.first_player, supplied)?;
        Ok(state)
    }
}

impl ForwardModel for LoveLetterForwardModel {
    type State = LoveLetterState;
    type Observation = LoveLetterObservation;
    type Action = LoveLetterAction;
    type Parameters = LoveLetterParameters;
    type Error = LoveLetterError;
    fn init_state(&self, p: &Self::Parameters, n: usize) -> Result<Self::State, Self::Error> {
        self.initialize(p, n, None)
    }
    fn player_count(&self, s: &Self::State) -> usize {
        s.public.player_count()
    }
    fn next_player(&self, s: &Self::State) -> Decision {
        s.public.next_player()
    }
    fn observe(&self, s: &Self::State, p: PlayerId) -> Result<Self::Observation, Self::Error> {
        s.public.validate_player(p)?;
        Ok(LoveLetterObservation {
            observer: p,
            public: s.public.clone(),
            hand: s.hands[p.index()].clone(),
            private: s.private[p.index()].clone(),
            last_round_private: s.completed_private[p.index()].clone(),
        })
    }
    fn offered_actions(&self, o: &Self::Observation) -> Vec<Self::Action> {
        if o.public.next_player() != Decision::Player(o.observer) {
            return vec![];
        }
        actions(&o.public, &o.hand, o.observer)
    }
    fn available_actions(
        &self,
        s: &Self::State,
        actor: PlayerId,
    ) -> Result<Vec<Self::Action>, Self::Error> {
        s.public.validate_player(actor)?;
        if self.next_player(s) == Decision::Finished {
            return Ok(vec![]);
        }
        if actor != s.public.current_player {
            return Err(LoveLetterError::WrongPlayer {
                expected: s.public.current_player,
                actual: actor,
            });
        }
        Ok(actions(&s.public, &s.hands[actor.index()], actor))
    }
    fn next(
        &self,
        s: &mut Self::State,
        actor: PlayerId,
        action: &Self::Action,
    ) -> Result<(), Self::Error> {
        s.public.validate_player(actor)?;
        if self.next_player(s) == Decision::Finished {
            return Err(LoveLetterError::Finished);
        }
        if !self.available_actions(s, actor)?.contains(action) {
            return Err(LoveLetterError::IllegalAction);
        }
        // Legality was checked above. Apply directly to this real or simulated
        // state; an internal failure aborts its execution and must be reported.
        // Such a failure may leave partial effects, so callers must discard the
        // failed simulation or stop the live game rather than retry this state.
        apply(s, actor, *action)?;
        Ok(())
    }
    fn player_result(&self, s: &Self::State, p: PlayerId) -> Result<PlayerResult, Self::Error> {
        s.public.player_result(p)
    }
    fn player_score(&self, s: &Self::State, p: PlayerId) -> Result<f64, Self::Error> {
        s.public.player_score(p)
    }
}

fn actions(p: &LoveLetterPublicState, hand: &[Card], actor: PlayerId) -> Vec<LoveLetterAction> {
    if p.next_player() != Decision::Player(actor) {
        return vec![];
    }
    let forced = hand.contains(&Card::Countess)
        && (hand.contains(&Card::Prince) || hand.contains(&Card::King));
    let mut offered = Vec::new();
    for card in Card::ALL {
        if !hand.contains(&card) || (forced && card != Card::Countess) {
            continue;
        }
        match card {
            Card::Handmaid | Card::Countess | Card::Princess => {
                offered.push(LoveLetterAction::Play(card))
            }
            _ => {
                let mut targeted = false;
                for seat in 0..p.player_count() {
                    let target = PlayerId::new(seat);
                    if !p.active[seat]
                        || p.protected[seat]
                        || (target == actor && card != Card::Prince)
                    {
                        continue;
                    }
                    targeted = true;
                    if card == Card::Guard {
                        for guess in Card::ALL {
                            if guess != Card::Guard {
                                offered.push(LoveLetterAction::Guess {
                                    target,
                                    card: guess,
                                });
                            }
                        }
                    } else {
                        offered.push(LoveLetterAction::Target { card, target });
                    }
                }
                if !targeted {
                    offered.push(LoveLetterAction::Play(card));
                }
            }
        }
    }
    offered
}
fn emit(s: &mut LoveLetterState, event: PublicEvent) -> usize {
    let index = s.public.events.len();
    s.public.events.push(event);
    index
}
fn private(s: &mut LoveLetterState, p: PlayerId, event: usize, information: PrivateInformation) {
    s.private[p.index()].push(PrivateEvent { event, information });
}
fn receive(s: &mut LoveLetterState, p: PlayerId, card: Card, source: DrawSource) {
    s.hands[p.index()].push(card);
    let event = emit(s, PublicEvent::Received { player: p, source });
    private(s, p, event, PrivateInformation::Received { card, source });
}
fn draw(s: &mut LoveLetterState, p: PlayerId) -> Result<(), LoveLetterError> {
    let (card, source) = if let Some(card) = s.pile.pop_front() {
        (card, DrawSource::Pile)
    } else {
        (
            s.removed
                .take()
                .ok_or(LoveLetterError::InvalidState("replacement card missing"))?,
            DrawSource::Removed,
        )
    };
    receive(s, p, card, source);
    refresh_counts(s);
    Ok(())
}
fn refresh_counts(s: &mut LoveLetterState) {
    s.public.draw_count = s.pile.len();
    s.public.removed_count = usize::from(s.removed.is_some());
}
fn start_round(
    s: &mut LoveLetterState,
    starter: PlayerId,
    supplied: Option<&[Card; 16]>,
) -> Result<(), LoveLetterError> {
    let n = s.public.player_count();
    s.public.round = s
        .public
        .counters
        .round_count()
        .checked_add(1)
        .ok_or(LoveLetterError::CounterOverflow)?;
    s.public.current_player = starter;
    s.public.active.fill(true);
    s.public.protected.fill(false);
    s.public.events.clear();
    s.private.iter_mut().for_each(Vec::clear);
    s.hands.iter_mut().for_each(Vec::clear);
    s.public.discards.iter_mut().for_each(Vec::clear);
    s.public.reserve.clear();
    let mut deck = Card::DECK;
    for i in (1..deck.len()).rev() {
        let j = s.random.index(i + 1).expect("nonempty shuffle");
        deck.swap(i, j);
    }
    s.pile = supplied.unwrap_or(&deck).iter().copied().collect();
    s.removed = s.pile.pop_front();
    if n == 2 {
        for _ in 0..3 {
            s.public
                .reserve
                .push(s.pile.pop_front().expect("classic reserve"));
        }
    }
    emit(
        s,
        PublicEvent::RoundStarted {
            round: s.public.round,
            starter,
        },
    );
    for seat in 0..n {
        let card = s.pile.pop_front().expect("classic deal");
        receive(s, PlayerId::new(seat), card, DrawSource::Deal);
    }
    draw(s, starter)?;
    refresh_counts(s);
    Ok(())
}
fn held(s: &LoveLetterState, p: PlayerId) -> Result<Card, LoveLetterError> {
    match s.hands[p.index()].as_slice() {
        [card] => Ok(*card),
        _ => Err(LoveLetterError::InvalidState("expected one retained card")),
    }
}
fn discard(s: &mut LoveLetterState, p: PlayerId, card: Card) {
    s.public.discards[p.index()].push(card);
    emit(s, PublicEvent::Discarded { player: p, card });
}
fn eliminate(s: &mut LoveLetterState, p: PlayerId) {
    s.public.active[p.index()] = false;
    let cards = std::mem::take(&mut s.hands[p.index()]);
    for card in cards {
        discard(s, p, card);
    }
    emit(s, PublicEvent::Eliminated { player: p });
}
fn apply(
    s: &mut LoveLetterState,
    actor: PlayerId,
    action: LoveLetterAction,
) -> Result<(), LoveLetterError> {
    let counters = s.public.counters;
    if [
        counters.move_count(),
        counters.completed_turn_count(),
        counters.turn_count_in_round(),
    ]
    .contains(&usize::MAX)
    {
        return Err(LoveLetterError::CounterOverflow);
    }
    emit(s, PublicEvent::Played { actor, action });
    let card = action.played_card();
    let index = s.hands[actor.index()]
        .iter()
        .position(|c| *c == card)
        .ok_or(LoveLetterError::InvalidState("played card missing"))?;
    s.hands[actor.index()].remove(index);
    discard(s, actor, card);
    match action {
        LoveLetterAction::Guess {
            target,
            card: guess,
        } => {
            let correct = held(s, target)? == guess;
            emit(
                s,
                PublicEvent::GuardResult {
                    actor,
                    target,
                    guess,
                    correct,
                },
            );
            if correct {
                eliminate(s, target);
            }
        }
        LoveLetterAction::Target {
            card: Card::Priest,
            target,
        } => {
            let card = held(s, target)?;
            let event = s.public.events.len() - 1;
            private(s, actor, event, PrivateInformation::Priest { target, card });
        }
        LoveLetterAction::Target {
            card: Card::Baron,
            target,
        } => {
            let own = held(s, actor)?;
            let other = held(s, target)?;
            let loser = match own.cmp(&other) {
                std::cmp::Ordering::Less => Some(actor),
                std::cmp::Ordering::Greater => Some(target),
                _ => None,
            };
            let event = emit(
                s,
                PublicEvent::Compared {
                    actor,
                    target,
                    eliminated: loser,
                },
            );
            private(
                s,
                actor,
                event,
                PrivateInformation::Comparison {
                    opponent: target,
                    own_card: own,
                    other_card: other,
                },
            );
            private(
                s,
                target,
                event,
                PrivateInformation::Comparison {
                    opponent: actor,
                    own_card: other,
                    other_card: own,
                },
            );
            if let Some(loser) = loser {
                eliminate(s, loser);
            }
        }
        LoveLetterAction::Target {
            card: Card::Prince,
            target,
        } => {
            let card = s.hands[target.index()]
                .pop()
                .ok_or(LoveLetterError::InvalidState("Prince target hand empty"))?;
            discard(s, target, card);
            if card == Card::Princess {
                eliminate(s, target);
            } else {
                draw(s, target)?;
            }
        }
        LoveLetterAction::Target {
            card: Card::King,
            target,
        } => {
            let own = held(s, actor)?;
            let other = held(s, target)?;
            s.hands.swap(actor.index(), target.index());
            let event = emit(s, PublicEvent::Exchanged { actor, target });
            private(
                s,
                actor,
                event,
                PrivateInformation::Exchange {
                    other: target,
                    given: own,
                    received: other,
                },
            );
            private(
                s,
                target,
                event,
                PrivateInformation::Exchange {
                    other: actor,
                    given: other,
                    received: own,
                },
            );
        }
        LoveLetterAction::Play(Card::Handmaid) => {
            s.public.protected[actor.index()] = true;
            emit(
                s,
                PublicEvent::Protection {
                    player: actor,
                    enabled: true,
                },
            );
        }
        LoveLetterAction::Play(Card::Princess) => eliminate(s, actor),
        LoveLetterAction::Play(_) => {}
        LoveLetterAction::Target { .. } => {
            return Err(LoveLetterError::InvalidState("unsupported targeted card"));
        }
    }
    s.public.last_actor = Some(actor);
    s.public.counters.record_move();
    s.public.counters.complete_turn();
    if s.public.active.iter().filter(|&&v| v).count() == 1 || s.pile.is_empty() {
        finish_round(s)?;
    } else {
        let n = s.public.player_count();
        let mut next = (actor.index() + 1) % n;
        while !s.public.active[next] {
            next = (next + 1) % n;
        }
        let next = PlayerId::new(next);
        s.public.current_player = next;
        if s.public.protected[next.index()] {
            s.public.protected[next.index()] = false;
            emit(
                s,
                PublicEvent::Protection {
                    player: next,
                    enabled: false,
                },
            );
        }
        draw(s, next)?;
    }
    refresh_counts(s);
    Ok(())
}
fn finish_round(s: &mut LoveLetterState) -> Result<(), LoveLetterError> {
    if s.public.counters.round_count() == usize::MAX {
        return Err(LoveLetterError::CounterOverflow);
    }
    let active: Vec<_> = s
        .public
        .active
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| a.then_some(PlayerId::new(i)))
        .collect();
    let (reason, winners, revealed_hands) = if active.len() == 1 {
        (RoundEnd::LastPlayer, active, vec![])
    } else {
        let hands: Vec<_> = active
            .iter()
            .map(|&p| Ok((p, held(s, p)?)))
            .collect::<Result<_, LoveLetterError>>()?;
        let best = hands
            .iter()
            .map(|(_, c)| c.rank())
            .max()
            .ok_or(LoveLetterError::InvalidState("no round survivors"))?;
        let winners = hands
            .iter()
            .filter_map(|(p, c)| (c.rank() == best).then_some(*p))
            .collect();
        (RoundEnd::DeckEmpty, winners, hands)
    };
    for winner in &winners {
        s.public.tokens[winner.index()] = s.public.tokens[winner.index()]
            .checked_add(1)
            .ok_or(LoveLetterError::CounterOverflow)?;
    }
    emit(
        s,
        PublicEvent::RoundEnded {
            winners: winners.clone(),
            reason,
            revealed_hands: revealed_hands.clone(),
        },
    );
    s.public.counters.complete_round();
    let target = s.public.rules.winning_tokens[s.public.player_count() - 2];
    let game_winners: Vec<_> = s
        .public
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v >= target).then_some(PlayerId::new(i)))
        .collect();
    if !game_winners.is_empty() {
        s.public.outcome = LoveLetterOutcome::Won(game_winners.clone());
        emit(
            s,
            PublicEvent::GameEnded {
                winners: game_winners,
            },
        );
    }
    let public = PublicRound {
        summary: RoundSummary {
            round: s.public.round,
            reason,
            winners: winners.clone(),
            revealed_hands,
            tokens: s.public.tokens.clone(),
        },
        events: s.public.events.clone(),
    };
    s.public.last_round = Some(public);
    s.completed_private = s.private.clone();
    if s.public.outcome == LoveLetterOutcome::Ongoing {
        let index = s
            .random
            .index(winners.len())
            .ok_or(LoveLetterError::InvalidState("no round winner"))?;
        start_round(s, winners[index], None)?;
    }
    Ok(())
}
