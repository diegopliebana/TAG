//! Validate complete snapshots using at most the current and last retained round.
use super::*;
use crate::core::{
    progress::TurnCounters,
    random::{ChanceRng, ChanceRngSnapshot},
};
use crate::replay::{RngData, counter_data};

/// Privileged persistence data, created only when saving. No match-long action
/// vector is kept in LoveLetterState or in an agent's observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedSnapshot {
    pub public: LoveLetterPublicState,
    pub hands: Vec<Vec<Card>>,
    pub pile: Vec<Card>,
    pub removed: Option<Card>,
    pub chance: RngData,
    pub private: Vec<Vec<PrivateEvent>>,
    pub completed_private: Vec<Vec<PrivateEvent>>,
}

pub(crate) mod counters {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &TurnCounters,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        counter_data(*value).serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<TurnCounters, D::Error> {
        let counts = <[usize; 4]>::deserialize(deserializer)?;
        TurnCounters::from_counts(counts)
            .ok_or_else(|| serde::de::Error::custom("invalid Love Letter counters"))
    }
}

impl SavedSnapshot {
    pub fn from_state(s: &LoveLetterState) -> Self {
        Self {
            public: s.public.clone(),
            hands: s.hands.clone(),
            pile: s.pile.iter().copied().collect(),
            removed: s.removed,
            chance: RngData::from_snapshot(s.random.snapshot().generator),
            private: s.private.clone(),
            completed_private: s.completed_private.clone(),
        }
    }
    pub fn validation_work(&self) -> Option<usize> {
        let mut count = self.public.events.len().checked_add(
            self.public
                .last_round
                .as_ref()
                .map_or(0, |r| r.events.len()),
        )?;
        for rows in self.private.iter().chain(&self.completed_private) {
            count = count.checked_add(rows.len())?;
        }
        count.checked_mul(64)?.checked_add(4096)
    }
    pub fn restore(
        &self,
        model: &LoveLetterForwardModel,
        parameters: &LoveLetterParameters,
    ) -> Result<LoveLetterState, PersistenceError> {
        let p = &self.public;
        let n = p.player_count();
        // Validate dimensions before indexing any supplied seat/collection.
        if !(2..=4).contains(&n)
            || p.rules != parameters.rules
            || parameters.first_player.index() >= n
            || parameters.rules.winning_tokens.contains(&0)
            || p.round == 0
            || p.round == usize::MAX
            || [
                self.hands.len(),
                self.private.len(),
                self.completed_private.len(),
                p.active.len(),
                p.protected.len(),
                p.discards.len(),
            ]
            .iter()
            .any(|&len| len != n)
            || self.hands.iter().any(|h| h.len() > 2)
            || self.pile.len() > 16
            || p.discards.iter().any(|d| d.len() > 16)
            || p.current_player.index() >= n
            || p.last_actor.is_some_and(|a| a.index() >= n)
            || p.reserve.len() != if n == 2 { 3 } else { 0 }
        {
            return Err(invalid(
                "invalid Love Letter snapshot rules, players or dimensions",
            ));
        }
        bounded(&p.events, &self.private)?;
        if let Some(last) = &p.last_round {
            bounded(&last.events, &self.completed_private)?;
        } else if self.completed_private.iter().any(|v| !v.is_empty()) {
            return Err(invalid("private memory has no completed round"));
        }
        let terminal = !matches!(p.outcome, LoveLetterOutcome::Ongoing);
        let played = play_count(&p.events);
        let [moves, turns, within, rounds] = counter_data(p.counters);
        if moves != turns
            || moves == usize::MAX
            || rounds != p.round - usize::from(!terminal)
            || within != if terminal { 0 } else { played }
        {
            return Err(invalid(
                "Love Letter counters disagree with round/turn history",
            ));
        }
        let before = moves
            .checked_sub(played)
            .ok_or_else(|| invalid("too many current-round actions"))?;
        let earlier_rounds = p.round - 1;
        if before < earlier_rounds
            || before as u128 > (earlier_rounds as u128) * 16
            || p.tokens.iter().any(|&v| u128::from(v) > rounds as u128)
            || p.tokens.iter().map(|&v| u128::from(v)).sum::<u128>() < rounds as u128
            || p.tokens.iter().map(|&v| u128::from(v)).sum::<u128>() > (rounds as u128) * n as u128
        {
            return Err(invalid(
                "Love Letter scores/counters cannot fit the completed rounds",
            ));
        }
        let rng = self.chance.snapshot()?;
        let initial = ChanceRng::new(parameters.seed).snapshot().generator;
        if rng.initial_seed != parameters.seed
            || rng.stream != initial.stream
            || rng.word_position % 2 != 0
            || rng.word_position < (p.round as u128) * 30
        {
            return Err(invalid(
                "Love Letter real-game chance seed/stream/progress disagrees",
            ));
        }
        // All first-round chance is the initial shuffle, including controlled deals.
        if p.round == 1
            && rng
                != model
                    .init_state(parameters, n)
                    .map_err(invalid)?
                    .random
                    .snapshot()
                    .generator
        {
            return Err(invalid("Love Letter first-round chance progress disagrees"));
        }
        let starter = starter(&p.events, p.round, n)?;
        if p.round == 1 && starter != parameters.first_player {
            return Err(invalid("Love Letter initial starter disagrees"));
        }
        let tokens_before = initial_tokens(&p.tokens, &p.events, terminal)?;
        score_bounds(&tokens_before, earlier_rounds)?;
        let target = p.rules.winning_tokens[n - 2];
        if tokens_before.iter().any(|&v| v >= target) {
            return Err(invalid("round started after a game-winning score"));
        }
        let mut previous = None;
        let mut previous_private = vec![vec![]; n];
        let mut previous_actor = None;
        if !terminal && p.round > 1 {
            let last = p
                .last_round
                .as_ref()
                .ok_or_else(|| invalid("missing previous round"))?;
            if last.summary.round != p.round - 1
                || last.summary.tokens != p.tokens
                || !last.summary.winners.contains(&starter)
            {
                return Err(invalid("previous round scores/starter disagree"));
            }
            let last_moves = play_count(&last.events);
            let prior = before
                .checked_sub(last_moves)
                .ok_or_else(|| invalid("previous-round counters disagree"))?;
            if prior < p.round - 2 || prior as u128 > (p.round as u128 - 2) * 16 {
                return Err(invalid("previous-round progress disagrees"));
            }
            let deck = deck(&last.events, &self.completed_private, n, None, None, None)?;
            let base_tokens = initial_tokens(&p.tokens, &last.events, true)?;
            score_bounds(&base_tokens, p.round - 2)?;
            if base_tokens.iter().any(|&v| v >= target) {
                return Err(invalid("previous round started after game end"));
            }
            let replayed = replay_round(
                model,
                parameters,
                &last.events,
                &deck,
                base_tokens,
                [prior, prior, 0, p.round - 2],
                n,
                None,
                vec![vec![]; n],
                None,
            )?;
            if replayed.public.outcome != LoveLetterOutcome::Ongoing
                || replayed.public.last_round.as_ref() != Some(last)
                || replayed.completed_private != self.completed_private
            {
                return Err(invalid(
                    "previous-round public/private information disagrees with rules",
                ));
            }
            previous_actor = last.events.iter().rev().find_map(|e| {
                if let PublicEvent::Played { actor, .. } = e {
                    Some(*actor)
                } else {
                    None
                }
            });
            previous = Some(last.clone());
            previous_private = self.completed_private.clone();
        } else if !terminal
            && (p.last_round.is_some() || self.completed_private.iter().any(|v| !v.is_empty()))
        {
            return Err(invalid("first round has previous-round information"));
        }
        let deck = deck(
            &p.events,
            &self.private,
            n,
            Some(&p.reserve),
            Some(&self.pile),
            Some(self.removed),
        )?;
        let mut restored = replay_round(
            model,
            parameters,
            &p.events,
            &deck,
            tokens_before,
            [before, before, 0, earlier_rounds],
            n,
            previous,
            previous_private,
            previous_actor,
        )?;
        restored.random =
            ChanceRng::from_snapshot(&ChanceRngSnapshot { generator: rng }).map_err(invalid)?;
        if restored.public != self.public
            || restored.hands != self.hands
            || restored.pile.iter().copied().collect::<Vec<_>>() != self.pile
            || restored.removed != self.removed
            || restored.private != self.private
            || restored.completed_private != self.completed_private
        {
            return Err(invalid(
                "Love Letter snapshot facts or private knowledge disagree with rule replay",
            ));
        }
        Ok(restored)
    }
}

fn bounded(events: &[PublicEvent], private: &[Vec<PrivateEvent>]) -> Result<(), PersistenceError> {
    if events.len() > 256 || private.iter().any(|rows| rows.len() > 128) {
        return Err(invalid(
            "Love Letter retained-round history exceeds its bound",
        ));
    }
    Ok(())
}
fn play_count(events: &[PublicEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, PublicEvent::Played { .. }))
        .count()
}

fn score_bounds(tokens: &[u32], rounds: usize) -> Result<(), PersistenceError> {
    let total = tokens.iter().map(|&v| u128::from(v)).sum::<u128>();
    if tokens.iter().any(|&v| u128::from(v) > rounds as u128)
        || total < rounds as u128
        || total > (rounds as u128) * tokens.len() as u128
    {
        return Err(invalid("scores cannot fit the earlier rounds"));
    }
    Ok(())
}
fn starter(events: &[PublicEvent], round: usize, n: usize) -> Result<PlayerId, PersistenceError> {
    match events.first() {
        Some(PublicEvent::RoundStarted { round: r, starter })
            if *r == round && starter.index() < n =>
        {
            Ok(*starter)
        }
        _ => Err(invalid("invalid Love Letter round start")),
    }
}
fn initial_tokens(
    tokens: &[u32],
    events: &[PublicEvent],
    ended: bool,
) -> Result<Vec<u32>, PersistenceError> {
    let mut result = tokens.to_vec();
    if ended {
        let winners = events
            .iter()
            .find_map(|e| {
                if let PublicEvent::RoundEnded { winners, .. } = e {
                    Some(winners)
                } else {
                    None
                }
            })
            .ok_or_else(|| invalid("missing round winners"))?;
        let mut seen = vec![false; tokens.len()];
        if winners.is_empty() {
            return Err(invalid("empty round winners"));
        }
        for &winner in winners {
            let i = winner.index();
            if i >= result.len() || seen[i] {
                return Err(invalid("invalid/duplicate round winner"));
            }
            seen[i] = true;
            result[i] = result[i]
                .checked_sub(1)
                .ok_or_else(|| invalid("round score underflow"))?;
        }
    }
    Ok(result)
}

/// Recover only a retained round's deal from its actual private draws. Unknown
/// unused cards in the previous round can be completed arbitrarily: they never
/// influenced its public/private effects and are not part of the saved state.
fn deck(
    events: &[PublicEvent],
    private: &[Vec<PrivateEvent>],
    n: usize,
    reserve: Option<&[Card]>,
    pile: Option<&[Card]>,
    removed: Option<Option<Card>>,
) -> Result<[Card; 16], PersistenceError> {
    let reserve_count = if n == 2 { 3 } else { 0 };
    let mut slots = [None; 16];
    if let Some(reserve) = reserve {
        for (i, &card) in reserve.iter().enumerate() {
            slots[i + 1] = Some(card);
        }
    }
    let mut next = 1 + reserve_count;
    let mut used_removed = false;
    for (index, event) in events.iter().enumerate() {
        if let PublicEvent::Received { player, source } = event {
            let rows = private
                .get(player.index())
                .ok_or_else(|| invalid("unknown draw recipient"))?;
            let card = rows
                .iter()
                .find_map(|e| match e.information {
                    PrivateInformation::Received { card, source: s }
                        if e.event == index && s == *source =>
                    {
                        Some(card)
                    }
                    _ => None,
                })
                .ok_or_else(|| invalid("missing private draw information"))?;
            if *source == DrawSource::Removed {
                if used_removed {
                    return Err(invalid("removed card drawn twice"));
                }
                used_removed = true;
                slots[0] = Some(card);
            } else {
                let slot = slots
                    .get_mut(next)
                    .ok_or_else(|| invalid("too many round draws"))?;
                *slot = Some(card);
                next += 1;
            }
        }
    }
    if let Some(pile) = pile {
        if pile.len() != 16 - next {
            return Err(invalid("draw pile length disagrees with draws"));
        }
        for (slot, &card) in slots[next..].iter_mut().zip(pile) {
            *slot = Some(card);
        }
    }
    if let Some(removed) = removed {
        if used_removed != removed.is_none() {
            return Err(invalid("removed-card zone disagrees with draws"));
        }
        if let Some(card) = removed {
            slots[0] = Some(card);
        }
    }
    let mut remaining = Card::DECK.to_vec();
    for &card in slots.iter().flatten() {
        let index = remaining
            .iter()
            .position(|&c| c == card)
            .ok_or_else(|| invalid("Love Letter card multiset disagrees"))?;
        remaining.remove(index);
    }
    let mut remaining = remaining.into_iter();
    Ok(slots.map(|card| {
        card.unwrap_or_else(|| remaining.next().expect("sixteen validated card slots"))
    }))
}

#[allow(clippy::too_many_arguments)] // Explicit bounded snapshot context; no match-wide proof/state is retained.
fn replay_round(
    model: &LoveLetterForwardModel,
    parameters: &LoveLetterParameters,
    events: &[PublicEvent],
    deck: &[Card; 16],
    tokens: Vec<u32>,
    counts: [usize; 4],
    n: usize,
    previous: Option<PublicRound>,
    previous_private: Vec<Vec<PrivateEvent>>,
    last_actor: Option<PlayerId>,
) -> Result<LoveLetterState, PersistenceError> {
    let round = counts[3]
        .checked_add(1)
        .ok_or_else(|| invalid("round overflow"))?;
    let first = starter(events, round, n)?;
    let setup = LoveLetterParameters {
        first_player: first,
        ..parameters.clone()
    };
    let mut state = model.init_with_deck(&setup, n, deck).map_err(invalid)?;
    state.public.round = round;
    state.public.events[0] = PublicEvent::RoundStarted {
        round,
        starter: first,
    };
    state.public.tokens = tokens;
    state.public.counters =
        TurnCounters::from_counts(counts).ok_or_else(|| invalid("invalid round counters"))?;
    state.public.last_round = previous;
    state.completed_private = previous_private;
    state.public.last_actor = last_actor;
    for event in events {
        if let PublicEvent::Played { actor, action } = event {
            model.next(&mut state, *actor, action).map_err(invalid)?;
        }
    }
    Ok(state)
}
