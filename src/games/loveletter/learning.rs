//! Versioned, observer-only numeric input. Historical ranks describe their event,
//! not an assertion that the same player still holds that card.
use super::*;
use crate::{
    core::{Decision, ForwardModel, PlayerId, PlayerResult},
    learning::*,
};
const ACTIONS: usize = 52;
const EVENTS: usize = 256;
const PUBLIC_WIDTH: usize = 22;
const PRIVATE_WIDTH: usize = 7;
const TARGETED: [Card; 4] = [Card::Priest, Card::Baron, Card::Prince, Card::King];

#[cfg(test)]
mod tests;
fn invalid(s: impl std::fmt::Display) -> LearningError {
    LearningError::Invalid(s.to_string())
}
fn card_index(c: Card) -> usize {
    Card::ALL.iter().position(|&v| v == c).expect("card type")
}
fn parts(value: u32) -> [f32; 2] {
    [(value >> 16) as f32, (value & 65535) as f32]
}
fn identity(r: &LoveLetterRules, n: usize) -> EncodingIdentity {
    EncodingIdentity {
        game: "loveletter",
        rules_version: 1,
        encoding_version: 1,
        player_count: n,
        configuration: EncodingConfiguration::Named(
            [(
                "winning_tokens".into(),
                r.winning_tokens.map(|v| v.to_string()).join(","),
            )]
            .into(),
        ),
    }
}
fn layouts(n: usize, limits: LearningLimits) -> Result<Vec<TensorSpec>, LearningError> {
    if !(2..=4).contains(&n) {
        return Err(invalid("Love Letter requires 2–4 players"));
    }
    if limits.max_cells < 16 {
        return Err(LearningError::ResourceLimit);
    }
    let tensor = |name, shape, axes, components, bounds| TensorSpec {
        name,
        shape,
        axes,
        components,
        bounds: Some(bounds),
        dtype: DType::Float32,
    };
    let public_components = vec![
        "valid",
        "event_type",
        "actor_plus_one",
        "target_plus_one",
        "card_rank",
        "guess_rank",
        "correct_or_enabled",
        "eliminated_plus_one",
        "draw_source",
        "round_end_reason",
        "round_limb_0",
        "round_limb_1",
        "round_limb_2",
        "round_limb_3",
        "winner_seat_0",
        "winner_seat_1",
        "winner_seat_2",
        "winner_seat_3",
        "revealed_rank_seat_0",
        "revealed_rank_seat_1",
        "revealed_rank_seat_2",
        "revealed_rank_seat_3",
    ];
    let private_components = vec![
        "valid",
        "public_event_index",
        "event_type",
        "other_plus_one",
        "card_rank",
        "other_card_rank",
        "draw_source",
    ];
    Ok(vec![
        tensor(
            "hands",
            vec![n, 2],
            vec!["absolute_seat", "hand_slot"],
            vec!["unknown=-1; absent=0; known=rank"],
            (-1, 9),
        ),
        tensor(
            "players",
            vec![n, 6],
            vec!["absolute_seat", "component"],
            vec![
                "active",
                "protected",
                "tokens_high16",
                "tokens_low16",
                "observer",
                "next_player",
            ],
            (0, 65535),
        ),
        tensor(
            "reserve",
            vec![3],
            vec!["reserve_slot"],
            vec!["absent=0; card=rank"],
            (0, 9),
        ),
        tensor(
            "discards",
            vec![n, 16],
            vec!["absolute_seat", "discard_order"],
            vec!["absent=0; card=rank"],
            (0, 9),
        ),
        tensor(
            "context",
            vec![10],
            vec!["component"],
            vec![
                "draw_count",
                "removed_count",
                "round_limb_0",
                "round_limb_1",
                "round_limb_2",
                "round_limb_3",
                "target_high16",
                "target_low16",
                "last_round_present",
                "last_actor_plus_one",
            ],
            (0, 65535),
        ),
        tensor(
            "public_history",
            vec![EVENTS, PUBLIC_WIDTH],
            vec!["event", "component"],
            public_components.clone(),
            (0, 65535),
        ),
        tensor(
            "private_history",
            vec![EVENTS, PRIVATE_WIDTH],
            vec!["private_event", "component"],
            private_components.clone(),
            (0, 65535),
        ),
        tensor(
            "last_public_history",
            vec![EVENTS, PUBLIC_WIDTH],
            vec!["event", "component"],
            public_components,
            (0, 65535),
        ),
        tensor(
            "last_private_history",
            vec![EVENTS, PRIVATE_WIDTH],
            vec!["private_event", "component"],
            private_components,
            (0, 65535),
        ),
        tensor(
            "last_tokens",
            vec![n, 2],
            vec!["absolute_seat", "high16_low16"],
            vec!["zero_when_no_last_round"],
            (0, 65535),
        ),
        tensor(
            "terminal_values",
            vec![n],
            vec!["absolute_seat"],
            vec!["ongoing=0; whole_game_win=1; whole_game_loss=-1"],
            (-1, 1),
        ),
    ])
}
fn budget(
    specs: &[TensorSpec],
    names: &[&str],
    n: usize,
    offered: usize,
    limits: LearningLimits,
) -> Result<(), LearningError> {
    let mut bytes = ACTIONS + 32 + n + offered * 8;
    for (i, name) in names.iter().enumerate() {
        if names[..i].contains(name) {
            return Err(invalid(format!("duplicate payload: {name}")));
        }
        let spec = specs
            .iter()
            .find(|s| s.name == *name)
            .ok_or_else(|| LearningError::UnsupportedPayload((*name).into()))?;
        bytes += spec.shape.iter().product::<usize>() * 4;
    }
    if bytes > limits.max_bytes {
        return Err(LearningError::ResourceLimit);
    }
    Ok(())
}
fn source(s: DrawSource) -> f32 {
    match s {
        DrawSource::Deal => 1.,
        DrawSource::Pile => 2.,
        DrawSource::Removed => 3.,
    }
}
fn seat(p: PlayerId) -> f32 {
    (p.index() + 1) as f32
}
fn public_rows(events: &[PublicEvent]) -> Result<Vec<f32>, LearningError> {
    if events.len() > EVENTS {
        return Err(invalid("public history capacity exceeded"));
    }
    let mut data = vec![0.; EVENTS * PUBLIC_WIDTH];
    for (event, row) in events
        .iter()
        .zip(data.as_chunks_mut::<PUBLIC_WIDTH>().0.iter_mut())
    {
        row[0] = 1.;
        match event {
            PublicEvent::RoundStarted { round, starter } => {
                row[1] = 1.;
                row[2] = seat(*starter);
                for i in 0..4 {
                    row[10 + i] = ((*round as u64 >> (16 * i)) & 65535) as f32;
                }
            }
            PublicEvent::Received { player, source: s } => {
                row[1] = 2.;
                row[2] = seat(*player);
                row[8] = source(*s);
            }
            PublicEvent::Played { actor, action } => {
                row[1] = 3.;
                row[2] = seat(*actor);
                row[4] = action.played_card().rank() as f32;
                match action {
                    LoveLetterAction::Play(_) => {}
                    LoveLetterAction::Target { target, .. } => row[3] = seat(*target),
                    LoveLetterAction::Guess { target, card } => {
                        row[3] = seat(*target);
                        row[5] = card.rank() as f32;
                    }
                }
            }
            PublicEvent::Discarded { player, card } => {
                row[1] = 4.;
                row[2] = seat(*player);
                row[4] = card.rank() as f32;
            }
            PublicEvent::GuardResult {
                actor,
                target,
                guess,
                correct,
            } => {
                row[1] = 5.;
                row[2] = seat(*actor);
                row[3] = seat(*target);
                row[5] = guess.rank() as f32;
                row[6] = f32::from(*correct);
            }
            PublicEvent::Compared {
                actor,
                target,
                eliminated,
            } => {
                row[1] = 6.;
                row[2] = seat(*actor);
                row[3] = seat(*target);
                row[7] = eliminated.map_or(0., seat);
            }
            PublicEvent::Exchanged { actor, target } => {
                row[1] = 7.;
                row[2] = seat(*actor);
                row[3] = seat(*target);
            }
            PublicEvent::Protection { player, enabled } => {
                row[1] = 8.;
                row[2] = seat(*player);
                row[6] = f32::from(*enabled);
            }
            PublicEvent::Eliminated { player } => {
                row[1] = 9.;
                row[2] = seat(*player);
            }
            PublicEvent::RoundEnded {
                winners,
                reason,
                revealed_hands,
            } => {
                row[1] = 10.;
                row[9] = match reason {
                    RoundEnd::LastPlayer => 1.,
                    RoundEnd::DeckEmpty => 2.,
                };
                for p in winners {
                    row[14 + p.index()] = 1.;
                }
                for (p, c) in revealed_hands {
                    row[18 + p.index()] = c.rank() as f32;
                }
            }
            PublicEvent::GameEnded { winners } => {
                row[1] = 11.;
                for p in winners {
                    row[14 + p.index()] = 1.;
                }
            }
        }
    }
    Ok(data)
}
fn private_rows(events: &[PrivateEvent]) -> Result<Vec<f32>, LearningError> {
    if events.len() > EVENTS {
        return Err(invalid("private history capacity exceeded"));
    }
    let mut data = vec![0.; EVENTS * PRIVATE_WIDTH];
    for (event, row) in events
        .iter()
        .zip(data.as_chunks_mut::<PRIVATE_WIDTH>().0.iter_mut())
    {
        if event.event >= EVENTS {
            return Err(invalid("private event index exceeds capacity"));
        }
        row[0] = 1.;
        row[1] = event.event as f32;
        match event.information {
            PrivateInformation::Received { card, source: s } => {
                row[2] = 1.;
                row[4] = card.rank() as f32;
                row[6] = source(s);
            }
            PrivateInformation::Priest { target, card } => {
                row[2] = 2.;
                row[3] = seat(target);
                row[4] = card.rank() as f32;
            }
            PrivateInformation::Comparison {
                opponent,
                own_card,
                other_card,
            } => {
                row[2] = 3.;
                row[3] = seat(opponent);
                row[4] = own_card.rank() as f32;
                row[5] = other_card.rank() as f32;
            }
            PrivateInformation::Exchange {
                other,
                given,
                received,
            } => {
                row[2] = 4.;
                row[3] = seat(other);
                row[4] = given.rank() as f32;
                row[5] = received.rank() as f32;
            }
        }
    }
    Ok(data)
}
impl LearningGame for LoveLetterForwardModel {
    fn learning_spec(
        &self,
        p: &LoveLetterParameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        self.learning_spec_for_players(p, 2, limits)
    }
    fn learning_spec_for_players(
        &self,
        p: &LoveLetterParameters,
        n: usize,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        let specs = layouts(n, limits)?;
        if p.rules.winning_tokens.contains(&0) || p.first_player.index() >= n {
            return Err(invalid("invalid targets or first player"));
        }
        budget(&specs, &[], n, 0, limits)?;
        Ok(LearningSpec {
            identity: identity(&p.rules, n),
            payloads: specs.into_iter().map(PayloadSpec::Features).collect(),
            metadata: metadata_spec(n),
            actions: ActionSpec {
                size: ACTIONS,
                mapping: ActionMapping::Indexed,
            },
            heuristic: None,
        })
    }
    fn encode(
        &self,
        o: &LoveLetterObservation,
        names: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let p = &o.public;
        let n = p.player_count();
        let specs = layouts(n, limits)?;
        let offered = self
            .offered_actions(o)
            .iter()
            .map(|a| self.action_id(o, a))
            .collect::<Result<Vec<_>, _>>()?;
        budget(&specs, names, n, offered.len(), limits)?;
        let mut tensors = vec![];
        for name in names {
            let spec = specs
                .iter()
                .find(|s| s.name == *name)
                .expect("validated name");
            let values = match *name {
                "hands" => {
                    let mut values = vec![0.; n * 2];
                    for s in 0..n {
                        if p.active[s] {
                            values[s * 2] = -1.;
                            if p.next_player() == Decision::Player(PlayerId::new(s)) {
                                values[s * 2 + 1] = -1.;
                            }
                        }
                    }
                    for e in &p.events {
                        if let PublicEvent::RoundEnded { revealed_hands, .. } = e {
                            for (player, card) in revealed_hands {
                                values[player.index() * 2] = card.rank() as f32;
                            }
                        }
                    }
                    for slot in 0..2 {
                        values[o.observer.index() * 2 + slot] =
                            o.hand.get(slot).map_or(0., |c| c.rank() as f32);
                    }
                    values
                }
                "players" => (0..n)
                    .flat_map(|s| {
                        let [hi, lo] = parts(p.tokens[s]);
                        [
                            f32::from(p.active[s]),
                            f32::from(p.protected[s]),
                            hi,
                            lo,
                            f32::from(s == o.observer.index()),
                            f32::from(p.next_player() == Decision::Player(PlayerId::new(s))),
                        ]
                    })
                    .collect(),
                "reserve" => (0..3)
                    .map(|i| p.reserve.get(i).map_or(0., |c| c.rank() as f32))
                    .collect(),
                "discards" => (0..n)
                    .flat_map(|s| {
                        (0..16).map(move |i| p.discards[s].get(i).map_or(0., |c| c.rank() as f32))
                    })
                    .collect(),
                "context" => {
                    let mut v = vec![p.draw_count as f32, p.removed_count as f32];
                    for i in 0..4 {
                        v.push(((p.round as u64 >> (16 * i)) & 65535) as f32);
                    }
                    v.extend(parts(p.rules.winning_tokens[n - 2]));
                    v.extend([
                        f32::from(p.last_round.is_some()),
                        p.last_actor.map_or(0., seat),
                    ]);
                    v
                }
                "public_history" => public_rows(&p.events)?,
                "private_history" => private_rows(&o.private)?,
                "last_public_history" => {
                    public_rows(p.last_round.as_ref().map_or(&[], |r| r.events.as_slice()))?
                }
                "last_private_history" => private_rows(&o.last_round_private)?,
                "last_tokens" => (0..n)
                    .flat_map(|s| parts(p.last_round.as_ref().map_or(0, |r| r.summary.tokens[s])))
                    .collect(),
                "terminal_values" => (0..n)
                    .map(|s| match &p.outcome {
                        LoveLetterOutcome::Ongoing => 0.,
                        LoveLetterOutcome::Won(winners) => {
                            if winners.contains(&PlayerId::new(s)) {
                                1.
                            } else {
                                -1.
                            }
                        }
                    })
                    .collect(),
                _ => unreachable!("validated name"),
            };
            tensors.push(Float32Tensor {
                name: spec.name,
                shape: spec.shape.clone(),
                values,
            });
        }
        let mut mask = vec![false; ACTIONS];
        for &id in &offered {
            mask[id] = true;
        }
        let c = p.counters;
        let progress = [
            c.move_count(),
            c.completed_turn_count(),
            c.turn_count_in_round(),
            c.round_count(),
        ]
        .map(|v| i64::try_from(v).map_err(|_| LearningError::Overflow));
        let [a, b, c, d] = progress;
        let results = (0..n)
            .map(|s| {
                Ok(match p.player_result(PlayerId::new(s)).map_err(invalid)? {
                    PlayerResult::Ongoing => ResultTag::Ongoing,
                    PlayerResult::Win => ResultTag::Win,
                    PlayerResult::Loss => ResultTag::Loss,
                    PlayerResult::Draw => ResultTag::Draw,
                })
            })
            .collect::<Result<Vec<_>, LearningError>>()?;
        Ok(EncodedObservation {
            identity: identity(&p.rules, n),
            observer: o.observer,
            next_player: match p.next_player() {
                Decision::Finished => None,
                Decision::Player(p) => Some(p),
            },
            tensors,
            progress: [a?, b?, c?, d?],
            results,
            action_mask: mask,
            offered_action_ids: offered,
        })
    }
    fn action_id(
        &self,
        _: &LoveLetterObservation,
        a: &LoveLetterAction,
    ) -> Result<usize, LearningError> {
        Ok(match *a {
            LoveLetterAction::Play(c) => card_index(c),
            LoveLetterAction::Target { card, target } => {
                if target.index() >= 4 {
                    return Err(invalid("target outside four-seat action domain"));
                }
                8 + 4 * TARGETED
                    .iter()
                    .position(|&c| c == card)
                    .ok_or_else(|| invalid("unsupported targeted card"))?
                    + target.index()
            }
            LoveLetterAction::Guess { target, card } => {
                if target.index() >= 4 || card == Card::Guard {
                    return Err(invalid("invalid Guard guess coordinate"));
                }
                24 + 7 * target.index() + card_index(card) - 1
            }
        })
    }
    fn action_from_id(
        &self,
        _: &LoveLetterObservation,
        id: usize,
    ) -> Result<LoveLetterAction, LearningError> {
        Ok(match id {
            0..=7 => LoveLetterAction::Play(Card::ALL[id]),
            8..=23 => LoveLetterAction::Target {
                card: TARGETED[(id - 8) / 4],
                target: PlayerId::new((id - 8) % 4),
            },
            24..=51 => LoveLetterAction::Guess {
                target: PlayerId::new((id - 24) / 7),
                card: Card::ALL[1 + (id - 24) % 7],
            },
            _ => return Err(invalid("action ID must be in 0..52")),
        })
    }
}
