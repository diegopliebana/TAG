//! Fixed action IDs and observation-only arrays for Python and neural search.
use super::*;
use crate::learning::{
    ActionMapping, ActionSpec, DType, EncodedObservation, EncodingConfiguration, EncodingIdentity,
    Float32Tensor, LearningError, LearningGame, LearningLimits, LearningSpec, PayloadSpec,
    ResultTag, TensorSpec, metadata_spec,
};

const ACTIONS: usize = 13 + CARD_COUNT * PILE_COUNT + 1;
const END_TURN_ID: usize = ACTIONS - 1;
const HISTORY: usize = CARD_COUNT;

fn invalid(message: impl Into<String>) -> LearningError {
    LearningError::Invalid(message.into())
}
fn identity(n: usize, communication: CommunicationMode) -> EncodingIdentity {
    EncodingIdentity {
        game: "thegame",
        rules_version: 1,
        encoding_version: 1,
        player_count: n,
        configuration: EncodingConfiguration::Named(
            [(
                "communication".into(),
                match communication {
                    CommunicationMode::LimitedChoices => "limited_choices".into(),
                    CommunicationMode::Silent => "silent".into(),
                },
            )]
            .into(),
        ),
    }
}
fn tensor(
    name: &'static str,
    shape: Vec<usize>,
    axes: Vec<&'static str>,
    components: Vec<&'static str>,
    bounds: (i64, i64),
) -> PayloadSpec {
    PayloadSpec::Features(TensorSpec {
        name,
        dtype: DType::Float32,
        shape,
        axes,
        components,
        bounds: Some(bounds),
    })
}
fn layouts(n: usize) -> Vec<PayloadSpec> {
    vec![
        tensor(
            "hand",
            vec![CARD_COUNT],
            vec!["card_2_to_99"],
            vec!["own_card_present"],
            (0, 1),
        ),
        tensor(
            "piles",
            vec![PILE_COUNT],
            vec!["pile"],
            vec![
                "ascending_one",
                "ascending_two",
                "descending_one",
                "descending_two",
            ],
            (1, 100),
        ),
        tensor(
            "players",
            vec![n, 2],
            vec!["absolute_seat", "component"],
            vec!["hand_count", "is_observer"],
            (0, 8),
        ),
        tensor(
            "context",
            vec![9],
            vec!["component"],
            vec![
                "first_player_plus_one",
                "active_player_plus_one",
                "next_speaker_plus_one_or_zero",
                "draw_count",
                "played_this_turn",
                "turns_completed",
                "placement_count",
                "communication_limited",
                "outcome_ongoing_1_win_2_loss_3",
            ],
            (0, 100),
        ),
        tensor(
            "placements",
            vec![HISTORY, 3],
            vec!["placement", "component"],
            vec!["actor_plus_one", "pile_plus_one", "card"],
            (0, 99),
        ),
        tensor(
            "messages",
            vec![HISTORY, n + 1],
            vec!["round", "component"],
            vec!["active_player_plus_one_then_message_ids_by_seat"],
            (0, 13),
        ),
        tensor(
            "terminal_values",
            vec![n],
            vec!["absolute_seat"],
            vec!["ongoing=0; team_win=1; team_loss=-1"],
            (-1, 1),
        ),
    ]
}
fn message_id(message: Message) -> usize {
    match message {
        Message::NoRequest => 0,
        Message::LeavePile(pile) => 1 + 3 * pile.index(),
        Message::SmallJump(pile) => 2 + 3 * pile.index(),
        Message::CanUsePile(pile) => 3 + 3 * pile.index(),
    }
}
fn message_from_id(id: usize) -> Message {
    if id == 0 {
        return Message::NoRequest;
    }
    let pile = Pile::ALL[(id - 1) / 3];
    match (id - 1) % 3 {
        0 => Message::LeavePile(pile),
        1 => Message::SmallJump(pile),
        _ => Message::CanUsePile(pile),
    }
}
fn validate(o: &Observation) -> Result<usize, LearningError> {
    let p = &o.public;
    let n = p.hand_counts.len();
    if !(1..=5).contains(&n)
        || o.observer.index() >= n
        || p.first_player.index() >= n
        || p.active_player.index() >= n
        || o.hand.len() != p.hand_counts[o.observer.index()]
        || p.placements.len() > HISTORY
        || (p.outcome == TeamOutcome::Won) != (p.placements.len() == CARD_COUNT)
        || p.messages.len() > HISTORY
        || p.hand_counts.iter().sum::<usize>() + p.draw_count + p.placements.len() != CARD_COUNT
        || p.messages.iter().any(|r| r.messages.len() != n)
    {
        return Err(invalid("inconsistent The Game observation"));
    }
    Ok(n)
}
impl LearningGame for TheGameForwardModel {
    fn learning_spec(
        &self,
        p: &Parameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        self.learning_spec_for_players(p, 2, limits)
    }
    fn learning_spec_for_players(
        &self,
        p: &Parameters,
        n: usize,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        if !(1..=5).contains(&n) || p.first_player.index() >= n {
            return Err(invalid(
                "The Game requires 1–5 players and a valid first player",
            ));
        }
        if limits.max_cells < CARD_COUNT {
            return Err(LearningError::ResourceLimit);
        }
        if ACTIONS + 32 + n > limits.max_bytes {
            return Err(LearningError::ResourceLimit);
        }
        Ok(LearningSpec {
            identity: identity(n, p.communication),
            payloads: layouts(n),
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
        o: &Observation,
        names: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let n = validate(o)?;
        if limits.max_cells < CARD_COUNT {
            return Err(LearningError::ResourceLimit);
        }
        let specs = layouts(n);
        let offered_action_ids = self
            .offered_actions(o)
            .iter()
            .map(|a| self.action_id(o, a))
            .collect::<Result<Vec<_>, _>>()?;
        let mut bytes = ACTIONS + 32 + n + offered_action_ids.len() * 8;
        let mut tensors = Vec::new();
        for (i, &name) in names.iter().enumerate() {
            if names[..i].contains(&name) {
                return Err(invalid(format!("duplicate payload: {name}")));
            }
            let spec = specs
                .iter()
                .find(|s| s.name() == name)
                .ok_or_else(|| LearningError::UnsupportedPayload(name.into()))?
                .tensor()
                .expect("feature tensor");
            bytes = bytes
                .checked_add(spec.shape.iter().try_fold(4usize, |size, &axis| {
                    size.checked_mul(axis).ok_or(LearningError::Overflow)
                })?)
                .ok_or(LearningError::Overflow)?;
            if bytes > limits.max_bytes {
                return Err(LearningError::ResourceLimit);
            }
            let p = &o.public;
            let values = match name {
                "hand" => (2..=99)
                    .map(|card| f32::from(o.hand.contains(&card)))
                    .collect(),
                "piles" => p.pile_tops.iter().map(|&top| f32::from(top)).collect(),
                "players" => (0..n)
                    .flat_map(|seat| {
                        [
                            p.hand_counts[seat] as f32,
                            f32::from(seat == o.observer.index()),
                        ]
                    })
                    .collect(),
                "context" => vec![
                    (p.first_player.index() + 1) as f32,
                    (p.active_player.index() + 1) as f32,
                    match p.phase {
                        Phase::Communicate { speaker } => (speaker.index() + 1) as f32,
                        _ => 0.,
                    },
                    p.draw_count as f32,
                    p.played_this_turn as f32,
                    p.turns_completed as f32,
                    p.placements.len() as f32,
                    f32::from(p.communication == CommunicationMode::LimitedChoices),
                    match p.outcome {
                        TeamOutcome::Ongoing => 1.,
                        TeamOutcome::Won => 2.,
                        TeamOutcome::Lost => 3.,
                    },
                ],
                "placements" => {
                    let mut data = vec![0.; HISTORY * 3];
                    for (i, entry) in p.placements.iter().enumerate() {
                        data[i * 3..i * 3 + 3].copy_from_slice(&[
                            (entry.player.index() + 1) as f32,
                            (entry.pile.index() + 1) as f32,
                            entry.card as f32,
                        ]);
                    }
                    data
                }
                "messages" => {
                    let mut data = vec![0.; HISTORY * (n + 1)];
                    for (i, round) in p.messages.iter().enumerate() {
                        let row = &mut data[i * (n + 1)..(i + 1) * (n + 1)];
                        row[0] = (round.active_player.index() + 1) as f32;
                        for (seat, &message) in round.messages.iter().enumerate() {
                            row[seat + 1] = (message_id(message) + 1) as f32;
                        }
                    }
                    data
                }
                "terminal_values" => vec![
                    match p.outcome {
                        TeamOutcome::Ongoing => 0.,
                        TeamOutcome::Won => 1.,
                        TeamOutcome::Lost => -1.,
                    };
                    n
                ],
                _ => unreachable!("validated payload"),
            };
            tensors.push(Float32Tensor {
                name: spec.name,
                shape: spec.shape.clone(),
                values,
            });
        }
        let mut action_mask = vec![false; ACTIONS];
        for &id in &offered_action_ids {
            action_mask[id] = true;
        }
        let next_player = match o.public.phase {
            Phase::Communicate { speaker } => Some(speaker),
            Phase::Play => Some(o.public.active_player),
            Phase::Finished => None,
        };
        let pending_messages = match o.public.phase {
            Phase::Communicate { speaker } => speaker.index(),
            _ => 0,
        };
        let progress = [
            o.public.placements.len()
                + o.public.messages.len() * n
                + o.public.turns_completed
                + pending_messages,
            o.public.turns_completed,
            o.public.turns_completed,
            0,
        ]
        .map(|value| i64::try_from(value).map_err(|_| LearningError::Overflow));
        let [a, b, c, d] = progress;
        Ok(EncodedObservation {
            identity: identity(n, o.public.communication),
            observer: o.observer,
            next_player,
            tensors,
            progress: [a?, b?, c?, d?],
            results: vec![
                match o.public.outcome {
                    TeamOutcome::Ongoing => ResultTag::Ongoing,
                    TeamOutcome::Won => ResultTag::Win,
                    TeamOutcome::Lost => ResultTag::Loss,
                };
                n
            ],
            action_mask,
            offered_action_ids,
        })
    }
    fn action_id(&self, _: &Observation, action: &Action) -> Result<usize, LearningError> {
        Ok(match *action {
            Action::Speak(message) => message_id(message),
            Action::Play { card, pile } if (2..=99).contains(&card) => {
                13 + (usize::from(card) - 2) * PILE_COUNT + pile.index()
            }
            Action::Play { .. } => return Err(invalid("card outside 2..99")),
            Action::EndTurn => END_TURN_ID,
        })
    }
    fn action_from_id(&self, _: &Observation, id: usize) -> Result<Action, LearningError> {
        Ok(match id {
            0..=12 => Action::Speak(message_from_id(id)),
            13..=404 => Action::Play {
                card: (2 + (id - 13) / PILE_COUNT) as u8,
                pile: Pile::ALL[(id - 13) % PILE_COUNT],
            },
            END_TURN_ID => Action::EndTurn,
            _ => return Err(invalid("action ID outside 0..406")),
        })
    }
}
