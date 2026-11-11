//! Fixed policy IDs and numeric input derived solely from one seat's observation.
use super::*;
use crate::learning::{
    ActionMapping, ActionSpec, DType, EncodedObservation, EncodingConfiguration, EncodingIdentity,
    Float32Tensor, LearningError, LearningGame, LearningLimits, LearningSpec, PayloadSpec,
    ResultTag, TensorSpec, metadata_spec,
};

const CARD_TYPES: usize = Card::ALL.len();

fn invalid(message: impl Into<String>) -> LearningError {
    LearningError::Invalid(message.into())
}

fn identity(players: usize) -> EncodingIdentity {
    EncodingIdentity {
        game: "sushigo",
        rules_version: 1,
        encoding_version: 1,
        player_count: players,
        configuration: EncodingConfiguration::Named(
            [("edition".into(), "original-108".into())].into(),
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

fn payloads(players: usize) -> Vec<PayloadSpec> {
    vec![
        tensor(
            "hand",
            vec![CARD_TYPES],
            vec!["card_id"],
            vec!["own_card_count"],
            (0, 14),
        ),
        tensor(
            "known_hands",
            vec![players, CARD_TYPES],
            vec!["hand_id", "card_id"],
            vec!["unknown=-1; otherwise exact remaining count"],
            (-1, 14),
        ),
        tensor(
            "played",
            vec![players, CARD_TYPES],
            vec!["absolute_seat", "card_id"],
            vec!["visible_card_count"],
            (0, 14),
        ),
        tensor(
            "paired_nigiri",
            vec![players, 3],
            vec!["absolute_seat", "nigiri_type"],
            vec!["egg", "salmon", "squid"],
            (0, 10),
        ),
        tensor(
            "players",
            vec![players, 4],
            vec!["absolute_seat", "component"],
            vec!["score", "pudding_count", "current_hand_id", "is_observer"],
            (-6, 200),
        ),
        tensor(
            "context",
            vec![5],
            vec!["component"],
            vec![
                "round",
                "hand_size",
                "draw_count",
                "finished",
                "completed_rounds",
            ],
            (0, 108),
        ),
        tensor(
            "discard",
            vec![CARD_TYPES],
            vec!["card_id"],
            vec!["discarded_card_count"],
            (0, 14),
        ),
        tensor(
            "last_reveal",
            vec![players, 3],
            vec!["absolute_seat", "component"],
            vec![
                "first_card_id_plus_one",
                "second_card_id_plus_one",
                "used_chopsticks",
            ],
            (0, 12),
        ),
        tensor(
            "round_scores",
            vec![3, players],
            vec!["round_index", "absolute_seat"],
            vec!["zero_for_unfinished_round"],
            (0, 100),
        ),
        tensor(
            "terminal_values",
            vec![players],
            vec!["absolute_seat"],
            vec!["ongoing=0; win=1; loss=-1"],
            (-1, 1),
        ),
    ]
}

fn counts(cards: impl IntoIterator<Item = Card>) -> Vec<f32> {
    let mut result = vec![0.; CARD_TYPES];
    for card in cards {
        result[card.id()] += 1.;
    }
    result
}

fn validate(observation: &Observation) -> Result<usize, LearningError> {
    let p = &observation.public;
    let n = p.scores.len();
    if !(2..=5).contains(&n)
        || observation.observer.index() >= n
        || p.pudding.len() != n
        || p.played.len() != n
        || p.hand_ids.len() != n
        || observation.known_hands.len() != n
        || p.round_scores.len() > 3
        || p.round_scores.iter().any(|r| r.len() != n)
        || p.last_reveal.len() > n
        || p.hand_size != observation.hand.len()
        || p.hand_size > 12 - n
        || (p.hand_size == 0) != p.finished
        || !(1..=3).contains(&p.round)
        || p.hand_ids.iter().any(|&id| id >= n)
        || (0..n).any(|id| p.hand_ids.iter().filter(|&&v| v == id).count() != 1)
        || observation.known_hands[p.hand_ids[observation.observer.index()]].as_ref()
            != Some(&observation.hand)
        || observation
            .known_hands
            .iter()
            .flatten()
            .any(|hand| hand.len() != p.hand_size)
    {
        return Err(invalid("inconsistent Sushi Go! observation"));
    }
    Ok(n)
}

fn result_tag(result: PlayerResult) -> ResultTag {
    match result {
        PlayerResult::Ongoing => ResultTag::Ongoing,
        PlayerResult::Win => ResultTag::Win,
        PlayerResult::Loss => ResultTag::Loss,
        PlayerResult::Draw => ResultTag::Draw,
    }
}

impl LearningGame for SushiGoForwardModel {
    fn learning_spec(
        &self,
        parameters: &Parameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        self.learning_spec_for_players(parameters, 2, limits)
    }

    fn learning_spec_for_players(
        &self,
        _: &Parameters,
        players: usize,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        if !(2..=5).contains(&players) {
            return Err(invalid("Sushi Go! requires 2–5 players"));
        }
        if limits.max_cells < players * CARD_TYPES {
            return Err(LearningError::ResourceLimit);
        }
        let specs = payloads(players);
        let bytes = Action::ACTION_COUNT + 32 + players;
        if bytes > limits.max_bytes {
            return Err(LearningError::ResourceLimit);
        }
        Ok(LearningSpec {
            identity: identity(players),
            payloads: specs,
            metadata: metadata_spec(players),
            actions: ActionSpec {
                size: Action::ACTION_COUNT,
                mapping: ActionMapping::Indexed,
            },
            heuristic: None,
        })
    }

    fn encode(
        &self,
        observation: &Observation,
        names: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let n = validate(observation)?;
        if limits.max_cells < n * CARD_TYPES {
            return Err(LearningError::ResourceLimit);
        }
        let p = &observation.public;
        let specs = payloads(n);
        let offered_action_ids = self
            .offered_actions(observation)
            .into_iter()
            .map(Action::id)
            .collect::<Vec<_>>();
        let mut size = Action::ACTION_COUNT + 32 + n;
        size = size
            .checked_add(
                offered_action_ids
                    .len()
                    .checked_mul(8)
                    .ok_or(LearningError::Overflow)?,
            )
            .ok_or(LearningError::Overflow)?;
        let mut selected = Vec::new();
        for &name in names {
            if selected.contains(&name) {
                return Err(invalid(format!("duplicate payload: {name}")));
            }
            let spec = specs
                .iter()
                .find(|spec| spec.name() == name)
                .ok_or_else(|| LearningError::UnsupportedPayload(name.into()))?;
            size = size
                .checked_add(
                    spec.tensor()
                        .expect("feature tensor")
                        .shape
                        .iter()
                        .try_fold(4usize, |acc, &axis| {
                            acc.checked_mul(axis).ok_or(LearningError::Overflow)
                        })?,
                )
                .ok_or(LearningError::Overflow)?;
            selected.push(name);
        }
        if size > limits.max_bytes {
            return Err(LearningError::ResourceLimit);
        }
        let mut tensors = Vec::new();
        for &name in names {
            let spec = specs
                .iter()
                .find_map(|spec| spec.tensor().filter(|spec| spec.name == name))
                .expect("validated payload");
            let values = match name {
                "hand" => counts(observation.hand.iter().copied()),
                "known_hands" => observation
                    .known_hands
                    .iter()
                    .flat_map(|hand| match hand {
                        Some(cards) => counts(cards.iter().copied()),
                        None => vec![-1.; CARD_TYPES],
                    })
                    .collect(),
                "played" => p
                    .played
                    .iter()
                    .flat_map(|cards| counts(cards.iter().map(|c| c.card)))
                    .collect(),
                "paired_nigiri" => p
                    .played
                    .iter()
                    .flat_map(|cards| {
                        [Card::EggNigiri, Card::SalmonNigiri, Card::SquidNigiri].map(|card| {
                            cards
                                .iter()
                                .filter(|c| c.card == card && c.on_wasabi)
                                .count() as f32
                        })
                    })
                    .collect(),
                "players" => (0..n)
                    .flat_map(|seat| {
                        [
                            p.scores[seat] as f32,
                            p.pudding[seat] as f32,
                            p.hand_ids[seat] as f32,
                            f32::from(seat == observation.observer.index()),
                        ]
                    })
                    .collect(),
                "context" => vec![
                    p.round as f32,
                    p.hand_size as f32,
                    p.draw_count as f32,
                    f32::from(p.finished),
                    p.round_scores.len() as f32,
                ],
                "discard" => counts(p.discard.iter().copied()),
                "last_reveal" => (0..n)
                    .flat_map(|seat| {
                        let action = p
                            .last_reveal
                            .iter()
                            .find(|(actor, _)| actor.index() == seat)
                            .map(|(_, action)| *action);
                        match action {
                            None => [0., 0., 0.],
                            Some(Action::Pick(card)) => [(card.id() + 1) as f32, 0., 0.],
                            Some(Action::Chopsticks { first, second }) => {
                                [(first.id() + 1) as f32, (second.id() + 1) as f32, 1.]
                            }
                        }
                    })
                    .collect(),
                "round_scores" => (0..3)
                    .flat_map(|round| {
                        (0..n).map(move |seat| {
                            p.round_scores
                                .get(round)
                                .map_or(0., |scores| scores[seat] as f32)
                        })
                    })
                    .collect(),
                "terminal_values" => {
                    let best = p.scores.iter().copied().max().unwrap_or_default();
                    let best_pudding = p
                        .scores
                        .iter()
                        .enumerate()
                        .filter(|(_, score)| **score == best)
                        .map(|(seat, _)| p.pudding[seat])
                        .max()
                        .unwrap_or_default();
                    (0..n)
                        .map(|seat| {
                            if !p.finished {
                                0.
                            } else if p.scores[seat] == best && p.pudding[seat] == best_pudding {
                                1.
                            } else {
                                -1.
                            }
                        })
                        .collect()
                }
                _ => unreachable!("validated payload"),
            };
            tensors.push(Float32Tensor {
                name: spec.name,
                shape: spec.shape.clone(),
                values,
            });
        }
        let mut action_mask = vec![false; Action::ACTION_COUNT];
        for &id in &offered_action_ids {
            action_mask[id] = true;
        }
        let initial = 12 - n;
        let turns_this_round = if p.finished {
            initial - 1
        } else {
            initial - p.hand_size
        };
        let completed = (p.round - 1) * (initial - 1) + turns_this_round;
        let progress = [
            n * completed,
            completed,
            turns_this_round,
            p.round_scores.len(),
        ]
        .map(|v| i64::try_from(v).map_err(|_| LearningError::Overflow));
        let [a, b, c, d] = progress;
        let best = p.scores.iter().copied().max().unwrap_or_default();
        let best_pudding = p
            .scores
            .iter()
            .enumerate()
            .filter(|(_, score)| **score == best)
            .map(|(seat, _)| p.pudding[seat])
            .max()
            .unwrap_or_default();
        let results = (0..n)
            .map(|seat| {
                result_tag(if !p.finished {
                    PlayerResult::Ongoing
                } else if p.scores[seat] == best && p.pudding[seat] == best_pudding {
                    PlayerResult::Win
                } else {
                    PlayerResult::Loss
                })
            })
            .collect();
        Ok(EncodedObservation {
            identity: identity(n),
            observer: observation.observer,
            next_player: (!p.finished).then_some(observation.observer),
            tensors,
            progress: [a?, b?, c?, d?],
            results,
            action_mask,
            offered_action_ids,
        })
    }

    fn action_id(&self, _: &Observation, action: &Action) -> Result<usize, LearningError> {
        Ok(action.id())
    }

    fn action_from_id(&self, _: &Observation, id: usize) -> Result<Action, LearningError> {
        match id {
            0..12 => Ok(Action::Pick(Card::ALL[id])),
            12..156 => {
                let offset = id - 12;
                Ok(Action::Chopsticks {
                    first: Card::ALL[offset / 12],
                    second: Card::ALL[offset % 12],
                })
            }
            _ => Err(invalid("Sushi Go! action ID must be in 0..156")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::joint;

    #[test]
    fn all_fixed_actions_roundtrip_and_only_offered_ids_are_masked() {
        let model = SushiGoForwardModel;
        let state = model.init_state(&Parameters { seed: 7 }, 3).unwrap();
        let observation = model.observe(&state, PlayerId::new(1)).unwrap();
        let encoded = model
            .encode(
                &observation,
                &["hand", "known_hands", "players"],
                LearningLimits::default(),
            )
            .unwrap();
        assert_eq!(encoded.identity.player_count, 3);
        assert_eq!(encoded.action_mask.len(), 156);
        assert_eq!(
            encoded.offered_action_ids.len(),
            model.offered_actions(&observation).len()
        );
        assert_eq!(
            encoded.action_mask.iter().filter(|&&valid| valid).count(),
            encoded.offered_action_ids.len()
        );
        for id in 0..156 {
            let action = model.action_from_id(&observation, id).unwrap();
            assert_eq!(model.action_id(&observation, &action).unwrap(), id);
        }
        assert!(model.action_from_id(&observation, 156).is_err());
        let known = &encoded.tensors[1].values;
        assert_eq!(&known[12..24], &counts(observation.hand.iter().copied()));
        assert_eq!(&known[0..12], &[-1.; 12]);
    }

    #[test]
    fn encoding_keeps_hidden_worlds_and_pending_choices_indistinguishable() {
        let model = SushiGoForwardModel;
        let state = model.init_state(&Parameters { seed: 11 }, 3).unwrap();
        let observation = model.observe(&state, PlayerId::new(0)).unwrap();
        let names = [
            "hand",
            "known_hands",
            "played",
            "players",
            "context",
            "discard",
            "last_reveal",
        ];
        let expected = model
            .encode(&observation, &names, LearningLimits::default())
            .unwrap();
        for seed in [101, 202] {
            let world = model
                .simulation_fork(&observation, &mut SimulationRng::new(seed))
                .unwrap();
            let view = model.observe(&world, PlayerId::new(0)).unwrap();
            assert_eq!(
                model
                    .encode(&view, &names, LearningLimits::default())
                    .unwrap(),
                expected
            );
        }
        let mut resolved = state.clone();
        let choices = (0..3)
            .map(|seat| {
                (
                    PlayerId::new(seat),
                    Action::Pick(state.privileged_hand(PlayerId::new(seat)).unwrap()[0]),
                )
            })
            .collect::<Vec<_>>();
        joint::apply(&model, &mut resolved, &choices).unwrap();
        let after = model.observe(&resolved, PlayerId::new(0)).unwrap();
        let encoded = model
            .encode(&after, &["last_reveal"], LearningLimits::default())
            .unwrap();
        assert_eq!(encoded.tensors[0].values.len(), 9);
        for seat in 0..3 {
            assert_eq!(
                encoded.tensors[0].values[seat * 3],
                (state.privileged_hand(PlayerId::new(seat)).unwrap()[0].id() + 1) as f32
            );
        }
    }

    #[test]
    fn payload_selection_and_limits_reject_without_mutating_observation() {
        let model = SushiGoForwardModel;
        let state = model.init_state(&Parameters { seed: 3 }, 2).unwrap();
        let observation = model.observe(&state, PlayerId::new(0)).unwrap();
        let original = observation.clone();
        assert!(
            model
                .encode(&observation, &["hand", "hand"], LearningLimits::default())
                .is_err()
        );
        assert!(
            model
                .encode(&observation, &["unknown"], LearningLimits::default())
                .is_err()
        );
        assert!(
            model
                .encode(
                    &observation,
                    &["known_hands"],
                    LearningLimits {
                        max_cells: 4096,
                        max_bytes: 156,
                    }
                )
                .is_err()
        );
        let bare = model
            .encode(&observation, &[], LearningLimits::default())
            .unwrap();
        let exact_bytes = 156 + 32 + 2 + bare.offered_action_ids.len() * 8;
        assert!(
            model
                .encode(
                    &observation,
                    &[],
                    LearningLimits {
                        max_cells: 4096,
                        max_bytes: exact_bytes - 1,
                    },
                )
                .is_err()
        );
        assert!(
            model
                .encode(
                    &observation,
                    &[],
                    LearningLimits {
                        max_cells: 4096,
                        max_bytes: exact_bytes,
                    },
                )
                .is_ok()
        );
        assert_eq!(observation, original);
    }

    #[test]
    fn terminal_arrays_use_absolute_seats_and_shared_winners() {
        let model = SushiGoForwardModel;
        let mut state = model.init_state(&Parameters { seed: 4 }, 3).unwrap();
        state.public.finished = true;
        state.public.hand_size = 0;
        state.public.round = 3;
        state.public.round_scores = vec![vec![0; 3]; 3];
        state.public.scores = vec![20, 18, 20];
        state.public.pudding = vec![2, 4, 2];
        state.hands = vec![Vec::new(); 3];
        state.known_by = vec![vec![Some(Vec::new()); 3]; 3];
        for seat in 0..3 {
            let observation = model.observe(&state, PlayerId::new(seat)).unwrap();
            let encoded = model
                .encode(
                    &observation,
                    &["terminal_values"],
                    LearningLimits::default(),
                )
                .unwrap();
            assert_eq!(encoded.tensors[0].values, [1., -1., 1.]);
            assert_eq!(
                encoded.results,
                [ResultTag::Win, ResultTag::Loss, ResultTag::Win]
            );
            assert_eq!(encoded.next_player, None);
            assert!(encoded.offered_action_ids.is_empty());
        }
    }
}
