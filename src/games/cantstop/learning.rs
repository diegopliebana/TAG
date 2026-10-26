//! Player-visible arrays and stable action IDs; no authoritative RNG is encoded.
use super::*;
use crate::core::{Decision, ForwardModel, PlayerResult};
use crate::learning::*;

const ACTIONS: usize = 80;
fn invalid(error: impl std::fmt::Display) -> LearningError {
    LearningError::Invalid(error.to_string())
}
fn identity(rules: &CantStopRules, players: usize) -> EncodingIdentity {
    EncodingIdentity {
        game: "cantstop",
        rules_version: 1,
        encoding_version: 1,
        player_count: players,
        configuration: EncodingConfiguration::Named(
            [
                (
                    "column_heights".into(),
                    rules
                        .column_heights
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                ),
                ("markers".into(), rules.markers.to_string()),
                ("columns_to_win".into(), rules.columns_to_win.to_string()),
            ]
            .into(),
        ),
    }
}
fn layouts(players: usize) -> Vec<TensorSpec> {
    let tensor = |name, shape, axes, components, bounds| TensorSpec {
        name,
        shape,
        axes,
        components,
        bounds: Some(bounds),
        dtype: DType::Float32,
    };
    vec![
        tensor(
            "permanent",
            vec![players, 11, 2],
            vec!["absolute_seat", "column_2_to_12", "component"],
            vec!["high16", "low16"],
            (0, 65535),
        ),
        tensor(
            "temporary",
            vec![11, 3],
            vec!["column_2_to_12", "component"],
            vec!["present", "high16", "low16"],
            (0, 65535),
        ),
        tensor(
            "owners",
            vec![11, players + 1],
            vec!["column_2_to_12", "unclaimed_then_absolute_seat"],
            vec!["one_hot"],
            (0, 1),
        ),
        tensor(
            "dice",
            vec![4, 7],
            vec!["die", "absent_then_face_1_to_6"],
            vec!["one_hot"],
            (0, 1),
        ),
        tensor(
            "context",
            vec![3 + 2 * players],
            vec!["component"],
            vec![
                "phase_StartTurn_Allocation_Decision",
                "observer_one_hot",
                "next_player_one_hot",
            ],
            (0, 1),
        ),
    ]
}
fn checked_layouts(
    rules: &CantStopRules,
    players: usize,
    limits: LearningLimits,
) -> Result<Vec<TensorSpec>, LearningError> {
    rules.validate(players).map_err(invalid)?;
    if players * 11 > limits.max_cells {
        return Err(LearningError::ResourceLimit);
    }
    Ok(layouts(players))
}
fn budget(
    specs: &[TensorSpec],
    names: &[&str],
    players: usize,
    offered: usize,
    limits: LearningLimits,
) -> Result<(), LearningError> {
    let mut bytes = ACTIONS + 32 + players + offered * 8;
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
fn parts(position: u32) -> [f32; 2] {
    [(position >> 16) as f32, (position & 65535) as f32]
}
impl LearningGame for CantStopForwardModel {
    fn learning_spec(
        &self,
        p: &CantStopParameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        self.learning_spec_for_players(p, 2, limits)
    }
    fn learning_spec_for_players(
        &self,
        p: &CantStopParameters,
        players: usize,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        let specs = checked_layouts(&p.rules, players, limits)?;
        budget(&specs, &[], players, 0, limits)?;
        Ok(LearningSpec {
            identity: identity(&p.rules, players),
            payloads: specs.into_iter().map(PayloadSpec::Features).collect(),
            metadata: crate::learning::metadata_spec(players),
            actions: ActionSpec {
                size: ACTIONS,
                mapping: ActionMapping::Indexed,
            },
            heuristic: None,
        })
    }
    fn encode(
        &self,
        o: &CantStopObservation,
        names: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let p = o.position();
        let players = p.player_count();
        let specs = checked_layouts(p.rules(), players, limits)?;
        let offered = self
            .offered_actions(o)
            .iter()
            .map(|a| self.action_id(o, a))
            .collect::<Result<Vec<_>, _>>()?;
        budget(&specs, names, players, offered.len(), limits)?;
        let mut tensors = Vec::with_capacity(names.len());
        for name in names {
            let spec = specs
                .iter()
                .find(|s| s.name == *name)
                .expect("validated payload");
            let mut values = Vec::with_capacity(spec.shape.iter().product());
            match *name {
                "permanent" => {
                    for row in &p.permanent {
                        for &value in row {
                            values.extend(parts(value));
                        }
                    }
                }
                "temporary" => {
                    for value in p.temporary {
                        match value {
                            Some(v) => {
                                values.push(1.);
                                values.extend(parts(v));
                            }
                            None => values.extend([0.; 3]),
                        }
                    }
                }
                "owners" => {
                    for owner in p.owners {
                        for index in 0..=players {
                            values.push(f32::from(owner.map_or(0, |p| p.index() + 1) == index));
                        }
                    }
                }
                "dice" => {
                    for die in 0..4 {
                        let face = p.dice.map_or(0, |dice| dice[die] as usize);
                        for index in 0..7 {
                            values.push(f32::from(index == face));
                        }
                    }
                }
                "context" => {
                    let phase = match p.phase {
                        CantStopPhase::StartTurn => 0,
                        CantStopPhase::Allocation => 1,
                        CantStopPhase::Decision => 2,
                    };
                    for index in 0..3 {
                        values.push(f32::from(index == phase));
                    }
                    for seat in 0..players {
                        values.push(f32::from(seat == o.observer().index()));
                    }
                    for seat in 0..players {
                        values.push(f32::from(
                            p.next_player() == Decision::Player(crate::core::PlayerId::new(seat)),
                        ));
                    }
                }
                _ => unreachable!("validated payload"),
            }
            tensors.push(Float32Tensor {
                name: spec.name,
                shape: spec.shape.clone(),
                values,
            });
        }
        let counters = p.counters();
        let progress = [
            counters.move_count(),
            counters.completed_turn_count(),
            counters.turn_count_in_round(),
            counters.round_count(),
        ]
        .map(|n| i64::try_from(n).map_err(|_| LearningError::Overflow));
        let results = (0..players)
            .map(|seat| {
                Ok(
                    match p
                        .player_result(crate::core::PlayerId::new(seat))
                        .map_err(invalid)?
                    {
                        PlayerResult::Ongoing => ResultTag::Ongoing,
                        PlayerResult::Win => ResultTag::Win,
                        PlayerResult::Draw => ResultTag::Draw,
                        PlayerResult::Loss => ResultTag::Loss,
                    },
                )
            })
            .collect::<Result<Vec<_>, LearningError>>()?;
        let mut mask = vec![false; ACTIONS];
        for &id in &offered {
            mask[id] = true;
        }
        let [moves, turns, in_round, rounds] = progress;
        Ok(EncodedObservation {
            identity: identity(p.rules(), players),
            observer: o.observer(),
            next_player: match p.next_player() {
                Decision::Finished => None,
                Decision::Player(actor) => Some(actor),
            },
            tensors,
            progress: [moves?, turns?, in_round?, rounds?],
            results,
            action_mask: mask,
            offered_action_ids: offered,
        })
    }
    fn action_id(
        &self,
        _: &CantStopObservation,
        action: &CantStopAction,
    ) -> Result<usize, LearningError> {
        Ok(match action {
            CantStopAction::Roll => 0,
            CantStopAction::Stop => 1,
            CantStopAction::Bust => 2,
            CantStopAction::Allocate(a) => match *a.columns() {
                [single] => 3 + usize::from(single - 2),
                [a, b] => {
                    14 + (2..a).map(|first| usize::from(13 - first)).sum::<usize>()
                        + usize::from(b - a)
                }
                _ => return Err(invalid("invalid allocation")),
            },
        })
    }
    fn action_from_id(
        &self,
        _: &CantStopObservation,
        id: usize,
    ) -> Result<CantStopAction, LearningError> {
        Ok(match id {
            0 => CantStopAction::Roll,
            1 => CantStopAction::Stop,
            2 => CantStopAction::Bust,
            3..=13 => {
                CantStopAction::Allocate(Allocation::single((id - 3 + 2) as u8).map_err(invalid)?)
            }
            14..=79 => {
                let mut offset = id - 14;
                let mut first = 2u8;
                while offset >= usize::from(13 - first) {
                    offset -= usize::from(13 - first);
                    first += 1;
                }
                CantStopAction::Allocate(
                    Allocation::pair(first, first + offset as u8).map_err(invalid)?,
                )
            }
            _ => return Err(invalid("action ID must be in 0..80")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{PlayerId, random::ChanceRng};
    const NAMES: &[&str] = &["permanent", "temporary", "owners", "dice", "context"];
    fn initial(players: usize) -> CantStopState {
        CantStopForwardModel
            .init_state(&CantStopParameters::default(), players)
            .unwrap()
    }
    fn encode(state: &CantStopState, seat: usize) -> EncodedObservation {
        let model = CantStopForwardModel;
        model
            .encode(
                &model.observe(state, PlayerId::new(seat)).unwrap(),
                NAMES,
                LearningLimits::default(),
            )
            .unwrap()
    }
    #[test]
    fn descriptors_and_all_80_actions_have_stable_player_count_aware_meanings() {
        let model = CantStopForwardModel;
        for players in 2..=4 {
            let state = initial(players);
            let obs = model.observe(&state, PlayerId::new(0)).unwrap();
            let spec = model
                .learning_spec_for_players(
                    &CantStopParameters::default(),
                    players,
                    LearningLimits::default(),
                )
                .unwrap();
            assert_eq!(spec.identity.player_count, players);
            assert_eq!(spec.actions.size, 80);
            assert!(spec.heuristic.is_none());
            assert_eq!(spec.payloads[0].tensor().unwrap().shape, [players, 11, 2]);
            let mut actions = vec![
                CantStopAction::Roll,
                CantStopAction::Stop,
                CantStopAction::Bust,
            ];
            actions
                .extend((2..=12).map(|n| CantStopAction::Allocate(Allocation::single(n).unwrap())));
            for a in 2..=12 {
                for b in a..=12 {
                    actions.push(CantStopAction::Allocate(Allocation::pair(a, b).unwrap()));
                }
            }
            for (id, action) in actions.iter().enumerate() {
                assert_eq!(model.action_id(&obs, action).unwrap(), id);
                assert_eq!(model.action_from_id(&obs, id).unwrap(), *action);
            }
            assert!(model.action_from_id(&obs, 80).is_err());
            assert!(model.action_from_id(&obs, usize::MAX).is_err());
            let value = encode(&state, 0);
            assert_eq!(value.identity, spec.identity);
            assert_eq!(value.offered_action_ids, [0]);
            assert_eq!(
                value.tensors[3]
                    .values
                    .chunks(7)
                    .map(|row| row[0])
                    .collect::<Vec<_>>(),
                [1.; 4]
            );
            assert_eq!(
                value.tensors[2]
                    .values
                    .chunks(players + 1)
                    .map(|row| row[0])
                    .sum::<f32>(),
                11.
            );
            assert!(encode(&state, 1).offered_action_ids.is_empty());
        }
        assert!(
            model
                .learning_spec_for_players(
                    &CantStopParameters::default(),
                    1,
                    LearningLimits::default()
                )
                .is_err()
        );
        assert!(
            model
                .learning_spec_for_players(
                    &CantStopParameters::default(),
                    5,
                    LearningLimits::default()
                )
                .is_err()
        );
    }
    #[test]
    fn allocation_stop_bust_and_terminal_masks_follow_decisions_not_seat_alternation() {
        let model = CantStopForwardModel;
        let actor = PlayerId::new(0);
        let mut state = initial(3);
        model
            .next_with_roll(&mut state, actor, [1, 2, 3, 4])
            .unwrap();
        assert_eq!(encode(&state, 0).offered_action_ids, [29, 37, 44]);
        let obs = model.observe(&state, actor).unwrap();
        model
            .next(&mut state, actor, &model.action_from_id(&obs, 29).unwrap())
            .unwrap();
        assert_eq!(encode(&state, 0).offered_action_ids, [1, 0]);
        assert_eq!(state.position.current_player(), actor);
        model
            .next(&mut state, actor, &CantStopAction::Stop)
            .unwrap();
        assert_eq!(encode(&state, 1).offered_action_ids, [0]);
        let parameters = CantStopParameters {
            rules: CantStopRules {
                markers: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut state = model.init_state(&parameters, 4).unwrap();
        model.next_with_roll(&mut state, actor, [1; 4]).unwrap();
        model
            .next(
                &mut state,
                actor,
                &CantStopAction::Allocate(Allocation::pair(2, 2).unwrap()),
            )
            .unwrap();
        model.next_with_roll(&mut state, actor, [6; 4]).unwrap();
        assert_eq!(encode(&state, 0).offered_action_ids, [2]);
        model
            .next(&mut state, actor, &CantStopAction::Bust)
            .unwrap();
        assert!(encode(&state, 1).tensors[1].values.iter().all(|&v| v == 0.));
        let parameters = CantStopParameters {
            rules: CantStopRules {
                column_heights: [1; 11],
                columns_to_win: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut state = model.init_state(&parameters, 4).unwrap();
        model.next_with_roll(&mut state, actor, [1; 4]).unwrap();
        let action = model.available_actions(&state, actor).unwrap()[0];
        model.next(&mut state, actor, &action).unwrap();
        model
            .next(&mut state, actor, &CantStopAction::Stop)
            .unwrap();
        for seat in 0..4 {
            let value = encode(&state, seat);
            assert_eq!(
                value.results,
                [
                    ResultTag::Win,
                    ResultTag::Loss,
                    ResultTag::Loss,
                    ResultTag::Loss
                ]
            );
            assert!(value.offered_action_ids.is_empty());
            assert!(!value.action_mask.iter().any(|&v| v));
            assert_eq!(&value.tensors[4].values[7..], &[0.; 4]);
        }
    }
    #[test]
    fn exact_large_positions_and_privileged_rng_independence() {
        let model = CantStopForwardModel;
        let parameters = CantStopParameters {
            rules: CantStopRules {
                column_heights: [u32::MAX; 11],
                ..Default::default()
            },
            seed: 42,
        };
        let mut state = model.init_state(&parameters, 4).unwrap();
        state.position.permanent[1][0] = (1 << 24) + 1;
        state.position.temporary[0] = Some(u32::MAX);
        state.position.phase = CantStopPhase::Decision;
        state.position.dice = Some([1, 2, 3, 4]);
        let before = state.clone();
        let value = encode(&state, 0);
        let pair = &value.tensors[0].values[22..24];
        assert_eq!(pair[0] as u32 * 65536 + pair[1] as u32, (1 << 24) + 1);
        assert_eq!(&value.tensors[1].values[..3], &[1., 65535., 65535.]);
        assert_eq!(state, before);
        let mut other = state.clone();
        other.random = ChanceRng::new(123);
        for seat in 0..4 {
            assert_eq!(encode(&state, seat), encode(&other, seat));
        }
        let a = model
            .learning_spec_for_players(&parameters, 4, LearningLimits::default())
            .unwrap();
        let b = model
            .learning_spec_for_players(
                &CantStopParameters {
                    seed: 99,
                    ..parameters.clone()
                },
                4,
                LearningLimits::default(),
            )
            .unwrap();
        assert_eq!(a, b);
        let mut changed = parameters;
        changed.rules.markers = 2;
        assert_ne!(
            a.identity,
            model
                .learning_spec_for_players(&changed, 4, LearningLimits::default())
                .unwrap()
                .identity
        );
    }
    #[test]
    fn encoding_limits_names_and_buffer_ownership_are_checked() {
        let model = CantStopForwardModel;
        let state = initial(2);
        let o = model.observe(&state, PlayerId::new(0)).unwrap();
        for names in [vec!["permanent", "permanent"], vec!["cells"], vec!["image"]] {
            assert!(model.encode(&o, &names, LearningLimits::default()).is_err());
        }
        let limit = LearningLimits {
            max_bytes: 200,
            ..Default::default()
        };
        assert!(
            model
                .learning_spec(&CantStopParameters::default(), limit)
                .is_ok()
        );
        assert!(model.encode(&o, &[], limit).is_ok());
        assert!(model.encode(&o, NAMES, limit).is_err());
        assert!(
            model
                .learning_spec(
                    &CantStopParameters::default(),
                    LearningLimits {
                        max_cells: 21,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        let mut value = encode(&state, 0);
        value.tensors[0].values[0] = 99.;
        assert_eq!(encode(&state, 0).tensors[0].values[0], 0.);
        assert_eq!(
            model
                .encode(&o, &["context", "dice"], LearningLimits::default())
                .unwrap()
                .tensors
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>(),
            ["context", "dice"]
        );
    }
}
