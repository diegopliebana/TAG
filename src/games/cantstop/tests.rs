//! Controlled rules and independent chance tests. Expected actions and values
//! are calculated directly rather than by reusing production legality helpers.
use super::*;
use crate::core::{
    Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel,
    checkpoint::Checkpoint,
    evaluation::Evaluator,
    outcomes::SingleWinnerOutcome,
    random::{AgentRng, ChanceRng, SimulationRng},
    simulation::SimulationModel,
};
use std::collections::HashSet;

const P0: PlayerId = PlayerId::new(0);
const P1: PlayerId = PlayerId::new(1);
fn initial(players: usize) -> CantStopState {
    CantStopForwardModel
        .init_state(&CantStopParameters::default(), players)
        .unwrap()
}
fn pair(a: u8, b: u8) -> CantStopAction {
    CantStopAction::Allocate(Allocation::pair(a, b).unwrap())
}
fn single(a: u8) -> CantStopAction {
    CantStopAction::Allocate(Allocation::single(a).unwrap())
}
fn roll(state: &mut CantStopState, dice: [u8; 4]) {
    let actor = state.position.current_player;
    CantStopForwardModel
        .next_with_roll(state, actor, dice)
        .unwrap();
}
fn play(state: &mut CantStopState, action: CantStopAction) {
    let actor = state.position.current_player;
    CantStopForwardModel.next(state, actor, &action).unwrap();
}
fn offers(state: &CantStopState) -> Vec<CantStopAction> {
    CantStopForwardModel
        .available_actions(state, state.position.current_player)
        .unwrap()
}

#[test]
fn published_defaults_and_parameter_boundaries() {
    for players in 2..=4 {
        let state = initial(players);
        assert_eq!(state.position.player_count(), players);
        assert_eq!(
            state.position.rules.column_heights,
            [3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]
        );
        assert_eq!(state.position.phase, CantStopPhase::StartTurn);
        assert_eq!(state.position.dice, None);
        assert_eq!(state.position.next_player(), Decision::Player(P0));
        assert_eq!(state.position.counters.move_count(), 0);
        assert_eq!(offers(&state), vec![CantStopAction::Roll]);
        assert_eq!(
            state.position.player_result(P0).unwrap(),
            PlayerResult::Ongoing
        );
        assert_eq!(state.position.player_score(P0).unwrap(), 0.0);
        // Broad valid research domain, rather than Java's tuning-list values.
        let parameters = CantStopParameters {
            rules: CantStopRules {
                column_heights: [u32::MAX; 11],
                markers: 11,
                columns_to_win: 11usize.div_ceil(players),
            },
            seed: u64::MAX,
        };
        assert!(
            CantStopForwardModel
                .init_state(&parameters, players)
                .is_ok()
        );
    }
    for players in [0, 1, 5, usize::MAX] {
        assert!(
            CantStopForwardModel
                .init_state(&CantStopParameters::default(), players)
                .is_err()
        );
    }
    for (markers, target) in [(0, 3), (12, 3), (3, 0), (3, 12), (3, 7)] {
        let parameters = CantStopParameters {
            rules: CantStopRules {
                markers,
                columns_to_win: target,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(CantStopForwardModel.init_state(&parameters, 2).is_err());
    }
    let mut parameters = CantStopParameters::default();
    parameters.rules.column_heights[0] = 0;
    assert!(CantStopForwardModel.init_state(&parameters, 2).is_err());
    assert!(Allocation::single(1).is_err());
    assert!(Allocation::pair(2, 13).is_err());
    assert_eq!(
        Allocation::pair(2, 12).unwrap(),
        Allocation::pair(12, 2).unwrap()
    );
}

#[test]
fn compulsory_roll_pairing_order_banking_and_multiaction_turns() {
    let mut state = initial(3);
    roll(&mut state, [1, 5, 4, 6]);
    assert_eq!(offers(&state), vec![pair(6, 10), pair(5, 11), pair(7, 9)]);
    play(&mut state, pair(10, 6));
    assert_eq!(state.position.temporary_position(6).unwrap(), Some(1));
    assert_eq!(
        offers(&state),
        vec![CantStopAction::Stop, CantStopAction::Roll]
    );
    assert_eq!(state.position.counters.completed_turn_count(), 0);
    play(&mut state, CantStopAction::Stop);
    assert_eq!(state.position.permanent_position(P0, 6).unwrap(), 1);
    assert_eq!(state.position.permanent_position(P0, 10).unwrap(), 1);
    assert_eq!(state.position.temporary, [None; 11]);
    assert_eq!(state.position.current_player, P1);
    assert_eq!(state.position.counters.move_count(), 3);
    assert_eq!(state.position.counters.completed_turn_count(), 1);
    assert_eq!(offers(&state), vec![CantStopAction::Roll]);
    // Banking again starts one space above this player's own retained marker.
    state.position.current_player = P0;
    roll(&mut state, [1, 5, 4, 6]);
    play(&mut state, pair(6, 10));
    assert_eq!(state.position.temporary_position(6).unwrap(), Some(2));
}

#[test]
fn repeated_sum_near_top_uses_one_step_then_bust_discards_entire_turn() {
    let mut state = initial(2);
    roll(&mut state, [1; 4]);
    assert_eq!(offers(&state), vec![pair(2, 2)]);
    play(&mut state, pair(2, 2));
    assert_eq!(state.position.temporary_position(2).unwrap(), Some(2));
    play(&mut state, CantStopAction::Roll);
    // Override only the observed values here; first Roll above already advanced RNG.
    state.position.dice = Some([1; 4]);
    assert_eq!(offers(&state), vec![single(2)]);
    play(&mut state, single(2));
    assert_eq!(state.position.column_owner(2).unwrap(), None);
    assert_eq!(state.position.player_score(P0).unwrap(), 0.0);
    roll(&mut state, [1; 4]);
    assert_eq!(offers(&state), vec![CantStopAction::Bust]);
    play(&mut state, CantStopAction::Bust);
    assert_eq!(state.position.permanent_position(P0, 2).unwrap(), 0);
    assert_eq!(state.position.temporary, [None; 11]);
    assert_eq!(state.position.current_player, P1);
    assert_eq!(state.position.counters.completed_turn_count(), 1);
}

#[test]
fn marker_limits_and_required_pair_use_match_published_example() {
    let mut state = initial(2);
    state.position.temporary[1] = Some(1); // column 3
    state.position.temporary[4] = Some(1); // column 6
    roll(&mut state, [2, 4, 5, 5]);
    assert_eq!(offers(&state), vec![pair(6, 10), single(7), single(9)]);
    assert!(
        !offers(&state).contains(&single(6)),
        "must also use the available 10"
    );
    // With one marker, a repeated sum uses one marker but advances twice.
    let parameters = CantStopParameters {
        rules: CantStopRules {
            markers: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut state = CantStopForwardModel.init_state(&parameters, 2).unwrap();
    roll(&mut state, [1, 2, 3, 4]);
    assert_eq!(
        offers(&state),
        vec![pair(5, 5), single(3), single(7), single(4), single(6)]
    );
    play(&mut state, pair(5, 5));
    assert_eq!(state.position.temporary.iter().flatten().count(), 1);
    assert_eq!(state.position.temporary_position(5).unwrap(), Some(2));
}

#[test]
fn claimed_columns_remove_opponents_and_terminal_bank_commits_all_columns() {
    let mut state = initial(4);
    state.position.rules.columns_to_win = 1;
    state.position.permanent[1][0] = 1;
    state.position.permanent[2][0] = 2;
    state.position.temporary[0] = Some(3);
    state.position.temporary[10] = Some(3);
    state.position.phase = CantStopPhase::Decision;
    play(&mut state, CantStopAction::Stop);
    assert_eq!(state.position.player_score(P0).unwrap(), 2.0);
    for column in [2, 12] {
        assert_eq!(state.position.column_owner(column).unwrap(), Some(P0));
        for player in 1..4 {
            assert_eq!(
                state
                    .position
                    .permanent_position(PlayerId::new(player), column)
                    .unwrap(),
                0
            );
        }
    }
    assert_eq!(state.position.next_player(), Decision::Finished);
    assert_eq!(state.position.current_player, P0);
    assert_eq!(state.position.counters.move_count(), 1);
    assert_eq!(state.position.counters.completed_turn_count(), 0);
    assert_eq!(state.position.player_result(P0).unwrap(), PlayerResult::Win);
    for player in 1..4 {
        assert_eq!(
            state.position.player_result(PlayerId::new(player)).unwrap(),
            PlayerResult::Loss
        );
    }
    assert!(offers(&state).is_empty());
    let before = state.clone();
    assert_eq!(
        CantStopForwardModel.next(&mut state, P0, &CantStopAction::Roll),
        Err(CantStopError::Finished)
    );
    assert_eq!(state, before);
}

#[test]
fn completed_columns_are_unavailable_and_same_space_as_opponent_is_allowed() {
    let mut state = initial(2);
    state.position.rules.columns_to_win = 4;
    state.position.owners[4] = Some(P1); // 6
    state.position.owners[6] = Some(P1); // 8
    state.position.owners[8] = Some(P1); // 10
    for n in [4, 6, 8] {
        state.position.permanent[1][n] = state.position.rules.column_heights[n];
    }
    roll(&mut state, [2, 4, 4, 6]);
    assert_eq!(offers(&state), vec![CantStopAction::Bust]);
    let mut state = initial(2);
    state.position.permanent[1][0] = 2;
    roll(&mut state, [1; 4]);
    play(&mut state, pair(2, 2));
    assert_eq!(state.position.temporary_position(2).unwrap(), Some(2));
    assert!(offers(&state).contains(&CantStopAction::Stop));
}

#[test]
fn illegal_input_preserves_position_and_chance_progress() {
    let mut state = initial(2);
    for (player, action) in [
        (P0, CantStopAction::Stop),
        (P0, CantStopAction::Bust),
        (P0, pair(2, 2)),
        (P1, CantStopAction::Roll),
        (PlayerId::new(2), CantStopAction::Roll),
    ] {
        let before = state.clone();
        assert!(
            CantStopForwardModel
                .next(&mut state, player, &action)
                .is_err()
        );
        assert_eq!(state, before);
    }
    for dice in [[0, 1, 1, 1], [1, 1, 1, 7], [255; 4]] {
        let before = state.clone();
        assert!(
            CantStopForwardModel
                .next_with_roll(&mut state, P0, dice)
                .is_err()
        );
        assert_eq!(state, before);
    }
    assert!(
        CantStopForwardModel
            .observe(&state, PlayerId::new(2))
            .is_err()
    );
    assert!(state.position.permanent_position(P0, 13).is_err());
    assert!(
        state
            .position
            .permanent_position(PlayerId::new(2), 2)
            .is_err()
    );
    roll(&mut state, [1; 4]);
    let before = state.clone();
    assert!(
        CantStopForwardModel
            .next_with_roll(&mut state, P0, [2; 4])
            .is_err()
    );
    assert_eq!(state, before);
    assert!(
        CantStopForwardModel
            .next(&mut state, P0, &single(3))
            .is_err()
    );
    assert_eq!(state, before);
}

#[test]
fn player_cycles_are_not_implicitly_rounds() {
    for count in 2..=4 {
        let mut state = initial(count);
        for player in 0..count {
            assert_eq!(state.position.current_player, PlayerId::new(player));
            roll(&mut state, [3, 3, 4, 4]);
            play(&mut state, pair(6, 8));
            play(&mut state, CantStopAction::Stop);
        }
        assert_eq!(state.position.current_player, P0);
        assert_eq!(state.position.counters.completed_turn_count(), count);
        assert_eq!(state.position.counters.turn_count_in_round(), count);
        assert_eq!(state.position.counters.round_count(), 0);
    }
}

#[test]
fn exact_checkpoints_and_recorded_rolls_reproduce_continuation() {
    let model = CantStopForwardModel;
    let mut live = initial(3);
    let mut saved = CantStopState::from_snapshot(&live.snapshot()).unwrap();
    let initial_state = live.clone();
    for _ in 0..80 {
        let choices = offers(&live);
        let action = if live.position.phase == CantStopPhase::Decision {
            CantStopAction::Stop
        } else {
            choices[0]
        };
        let actor = live.position.current_player;
        let event = model.next_recorded(&mut live, actor, &action).unwrap();
        if let Some(event) = event {
            model.next_with_roll(&mut saved, actor, event.dice).unwrap();
        } else {
            model.next(&mut saved, actor, &action).unwrap();
        }
        assert_eq!(live, saved);
        if live.position.next_player() == Decision::Finished {
            break;
        }
    }
    assert_ne!(live, initial_state);
    assert_eq!(initial_state.position.counters.move_count(), 0);
    // Controlled values replace outcomes but consume the same RNG work.
    let mut normal = initial(2);
    let mut controlled = normal.clone();
    model.next(&mut normal, P0, &CantStopAction::Roll).unwrap();
    model.next_with_roll(&mut controlled, P0, [1; 4]).unwrap();
    assert_eq!(normal.chance_snapshot(), controlled.chance_snapshot());
    assert_eq!(controlled.position.dice, Some([1; 4]));
}

#[test]
fn observations_do_not_depend_on_live_rng_and_forks_have_independent_reproducible_dice() {
    let model = CantStopForwardModel;
    let live = initial(2);
    let before = live.clone();
    let mut other = live.clone();
    other.random = ChanceRng::new(999);
    assert_eq!(
        model.observe(&live, P0).unwrap(),
        model.observe(&other, P0).unwrap()
    );
    let observation = model.observe(&live, P0).unwrap();
    let inactive = model.observe(&live, P1).unwrap();
    assert!(model.offered_actions(&inactive).is_empty());
    assert_eq!(model.offered_actions(&observation), offers(&live));
    let mut seeds = SimulationRng::new(42);
    let mut restored = SimulationRng::from_snapshot(&seeds.snapshot()).unwrap();
    let mut outcomes = HashSet::new();
    for _ in 0..40 {
        let mut simulation = model.simulation_fork(&observation, &mut seeds).unwrap();
        let mut repeated = model.simulation_fork(&observation, &mut restored).unwrap();
        assert_eq!(simulation.position(), live.position());
        assert_ne!(simulation.chance_snapshot(), live.chance_snapshot());
        model
            .next(&mut simulation, P0, &CantStopAction::Roll)
            .unwrap();
        model
            .next(&mut repeated, P0, &CantStopAction::Roll)
            .unwrap();
        assert_eq!(simulation, repeated);
        outcomes.insert(simulation.position.dice);
    }
    assert!(outcomes.len() > 1);
    assert_eq!(
        live, before,
        "queries and discarded simulations must not mutate the real game"
    );
    assert_eq!(observation, model.observe(&live, P0).unwrap());
}

#[test]
fn deterministic_simulation_forks_consume_no_randomness_and_keep_exact_state() {
    use crate::games::connect4::{Connect4ForwardModel, Connect4Parameters};
    use crate::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters};
    fn check<F: PublicInformationModel>(model: F, parameters: F::Parameters)
    where
        F::State: PartialEq + std::fmt::Debug,
        F::Error: std::fmt::Debug,
    {
        let state = model.init_state(&parameters, 2).unwrap();
        let obs = model.observe(&state, P0).unwrap();
        let mut random = SimulationRng::new(1);
        let before = random.snapshot();
        assert_eq!(model.simulation_fork(&obs, &mut random).unwrap(), state);
        assert_eq!(random.snapshot(), before);
    }
    check(TicTacToeForwardModel, TicTacToeParameters::default());
    check(Connect4ForwardModel, Connect4Parameters::default());
}

#[test]
fn chance_stream_domains_and_snapshot_validation() {
    let mut game = ChanceRng::new(42);
    let mut source = SimulationRng::new(42);
    assert_ne!(
        game.snapshot().generator.stream,
        AgentRng::new(42).snapshot().stream
    );
    let simulated = source.fork_chance();
    assert_ne!(
        game.snapshot().generator.stream,
        simulated.snapshot().generator.stream
    );
    let mut saved = game.snapshot();
    saved.generator.version += 1;
    assert!(ChanceRng::from_snapshot(&saved).is_err());
    let mut saved = game.snapshot();
    saved.generator.word_position = 1u128 << 68;
    assert!(ChanceRng::from_snapshot(&saved).is_err());
    let mut saved = game.snapshot();
    saved.generator.stream = 0;
    assert!(ChanceRng::from_snapshot(&saved).is_err());
    let mut saved = source.snapshot();
    saved.generator.stream = 0;
    assert!(SimulationRng::from_snapshot(&saved).is_err());
    let mut restored = ChanceRng::from_snapshot(&game.snapshot()).unwrap();
    let before = game.snapshot();
    assert_eq!(game.index(0), None);
    assert_eq!(game.index(1), Some(0));
    assert_eq!(game.snapshot(), before);
    for _ in 0..100 {
        assert_eq!(game.index(6), restored.index(6));
    }
}

#[test]
fn approved_heuristic_arithmetic_perspectives_and_extreme_weights() {
    let mut state = initial(3);
    state.position.owners[0] = Some(P0);
    state.position.permanent[0][0] = 3;
    state.position.permanent[0][5] = 6;
    state.position.temporary[5] = Some(9);
    state.position.permanent[1][5] = 3;
    let mut evaluator = CantStopProgress::default();
    let expected = (1.0 / 3.0 + 0.5 * (6.0 / 13.0 / 11.0) + 0.25 * (3.0 / 13.0 / 11.0)) / 1.75;
    assert!((evaluator.evaluate(&state, P0).unwrap() - expected).abs() < 1e-14);
    let expected = 0.5 * (3.0 / 13.0 / 11.0) / 1.75;
    assert!((evaluator.evaluate(&state, P1).unwrap() - expected).abs() < 1e-14);
    assert!((CantStopBaseline.evaluate(&state, P0).unwrap() - 1.0 / 3.0).abs() < 1e-14);
    assert_eq!(
        CantStopProgress::new(0.0, 0.0)
            .unwrap()
            .evaluate(&state, P0)
            .unwrap(),
        1.0 / 3.0
    );
    assert!(
        CantStopProgress::new(f64::MAX, f64::MAX)
            .unwrap()
            .evaluate(&state, P0)
            .unwrap()
            .is_finite()
    );
    for (a, b) in [(f64::NAN, 0.0), (0.0, f64::INFINITY), (-1.0, 0.0)] {
        assert!(CantStopProgress::new(a, b).is_err());
    }
    assert!(evaluator.evaluate(&state, PlayerId::new(3)).is_err());
    assert_eq!(
        CantStopProgress::from_snapshot(&evaluator.snapshot()).unwrap(),
        evaluator
    );
    state.position.status = SingleWinnerOutcome::Won(P0);
    assert_eq!(evaluator.evaluate(&state, P0).unwrap(), 1.0);
    assert_eq!(evaluator.evaluate(&state, P1).unwrap(), -1.0);
    assert_eq!(CantStopBaseline.evaluate(&state, P1).unwrap(), -1.0);
}

// Independent oracle: enumerate subsets of each actual dice pairing and apply
// them step by step to copied counters. Keep only maximal feasible subsets for
// that pairing. This differs from production's count/marker legality arithmetic.
fn oracle(position: &CantStopPosition, dice: [u8; 4]) -> HashSet<CantStopAction> {
    let [a, b, c, d] = dice;
    let mut expected = HashSet::new();
    for sums in [[a + b, c + d], [a + c, b + d], [a + d, b + c]] {
        let mut legal = vec![];
        for mask in 1u8..4 {
            let mut markers = position.temporary;
            let mut success = true;
            for (index, sum) in sums.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    continue;
                }
                let n = (*sum - 2) as usize;
                let old =
                    markers[n].unwrap_or(position.permanent[position.current_player.index()][n]);
                if position.owners[n].is_some()
                    || old == position.rules.column_heights[n]
                    || (markers[n].is_none()
                        && markers.iter().flatten().count() == position.rules.markers)
                {
                    success = false;
                    break;
                }
                markers[n] = Some(old + 1);
            }
            if success {
                legal.push(mask);
            }
        }
        for &mask in &legal {
            if legal
                .iter()
                .any(|&other| other != mask && other & mask == mask)
            {
                continue;
            }
            expected.insert(match mask {
                1 => single(sums[0]),
                2 => single(sums[1]),
                3 => pair(sums[0], sums[1]),
                _ => unreachable!(),
            });
        }
    }
    if expected.is_empty() {
        expected.insert(CantStopAction::Bust);
    }
    expected
}

#[test]
fn all_1296_rolls_match_independent_legality_oracle_on_representative_states() {
    let base = initial(2);
    let mut near_top = base.clone();
    near_top.position.permanent[0][0] = 2;
    let mut no_spare = base.clone();
    no_spare.position.temporary[0] = Some(2);
    no_spare.position.temporary[5] = Some(12);
    no_spare.position.temporary[10] = Some(3);
    let mut claimed = base.clone();
    claimed.position.owners[4] = Some(P1);
    claimed.position.owners[6] = Some(P1);
    let mut one_marker = base.clone();
    one_marker.position.rules.markers = 1;
    for template in [base, near_top, no_spare, claimed, one_marker] {
        for a in 1..=6 {
            for b in 1..=6 {
                for c in 1..=6 {
                    for d in 1..=6 {
                        let dice = [a, b, c, d];
                        let mut state = template.clone();
                        roll(&mut state, dice);
                        let choices = offers(&state);
                        let unique: HashSet<_> = choices.iter().copied().collect();
                        assert_eq!(unique.len(), choices.len());
                        assert_eq!(
                            unique,
                            oracle(&state.position, dice),
                            "dice {dice:?}, position {:?}",
                            state.position
                        );
                        for action in choices {
                            let mut next = state.clone();
                            play(&mut next, action);
                            assert!(
                                next.position.temporary.iter().flatten().count()
                                    <= next.position.rules.markers
                            );
                            for n in 0..11 {
                                if let Some(value) = next.position.temporary[n] {
                                    assert!(value <= next.position.rules.column_heights[n]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn complete_seeded_matches_for_all_supported_player_counts() {
    use crate::agents::{Agent, RandomAgent};
    for count in 2..=4 {
        for seed in 0..12 {
            let parameters = CantStopParameters {
                seed,
                ..Default::default()
            };
            let mut state = CantStopForwardModel.init_state(&parameters, count).unwrap();
            let mut agents: Vec<_> = (0..count)
                .map(|seat| RandomAgent::new(seed + seat as u64))
                .collect();
            for _ in 0..100_000 {
                let Decision::Player(actor) = state.position.next_player() else {
                    break;
                };
                let observation = CantStopForwardModel.observe(&state, actor).unwrap();
                let choices = CantStopForwardModel.offered_actions(&observation);
                let selected = agents[actor.index()]
                    .choose(&CantStopForwardModel, actor, &observation, &choices)
                    .unwrap();
                play(&mut state, choices[selected]);
            }
            // This test bound is an assertion on these seeds, not a game rule or
            // an experimental claim that all stochastic games finish within it.
            let SingleWinnerOutcome::Won(winner) = state.position.status else {
                panic!("seeded game did not finish");
            };
            assert!(state.position.player_score(winner).unwrap() >= 3.0);
            assert_eq!(state.position.temporary, [None; 11]);
            for player in 0..count {
                assert_ne!(
                    state.position.player_result(PlayerId::new(player)).unwrap(),
                    PlayerResult::Ongoing
                );
            }
        }
    }
}

#[test]
fn middle_of_turn_snapshots_preserve_pending_dice_and_random_progress() {
    let mut state = initial(2);
    roll(&mut state, [1, 5, 4, 6]);
    play(&mut state, pair(6, 10));
    let mut resumed = CantStopState::from_snapshot(&state.snapshot()).unwrap();
    assert!(state.chance_snapshot().generator.word_position > 0);
    for _ in 0..30 {
        if state.position.next_player() == Decision::Finished {
            break;
        }
        let action = offers(&state)[0];
        play(&mut state, action);
        play(&mut resumed, action);
        assert_eq!(state, resumed);
    }
    let mut source = SimulationRng::new(3);
    source.fork_chance();
    let mut resumed = SimulationRng::from_snapshot(&source.snapshot()).unwrap();
    for _ in 0..10 {
        assert_eq!(source.fork_chance(), resumed.fork_chance());
    }
}
