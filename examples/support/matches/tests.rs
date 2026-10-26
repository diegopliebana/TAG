use super::config::ResolvedMatch;
use super::*;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

fn configuration(game: &str, first: Value, second: Value) -> Value {
    json!({"schema_version":1, "game":{"name":game}, "players":[
        {"name":"alpha", "agent":first}, {"name":"beta", "agent":second}
    ], "games":3, "seed":42, "reporting":{"include_actions":true}})
}
fn resolve(value: Value) -> Result<ResolvedMatch> {
    serde_json::from_value::<MatchConfig>(value)?.resolve(Path::new("."))
}
fn mcts(policy: &str) -> Value {
    json!({"type":"mcts", "config":{"schema_version":1,"tree_policy":policy,"budgets":{"iterations":12},"rollout_length":2}})
}
fn random() -> Value {
    json!({"type":"random"})
}

#[cfg(feature = "inference")]
#[test]
fn neural_matches_validate_models_and_run_both_heads_on_each_game() {
    for game in ["tictactoe", "connect4"] {
        for head in ["scalar", "vector"] {
            let model = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("web/tests/fixtures/mcts/{game}-{head}.json"));
            let agent = json!({"type":"neural_mcts","model_file":model,
                "config":{"schema_version":1,"tree_policy":"puct","rollout_length":0,
                    "budgets":{"iterations":16},"state_evaluator":{"name":"network_value"},
                    "puct_preferences":{"source":"probabilities","evaluator":{"name":"network_policy"}}}});
            let mut value = configuration(game, agent, random());
            value["games"] = json!(2);
            let settings = resolve(value.clone()).unwrap();
            let first = run(&settings).unwrap();
            let second = run(&resolve(value.clone()).unwrap()).unwrap();
            assert_eq!(first.completed_games, 2);
            assert!(first.failure.is_none(), "{:?}", first.failure);
            assert_eq!(first.games, second.games);
            let other = if game == "tictactoe" {
                "connect4"
            } else {
                "tictactoe"
            };
            value["game"]["name"] = json!(other);
            assert!(run(&resolve(value).unwrap()).is_err());
        }
    }
}
fn osla() -> Value {
    json!({"type":"osla"})
}

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tag-matches-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn both_games_run_mixed_agents_and_independent_mcts_configurations() {
    for game in ["tictactoe", "connect4"] {
        for (first, second) in [
            (random(), osla()),
            (mcts("ucb1"), osla()),
            (mcts("ucb1"), mcts("puct")),
        ] {
            let settings = resolve(configuration(game, first, second)).unwrap();
            let first = run(&settings).unwrap();
            let repeat = run(&settings).unwrap();
            assert_eq!(
                first, repeat,
                "fixed-work seeds must reproduce actions, counts and outcomes"
            );
            assert_eq!(first.completed_games, 3);
            assert!(first.failure.is_none());
            assert_eq!(first.games[0].seats, [0, 1]);
            assert_eq!(first.games[1].seats, [1, 0]);
            assert_eq!(first.games[2].seats, [0, 1]);
            for game in &first.games {
                assert_eq!(game.actions.as_ref().unwrap().len(), game.accepted_moves);
                assert_ne!(game.agent_seeds[0], game.agent_seeds[1]);
            }
            assert_ne!(first.games[0].agent_seeds, first.games[1].agent_seeds);
            assert_eq!(first.agents[0].by_seat[0].games, 2);
            assert_eq!(first.agents[1].by_seat[0].games, 1);
            for agent in &first.agents {
                assert_eq!(agent.combined.games, 3);
                assert_eq!(
                    agent.combined.wins + agent.combined.draws + agent.combined.losses,
                    3
                );
            }
        }
    }
}

#[test]
fn fixed_positions_and_nondefault_rules_use_the_same_runner() {
    for (game, parameters) in [
        ("tictactoe", json!({"grid_size":2})),
        ("connect4", json!({"width":4,"height":3,"winning_length":3})),
    ] {
        let mut value = configuration(game, json!({"type":"first_available"}), osla());
        value["positions"] = json!("fixed");
        value["game"]["parameters"] = parameters.clone();
        let result = run(&resolve(value).unwrap()).unwrap();
        assert!(result.games.iter().all(|game| game.seats == [0, 1]));
        assert_eq!(result.agents[0].by_seat[1].games, 0);
        for (key, expected) in parameters.as_object().unwrap() {
            assert_eq!(&result.parameters[key], expected);
        }
    }
}

#[test]
fn game_validation_and_unknown_factories_fail_before_outputs_are_created() {
    let temporary = Temporary::new();
    let mut invalid = Vec::new();
    let base = configuration("tictactoe", mcts("ucb1"), random());
    let mut value = base.clone();
    value["players"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"third","agent":{"type":"random"}}));
    invalid.push(value);
    let mut value = base.clone();
    value["game"]["parameters"] = json!({"grid_size":0});
    invalid.push(value);
    let mut value = base.clone();
    value["game"]["parameters"] = json!({"unknown":4});
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"]["config"]["state_evaluator"] = json!({"name":"unknown"});
    invalid.push(value);
    for mut value in invalid {
        value["reporting"]["path"] = json!(temporary.0.join("must-not-exist.json"));
        assert!(run(&resolve(value).unwrap()).is_err());
        assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
    }
}

#[test]
fn malformed_and_unsupported_match_settings_are_rejected() {
    let base = configuration("tictactoe", random(), osla());
    let mut invalid = Vec::new();
    let mut value = base.clone();
    value["extra"] = json!(true);
    invalid.push(value);
    let mut value = base.clone();
    value["schema_version"] = json!(2);
    invalid.push(value);
    let mut value = base.clone();
    value["games"] = json!(0);
    invalid.push(value);
    let mut value = base.clone();
    value["games"] = json!(10001);
    invalid.push(value);
    let mut value = base.clone();
    value["players"] = json!([]);
    invalid.push(value);
    let mut value = base.clone();
    value["players"][1]["name"] = json!("alpha");
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"] = json!({"type":"random","ignored":1});
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"] = json!({"type":"external_policy_value"});
    invalid.push(value);
    let mut value = base.clone();
    value["game"]["name"] = json!("unknown");
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"] = mcts("ucb1");
    value["players"][0]["agent"]["config"]["budgets"] = json!({});
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"] = mcts("ucb1");
    value["players"][0]["agent"]["config_file"] = json!("other.json");
    invalid.push(value);
    let mut value = base.clone();
    value["players"][0]["agent"] = mcts("ucb1");
    value["players"][0]["agent"]["config"]["seed"] = json!(3);
    value["players"][0]["seed"] = json!(4);
    invalid.push(value);
    let mut value = base;
    value["players"][0]["agent"] = mcts("ucb1");
    value["reporting"]["recordings"] = json!("unused");
    invalid.push(value);
    for value in invalid {
        assert!(resolve(value.clone()).is_err(), "accepted {value}");
    }
}

#[test]
fn missing_configuration_errors_identify_the_attempted_file() {
    let temporary = Temporary::new();
    let missing_match = temporary.0.join("missing-match.json");
    let error = MatchConfig::load(&missing_match).err().unwrap().to_string();
    assert!(error.contains("could not open configuration file"));
    assert!(error.contains(&missing_match.display().to_string()));

    let path = temporary.0.join("match.json");
    let value = configuration(
        "tictactoe",
        json!({"type":"mcts","config_file":"agents/ucb1.json"}),
        random(),
    );
    std::fs::write(&path, value.to_string()).unwrap();
    let error = MatchConfig::load(&path).err().unwrap().to_string();
    assert!(error.contains("could not open configuration file"));
    assert!(error.contains(&temporary.0.join("agents/ucb1.json").display().to_string()));
}

#[test]
fn relative_agent_files_and_output_paths_use_the_match_directory() {
    let temporary = Temporary::new();
    std::fs::create_dir(temporary.0.join("agents")).unwrap();
    std::fs::write(
        temporary.0.join("agents/search.json"),
        r#"{"schema_version":1,"budgets":{"iterations":3}}"#,
    )
    .unwrap();
    let mut value = configuration(
        "tictactoe",
        json!({"type":"mcts","config_file":"agents/search.json"}),
        random(),
    );
    value["reporting"]["format"] = json!("json");
    value["reporting"]["path"] = json!("results/report.json");
    let path = temporary.0.join("match.json");
    std::fs::write(&path, value.to_string()).unwrap();
    let settings = MatchConfig::load(&path).unwrap();
    assert!(matches!(settings.config.reporting.format, Format::Json));
    let report = run(&settings).unwrap();
    let destination = settings.config.reporting.path.as_ref().unwrap();
    assert_eq!(destination, &temporary.0.join("results/report.json"));
    write_report(destination, &serde_json::to_string(&report).unwrap()).unwrap();
    assert!(run(&settings).is_err(), "existing report must be preserved");
    assert!(write_report(destination, "replacement").is_err());
    assert_eq!(
        serde_json::from_str::<Value>(&std::fs::read_to_string(destination).unwrap()).unwrap()["completed_games"],
        3
    );
    assert!(report.summary().contains("alpha (mcts)"));
}

#[test]
fn conflicting_report_and_recording_paths_fail_before_play() {
    let temporary = Temporary::new();
    for report_path in [
        temporary.0.join("replays"),
        temporary.0.join("replays/../replays/game-000000.json"),
    ] {
        let mut value = configuration("tictactoe", random(), osla());
        value["reporting"]["path"] = json!(report_path);
        value["reporting"]["recordings"] = json!(temporary.0.join("replays"));
        assert!(run(&resolve(value).unwrap()).is_err());
        assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
    }
}

#[test]
fn runtime_evaluation_failure_is_not_a_game_loss() {
    let mut value = configuration("tictactoe", mcts("puct"), random());
    value["players"][0]["agent"]["config"]["puct_preferences"] =
        json!({"source":"probabilities","evaluator":{"name":"uniform_actions"}});
    let report = run(&resolve(value).unwrap()).unwrap();
    assert_eq!(report.completed_games, 0);
    assert!(
        report
            .failure
            .as_ref()
            .unwrap()
            .message
            .contains("zero mass")
    );
    assert!(
        report
            .agents
            .iter()
            .all(|agent| agent.combined.games == 0 && agent.combined.losses == 0)
    );
}

#[test]
fn later_failure_retains_completed_results_and_stops_the_schedule() {
    let settings = resolve(configuration("tictactoe", random(), osla())).unwrap();
    let mut success = run(&settings).unwrap();
    let first = success.games.remove(0);
    let initial = super::report::create(&settings, success.parameters).unwrap();
    let mut calls = 0;
    let mut first = Some(first);
    let report = super::execution::run_games(&settings, initial, |_| {
        calls += 1;
        if calls == 1 {
            Ok((first.take().unwrap(), None))
        } else {
            Err("controlled second-game failure".into())
        }
    });
    assert_eq!(calls, 2);
    assert_eq!(report.completed_games, 1);
    assert_eq!(report.games.len(), 1);
    assert_eq!(report.failure.unwrap().game_index, 1);
    assert!(report.agents.iter().all(|agent| agent.combined.games == 1));
}

#[test]
fn recording_failure_preserves_the_finished_games_outcome() {
    let temporary = Temporary::new();
    let settings = resolve(configuration("tictactoe", random(), osla())).unwrap();
    let mut success = run(&settings).unwrap();
    let first = success.games.remove(0);
    let initial = super::report::create(&settings, success.parameters).unwrap();
    let mut settings = settings;
    settings.config.reporting.recordings = Some(temporary.0.clone());
    let mut first = Some(first);
    let result = super::execution::run_games(&settings, initial, |_| {
        Ok((
            first.take().unwrap(),
            Some(Err("controlled replay encoding failure".into())),
        ))
    });
    assert_eq!(result.completed_games, 1);
    assert_eq!(result.games[0].recording, None);
    assert!(result.agents.iter().all(|agent| agent.combined.games == 1));
    assert!(
        result
            .failure
            .unwrap()
            .message
            .contains("game completed but recording failed")
    );
    assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
}

#[test]
fn saved_recordings_reload_with_the_same_rules_and_controller_progress() {
    use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel};
    use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel};
    use tag_rust::replay::{ReplayLimits, import_replay};
    for game in ["tictactoe", "connect4"] {
        let temporary = Temporary::new();
        let mut value = configuration(game, random(), osla());
        value["reporting"]["recordings"] = json!(temporary.0);
        let settings = resolve(value).unwrap();
        let recorded = run(&settings).unwrap();
        let mut unrecorded = configuration(game, random(), osla());
        unrecorded["reporting"]["recordings"] = Value::Null;
        let unrecorded = run(&resolve(unrecorded).unwrap()).unwrap();
        for (recorded, direct) in recorded.games.iter().zip(unrecorded.games) {
            assert_eq!(recorded.actions, direct.actions);
            assert_eq!(recorded.results, direct.results);
            assert_eq!(recorded.agent_seeds, direct.agent_seeds);
            let text = std::fs::read_to_string(recorded.recording.as_ref().unwrap()).unwrap();
            let next_player = if game == "tictactoe" {
                import_replay::<_, TicTacToeEvaluator>(
                    &text,
                    TicTacToeForwardModel,
                    ReplayLimits::default(),
                )
                .unwrap()
                .session
                .active()
                .session()
                .next_player()
            } else {
                import_replay::<_, Connect4Evaluator>(
                    &text,
                    Connect4ForwardModel,
                    ReplayLimits::default(),
                )
                .unwrap()
                .session
                .active()
                .session()
                .next_player()
            };
            assert_eq!(next_player, tag_rust::core::Decision::Finished);
        }
        assert!(
            run(&settings).is_err(),
            "existing recordings must not be overwritten"
        );
    }
}

#[test]
fn seed_bases_remain_attached_to_agent_identity_when_seats_change() {
    let mut value = configuration("tictactoe", mcts("ucb1"), random());
    value["seed"] = Value::Null;
    value["players"][0]["agent"]["config"]["seed"] = json!(17);
    let settings = resolve(value).unwrap();
    assert_eq!(settings.players[0].base_seed, 17);
    let report = run(&settings).unwrap();
    for game in report.games {
        assert_eq!(game.agent_seeds[0], derive_seed(17, game.index as u64, 1));
        assert_eq!(game.game_seed, None);
    }
}

#[test]
fn shared_file_reader_bounds_bytes_and_preserves_error_context() {
    let temporary = Temporary::new();
    let path = temporary.0.join("input.json");
    std::fs::write(&path, "123456789").unwrap();
    // One extra byte lets each caller detect its own exceeded size limit.
    assert_eq!(
        super::files::read_text(&path, 4, "configuration file").unwrap(),
        "12345"
    );
    std::fs::write(&path, [0xff]).unwrap();
    let error = super::files::read_text(&path, 4, "neural model").unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("could not read neural model"));
    assert!(error.to_string().contains(&path.display().to_string()));
}

#[test]
fn cantstop_matches_rotate_all_seats_record_dice_and_report_truncation() {
    use tag_rust::core::{ForwardModel, PlayerId};
    use tag_rust::games::cantstop::*;
    for players in 2..=4 {
        let mut value = configuration(
            "cantstop",
            json!({"type":"osla", "repetitions_per_action":4}),
            mcts("ucb1"),
        );
        value["game"]["parameters"] =
            json!({"rules":{"column_heights":[2,2,2,2,2,2,2,2,2,2,2],"columns_to_win":1}});
        value["game"]["ranking"] = json!("claimed_columns");
        value["players"] = json!((0..players).map(|seat| json!({"name":format!("p{seat}"),"agent":if seat == 0 { json!({"type":"osla","repetitions_per_action":4}) } else { mcts("ucb1") }})).collect::<Vec<_>>());
        value["games"] = json!(players);
        value["max_actions"] = json!(1000);
        let report = run(&resolve(value.clone()).unwrap()).unwrap();
        assert!(report.failure.is_none());
        assert_eq!(report.completed_games, players);
        assert_eq!(report, run(&resolve(value.clone()).unwrap()).unwrap());
        let mut seeds = std::collections::BTreeSet::new();
        for game in &report.games {
            assert!(!game.truncated);
            assert!(game.ranks.is_some());
            assert!(seeds.insert(game.game_seed.unwrap()));
            let model = CantStopForwardModel;
            let mut parameters: CantStopParameters =
                serde_json::from_value(report.parameters.clone()).unwrap();
            parameters.seed = game.game_seed.unwrap();
            let mut state = model.init_state(&parameters, players).unwrap();
            for entry in game.actions.as_ref().unwrap() {
                let action: CantStopAction = serde_json::from_value(entry.action.clone()).unwrap();
                model
                    .next(&mut state, PlayerId::new(entry.seat), &action)
                    .unwrap();
                if action == CantStopAction::Roll {
                    assert_eq!(
                        entry.chance.as_ref().unwrap()["dice"],
                        json!(state.position().dice().unwrap())
                    );
                } else {
                    assert!(entry.chance.is_none());
                }
            }
            for (agent, &seat) in game.seats.iter().enumerate() {
                assert_eq!(
                    game.results[agent],
                    if model.player_result(&state, PlayerId::new(seat)).unwrap()
                        == tag_rust::core::PlayerResult::Win
                    {
                        "win"
                    } else {
                        "loss"
                    }
                );
            }
        }
        value["game"]["ranking"] = json!("score");
        assert_eq!(
            report.games,
            run(&resolve(value.clone()).unwrap()).unwrap().games
        );
        value["max_actions"] = json!(1);
        let truncated = run(&resolve(value).unwrap()).unwrap();
        assert!(truncated.failure.is_none());
        for game in &truncated.games {
            assert!(game.truncated);
            assert_eq!(game.accepted_moves, 1);
            assert!(game.ranks.is_none());
            assert!(game.results.iter().all(|r| r == "truncated"));
        }
        for agent in &truncated.agents {
            assert_eq!(agent.combined.truncated, players);
            assert_eq!(
                (
                    agent.combined.wins,
                    agent.combined.draws,
                    agent.combined.losses
                ),
                (0, 0, 0)
            );
        }
    }
}

#[test]
fn cantstop_optional_capabilities_and_invalid_settings_fail_before_output() {
    let temporary = Temporary::new();
    let mut value = configuration("cantstop", random(), random());
    value["reporting"]["recordings"] = json!(temporary.0.join("replays"));
    assert!(run(&resolve(value).unwrap()).is_err());
    for edit in [
        json!({"ranking":"unknown"}),
        json!({"parameters":{"rules":{"markers":0}}}),
    ] {
        let mut value = configuration("cantstop", random(), random());
        for (key, val) in edit.as_object().unwrap() {
            value["game"][key] = val.clone();
        }
        assert!(run(&resolve(value).unwrap()).is_err());
    }
    let mut value = configuration(
        "cantstop",
        json!({"type":"osla","repetitions_per_action":0}),
        random(),
    );
    assert!(resolve(value.clone()).is_err());
    value["players"][0]["agent"] = random();
    value["max_actions"] = json!(0);
    assert!(resolve(value).is_err());
    let mut value = configuration("tictactoe", random(), random());
    value["game"]["ranking"] = json!("score");
    assert!(run(&resolve(value).unwrap()).is_err());
    assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
}

#[test]
fn loveletter_uses_shared_rotation_seed_reporting_and_capability_checks() {
    use tag_rust::core::{ForwardModel, PlayerId, PlayerResult};
    use tag_rust::games::loveletter::*;
    for n in 2..=4 {
        let mut value = configuration("loveletter", osla(), mcts("ucb1"));
        value["players"] = json!(
            (0..n)
                .map(
                    |i| json!({"name":format!("p{i}"),"agent":if i==0 {osla()} else {mcts("ucb1")}})
                )
                .collect::<Vec<_>>()
        );
        value["game"]["parameters"] = json!({"rules":{"winning_tokens":[1,1,1]}});
        value["games"] = json!(n);
        let report = run(&resolve(value.clone()).unwrap()).unwrap();
        assert!(report.failure.is_none(), "{:?}", report.failure);
        assert_eq!(report.completed_games, n);
        assert_eq!(report, run(&resolve(value.clone()).unwrap()).unwrap());
        for agent in &report.agents {
            assert!(agent.by_seat.iter().all(|counts| counts.games == 1));
        }
        let model = LoveLetterForwardModel;
        for game in &report.games {
            let mut parameters: LoveLetterParameters =
                serde_json::from_value(report.parameters.clone()).unwrap();
            parameters.seed = game.game_seed.unwrap();
            let mut state = model.init_state(&parameters, n).unwrap();
            for entry in game.actions.as_ref().unwrap() {
                let action = serde_json::from_value(entry.action.clone()).unwrap();
                assert!(entry.chance.is_none()); // Never record hidden draws as public chance.
                model
                    .next(&mut state, PlayerId::new(entry.seat), &action)
                    .unwrap();
            }
            for (agent, &seat) in game.seats.iter().enumerate() {
                assert_eq!(
                    game.results[agent],
                    if model.player_result(&state, PlayerId::new(seat)).unwrap()
                        == PlayerResult::Win
                    {
                        "win"
                    } else {
                        "loss"
                    }
                );
            }
        }
        value["max_actions"] = json!(1);
        let limited = run(&resolve(value.clone()).unwrap()).unwrap();
        assert!(
            limited
                .games
                .iter()
                .all(|g| g.truncated || g.results.iter().any(|r| r == "win"))
        );
        value["reporting"]["recordings"] = json!("unused-loveletter-replays");
        assert!(
            resolve(value.clone())
                .and_then(|settings| run(&settings))
                .is_err()
        );
        value["reporting"]
            .as_object_mut()
            .unwrap()
            .remove("recordings");
        value["game"]["parameters"]["rules"]["winning_tokens"] = json!([0, 1, 1]);
        assert!(run(&resolve(value).unwrap()).is_err());
    }
}

#[test]
fn terminal_move_at_the_action_limit_takes_precedence_over_truncation() {
    let mut value = configuration("tictactoe", random(), random());
    value["game"]["parameters"] = json!({"grid_size":1});
    value["max_actions"] = json!(1);
    let report = run(&resolve(value).unwrap()).unwrap();
    assert!(
        report
            .games
            .iter()
            .all(|game| !game.truncated && game.accepted_moves == 1)
    );
}
