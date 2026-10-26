use super::config::{Metric, Mode, ResolvedTournament, SeatAllocation};
use super::execution::PlayedGame;
use super::schedule::ScheduleSettings;
use super::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tag-tournament-{}-{}",
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

fn settings(n: usize, p: usize) -> ScheduleSettings {
    ScheduleSettings {
        agent_count: n,
        player_count: p,
        mode: Mode::Exhaustive,
        matchups: None,
        seat_allocation: SeatAllocation::Permutations,
        allow_self_play: false,
        repetitions: 1,
        seed: 42,
        max_games: 10_000,
    }
}
fn configuration(path: &Path) -> Value {
    json!({"schema_version":1,"agents":[
        {"name":"random","agent":{"type":"random"}},
        {"name":"osla","agent":{"type":"osla"}},
        {"name":"mcts","agent":{"type":"mcts","config":{"schema_version":1,"budgets":{"iterations":5},"rollout_length":1}}}
    ],"games":[{"id":"ttt","name":"tictactoe"},{"id":"c4","name":"connect4"}],
    "mode":"exhaustive","seed":42,"results_file":path,"include_actions":true})
}
fn resolve(value: Value) -> Result<ResolvedTournament> {
    serde_json::from_value::<TournamentConfig>(value)?.resolve(Path::new("."))
}

#[cfg(feature = "inference")]
#[test]
fn neural_tournaments_select_models_by_entry_and_preserve_separate_replays() {
    let temp = Temporary::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("web/tests/fixtures/mcts");
    let mut value = configuration(&temp.0.join("results.json"));
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"separate","directory":temp.0.join("logs")});
    value["agents"][2]["agent"] = json!({"type":"neural_mcts", "model_files":{
        "ttt":root.join("tictactoe-scalar.json"),"c4":root.join("connect4-vector.json")},
        "config":{"schema_version":1,"tree_policy":"puct","rollout_length":0,
            "budgets":{"iterations":8},"state_evaluator":{"name":"network_value"},
            "puct_preferences":{"source":"probabilities","evaluator":{"name":"network_policy"}}}});
    let settings = resolve(value.clone()).unwrap();
    let report = run(&settings).unwrap();
    assert!(report.failure.is_none(), "{:?}", report.failure);
    assert_eq!(report.completed_games, 12);
    for game in &report.games {
        assert!(game.game_log.is_some());
        assert!(game.replay_file.is_some());
    }
    value["agents"][2]["agent"]["model_files"]
        .as_object_mut()
        .unwrap()
        .remove("c4");
    assert!(run(&resolve(value.clone()).unwrap()).is_err());
    value["agents"][2]["agent"]["model_files"]["unknown"] =
        json!(root.join("tictactoe-scalar.json"));
    assert!(resolve(value).is_err());
}

#[test]
fn separate_logs_reconstruct_both_games_and_cross_reference_the_report() {
    use tag_rust::core::{PlayerId, PlayerResult};
    use tag_rust::replay::{ReplayLimits, import_replay};
    fn check<
        F: crate::matches::agents::MatchGame
            + tag_rust::core::PublicInformationModel
            + tag_rust::replay::ReplayGame,
    >(
        model: F,
        text: &str,
        outcome: &crate::matches::report::GameReport,
    ) where
        F::Action: serde::de::DeserializeOwned,
        F::SavedEvaluator: Default
            + tag_rust::core::checkpoint::Checkpoint<
                Snapshot = (),
                RestoreError: std::error::Error + 'static,
            >,
        <F::SavedEvaluator as tag_rust::core::evaluation::Evaluator<F::State>>::Error:
            std::error::Error + 'static,
    {
        let loaded = import_replay(text, model.clone(), ReplayLimits::default()).unwrap();
        assert_eq!(
            loaded
                .session
                .info(tag_rust::session::BranchId(0))
                .unwrap()
                .move_count,
            outcome.accepted_moves
        );
        let state = loaded.session.active().session().state();
        for (seat, result) in outcome.results.iter().enumerate() {
            let expected = match result.as_str() {
                "win" => PlayerResult::Win,
                "draw" => PlayerResult::Draw,
                _ => PlayerResult::Loss,
            };
            assert_eq!(
                model.player_result(state, PlayerId::new(seat)).unwrap(),
                expected
            );
        }
    }
    let temp = Temporary::new();
    let mut value = configuration(&temp.0.join("results.json"));
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"separate","directory":temp.0.join("logs")});
    let settings = resolve(value).unwrap();
    let report = run(&settings).unwrap();
    assert_eq!(report.completed_games, 12);
    assert!(report.failure.is_none(), "{:?}", report.failure);
    for record in &report.games {
        let log: Value = serde_json::from_str(
            &std::fs::read_to_string(temp.0.join(record.game_log.as_ref().unwrap())).unwrap(),
        )
        .unwrap();
        assert_eq!(
            log["tournament_id"],
            report.tournament_id.as_ref().unwrap().as_str()
        );
        assert_eq!(log["record"]["index"], record.index);
        assert_eq!(log["record"]["agent_seeds"], json!(record.agent_seeds));
        assert_eq!(
            log["record"]["outcome"]["results"],
            json!(record.outcome.as_ref().unwrap().results)
        );
        let text =
            std::fs::read_to_string(temp.0.join(record.replay_file.as_ref().unwrap())).unwrap();
        assert_eq!(
            std::fs::read_to_string(
                temp.0
                    .join("logs")
                    .join(log["replay_file"].as_str().unwrap())
            )
            .unwrap(),
            text
        );
        let outcome = record.outcome.as_ref().unwrap();
        assert!(outcome.actions.is_none());
        if record.entry_index == 0 {
            check(
                tag_rust::games::tictactoe::TicTacToeForwardModel,
                &text,
                outcome,
            );
        } else {
            check(
                tag_rust::games::connect4::Connect4ForwardModel,
                &text,
                outcome,
            );
        }
    }
    assert!(
        run(&settings)
            .unwrap_err()
            .to_string()
            .contains("directory")
    );
    let summary = report.summary();
    assert!(summary.contains("Outcome points="));
    assert!(!summary.contains("Entries with completed results"));
}

#[test]
fn logging_modes_preserve_outcomes_and_validate_configuration() {
    let temp = Temporary::new();
    let mut value = configuration(&temp.0.join("results.json"));
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"off"});
    let off = run(&resolve(value.clone()).unwrap()).unwrap();
    value["game_logs"] = json!({"mode":"inline"});
    let inline = run(&resolve(value.clone()).unwrap()).unwrap();
    for (a, b) in off.games.iter().zip(&inline.games) {
        assert_eq!(
            a.outcome.as_ref().unwrap().results,
            b.outcome.as_ref().unwrap().results
        );
        assert!(a.outcome.as_ref().unwrap().actions.is_none());
        assert_eq!(
            b.outcome.as_ref().unwrap().actions.as_ref().unwrap().len(),
            b.outcome.as_ref().unwrap().accepted_moves
        );
    }
    for invalid in [
        json!({"mode":"separate"}),
        json!({"mode":"off","directory":"logs"}),
        json!({"mode":"other"}),
        json!({"mode":"separate","directory":""}),
    ] {
        value["game_logs"] = invalid;
        assert!(resolve(value.clone()).is_err());
    }
    value["game_logs"] = json!({"mode":"inline"});
    value["include_actions"] = json!(true);
    assert!(resolve(value).is_err());
}

#[test]
fn separate_log_output_conflicts_fail_without_overwriting_files() {
    let temp = Temporary::new();
    let mut value = configuration(&temp.0.join("results.json"));
    let mut report = run(&resolve(value.clone()).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"separate","directory":temp.0.join("logs")});
    let settings = resolve(value.clone()).unwrap();
    std::fs::create_dir(temp.0.join("logs")).unwrap();
    let protected = temp.0.join("logs/game-000000.json");
    std::fs::write(&protected, "existing metadata").unwrap();
    let record = &mut report.games[0];
    assert!(super::logs::save(&settings, "test-id", record, &report.game_entries[0]).is_err());
    assert!(record.outcome.is_some());
    assert!(record.replay_file.is_some());
    assert!(record.game_log.is_none());
    assert_eq!(
        std::fs::read_to_string(&protected).unwrap(),
        "existing metadata"
    );

    // Conflicting destinations and browser-incompatible boards fail before play.
    value["game_logs"]["directory"] = json!(temp.0.join("new-logs"));
    value["results_file"] = json!(temp.0.join("new-logs/../new-logs/results.json"));
    assert!(
        run(&resolve(value.clone()).unwrap())
            .unwrap_err()
            .to_string()
            .contains("outside")
    );
    assert!(!temp.0.join("new-logs").exists());
    value["results_file"] = json!(temp.0.join("results.json"));
    value["games"][1]["parameters"] = json!({"width":129,"height":6});
    assert!(
        run(&resolve(value).unwrap())
            .unwrap_err()
            .to_string()
            .contains("128")
    );
    assert!(!temp.0.join("new-logs").exists());
}

#[test]
fn exhaustive_schedules_match_independent_counts_and_include_all_seats() {
    for (n, p, self_play, expected) in [
        (3, 2, false, 6),
        (4, 3, false, 24),
        (2, 3, true, 8),
        (3, 2, true, 9),
    ] {
        let mut input = settings(n, p);
        input.allow_self_play = self_play;
        let schedule = schedule::build(&input).unwrap();
        assert_eq!(schedule.len(), expected);
        let lineups: BTreeSet<_> = schedule.iter().map(|r| r.agents_by_seat.clone()).collect();
        assert_eq!(lineups.len(), expected);
        if !self_play {
            assert!(
                lineups
                    .iter()
                    .all(|g| g.iter().collect::<BTreeSet<_>>().len() == p)
            );
        }
    }
    let schedule = schedule::build(&settings(3, 3)).unwrap();
    let expected: BTreeSet<_> = [
        vec![0, 1, 2],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![2, 1, 0],
    ]
    .into_iter()
    .collect();
    assert_eq!(
        schedule
            .into_iter()
            .map(|r| r.agents_by_seat)
            .collect::<BTreeSet<_>>(),
        expected
    );
}

#[test]
fn random_count_is_multiplied_by_seat_allocations_and_repetitions() {
    let mut input = settings(2, 2);
    input.mode = Mode::Random;
    input.matchups = Some(5);
    assert_eq!(schedule::build(&input).unwrap().len(), 10);
    input.repetitions = 3;
    assert_eq!(schedule::build(&input).unwrap().len(), 30);
    input.repetitions = 1;
    input.seat_allocation = SeatAllocation::Random;
    assert_eq!(schedule::build(&input).unwrap().len(), 5);
    // Only one distinct group exists, but random sampling may repeat it five times.
    assert_eq!(
        schedule::build(&input)
            .unwrap()
            .iter()
            .map(|r| r.pairing)
            .collect::<BTreeSet<_>>()
            .len(),
        5
    );
}

#[test]
fn random_groups_are_reproducible_and_independent_of_seat_allocation_work() {
    let mut input = settings(4, 3);
    input.mode = Mode::Random;
    input.matchups = Some(30);
    let first = schedule::build(&input).unwrap();
    assert_eq!(first, schedule::build(&input).unwrap());
    let groups: Vec<_> = first
        .iter()
        .filter(|r| r.allocation == 0)
        .map(|r| {
            let mut ids = r.agents_by_seat.clone();
            ids.sort();
            ids
        })
        .collect();
    input.seat_allocation = SeatAllocation::Random;
    let random: Vec<_> = schedule::build(&input)
        .unwrap()
        .iter()
        .map(|r| {
            let mut ids = r.agents_by_seat.clone();
            ids.sort();
            ids
        })
        .collect();
    assert_eq!(groups, random);
    input.allow_self_play = true;
    assert!(
        schedule::build(&input).unwrap().iter().any(|r| r
            .agents_by_seat
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            < 3)
    );
}

#[test]
fn random_seats_select_one_allocation_for_each_exhaustive_group() {
    let mut input = settings(4, 3);
    input.seat_allocation = SeatAllocation::Random;
    let schedule = schedule::build(&input).unwrap();
    assert_eq!(schedule.len(), 4);
    assert_eq!(
        schedule
            .iter()
            .map(|r| {
                let mut v = r.agents_by_seat.clone();
                v.sort();
                v
            })
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
}

#[test]
fn schedule_limits_reject_complete_tournament_before_play() {
    let mut input = settings(4, 3);
    input.max_games = 23;
    assert!(schedule::build(&input).is_err());
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("result.json"));
    value["max_games"] = json!(11);
    assert!(run(&resolve(value).unwrap()).is_err());
    assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
}

#[test]
fn two_games_produce_all_raw_results_and_three_reproducible_rankings() {
    let temporary = Temporary::new();
    let input = resolve(configuration(&temporary.0.join("result.json"))).unwrap();
    let report: TournamentReport = run(&input).unwrap();
    assert_eq!(report, run(&input).unwrap());
    assert_eq!(report.planned_games, 12);
    assert_eq!(report.completed_games, 12);
    assert_eq!(report.rankings.per_game.len(), 2);
    assert_eq!(report.rankings.global.len(), 3);
    assert!(report.failure.is_none());
    for (index, record) in report.games.iter().enumerate() {
        assert_eq!(record.index, index);
        let outcome = record.outcome.as_ref().unwrap();
        assert_eq!(record.agent_seeds, outcome.agent_seeds);
        assert_eq!(outcome.seats, [0, 1]);
        assert_eq!(record.agent_names.len(), 2);
        assert_eq!(
            outcome.accepted_moves,
            outcome.actions.as_ref().unwrap().len()
        );
    }
    for row in &report.rankings.global {
        assert_eq!(row.games_covered, 2);
        assert_eq!(row.appearances, 8);
    }
    crate::matches::write_report(
        &input.config.results_file,
        &serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    let stored: Value =
        serde_json::from_str(&std::fs::read_to_string(&input.config.results_file).unwrap())
            .unwrap();
    assert_eq!(stored["games"].as_array().unwrap().len(), 12);
    assert_eq!(stored["rankings"]["per_game"].as_array().unwrap().len(), 2);
    assert!(
        run(&input).is_err(),
        "existing results must not be overwritten"
    );
}

#[test]
fn self_play_counts_each_independent_seat_appearance() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("self.json"));
    value["agents"] = json!([{"name":"same","agent":{"type":"first_available"}}]);
    value["games"] = json!([{"id":"ttt","name":"tictactoe"}]);
    value["allow_self_play"] = json!(true);
    let report = run(&resolve(value).unwrap()).unwrap();
    assert_eq!(report.completed_games, 1); // AA has one distinct seat allocation.
    assert_ne!(
        report.games[0].agent_seeds[0],
        report.games[0].agent_seeds[1]
    );
    let row = &report.rankings.global[0];
    assert_eq!(row.appearances, 2);
    assert_eq!(row.win_rate, Some(0.5));
}

#[test]
fn game_entries_with_the_same_player_count_share_the_random_schedule() {
    let temporary = Temporary::new();
    for self_play in [false, true] {
        let mut value = configuration(&temporary.0.join("same-schedule.json"));
        value["mode"] = json!("random");
        value["matchups"] = json!(5);
        value["allow_self_play"] = json!(self_play);
        let report = run(&resolve(value).unwrap()).unwrap();
        assert_eq!(
            report.game_entries[0].planned_games,
            report.game_entries[1].planned_games
        );
        let lineups = |entry| {
            report
                .games
                .iter()
                .filter(|r| r.entry_index == entry)
                .map(|r| (r.pairing, r.allocation, r.agents_by_seat.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(lineups(0), lineups(1));
        let first = report.games.iter().find(|r| r.entry_index == 0).unwrap();
        let second = report.games.iter().find(|r| r.entry_index == 1).unwrap();
        assert_ne!(first.agent_seeds, second.agent_seeds);
    }
}

#[test]
fn invalid_entries_and_agents_are_rejected_before_any_result_is_written() {
    let temporary = Temporary::new();
    let base = configuration(&temporary.0.join("no.json"));
    let mut invalid = Vec::new();
    let mut v = base.clone();
    v["games"][1]["parameters"] = json!({"width":0});
    invalid.push(v);
    let mut v = base.clone();
    v["games"][1]["player_count"] = json!(3);
    invalid.push(v);
    let mut v = base.clone();
    v["agents"][2]["agent"]["config"]["state_evaluator"] = json!({"name":"missing"});
    invalid.push(v);
    for value in invalid {
        assert!(run(&resolve(value).unwrap()).is_err());
    }
    let mut invalid = Vec::new();
    let mut v = base.clone();
    v["extra"] = json!(1);
    invalid.push(v);
    let mut v = base.clone();
    v["mode"] = json!("random");
    invalid.push(v);
    let mut v = base.clone();
    v["matchups"] = json!(5);
    invalid.push(v);
    let mut v = base.clone();
    v["mode"] = json!("random");
    v["matchups"] = json!(0);
    invalid.push(v);
    let mut v = base.clone();
    v["games"][1]["id"] = json!("ttt");
    invalid.push(v);
    let mut v = base.clone();
    v["agents"][1]["name"] = json!("random");
    invalid.push(v);
    let mut v = base.clone();
    v["repetitions"] = json!(0);
    invalid.push(v);
    let mut v = base;
    v["results_file"] = Value::Null;
    invalid.push(v);
    for value in invalid {
        assert!(resolve(value).is_err());
    }
    assert_eq!(std::fs::read_dir(&temporary.0).unwrap().count(), 0);
}

#[test]
fn runtime_failure_records_the_attempt_without_assigning_a_loss() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("partial.json"));
    value["agents"] = json!([
        {"name":"bad","agent":{"type":"mcts","config":{"schema_version":1,"tree_policy":"puct","budgets":{"iterations":30},"puct_preferences":{"source":"probabilities","evaluator":{"name":"uniform_actions"}}}}},
        {"name":"random","agent":{"type":"random"}}]);
    let input = resolve(value).unwrap();
    let report = run(&input).unwrap();
    assert_eq!(report.completed_games, 0);
    assert_eq!(report.games.len(), 1);
    assert!(report.failure.as_ref().unwrap().contains("zero mass"));
    assert!(
        report
            .rankings
            .global
            .iter()
            .all(|r| r.appearances == 0 && r.rank.is_none())
    );
    assert_eq!(report.games[0].agent_names, ["bad", "random"]);
    crate::matches::write_report(
        &input.config.results_file,
        &serde_json::to_string(&report).unwrap(),
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(&input.config.results_file)
            .unwrap()
            .contains("zero mass")
    );
}

#[test]
fn later_runtime_failure_preserves_completed_games_and_marks_unplayed_agents() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("partial.json"));
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"separate","directory":temporary.0.join("logs")});
    value["agents"][0]["agent"] = json!({"type":"first_available"});
    value["agents"][2]["agent"]["config"] = json!({"schema_version":1,"tree_policy":"puct","budgets":{"iterations":30},
        "puct_preferences":{"source":"probabilities","evaluator":{"name":"uniform_actions"}}});
    let input = resolve(value).unwrap();
    let report = run(&input).unwrap();
    assert_eq!(report.completed_games, 2);
    assert_eq!(report.games.len(), 3);
    assert!(report.games[..2].iter().all(|r| r.outcome.is_some()));
    assert!(
        report.games[..2]
            .iter()
            .all(|r| temporary.0.join(r.replay_file.as_ref().unwrap()).is_file())
    );
    assert!(report.games[2].game_log.is_none());
    assert!(
        report.games[2]
            .error
            .as_ref()
            .unwrap()
            .contains("zero mass")
    );
    let bad = report
        .rankings
        .global
        .iter()
        .find(|r| r.agent == "mcts")
        .unwrap();
    assert_eq!(bad.appearances, 0);
    assert_eq!(bad.rank, None);
    crate::matches::write_report(
        &input.config.results_file,
        &serde_json::to_string(&report).unwrap(),
    )
    .unwrap();
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(&input.config.results_file).unwrap())
            .unwrap();
    assert_eq!(saved["completed_games"], 2);
    assert_eq!(saved["games"].as_array().unwrap().len(), 3);
}

fn outcome(
    entry: usize,
    participants: [usize; 2],
    results: [&str; 2],
    points: [f64; 2],
) -> PlayedGame {
    PlayedGame {
        index: 0,
        entry_index: entry,
        game_id: format!("g{entry}"),
        pairing: 0,
        allocation: 0,
        repetition: 0,
        agents_by_seat: participants.to_vec(),
        agent_names: vec![],
        agent_seeds: vec![1, 2],
        outcome: Some(crate::matches::report::GameReport {
            index: 0,
            game_seed: None,
            max_actions: None,
            truncated: false,
            ranks: None,
            agent_seeds: vec![1, 2],
            seats: vec![0, 1],
            results: results.into_iter().map(String::from).collect(),
            accepted_moves: 0,
            mcts: vec![],
            actions: None,
            recording: None,
        }),
        rank_points: None,
        points: Some(points.to_vec()),
        game_log: None,
        replay_file: None,
        error: None,
    }
}
#[test]
fn global_rates_use_equal_game_weights_and_shared_ranks() {
    let mut records = vec![outcome(0, [0, 1], ["win", "loss"], [1., 0.])];
    for _ in 0..9 {
        records.push(outcome(1, [0, 1], ["loss", "win"], [0., 1.]));
    }
    let rankings = ranking::calculate(
        &["A".into(), "B".into(), "absent".into()],
        &["g0".into(), "g1".into()],
        &records,
        Metric::WinRate,
    )
    .unwrap();
    assert_eq!(rankings.global[0].win_rate, Some(0.5));
    assert_eq!(rankings.global[1].win_rate, Some(0.5));
    assert_eq!(rankings.global[0].rank, Some(1));
    assert_eq!(rankings.global[1].rank, Some(1));
    assert_eq!(rankings.global[0].wins, 1);
    assert_eq!(rankings.global[0].appearances, 10);
    assert_eq!(rankings.global[2].rank, None);
    assert_eq!(rankings.global[2].games_covered, 0);
}

#[test]
fn sorting_metric_changes_order_and_missing_games_are_not_losses() {
    let records = vec![
        outcome(0, [0, 2], ["win", "loss"], [1., 0.]),
        outcome(0, [0, 2], ["loss", "win"], [0., 1.]),
        outcome(0, [0, 2], ["loss", "win"], [0., 1.]),
        outcome(0, [1, 2], ["draw", "draw"], [0.5, 0.5]),
        outcome(0, [1, 2], ["draw", "draw"], [0.5, 0.5]),
    ];
    let agents = vec!["A".into(), "B".into(), "C".into()];
    let games = vec!["played".into(), "unplayed".into()];
    let wins = ranking::calculate(&agents, &games, &records, Metric::WinRate).unwrap();
    let points = ranking::calculate(&agents, &games, &records, Metric::PointsRate).unwrap();
    let position = |r: &super::ranking::Rankings, name: &str| {
        r.global.iter().position(|a| a.agent == name).unwrap()
    };
    assert!(position(&wins, "A") < position(&wins, "B"));
    assert!(position(&points, "B") < position(&points, "A"));
    let b = points.global.iter().find(|a| a.agent == "B").unwrap();
    assert_eq!(b.points_rate, Some(0.5));
    assert_eq!(b.games_covered, 1);
    assert_eq!(b.game_entries, 2);
    assert_eq!(b.win_rate, Some(0.0));
}

#[test]
fn file_references_are_relative_to_the_tournament_file() {
    let temporary = Temporary::new();
    std::fs::write(
        temporary.0.join("agent.json"),
        r#"{"schema_version":1,"budgets":{"iterations":2}}"#,
    )
    .unwrap();
    let mut value = configuration(Path::new("results/tournament.json"));
    value["agents"][2]["agent"] = json!({"type":"mcts","config_file":"agent.json"});
    let path = temporary.0.join("tournament.json");
    std::fs::write(&path, value.to_string()).unwrap();
    let input = TournamentConfig::load(&path).unwrap();
    assert_eq!(
        input.config.results_file,
        temporary.0.join("results/tournament.json")
    );
    let report = run(&input).unwrap();
    assert_eq!(report.completed_games, 12);
}

#[test]
fn rank_points_follow_finishing_places_and_average_ties() {
    assert_eq!(
        super::ranking::rank_points(&[1, 2, 3, 4]).unwrap(),
        [4.0, 3.0, 2.0, 1.0]
    );
    assert_eq!(
        super::ranking::rank_points(&[1, 2, 2, 4]).unwrap(),
        [4.0, 2.5, 2.5, 1.0]
    );
    assert_eq!(
        super::ranking::rank_points(&[1, 1, 3]).unwrap(),
        [2.5, 2.5, 1.0]
    );
    assert_eq!(super::ranking::rank_points(&[1, 1]).unwrap(), [1.5, 1.5]);
    for ranks in [vec![], vec![0, 1], vec![1, 1, 2], vec![1, 3], vec![2, 2]] {
        assert!(super::ranking::rank_points(&ranks).is_err());
    }
}

#[test]
fn cantstop_tournaments_offer_both_points_systems_and_separate_truncations() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("results.json"));
    value["agents"] = json!(
        (0..4)
            .map(|i| json!({"name":format!("p{i}"), "agent":{"type":"random"}}))
            .collect::<Vec<_>>()
    );
    value["games"] = json!([
        {"id":"natural","name":"cantstop","player_count":3,"max_actions":1000,"parameters":{"rules":{"column_heights":[1,1,1,1,1,1,1,1,1,1,1],"columns_to_win":1}}},
        {"id":"ranked","name":"cantstop","ranking":"score","player_count":4,"max_actions":1000,"parameters":{"rules":{"column_heights":[1,1,1,1,1,1,1,1,1,1,1],"columns_to_win":1}}},
        {"id":"limited","name":"cantstop","ranking":"claimed_columns","player_count":4,"max_actions":1}
    ]);
    value["ranking_metric"] = json!("rank_points_rate");
    let report = run(&resolve(value.clone()).unwrap()).unwrap();
    assert!(report.failure.is_none());
    assert!(report.truncated_games > 0);
    assert_eq!(
        report.completed_games + report.truncated_games,
        report.planned_games
    );
    assert_eq!(report, run(&resolve(value).unwrap()).unwrap());
    let summary = report.summary();
    assert!(summary.contains("Game entry 'natural': cantstop, 3 players"));
    assert!(
        summary.contains("Final ranking: game score (opt-in); finishing-place results available")
    );
    assert!(summary.contains("Combined results across 3 game entries"));
    assert!(!summary.contains("Entries with finishing-place results"));
    assert!(summary.contains("separately played configuration"));
    let mut seeds = std::collections::BTreeSet::new();
    for record in &report.games {
        let game = record.outcome.as_ref().unwrap();
        assert!(seeds.insert(game.game_seed.unwrap()));
        if game.truncated {
            assert!(record.points.is_none());
            assert!(record.rank_points.is_none());
            assert!(game.ranks.is_none());
        } else {
            assert_eq!(record.points.as_ref().unwrap().iter().sum::<f64>(), 1.0);
            if record.game_id == "ranked" {
                let points = record.rank_points.as_ref().unwrap();
                assert_eq!(points.iter().sum::<f64>(), 10.0);
                for (point, result) in points.iter().zip(&game.results) {
                    assert_eq!(*point, if result == "win" { 4.0 } else { 2.0 });
                }
            } else {
                assert!(record.rank_points.is_none());
            }
        }
    }
    for row in &report.rankings.global {
        assert!(row.rank_points_rate.is_some());
        assert_eq!(row.rank_games_covered, 1);
        assert_eq!(row.games_covered, 2);
        assert!(row.truncated > 0);
    }
    for row in &report.rankings.per_game[2].rows {
        assert_eq!(row.appearances, 0);
        assert!(row.points_rate.is_none());
        assert!(row.rank_points_rate.is_none());
    }
}

#[test]
fn loveletter_tournaments_preserve_shared_winners_and_split_outcome_points() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("results.json"));
    value["agents"] = json!([
        {"name":"a","agent":{"type":"random"}},
        {"name":"b","agent":{"type":"random"}},
        {"name":"c","agent":{"type":"random"}},
        {"name":"d","agent":{"type":"random"}}
    ]);
    value["games"] = json!([{"id":"ll","name":"loveletter","player_count":4,
        "parameters":{"rules":{"winning_tokens":[1,1,1]}}}]);
    value["repetitions"] = json!(64);
    let report = run(&resolve(value.clone()).unwrap()).unwrap();
    assert!(report.failure.is_none(), "{:?}", report.failure);
    assert_eq!(report.completed_games, 1536);
    assert_eq!(report, run(&resolve(value).unwrap()).unwrap());
    let mut shared = 0;
    for record in &report.games {
        let game = record.outcome.as_ref().unwrap();
        let winners = game.results.iter().filter(|r| *r == "win").count();
        assert!(winners > 0);
        if winners > 1 {
            shared += 1;
        }
        let points = record.points.as_ref().unwrap();
        for (result, point) in game.results.iter().zip(points) {
            assert_eq!(
                *point,
                if result == "win" {
                    1.0 / winners as f64
                } else {
                    0.0
                }
            );
        }
    }
    assert!(shared > 0, "seeded sample must exercise shared victories");
}

#[test]
fn separate_cantstop_logs_preserve_seed_rules_and_dice_without_a_browser_replay() {
    let temporary = Temporary::new();
    let mut value = configuration(&temporary.0.join("results.json"));
    value.as_object_mut().unwrap().remove("include_actions");
    value["game_logs"] = json!({"mode":"separate","directory":temporary.0.join("logs")});
    value["games"] = json!([{"id":"cs","name":"cantstop","player_count":3,"max_actions":3}]);
    value["agents"] = json!(
        (0..3)
            .map(|i| json!({"name":format!("p{i}"),"agent":{"type":"random"}}))
            .collect::<Vec<_>>()
    );
    let report = run(&resolve(value).unwrap()).unwrap();
    assert!(report.failure.is_none());
    for record in &report.games {
        assert!(record.replay_file.is_none());
        let path = temporary.0.join(record.game_log.as_ref().unwrap());
        let log: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(log["replay_file"].is_null());
        assert_eq!(log["actions"].as_array().unwrap().len(), 3);
        assert_eq!(
            log["parameters"]["seed"],
            json!(record.outcome.as_ref().unwrap().game_seed.unwrap())
        );
        assert_eq!(log["parameters"]["rules"]["markers"], json!(3));
        assert_eq!(
            log["actions"][0]["chance"]["dice"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }
}

#[test]
fn cantstop_example_uses_one_twelve_game_sample_for_both_sorting_metrics() {
    let temporary = Temporary::new();
    let mut value: Value =
        serde_json::from_str(include_str!("../../tournaments/cantstop.json")).unwrap();
    value["seed"] = json!(42);
    value["results_file"] = json!(temporary.0.join("results.json"));
    let outcome_sorted = run(&resolve(value.clone()).unwrap()).unwrap();
    value["ranking_metric"] = json!("average_finishing_position");
    let place_sorted = run(&resolve(value).unwrap()).unwrap();
    assert_eq!(outcome_sorted.games, place_sorted.games);
    assert_eq!(outcome_sorted.game_entries.len(), 1);
    assert_eq!(outcome_sorted.planned_games, 12);
    assert_eq!(outcome_sorted.completed_games, 12);
    assert!(outcome_sorted.failure.is_none());
    for row in &outcome_sorted.rankings.per_game[0].rows {
        assert_eq!(row.appearances, 12);
        assert_eq!(row.ranked_appearances, 12);
        assert!(row.points_rate.is_some());
        assert!(row.rank_points_rate.is_some());
        let other = place_sorted.rankings.per_game[0]
            .rows
            .iter()
            .find(|r| r.agent == row.agent)
            .unwrap();
        assert_eq!(row.points_rate, other.points_rate);
        assert_eq!(row.rank_points_rate, other.rank_points_rate);
    }
    let summary = outcome_sorted.summary();
    assert_eq!(summary.matches("Game entry '").count(), 1);
    assert!(!summary.contains("Combined results"));
    assert!(!summary.contains("truncated="));
    assert!(!summary.contains("appearances="));
    assert!(!summary.contains("Entries with"));
    assert!(summary.contains("games played=12"));
    assert!(summary.contains("Average finishing position="));
    assert!(summary.contains("Both statistics use the same completed games"));
}

#[test]
fn average_finishing_position_averages_ties_and_entries_and_sorts_lowest_first() {
    let mut first = outcome(0, [0, 1], ["win", "loss"], [1., 0.]);
    first.rank_points = Some(vec![2., 1.]);
    let mut draw = outcome(0, [0, 1], ["draw", "draw"], [0.5, 0.5]);
    draw.rank_points = Some(vec![1.5, 1.5]);
    let mut records = vec![first, draw];
    for _ in 0..9 {
        let mut game = outcome(1, [0, 1], ["loss", "win"], [0., 1.]);
        game.rank_points = Some(vec![1., 2.]);
        records.push(game);
    }
    let mut truncated = outcome(0, [0, 1], ["truncated", "truncated"], [0., 0.]);
    truncated.outcome.as_mut().unwrap().truncated = true;
    truncated.points = None;
    records.push(truncated);
    let ranks = ranking::calculate(
        &["A".into(), "B".into(), "unplayed".into()],
        &["first".into(), "second".into()],
        &records,
        Metric::AverageFinishingPosition,
    )
    .unwrap();
    assert_eq!(ranks.per_game[0].rows[0].agent, "A");
    assert_eq!(
        ranks.per_game[0].rows[0].average_finishing_position,
        Some(1.25)
    );
    assert_eq!(
        ranks.per_game[0].rows[1].average_finishing_position,
        Some(1.75)
    );
    assert_eq!(ranks.global[0].agent, "B");
    assert_eq!(ranks.global[0].average_finishing_position, Some(1.375));
    assert_eq!(ranks.global[1].average_finishing_position, Some(1.625));
    assert_eq!(ranks.global[2].average_finishing_position, None);
    assert_eq!(ranks.global[2].rank, None);
    assert_eq!(ranks.global[0].truncated, 1);
}
