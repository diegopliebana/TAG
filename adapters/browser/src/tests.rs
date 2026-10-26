use super::BrowserEngine;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tag_rust::core::{Decision, PlayerId};
use tag_rust::games::{connect4::Connect4ForwardModel, tictactoe::TicTacToeForwardModel};
use tag_rust::replay::{BoardStateData, PersistableGame};

mod cantstop;
mod capabilities;
mod files;
mod history;
mod learned;
mod loveletter;
mod mcts;
mod random;

fn send(
    engine: &mut BrowserEngine,
    generation: &str,
    version: Option<&str>,
    command: Value,
) -> Value {
    serde_json::from_str(&engine.handle(&json!({"request_id":"1","generation":generation,"position_version":version,"command":command}).to_string())).unwrap()
}

#[test]
fn decoding_failures_keep_request_identity_and_typed_session_errors_keep_state() {
    let mut engine = BrowserEngine::new();
    for command in [
        json!({"type":"unknown_command"}),
        json!({"type":"create","game":"tictactoe","controllers":false}),
    ] {
        let reply = send(&mut engine, "0", None, command);
        assert_eq!(reply["request_id"], "1");
        assert_eq!(reply["request_generation"], "0");
        assert_eq!(reply["error"]["code"], "invalid_request");
    }
    let initial = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe"}),
    );
    let before = &initial["view"];
    for (command, code) in [
        (
            json!({"type":"human_move","actor":1,"action":{"x":0,"y":0}}),
            "wrong_actor",
        ),
        (
            json!({"type":"human_move","actor":0,"action":{"x":99,"y":0}}),
            "action_not_offered",
        ),
        (json!({"type":"prepare_ai"}), "wrong_controller"),
        (json!({"type":"view","observer":2}), "invalid_observer"),
    ] {
        let reply = send(&mut engine, "1", Some("0"), command);
        assert_eq!(reply["error"]["code"], code);
        let current = send(&mut engine, "1", None, json!({"type":"view"}));
        assert_eq!(&current["view"], before);
    }
    // A one-cell game ends on its first move; further requests keep the finished board.
    send(
        &mut engine,
        "1",
        None,
        json!({"type":"create","game":"tictactoe","parameters":{"grid_size":1,"limits":{"max_cells":100}}}),
    );
    let end = send(
        &mut engine,
        "2",
        Some("0"),
        json!({"type":"human_move","actor":0,"action":{"x":0,"y":0}}),
    );
    assert_eq!(end["view"]["board"]["winning_cells"], json!([[0, 0]]));
    let rejected = send(
        &mut engine,
        "2",
        Some("1"),
        json!({"type":"human_move","actor":0,"action":{"x":0,"y":0}}),
    );
    assert_eq!(rejected["error"]["code"], "game_finished");
    assert_eq!(
        send(&mut engine, "2", None, json!({"type":"view"}))["view"],
        {
            let mut view = end["view"].clone();
            view["accepted_move"] = Value::Null;
            view
        }
    );
}
fn assert_board<F: PersistableGame<SavedState = BoardStateData>>(
    reply: &Value,
    model: &F,
    state: &F::State,
) {
    let data = model.state_data(state);
    let board = &reply["view"]["board"];
    assert_eq!(board["cells"], json!(data.cells));
    assert_eq!(board["counters"], json!(data.counters));
    assert_eq!(board["outcome"], json!(data.outcome));
    // Tic-Tac-Toe's browser query is derived, whereas its version-1 file field remains empty.
    if F::GAME_ID != "tictactoe" {
        assert_eq!(board["winning_cells"], json!(data.winning_cells));
    }
}
fn matched_recording<F>(model: F, text: &str)
where
    F: PersistableGame<SavedState = BoardStateData>,
    F::Parameters: DeserializeOwned,
    F::Action: DeserializeOwned,
    F::Error: std::fmt::Debug,
{
    let file: Value = serde_json::from_str(text).unwrap();
    let p = serde_json::from_value(file["parameters"].clone()).unwrap();
    let mut state = model.init_state(&p, 2).unwrap();
    let mut engine = BrowserEngine::new();
    let mut reply = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":F::GAME_ID,"parameters":file["parameters"],"controllers":[{"kind":"osla","seed":"7"},{"kind":"osla","seed":"8"}]}),
    );
    assert!(reply["error"].is_null(), "{reply}");
    assert_board(&reply, &model, &state);
    for record in file["branches"][0]["moves"].as_array().unwrap() {
        let version = reply["view"]["position_version"]
            .as_str()
            .unwrap()
            .to_string();
        let prepared = send(
            &mut engine,
            "1",
            Some(&version),
            json!({"type":"prepare_ai"}),
        );
        assert!(prepared["error"].is_null(), "{prepared}");
        assert_eq!(prepared["proposal"]["action"], record["action"]);
        assert_board(&prepared, &model, &state); // Preparation preserves live state.
        reply = send(
            &mut engine,
            "1",
            Some(&version),
            json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
        );
        assert!(reply["error"].is_null(), "{reply}");
        let actor = PlayerId::new(record["actor"].as_u64().unwrap() as usize);
        let action = serde_json::from_value(record["action"].clone()).unwrap();
        model.next(&mut state, actor, &action).unwrap();
        assert_board(&reply, &model, &state);
    }
    assert_eq!(model.next_player(&state), Decision::Finished);
    assert!(reply["view"]["actions"].as_array().unwrap().is_empty());
}
#[test]
fn browser_adapter_matches_recorded_native_ai_games_move_by_move() {
    matched_recording(
        TicTacToeForwardModel,
        include_str!("../../../examples/saves/tictactoe.tag-replay.json"),
    );
    matched_recording(
        Connect4ForwardModel,
        include_str!("../../../examples/saves/connect4.tag-replay.json"),
    );
}
#[test]
fn stale_requests_and_failed_replacements_preserve_live_session() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe"}),
    );
    assert_eq!(created["generation"], "1");
    let moved = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"human_move","actor":0,"action":{"x":1,"y":1}}),
    );
    assert!(moved["error"].is_null());
    let stale = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"human_move","actor":1,"action":{"x":0,"y":0}}),
    );
    assert_eq!(stale["error"]["code"], "stale_position");
    let invalid = send(
        &mut engine,
        "1",
        None,
        json!({"type":"create","game":"connect4","parameters":{"width":129,"height":1,"winning_length":1,"limits":{"max_cells":4096}}}),
    );
    assert_eq!(invalid["error"]["code"], "resource_limit");
    let current = send(&mut engine, "1", None, json!({"type":"view","observer":1}));
    assert_eq!(current["view"]["board"], moved["view"]["board"]);
    let reset = send(&mut engine, "1", Some("1"), json!({"type":"reset"}));
    assert!(reset["error"].is_null());
    assert_eq!(reset["view"]["board"]["counters"][0], 0);
    let replacement = send(
        &mut engine,
        "1",
        None,
        json!({"type":"create","game":"connect4"}),
    );
    assert_eq!(replacement["generation"], "2");
    let old = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"human_move","actor":0,"action":{"x":0,"y":0}}),
    );
    assert_eq!(old["error"]["code"], "stale_generation");
    assert_eq!(
        send(&mut engine, "2", None, json!({"type":"view"}))["view"]["board"],
        replacement["view"]["board"]
    );
}
#[test]
fn discarded_proposals_leave_rng_progress_unchanged_and_cannot_commit_twice() {
    let mut engine = BrowserEngine::new();
    send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe","controllers":[{"kind":"osla","seed":"18446744073709551615"},{"kind":"human"}]}),
    );
    let first = send(&mut engine, "1", Some("0"), json!({"type":"prepare_ai"}));
    let discarded = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"discard_ai","proposal":first["proposal"]["id"]}),
    );
    assert!(discarded["error"].is_null());
    let second = send(&mut engine, "1", Some("0"), json!({"type":"prepare_ai"}));
    assert_eq!(first["proposal"]["action"], second["proposal"]["action"]);
    let wrong = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"commit_ai","proposal":first["proposal"]["id"]}),
    );
    assert_eq!(wrong["error"]["code"], "invalid_proposal");
    let accepted = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"commit_ai","proposal":second["proposal"]["id"]}),
    );
    assert!(accepted["error"].is_null());
    assert_eq!(
        send(
            &mut engine,
            "1",
            Some("1"),
            json!({"type":"commit_ai","proposal":second["proposal"]["id"]})
        )["error"]["code"],
        "invalid_proposal"
    );
}
#[test]
fn observers_and_malformed_commands_respect_permitted_input_boundaries() {
    let mut engine = BrowserEngine::new();
    send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe"}),
    );
    let inactive = send(&mut engine, "1", None, json!({"type":"view","observer":1}));
    assert!(inactive["view"]["actions"].as_array().unwrap().is_empty());
    assert!(!send(&mut engine, "1", None, json!({"type":"view","observer":99}))["error"].is_null());
    let error: Value = serde_json::from_str(&engine.handle("{")).unwrap();
    assert_eq!(error["error"]["code"], "invalid_request");
    let error: Value = serde_json::from_str(&engine.handle(&"x".repeat(32769))).unwrap();
    assert_eq!(error["error"]["code"], "invalid_request");
}
