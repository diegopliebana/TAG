use super::{BrowserEngine, send};
use serde_json::{Value, json};

fn step(engine: &mut BrowserEngine, view: &Value) -> Value {
    let version = view["position_version"].as_str().unwrap();
    let prepared = send(engine, "1", Some(version), json!({"type":"prepare_ai"}));
    assert!(prepared["error"].is_null(), "{prepared}");
    let result = send(
        engine,
        "1",
        Some(version),
        json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
    );
    assert!(result["error"].is_null(), "{result}");
    result["view"].clone()
}
#[test]
fn historical_views_and_nested_forks_preserve_every_position_and_agent_continuation() {
    for game in ["tictactoe", "connect4"] {
        let mut engine = BrowserEngine::new();
        let mut live = send(&mut engine, "0", None, json!({"type":"create","game":game,"controllers":[{"kind":"osla","seed":"7"},{"kind":"osla","seed":"18446744073709551615"}]}))["view"].clone();
        let mut positions = vec![live.clone()];
        for _ in 0..4 {
            live = step(&mut engine, &live);
            positions.push(live.clone());
        }
        let baseline_next = step(&mut engine, &live);
        live = baseline_next.clone();
        positions.push(live.clone());
        let tip_version = live["position_version"].clone();
        for (cursor, expected) in positions.iter().enumerate() {
            let history = send(
                &mut engine,
                "1",
                None,
                json!({"type":"history","branch":0,"cursor":cursor}),
            );
            let view = &history["view"];
            assert!(history["error"].is_null(), "{history}");
            for key in ["board", "controllers", "next_player", "results", "scores"] {
                assert_eq!(view[key], expected[key], "{game} cursor {cursor} {key}");
            }
            assert_eq!(view["position_version"], tip_version);
            assert_eq!(view["moves"].as_array().unwrap().len(), 5);
            for index in 0..5 {
                assert_eq!(view["moves"][index], positions[index + 1]["accepted_move"]);
            }
            assert_eq!(view["read_only"], json!(cursor < 5));
            if cursor < 5 {
                assert_eq!(view["actions"], json!([]));
                let rejected = send(
                    &mut engine,
                    "1",
                    tip_version.as_str(),
                    json!({"type":"prepare_ai"}),
                );
                assert_eq!(rejected["error"]["code"], "historical_position");
            }
        }
        let fork = send(
            &mut engine,
            "1",
            tip_version.as_str(),
            json!({"type":"fork","branch":0,"cursor":4,"replacements":[null,null]}),
        );
        assert!(fork["error"].is_null(), "{fork}");
        assert_eq!(fork["view"]["tip"], 4);
        assert_eq!(fork["view"]["branches"][0]["move_count"], 5);
        let fork_next = step(&mut engine, &fork["view"]);
        assert_eq!(fork_next["board"], baseline_next["board"]);
        let nested = send(
            &mut engine,
            "1",
            fork_next["position_version"].as_str(),
            json!({"type":"fork","branch":1,"cursor":2,"replacements":[{"kind":"human"},null]}),
        );
        assert!(nested["error"].is_null(), "{nested}");
        assert_eq!(nested["view"]["controllers"][0]["kind"], "human");
        assert_eq!(nested["view"]["moves"].as_array().unwrap().len(), 2);
        // Before the replacement boundary, the inherited seat still reports its original AI.
        let prefix = send(
            &mut engine,
            "1",
            None,
            json!({"type":"history","branch":2,"cursor":1}),
        );
        assert_eq!(prefix["view"]["controllers"][0]["kind"], "osla");
        let root = send(
            &mut engine,
            "1",
            None,
            json!({"type":"history","branch":2,"cursor":2}),
        );
        assert_eq!(root["view"]["controllers"][0]["kind"], "human");
        let action = &root["view"]["actions"][0];
        let accepted = send(
            &mut engine,
            "1",
            root["view"]["position_version"].as_str(),
            json!({"type":"human_move","actor":0,"action":action}),
        );
        assert!(accepted["error"].is_null(), "{accepted}");
        assert_eq!(
            accepted["view"]["accepted_move"]["controller"]["kind"],
            "human"
        );
        let selected = send(
            &mut engine,
            "1",
            accepted["view"]["position_version"].as_str(),
            json!({"type":"select_branch","branch":0}),
        );
        assert_eq!(selected["view"]["board"], baseline_next["board"]);
        assert_eq!(selected["view"]["branches"][2]["move_count"], 3);
        assert_eq!(selected["view"]["read_only"], false);
        let bad = send(
            &mut engine,
            "1",
            selected["view"]["position_version"].as_str(),
            json!({"type":"fork","branch":0,"cursor":999,"replacements":[null,null]}),
        );
        assert_eq!(bad["error"]["code"], "invalid_branch");
        let current = send(&mut engine, "1", None, json!({"type":"view"}));
        assert_eq!(current["view"]["board"], selected["view"]["board"]);
        assert_eq!(
            current["view"]["position_version"],
            selected["view"]["position_version"]
        );
    }
}

#[test]
fn malformed_replacements_and_stale_proposals_cannot_cross_branches() {
    let mut engine = BrowserEngine::new();
    let initial = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe","controllers":[{"kind":"osla","seed":"7"},{"kind":"human"}]}),
    );
    let prepared = send(&mut engine, "1", Some("0"), json!({"type":"prepare_ai"}));
    for replacements in [json!([]), json!([{"kind":"osla","seed":"-1"}, null])] {
        let bad = send(
            &mut engine,
            "1",
            Some("0"),
            json!({"type":"fork","branch":0,"cursor":0,"replacements":replacements}),
        );
        assert!(!bad["error"].is_null());
        assert_eq!(
            send(&mut engine, "1", None, json!({"type":"view"}))["view"],
            initial["view"]
        );
    }
    let fork = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"fork","branch":0,"cursor":0,"replacements":[null,null]}),
    );
    let old = send(
        &mut engine,
        "1",
        Some("0"),
        json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
    );
    assert_eq!(old["error"]["code"], "stale_position");
    let foreign = send(
        &mut engine,
        "1",
        fork["view"]["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
    );
    assert_eq!(foreign["error"]["code"], "invalid_proposal");
}
