use super::files::file;
use super::{BrowserEngine, send};
use serde_json::{Value, json};

fn prepare(engine: &mut BrowserEngine, version: &str) -> Value {
    let mut reply = send(engine, "1", Some(version), json!({"type":"prepare_ai"}));
    let mut polls = 0;
    while reply["proposal"]["pending"] == true {
        assert!(polls < 100);
        polls += 1;
        reply = send(
            engine,
            "1",
            Some(version),
            json!({"type":"poll_ai","proposal":reply["proposal"]["id"]}),
        );
    }
    assert!(reply["error"].is_null(), "{reply}");
    reply
}

#[test]
fn neural_mcts_plays_both_heads_and_restores_self_contained_recordings() {
    for game in ["tictactoe", "connect4"] {
        for head in ["scalar", "vector"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../web/tests/fixtures/mcts/{game}-{head}.json"));
            let text = std::fs::read_to_string(path).unwrap();
            let mut engine = BrowserEngine::new();
            let loaded = file(&mut engine, "0", json!({"type":"load_model"}), &text);
            assert!(loaded["error"].is_null(), "{loaded}");
            let neural = json!({"kind":"neural_mcts","seed":"42","model_id":loaded["text"],
                "configuration":{"schema_version":1,"tree_policy":"puct","rollout_length":0,
                    "budgets":{"iterations":32},"state_evaluator":{"name":"network_value"},
                    "puct_preferences":{"source":"probabilities","evaluator":{"name":"network_policy"}}}});
            let native = json!({"kind":"mcts","seed":"17","configuration":{
                "schema_version":1,"rollout_length":0,"budgets":{"iterations":32}}});
            let mut view = send(
                &mut engine,
                "0",
                None,
                json!({"type":"create","game":game,
                "controllers":[neural,native]}),
            )["view"]
                .clone();
            for _ in 0..4 {
                let calculated = prepare(&mut engine, view["position_version"].as_str().unwrap());
                let committed = send(
                    &mut engine,
                    "1",
                    view["position_version"].as_str(),
                    json!({"type":"commit_ai","proposal":calculated["proposal"]["id"]}),
                );
                assert!(committed["error"].is_null(), "{committed}");
                view = committed["view"].clone();
            }
            let saved = file(
                &mut engine,
                "1",
                json!({"type":"export","scope":"session","cursor":4,"delay_ms":0}),
                "",
            );
            assert!(saved["error"].is_null(), "{saved}");
            let mut restored = BrowserEngine::new();
            let imported = file(
                &mut restored,
                "0",
                json!({"type":"import","controllers":null}),
                saved["text"].as_str().unwrap(),
            );
            assert!(imported["error"].is_null(), "{imported}");
            assert_eq!(imported["view"]["board"], view["board"]);
            assert_eq!(imported["view"]["controllers"], view["controllers"]);
            let expected = prepare(&mut engine, view["position_version"].as_str().unwrap());
            let actual = prepare(
                &mut restored,
                imported["view"]["position_version"].as_str().unwrap(),
            );
            assert_eq!(actual["proposal"]["action"], expected["proposal"]["action"]);
        }
    }
}

#[test]
fn native_mcts_yields_and_discard_restores_the_same_decision_in_both_games() {
    for game in ["tictactoe", "connect4"] {
        let mut engine = BrowserEngine::new();
        let seat = json!({"kind":"mcts","seed":"42","configuration":{
            "schema_version":1,"rollout_length":0,"budgets":{"iterations":64}}});
        let initial = send(
            &mut engine,
            "0",
            None,
            json!({"type":"create","game":game,"controllers":[seat, {"kind":"human"}]}),
        );
        assert!(initial["error"].is_null(), "{initial}");
        let pending = send(&mut engine, "1", Some("0"), json!({"type":"prepare_ai"}));
        assert_eq!(pending["proposal"]["pending"], true);
        assert_eq!(pending["view"]["board"], initial["view"]["board"]);
        send(
            &mut engine,
            "1",
            Some("0"),
            json!({"type":"discard_ai", "proposal":pending["proposal"]["id"]}),
        );
        let stale = send(
            &mut engine,
            "1",
            Some("0"),
            json!({"type":"poll_ai", "proposal":pending["proposal"]["id"]}),
        );
        assert!(!stale["error"].is_null());
        let first = prepare(&mut engine, "0");
        send(
            &mut engine,
            "1",
            Some("0"),
            json!({"type":"discard_ai", "proposal":first["proposal"]["id"]}),
        );
        let second = prepare(&mut engine, "0");
        assert_eq!(first["proposal"]["action"], second["proposal"]["action"]);
        let committed = send(
            &mut engine,
            "1",
            Some("0"),
            json!({"type":"commit_ai", "proposal":second["proposal"]["id"]}),
        );
        assert!(committed["error"].is_null(), "{committed}");
        assert_eq!(committed["view"]["cursor"], 1);
    }
}
