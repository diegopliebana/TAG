use super::files::{file, step};
use super::{BrowserEngine, send};
use serde_json::{Value, json};
fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../web/tests/fixtures/learned")
            .join(name),
    )
    .unwrap()
}
fn load(engine: &mut BrowserEngine, generation: &str, name: &str) -> String {
    let loaded = file(
        engine,
        generation,
        json!({"type":"load_model"}),
        &fixture(name),
    );
    assert!(loaded["error"].is_null(), "{loaded}");
    let doc: Value = serde_json::from_str(&fixture(name)).unwrap();
    if doc["model_identity"].is_string() {
        doc["model_identity"].as_str().unwrap().to_string()
    } else {
        loaded["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["kind"] == "q_learning")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .into()
    }
}
#[test]
fn neural_moves_match_python_traces_for_trained_and_controlled_zero_models() {
    let fixtures: Value = serde_json::from_str(&fixture("fixtures.json")).unwrap();
    for model in fixtures["models"].as_array().unwrap() {
        for trace in model["traces"].as_array().unwrap() {
            let mut engine = BrowserEngine::new();
            let id = load(&mut engine, "0", model["model_file"].as_str().unwrap());
            let seat = trace["seat"].as_u64().unwrap() as usize;
            let mut controllers = vec![json!({"kind":"osla","seed":"13"}); 2];
            controllers[seat] = json!({"kind":"neural_osla","seed":"7","model_id":id});
            let reply = send(
                &mut engine,
                "0",
                None,
                json!({"type":"create","game":model["game"],"controllers":controllers}),
            );
            assert!(reply["error"].is_null(), "{reply}");
            let mut view = reply["view"].clone();
            for expected in trace["actions"].as_array().unwrap() {
                view = step(&mut engine, "1", &view);
                let action = &view["accepted_move"]["action"];
                let id = if model["game"] == "tictactoe" {
                    action["x"].as_u64().unwrap() * 3 + action["y"].as_u64().unwrap()
                } else {
                    action["column"].as_u64().unwrap()
                };
                assert_eq!(
                    id,
                    expected.as_u64().unwrap(),
                    "{} seat {seat}",
                    model["model_file"]
                );
            }
            assert!(view["next_player"].is_null());
        }
    }
}
#[test]
fn learned_proposals_forks_reset_and_self_contained_files_keep_exact_continuation() {
    for (game, name, kind) in [
        ("tictactoe", "q-table.json", "q_learning"),
        ("tictactoe", "tictactoe-trained.json", "neural_osla"),
        ("connect4", "connect4-trained.json", "neural_osla"),
    ] {
        let mut engine = BrowserEngine::new();
        let id = load(&mut engine, "0", name);
        let setup = json!({"kind":kind,"seed":"7","model_id":id});
        let mut view = send(
            &mut engine,
            "0",
            None,
            json!({"type":"create","game":game,"controllers":[setup,setup]}),
        )["view"]
            .clone();
        let prepared = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        assert_eq!(prepared["view"]["board"], view["board"]);
        send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"discard_ai","proposal":prepared["proposal"]["id"]}),
        );
        let again = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        assert_eq!(again["proposal"]["action"], prepared["proposal"]["action"]);
        // Loading an unrelated model neither changes existing weights nor invalidates a valid proposal.
        load(
            &mut engine,
            "1",
            if game == "connect4" {
                "tictactoe-zero.json"
            } else {
                "connect4-zero.json"
            },
        );
        view = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"commit_ai","proposal":again["proposal"]["id"]}),
        )["view"]
            .clone();
        let first = view.clone();
        view = step(&mut engine, "1", &view);
        let second = view.clone();
        let fork = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"fork","branch":0,"cursor":1,"replacements":[null,null]}),
        );
        assert!(fork["error"].is_null(), "{fork}");
        view = step(&mut engine, "1", &fork["view"]);
        assert_eq!(view["board"], second["board"]);
        let saved = file(
            &mut engine,
            "1",
            json!({"type":"export","scope":"session","cursor":1,"delay_ms":0}),
            "",
        );
        assert!(saved["error"].is_null(), "{saved}");
        let bundle: Value = serde_json::from_str(saved["text"].as_str().unwrap()).unwrap();
        assert_eq!(bundle["kind"], "tag-learned-browser");
        assert_eq!(bundle["models"].as_array().unwrap().len(), 1);
        let expected = step(&mut engine, "1", &view);
        for scope in ["session", "branch"] {
            let text = if scope == "session" {
                saved["text"].as_str().unwrap().to_string()
            } else {
                file(
                    &mut engine,
                    "1",
                    json!({"type":"export","scope":scope,"cursor":1,"delay_ms":0}),
                    "",
                )["text"]
                    .as_str()
                    .unwrap()
                    .into()
            };
            let mut imported = BrowserEngine::new();
            let reply = file(
                &mut imported,
                "0",
                json!({"type":"import","controllers":null}),
                &text,
            );
            assert!(reply["error"].is_null(), "{reply}");
            assert_eq!(reply["view"]["board"], first["board"]);
            let selected = send(
                &mut imported,
                "1",
                reply["live_view"]["position_version"].as_str(),
                json!({"type":"select_branch","branch":0}),
            );
            let tip = &selected["view"];
            let actual = if scope == "session" {
                step(&mut imported, "1", tip)
            } else {
                tip.clone()
            };
            assert_eq!(actual["board"], expected["board"]);
        }
        let reset = send(
            &mut engine,
            "1",
            expected["position_version"].as_str(),
            json!({"type":"reset"}),
        );
        assert!(reset["error"].is_null(), "{reset}");
        assert_eq!(
            step(&mut engine, "1", &reset["view"])["board"],
            first["board"]
        );
    }
}
#[test]
fn malformed_models_and_bundles_leave_live_game_and_registry_unchanged() {
    let mut engine = BrowserEngine::new();
    load(&mut engine, "0", "tictactoe-trained.json");
    let view = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe"}),
    )["view"]
        .clone();
    let before = file(&mut engine, "1", json!({"type":"models"}), "")["models"].clone();
    let mut bad: Value = serde_json::from_str(&fixture("tictactoe-trained.json")).unwrap();
    bad["model_identity"] = json!("wrong");
    assert_eq!(
        file(
            &mut engine,
            "1",
            json!({"type":"load_model"}),
            &bad.to_string()
        )["error"]["code"],
        "invalid_model"
    );
    let broken = json!({"kind":"tag-learned-browser","version":1,"models":[serde_json::from_str::<Value>(&fixture("connect4-zero.json")).unwrap()],"replay":{}});
    assert!(
        !file(
            &mut engine,
            "1",
            json!({"type":"import","controllers":null}),
            &broken.to_string()
        )["error"]
            .is_null()
    );
    assert_eq!(
        file(&mut engine, "1", json!({"type":"models"}), "")["models"],
        before
    );
    assert_eq!(
        send(&mut engine, "1", None, json!({"type":"view"}))["view"],
        view
    );
    let invalid = send(
        &mut engine,
        "1",
        None,
        json!({"type":"create","game":"connect4","controllers":[{"kind":"neural_osla","seed":"1","model_id":before[0]["id"]},{"kind":"human"}]}),
    );
    assert_eq!(invalid["error"]["code"], "invalid_setup");
}
