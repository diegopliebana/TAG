use super::files::{file, step};
use super::{BrowserEngine, send};
use serde_json::{Value, json};

#[test]
fn random_cancellation_forks_and_files_restore_exact_decisions_in_both_games() {
    for game in ["tictactoe", "connect4"] {
        let mut engine = BrowserEngine::new();
        let mut view = send(&mut engine, "0", None, json!({"type":"create","game":game,"controllers":[{"kind":"random","seed":"7"},{"kind":"random","seed":"18446744073709551615"}]}))["view"].clone();
        let before = view.clone();
        let first = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        assert_eq!(first["view"]["board"], before["board"]);
        send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"discard_ai","proposal":first["proposal"]["id"]}),
        );
        let second = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        assert_eq!(first["proposal"]["action"], second["proposal"]["action"]);
        send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"discard_ai","proposal":second["proposal"]["id"]}),
        );
        let mut positions = vec![before];
        for _ in 0..4 {
            view = step(&mut engine, "1", &view);
            positions.push(view.clone());
        }
        view = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"fork","branch":0,"cursor":2,"replacements":[null,null]}),
        )["view"]
            .clone();
        view = step(&mut engine, "1", &view);
        view = step(&mut engine, "1", &view);
        assert_eq!(view["board"], positions[4]["board"]);
        // A takeover must retain the other Random's progress and the full inherited line.
        view = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"fork","branch":1,"cursor":2,"replacements":[{"kind":"human"},null]}),
        )["view"]
            .clone();
        view = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"human_move","actor":0,"action":view["actions"][0]}),
        )["view"]
            .clone();
        let saves = ["session", "branch"].map(|scope| {
            (
                scope,
                file(
                    &mut engine,
                    "1",
                    json!({"type":"export","scope":scope,"cursor":1,"delay_ms":23}),
                    "",
                ),
            )
        });
        let expected = step(&mut engine, "1", &view);
        view = expected.clone();
        for (scope, saved) in saves {
            assert!(saved["error"].is_null(), "{saved}");
            let mut data: Value = serde_json::from_str(saved["text"].as_str().unwrap()).unwrap();
            assert_eq!(data["versions"]["agents"], 2);
            if scope == "branch" {
                assert_eq!(data["branches"].as_array().unwrap().len(), 1);
                assert_eq!(data["versions"]["format"], 2);
            }
            let mut restored = BrowserEngine::new();
            let loaded = file(
                &mut restored,
                "0",
                json!({"type":"import","controllers":null}),
                saved["text"].as_str().unwrap(),
            );
            assert!(loaded["error"].is_null(), "{loaded}");
            let selected = loaded["live_view"]["branch"].clone();
            let tip = send(
                &mut restored,
                "1",
                None,
                json!({"type":"history","branch":selected,"cursor":3}),
            );
            let restored_next = step(&mut restored, "1", &tip["view"]);
            assert_eq!(restored_next["board"], expected["board"]);
            data["versions"]["agents"] = 1.into();
            let bad = file(
                &mut restored,
                "1",
                json!({"type":"import","controllers":null}),
                &data.to_string(),
            );
            assert!(
                bad["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("Random controllers require agent scheme version 2")
            );
            assert_eq!(bad["generation"], "1");
        }
        // The old root also restores the first random decision on Reset.
        let reset = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"reset"}),
        );
        assert_eq!(
            step(&mut engine, "1", &reset["view"])["board"],
            positions[1]["board"]
        );
    }
}
#[test]
fn random_supports_large_boards_without_osla_constraints_and_legacy_files_stay_compatible() {
    let mut engine = BrowserEngine::new();
    let parameters = tag_rust::games::tictactoe::TicTacToeParameters {
        grid_size: 64,
        ..Default::default()
    };
    let start = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"tictactoe","parameters":parameters,"controllers":[{"kind":"random","seed":"7"},{"kind":"random","seed":"8"}]}),
    );
    assert!(start["error"].is_null(), "{start}");
    assert_eq!(
        step(&mut engine, "1", &start["view"])["board"]["counters"][0],
        1
    );
    let state = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"state","cursor":1,"delay_ms":0}),
        "",
    );
    let loaded = file(
        &mut BrowserEngine::new(),
        "0",
        json!({"type":"import","controllers":[{"kind":"random","seed":"18446744073709551615"},{"kind":"human"}]}),
        state["text"].as_str().unwrap(),
    );
    assert!(loaded["error"].is_null(), "{loaded}");
    for text in [
        include_str!("../../../../examples/saves/tictactoe.tag-replay.json"),
        include_str!("../../../../examples/saves/connect4.tag-replay.json"),
    ] {
        let loaded = file(
            &mut BrowserEngine::new(),
            "0",
            json!({"type":"import","controllers":null}),
            text,
        );
        assert!(loaded["error"].is_null(), "{loaded}");
    }
}
