use super::{BrowserEngine, files::file, send};
use serde_json::{Value, json};

fn create(engine: &mut BrowserEngine, seats: Vec<Value>) -> Value {
    let reply = send(
        engine,
        "0",
        None,
        json!({"type":"create","game":"thegame","parameters":{"seed":"42","first_player":0,"communication":"LimitedChoices"},"controllers":seats}),
    );
    assert!(reply["error"].is_null(), "{reply}");
    reply["view"].clone()
}
fn play(engine: &mut BrowserEngine, view: &Value) -> Value {
    let reply = send(
        engine,
        "1",
        view["position_version"].as_str(),
        json!({"type":"human_move","actor":view["next_player"],"action":view["actions"][0]}),
    );
    assert!(reply["error"].is_null(), "{reply}");
    reply["view"].clone()
}

fn prepare_mcts(engine: &mut BrowserEngine, view: &Value) -> Value {
    let version = view["position_version"].as_str().unwrap();
    let mut reply = send(engine, "1", Some(version), json!({"type":"prepare_ai"}));
    for _ in 0..100 {
        if reply["proposal"]["pending"] != true {
            break;
        }
        reply = send(
            engine,
            "1",
            Some(version),
            json!({"type":"poll_ai","proposal":reply["proposal"]["id"]}),
        );
    }
    assert!(reply["error"].is_null(), "{reply}");
    assert_ne!(reply["proposal"]["pending"], true, "{reply}");
    reply
}

#[test]
fn neural_mcts_uses_a_loaded_two_seat_model_and_restores_search() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../web/tests/fixtures/mcts/thegame-two.json"),
    )
    .unwrap();
    let mut engine = BrowserEngine::new();
    let loaded = file(&mut engine, "0", json!({"type":"load_model"}), &text);
    assert!(loaded["error"].is_null(), "{loaded}");
    let neural = json!({"kind":"neural_mcts","seed":"17","model_id":loaded["text"],
        "bindings":{"value":"network_value","policy":"network_policy",
            "value_observers":"deciding_player"},
        "configuration":{"schema_version":1,"structured_actions":true,
            "tree_policy":"puct","rollout_length":0,"budgets":{"iterations":8},
            "state_evaluator":{"name":"network_value"},
            "puct_preferences":{"source":"probabilities",
                "evaluator":{"name":"network_policy"}}}});
    let wrong_seats = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"thegame","parameters":{"seed":"42"},
            "controllers":[neural.clone(),{"kind":"human"},{"kind":"human"}]}),
    );
    assert_eq!(wrong_seats["error"]["code"], "invalid_setup");
    let wrong_rules = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"thegame",
            "parameters":{"seed":"42","communication":"Silent"},
            "controllers":[neural.clone(),{"kind":"human"}]}),
    );
    assert_eq!(wrong_rules["error"]["code"], "invalid_setup");
    let mut view = create(
        &mut engine,
        vec![neural.clone(), json!({"kind":"random","seed":"9"})],
    );
    for _ in 0..3 {
        let proposal = prepare_mcts(&mut engine, &view);
        let committed = send(
            &mut engine,
            "1",
            view["position_version"].as_str(),
            json!({"type":"commit_ai","proposal":proposal["proposal"]["id"]}),
        );
        assert!(committed["error"].is_null(), "{committed}");
        view = committed["view"].clone();
    }
    assert_eq!(
        view["board"]["thegame"]["placements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let saved = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"session","cursor":view["cursor"],"delay_ms":0}),
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
    let expected = prepare_mcts(&mut engine, &view);
    let actual = prepare_mcts(&mut restored, &imported["view"]);
    assert_eq!(actual["proposal"]["action"], expected["proposal"]["action"]);
}

#[test]
fn mcts_plays_a_card_and_session_restores_its_next_choice() {
    let setup = json!({"kind":"mcts","seed":"17","configuration":{
        "schema_version":1,"structured_actions":true,"rollout_length":0,
        "budgets":{"iterations":16},"state_evaluator":{"name":"game_heuristic"}}});
    let mut original = BrowserEngine::new();
    let view = create(&mut original, vec![setup]);
    let proposal = prepare_mcts(&mut original, &view);
    assert_eq!(proposal["view"]["board"], view["board"]);
    let next = send(
        &mut original,
        "1",
        view["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":proposal["proposal"]["id"]}),
    );
    assert!(next["error"].is_null(), "{next}");
    let next_view = next["view"].clone();
    assert_eq!(
        next_view["board"]["thegame"]["placements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let saved = file(
        &mut original,
        "1",
        json!({"type":"export","scope":"session",
        "cursor":next_view["cursor"],"delay_ms":0}),
        "",
    );
    assert!(saved["error"].is_null(), "{saved}");
    let mut restored = BrowserEngine::new();
    let loaded = file(
        &mut restored,
        "0",
        json!({"type":"import","controllers":null}),
        saved["text"].as_str().unwrap(),
    );
    assert!(loaded["error"].is_null(), "{loaded}");
    assert_eq!(loaded["live_view"]["board"], next_view["board"]);
    let expected = prepare_mcts(&mut original, &next_view);
    let actual = prepare_mcts(&mut restored, &loaded["live_view"]);
    assert_eq!(actual["proposal"]["action"], expected["proposal"]["action"]);
}

#[test]
fn mcts_speaks_then_places_after_teammate_message() {
    let setup = json!({"kind":"mcts","seed":"17","configuration":{
        "schema_version":1,"structured_actions":true,"rollout_length":0,
        "budgets":{"iterations":16},"state_evaluator":{"name":"game_heuristic"}}});
    let mut engine = BrowserEngine::new();
    let first = create(&mut engine, vec![setup, json!({"kind":"human"})]);
    let speech = prepare_mcts(&mut engine, &first);
    let second = send(
        &mut engine,
        "1",
        first["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":speech["proposal"]["id"]}),
    );
    assert!(second["error"].is_null(), "{second}");
    assert_eq!(second["view"]["next_player"], 1);
    let after_human = play(&mut engine, &second["view"]);
    assert_eq!(
        after_human["board"]["thegame"]["messages"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let card = prepare_mcts(&mut engine, &after_human);
    let placed = send(
        &mut engine,
        "1",
        after_human["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":card["proposal"]["id"]}),
    );
    assert!(placed["error"].is_null(), "{placed}");
    assert_eq!(
        placed["view"]["board"]["thegame"]["placements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn communication_hands_and_team_view_are_separate() {
    let mut engine = BrowserEngine::new();
    let first = create(
        &mut engine,
        vec![json!({"kind":"human"}), json!({"kind":"human"})],
    );
    assert_eq!(first["game"], "thegame");
    assert_eq!(
        first["outcome_score"],
        json!({"scope":"team","values":[0.0],"unit":"cards_placed","maximum":98.0})
    );
    assert_eq!(first["next_player"], 0);
    assert_eq!(
        first["board"]["thegame"]["hand"].as_array().unwrap().len(),
        7
    );
    assert_eq!(first["actions"].as_array().unwrap().len(), 13);
    assert_eq!(
        first["board"]["thegame"]["action_paths"]
            .as_array()
            .unwrap()
            .len(),
        13
    );
    let second = play(&mut engine, &first);
    assert_eq!(second["next_player"], 1);
    assert_eq!(
        second["board"]["thegame"]["messages"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        second["board"]["thegame"]["hand"].as_array().unwrap().len(),
        7
    );
    assert_ne!(
        first["board"]["thegame"]["hand"],
        second["board"]["thegame"]["hand"]
    );
    let pending = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"state","cursor":second["cursor"],"delay_ms":0}),
        "",
    );
    assert!(pending["error"].is_null(), "{pending}");
    let mut resumed = BrowserEngine::new();
    let restored = file(
        &mut resumed,
        "0",
        json!({"type":"import","controllers":[{"kind":"human"},{"kind":"human"}]}),
        pending["text"].as_str().unwrap(),
    );
    assert!(restored["error"].is_null(), "{restored}");
    assert_eq!(
        restored["live_view"]["board"]["thegame"]["messages"],
        json!([])
    );
    assert_eq!(restored["live_view"]["next_player"], 1);
    let third = play(&mut engine, &second);
    let restored_third = play(&mut resumed, &restored["live_view"]);
    assert_eq!(
        restored_third["board"]["thegame"]["messages"],
        third["board"]["thegame"]["messages"]
    );
    assert_eq!(third["next_player"], 0);
    assert_eq!(
        third["board"]["thegame"]["messages"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let public = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":null,"public_only":true,"reveal_hidden_information":false}),
    );
    assert!(public["error"].is_null(), "{public}");
    assert_eq!(public["view"]["board"]["thegame"]["hand"], json!([]));
    assert_eq!(
        public["view"]["board"]["thegame"]["action_paths"],
        json!([])
    );
    assert_eq!(public["view"]["actions"], json!([]));
    let revealed = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":null,"public_only":true,"reveal_hidden_information":true}),
    );
    assert!(revealed["error"].is_null(), "{revealed}");
    let hands = &revealed["view"]["board"]["thegame"]["opponent_hands"];
    assert_eq!(hands.as_array().unwrap().len(), 2);
    assert_eq!(hands[1], second["board"]["thegame"]["hand"]);
    assert_eq!(hands[0].as_array().unwrap().len(), 7);
    assert_eq!(revealed["view"]["board"]["thegame"]["hand"], json!([]));
}
#[test]
fn state_and_session_files_restore_exact_progress() {
    let mut original = BrowserEngine::new();
    let mut view = create(&mut original, vec![json!({"kind":"human"})]);
    view = play(&mut original, &view);
    for scope in ["state", "branch", "session"] {
        let saved = file(
            &mut original,
            "1",
            json!({"type":"export","scope":scope,"cursor":view["cursor"],"delay_ms":0}),
            "",
        );
        assert!(saved["error"].is_null(), "{saved}");
        let mut restored = BrowserEngine::new();
        let controllers = if scope == "state" {
            json!([{"kind":"human"}])
        } else {
            Value::Null
        };
        let loaded = file(
            &mut restored,
            "0",
            json!({"type":"import","controllers":controllers}),
            saved["text"].as_str().unwrap(),
        );
        assert!(loaded["error"].is_null(), "{loaded}");
        assert_eq!(
            loaded["live_view"]["board"]["thegame"],
            view["board"]["thegame"]
        );
        assert_eq!(loaded["live_view"]["actions"], view["actions"]);
    }
}

#[test]
fn random_team_finishes_and_all_seats_share_the_result() {
    for seats in [1, 5] {
        let mut engine = BrowserEngine::new();
        let controllers = (0..seats)
            .map(|seat| json!({"kind":"random","seed":(seat+7).to_string()}))
            .collect();
        let mut view = create(&mut engine, controllers);
        let mut steps = 0;
        while !view["next_player"].is_null() {
            let prepared = send(
                &mut engine,
                "1",
                view["position_version"].as_str(),
                json!({"type":"prepare_ai"}),
            );
            assert!(prepared["error"].is_null(), "{prepared}");
            let reply = send(
                &mut engine,
                "1",
                view["position_version"].as_str(),
                json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
            );
            assert!(reply["error"].is_null(), "{reply}");
            view = reply["view"].clone();
            steps += 1;
            assert!(steps < 500);
        }
        assert!(matches!(
            view["board"]["outcome"]["kind"].as_str(),
            Some("shared_win" | "team_loss")
        ));
        let first = view["results"][0].clone();
        assert!(
            view["results"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| *r == first)
        );
    }
}

#[test]
fn random_branch_files_keep_the_next_decision() {
    let mut original = BrowserEngine::new();
    let controllers = vec![
        json!({"kind":"random","seed":"7"}),
        json!({"kind":"random","seed":"8"}),
    ];
    let initial = create(&mut original, controllers);
    let prepared = send(
        &mut original,
        "1",
        initial["position_version"].as_str(),
        json!({"type":"prepare_ai"}),
    );
    assert!(prepared["error"].is_null(), "{prepared}");
    let first = send(
        &mut original,
        "1",
        initial["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
    );
    assert!(first["error"].is_null(), "{first}");
    let view = &first["view"];
    for scope in ["branch", "session"] {
        let saved = file(
            &mut original,
            "1",
            json!({"type":"export","scope":scope,"cursor":view["cursor"],"delay_ms":0}),
            "",
        );
        assert!(saved["error"].is_null(), "{saved}");
        let mut restored = BrowserEngine::new();
        let loaded = file(
            &mut restored,
            "0",
            json!({"type":"import","controllers":null}),
            saved["text"].as_str().unwrap(),
        );
        assert!(loaded["error"].is_null(), "{loaded}");
        let next = send(
            &mut restored,
            "1",
            loaded["live_view"]["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        let expected = send(
            &mut original,
            "1",
            view["position_version"].as_str(),
            json!({"type":"prepare_ai"}),
        );
        assert!(next["error"].is_null(), "{next}");
        assert!(expected["error"].is_null(), "{expected}");
        assert_eq!(next["proposal"]["action"], expected["proposal"]["action"]);
        let discarded = send(
            &mut original,
            "1",
            view["position_version"].as_str(),
            json!({"type":"discard_ai","proposal":expected["proposal"]["id"]}),
        );
        assert!(discarded["error"].is_null(), "{discarded}");
    }
}
