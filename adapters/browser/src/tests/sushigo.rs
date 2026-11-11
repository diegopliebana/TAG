use super::{files::file, send};
use crate::BrowserEngine;
use serde_json::json;

#[test]
fn sushi_choices_resolve_once_and_cancel_restores_initial_view() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo",
        "parameters":{"seed":"42"},"controllers":[{"kind":"human"},{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let first = &created["view"];
    assert_eq!(first["waiting_seats"], json!([0, 1, 2]));
    assert!(first["board"]["sushigo"].get("known_hands").is_none());
    let version = first["position_version"].as_str().unwrap();
    let chosen = first["actions"][0].clone();
    let submitted = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":0,"action":chosen}),
    );
    assert!(submitted["error"].is_null(), "{submitted}");
    assert_eq!(submitted["view"]["position_version"], version);
    assert_eq!(submitted["view"]["waiting_seats"], json!([1, 2]));
    assert_eq!(submitted["view"]["cursor"], 0);
    assert_eq!(submitted["view"]["observer"], 1);
    assert!(
        submitted["view"]["board"]["sushigo"]
            .get("known_hands")
            .is_none()
    );
    assert!(
        !submitted["view"]["board"]["sushigo"]["hand"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(!submitted["view"]["actions"].as_array().unwrap().is_empty());
    let cancelled = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"cancel_joint"}),
    );
    assert!(cancelled["error"].is_null(), "{cancelled}");
    assert_eq!(cancelled["view"]["waiting_seats"], json!([0, 1, 2]));
    assert_ne!(cancelled["view"]["position_version"], version);
    assert_eq!(
        cancelled["view"]["board"]["sushigo"]["hand"],
        first["board"]["sushigo"]["hand"]
    );
}

#[test]
fn sushi_complete_choice_reveals_only_after_all_seats_submit() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo",
        "parameters":{"seed":"19"},"controllers":[{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let initial = created["view"].clone();
    let version = initial["position_version"].as_str().unwrap();
    let first = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":0,
        "action":initial["actions"][0]}),
    );
    assert!(first["error"].is_null(), "{first}");
    assert_eq!(first["view"]["cursor"], 0);
    assert_eq!(
        first["view"]["board"]["sushigo"]["last_reveal"],
        initial["board"]["sushigo"]["last_reveal"]
    );
    assert_eq!(first["view"]["observer"], 1);
    assert!(!first["view"]["actions"].as_array().unwrap().is_empty());
    let second = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":1,
        "action":first["view"]["actions"][0]}),
    );
    assert!(second["error"].is_null(), "{second}");
    assert_eq!(second["view"]["cursor"], 1);
    assert_eq!(second["view"]["accepted_move"]["position"], 1);
    assert_eq!(
        second["view"]["accepted_move"]["action"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_ne!(second["view"]["position_version"], version);
    assert_eq!(second["view"]["observer"], 0);
    assert!(
        !second["view"]["board"]["sushigo"]["hand"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(second["view"]["waiting_seats"], json!([0, 1]));
}

#[test]
fn sushi_state_proof_and_joint_replay_restore_complete_boundary() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo",
        "parameters":{"seed":"19"},"controllers":[{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let version = created["view"]["position_version"].as_str().unwrap();
    let a = created["view"]["actions"][0].clone();
    let one = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":0,"action":a}),
    );
    assert!(one["error"].is_null(), "{one}");
    let blocked = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"state","cursor":0,"delay_ms":0}),
        "",
    );
    assert_eq!(blocked["error"]["code"], "invalid_request");
    let handoff = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":1,
        "public_only":false,"reveal_hidden_information":false}),
    );
    let b = handoff["view"]["actions"][0].clone();
    let two = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":1,"action":b}),
    );
    assert!(two["error"].is_null(), "{two}");
    for scope in ["state", "session"] {
        let saved = file(
            &mut engine,
            "1",
            json!({"type":"export","scope":scope,"cursor":1,"delay_ms":0}),
            "",
        );
        assert!(saved["error"].is_null(), "{saved}");
        let text = saved["text"].as_str().unwrap();
        let mut imported = BrowserEngine::new();
        let controls = if scope == "state" {
            json!([{"kind":"human"},{"kind":"human"}])
        } else {
            serde_json::Value::Null
        };
        let loaded = file(
            &mut imported,
            "0",
            json!({"type":"import","controllers":controls}),
            text,
        );
        assert!(loaded["error"].is_null(), "{loaded}");
        assert_eq!(
            loaded["live_view"]["board"]["sushigo"]["scores"],
            two["view"]["board"]["sushigo"]["scores"]
        );
        assert_eq!(
            loaded["live_view"]["board"]["sushigo"]["played"],
            two["view"]["board"]["sushigo"]["played"]
        );
    }
}

#[test]
fn sushi_fork_and_replay_keep_complete_decisions_and_reject_tampering() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo",
        "parameters":{"seed":"89"},"controllers":[{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let version = created["view"]["position_version"].as_str().unwrap();
    let first = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":0,"action":created["view"]["actions"][0]}),
    );
    assert!(first["error"].is_null(), "{first}");
    let blocked = send(&mut engine, "1", None, json!({"type":"view","observer":1}));
    assert_eq!(blocked["view"]["board"]["sushigo"]["hand"], json!([]));
    let revealed = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":1,"public_only":false,"reveal_hidden_information":false}),
    );
    assert!(revealed["error"].is_null(), "{revealed}");
    let second = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"human_move","actor":1,"action":revealed["view"]["actions"][0]}),
    );
    assert!(second["error"].is_null(), "{second}");
    let next_version = second["view"]["position_version"].as_str().unwrap();
    let historical = send(
        &mut engine,
        "1",
        Some(next_version),
        json!({"type":"history","branch":0,"cursor":0,"observer":0}),
    );
    assert!(historical["error"].is_null(), "{historical}");
    assert_eq!(historical["view"]["read_only"], true);
    assert_eq!(historical["view"]["actions"], json!([]));
    let at_one = send(
        &mut engine,
        "1",
        None,
        json!({"type":"history","branch":0,"cursor":1,"observer":0}),
    );
    assert_eq!(at_one["view"]["moves"].as_array().unwrap().len(), 1);
    let fork = send(
        &mut engine,
        "1",
        Some(next_version),
        json!({"type":"fork","branch":0,"cursor":0,"replacements":[null,null]}),
    );
    assert!(fork["error"].is_null(), "{fork}");
    assert_eq!(fork["view"]["branch"], 1);
    assert_eq!(fork["view"]["branches"][0]["move_count"], 1);
    assert_eq!(fork["view"]["branches"][1]["move_count"], 0);
    let branch_file = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"branch","cursor":0,"delay_ms":0}),
        "",
    );
    assert!(branch_file["error"].is_null(), "{branch_file}");
    let mut branch_engine = BrowserEngine::new();
    let branch_loaded = file(
        &mut branch_engine,
        "0",
        json!({"type":"import","controllers":null}),
        branch_file["text"].as_str().unwrap(),
    );
    assert!(branch_loaded["error"].is_null(), "{branch_loaded}");
    assert_eq!(
        branch_loaded["view"]["branches"].as_array().unwrap().len(),
        1
    );
    let saved = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"session","cursor":0,"delay_ms":0}),
        "",
    );
    assert!(saved["error"].is_null(), "{saved}");
    let text = saved["text"].as_str().unwrap();
    let mut imported = BrowserEngine::new();
    let loaded = file(
        &mut imported,
        "0",
        json!({"type":"import","controllers":null}),
        text,
    );
    assert!(loaded["error"].is_null(), "{loaded}");
    assert_eq!(loaded["view"]["branches"].as_array().unwrap().len(), 2);
    let mut changed: serde_json::Value = serde_json::from_str(text).unwrap();
    changed["branches"][0]["decisions"][0]["state_after"] = json!("0000000000000000");
    let rejected = file(
        &mut imported,
        "1",
        json!({"type":"import","controllers":null}),
        &changed.to_string(),
    );
    assert_eq!(rejected["error"]["code"], "invalid_file");
}

#[test]
fn sushi_ai_preparation_can_be_discarded_without_advancing_its_rng() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo",
        "parameters":{"seed":"27"},"controllers":[{"kind":"random","seed":"81"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let version = created["view"]["position_version"].as_str().unwrap();
    let first = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"prepare_ai"}),
    );
    assert!(first["error"].is_null(), "{first}");
    let id = first["proposal"]["id"].as_str().unwrap();
    let discarded = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"discard_ai","proposal":id}),
    );
    assert!(discarded["error"].is_null(), "{discarded}");
    let stale = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"commit_ai","proposal":id}),
    );
    assert_eq!(stale["error"]["code"], "invalid_proposal");
    let prepared = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"prepare_ai"}),
    );
    assert!(prepared["error"].is_null(), "{prepared}");
    let next = prepared["proposal"]["id"].as_str().unwrap();
    let committed = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"commit_ai","proposal":next}),
    );
    assert!(committed["error"].is_null(), "{committed}");
    assert_eq!(committed["view"]["waiting_seats"], json!([1]));
}

#[test]
fn sushi_native_mcts_prepares_a_complete_joint_choice() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo","parameters":{"seed":"42"},
        "controllers":[{"kind":"mcts","seed":"17","configuration":{"schema_version":1,"tree_policy":"ucb1","rollout_length":0,"budgets":{"iterations":8}}},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let version = created["view"]["position_version"].as_str().unwrap();
    let mut result = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"prepare_ai"}),
    );
    assert!(result["error"].is_null(), "{result}");
    for _ in 0..1000 {
        if result["proposal"]["pending"] == false {
            break;
        }
        let id = result["proposal"]["id"].as_str().unwrap();
        result = send(
            &mut engine,
            "1",
            Some(version),
            json!({"type":"poll_ai","proposal":id}),
        );
        assert!(result["error"].is_null(), "{result}");
    }
    assert_eq!(result["proposal"]["pending"], false);
    let committed = send(
        &mut engine,
        "1",
        Some(version),
        json!({"type":"commit_ai","proposal":result["proposal"]["id"]}),
    );
    assert!(committed["error"].is_null(), "{committed}");
    assert_eq!(committed["view"]["waiting_seats"], json!([1]));
}

#[test]
fn sushi_reset_branch_roundtrips_with_its_original_root() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo","parameters":{"seed":"5"},
        "controllers":[{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let initial = created["view"].clone();
    let reset = send(
        &mut engine,
        "1",
        Some(initial["position_version"].as_str().unwrap()),
        json!({"type":"reset"}),
    );
    assert!(reset["error"].is_null(), "{reset}");
    assert_eq!(reset["view"]["branches"][1]["origin"], "reset");
    let saved = file(
        &mut engine,
        "1",
        json!({"type":"export","scope":"session","cursor":0,"delay_ms":0}),
        "",
    );
    assert!(saved["error"].is_null(), "{saved}");
    let mut loaded = BrowserEngine::new();
    let restored = file(
        &mut loaded,
        "0",
        json!({"type":"import","controllers":null}),
        saved["text"].as_str().unwrap(),
    );
    assert!(restored["error"].is_null(), "{restored}");
    assert_eq!(restored["view"]["branches"][1]["origin"], "reset");
}

#[test]
fn sushi_privileged_reveal_shows_opponent_hands_without_changing_choices() {
    let mut engine = BrowserEngine::new();
    let created = send(
        &mut engine,
        "0",
        None,
        json!({"type":"create","game":"sushigo","parameters":{"seed":"17"},
        "controllers":[{"kind":"human"},{"kind":"human"},{"kind":"human"}]}),
    );
    assert!(created["error"].is_null(), "{created}");
    let initial = &created["view"];
    assert!(initial["board"]["sushigo"].get("opponent_hands").is_none());
    let actions = initial["actions"].clone();
    let version = initial["position_version"].as_str().unwrap();
    let revealed = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":null,
        "public_only":false,"reveal_hidden_information":true}),
    );
    assert!(revealed["error"].is_null(), "{revealed}");
    let hands = &revealed["view"]["board"]["sushigo"]["opponent_hands"];
    assert_eq!(hands.as_array().unwrap().len(), 3);
    assert!(hands[0].is_null());
    assert_eq!(hands[1].as_array().unwrap().len(), 9);
    assert_eq!(hands[2].as_array().unwrap().len(), 9);
    assert_eq!(revealed["view"]["actions"], actions);
    assert_eq!(revealed["view"]["position_version"], version);
    let hidden = send(
        &mut engine,
        "1",
        None,
        json!({"type":"presentation","observer":null,
        "public_only":false,"reveal_hidden_information":false}),
    );
    assert!(hidden["error"].is_null(), "{hidden}");
    assert!(
        hidden["view"]["board"]["sushigo"]
            .get("opponent_hands")
            .is_none()
    );
}
