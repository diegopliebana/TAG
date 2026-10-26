use super::{
    files::{file, step},
    send,
};
use crate::BrowserEngine;
use serde_json::{Value, json};

fn create(e: &mut BrowserEngine, n: usize, kind: &str) -> Value {
    let seats:Vec<_>=(0..n).map(|i|match kind {
        "human"=>json!({"kind":"human"}),
        "sampled_osla"=>json!({"kind":kind,"seed":(i+7).to_string(),"repetitions_per_action":2}),
        "mcts"=>json!({"kind":kind,"seed":(i+7).to_string(),"configuration":{"schema_version":1,"rollout_length":0,"budgets":{"iterations":8}}}),
        _=>json!({"kind":kind,"seed":(i+7).to_string()}),
    }).collect();
    let r = send(
        e,
        "0",
        None,
        json!({"type":"create","game":"cantstop","parameters":{"seed":"18446744073709551615"},"controllers":seats}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
fn human(e: &mut BrowserEngine, v: &Value, a: Value) -> Value {
    let r = send(
        e,
        "1",
        v["position_version"].as_str(),
        json!({"type":"human_move","actor":v["next_player"],"action":a}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
#[test]
fn human_phases_and_chance_recordings_roundtrip_and_reject_tampering_atomically() {
    for n in 2..=4 {
        let mut e = BrowserEngine::new();
        let mut v = create(&mut e, n, "human");
        assert_eq!(v["actions"], json!(["Roll"]));
        for _ in 0..25 {
            let action = v["actions"][0].clone();
            v = human(&mut e, &v, action);
            let saved = file(
                &mut e,
                "1",
                json!({"type":"export","scope":"session","cursor":v["cursor"],"delay_ms":0}),
                "",
            );
            assert!(saved["error"].is_null(), "{saved}");
            let text = saved["text"].as_str().unwrap();
            let mut restored = BrowserEngine::new();
            let load = file(
                &mut restored,
                "0",
                json!({"type":"import","controllers":null}),
                text,
            );
            assert!(load["error"].is_null(), "{load}");
            assert_eq!(load["view"]["board"], v["board"]);
            assert_eq!(load["parameters"]["seed"], "18446744073709551615");
            let next = v["actions"][0].clone();
            let a = human(&mut e, &v, next.clone());
            let b = human(&mut restored, &load["view"], next);
            assert_eq!(a["board"], b["board"]);
            v = a;
        }
        let saved = file(
            &mut e,
            "1",
            json!({"type":"export","scope":"session","cursor":v["cursor"],"delay_ms":0}),
            "",
        );
        let mut d: Value = serde_json::from_str(saved["text"].as_str().unwrap()).unwrap();
        assert!(d["branches"][0]["moves"][0]["chance"]["dice"].is_array());
        d["branches"][0]["moves"][0]["chance"]["dice"][0] = json!(0);
        let bad = file(
            &mut e,
            "1",
            json!({"type":"import","controllers":null}),
            &d.to_string(),
        );
        assert!(!bad["error"].is_null());
        assert_eq!(
            send(&mut e, "1", None, json!({"type":"view"}))["view"]["board"],
            v["board"]
        );
        let text = file(
            &mut e,
            "1",
            json!({"type":"export","scope":"state","cursor":v["cursor"],"delay_ms":0}),
            "",
        );
        assert!(text["error"].is_null(), "{text}");
        let mut restored = BrowserEngine::new();
        let r = file(
            &mut restored,
            "0",
            json!({"type":"import","controllers":null}),
            text["text"].as_str().unwrap(),
        );
        assert!(r["error"].is_null(), "{r}");
        assert_eq!(r["view"]["controllers"].as_array().unwrap().len(), n);
        assert_eq!(r["view"]["board"], v["board"]);
    }
}

#[test]
fn malformed_parameters_are_rejected_without_panicking_or_replacing_play() {
    let mut engine = BrowserEngine::new();
    let before = create(&mut engine, 2, "human");
    for parameters in [json!([]), json!(7), json!("invalid"), json!({"seed":7})] {
        let reply = send(
            &mut engine,
            "1",
            None,
            json!({"type":"create","game":"cantstop","parameters":parameters}),
        );
        assert!(!reply["error"].is_null());
        assert_eq!(
            send(&mut engine, "1", None, json!({"type":"view"}))["view"],
            before
        );
    }
}
#[test]
fn stochastic_controllers_discard_resume_history_fork_and_reset() {
    for kind in ["random", "first_available", "sampled_osla", "mcts"] {
        for n in 2..=4 {
            let mut e = BrowserEngine::new();
            let mut v = create(&mut e, n, kind);
            for _ in 0..12 {
                let version = v["position_version"].as_str();
                let mut p = send(&mut e, "1", version, json!({"type":"prepare_ai"}));
                while p["proposal"]["pending"] == true {
                    p = send(
                        &mut e,
                        "1",
                        version,
                        json!({"type":"poll_ai","proposal":p["proposal"]["id"]}),
                    );
                }
                assert!(p["error"].is_null(), "{p}");
                assert_eq!(p["view"]["board"], v["board"]);
                let discarded = send(
                    &mut e,
                    "1",
                    version,
                    json!({"type":"discard_ai","proposal":p["proposal"]["id"]}),
                );
                assert!(discarded["error"].is_null(), "{discarded}");
                let before = file(
                    &mut e,
                    "1",
                    json!({"type":"export","scope":"session","cursor":v["cursor"],"delay_ms":0}),
                    "",
                );
                let next = step(&mut e, "1", &v);
                assert_eq!(next["accepted_move"]["action"], p["proposal"]["action"]);
                let mut restored = BrowserEngine::new();
                let r = file(
                    &mut restored,
                    "0",
                    json!({"type":"import","controllers":null}),
                    before["text"].as_str().unwrap(),
                );
                assert!(r["error"].is_null(), "{r}");
                assert_eq!(step(&mut restored, "1", &r["view"])["board"], next["board"]);
                v = next;
            }
            let history = send(
                &mut e,
                "1",
                None,
                json!({"type":"history","branch":0,"cursor":4}),
            );
            assert_eq!(history["view"]["cursor"], 4);
            let fork = send(
                &mut e,
                "1",
                v["position_version"].as_str(),
                json!({"type":"fork","branch":0,"cursor":4,"replacements":vec![Value::Null;n]}),
            );
            assert!(fork["error"].is_null(), "{fork}");
            let branch_saved = file(
                &mut e,
                "1",
                json!({"type":"export","scope":"branch","cursor":4,"delay_ms":0}),
                "",
            );
            assert!(branch_saved["error"].is_null(), "{branch_saved}");
            let mut branch_restored = BrowserEngine::new();
            let branch_load = file(
                &mut branch_restored,
                "0",
                json!({"type":"import","controllers":null}),
                branch_saved["text"].as_str().unwrap(),
            );
            assert!(branch_load["error"].is_null(), "{branch_load}");
            let branch_next = step(&mut branch_restored, "1", &branch_load["view"]);
            let saved = file(
                &mut e,
                "1",
                json!({"type":"export","scope":"session","cursor":4,"delay_ms":0}),
                "",
            );
            assert!(saved["error"].is_null(), "{saved}");
            let mut restored = BrowserEngine::new();
            let r = file(
                &mut restored,
                "0",
                json!({"type":"import","controllers":null}),
                saved["text"].as_str().unwrap(),
            );
            assert!(r["error"].is_null(), "{r}");
            assert_eq!(r["view"]["board"], fork["view"]["board"]);
            let next = step(&mut e, "1", &fork["view"]);
            assert_eq!(step(&mut restored, "1", &r["view"])["board"], next["board"]);
            assert_eq!(branch_next["board"], next["board"]);
            let current = send(&mut e, "1", None, json!({"type":"view"}));
            let reset = send(
                &mut e,
                "1",
                current["view"]["position_version"].as_str(),
                json!({"type":"reset"}),
            );
            assert!(reset["error"].is_null(), "{reset}");
            assert_eq!(reset["view"]["actions"], json!(["Roll"]));
        }
    }
}
