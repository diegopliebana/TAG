use super::{files::file, send};
use crate::BrowserEngine;
use serde_json::{Value, json};
use tag_rust::{
    core::{ForwardModel, PlayerId},
    games::loveletter::*,
};

#[test]
fn effect_response_retains_one_participants_private_comparison_until_normal_view_is_requested() {
    let model = LoveLetterForwardModel;
    let p = LoveLetterParameters {
        seed: 42,
        ..Default::default()
    };
    let mut remaining = Card::DECK.to_vec();
    let mut deck = Vec::new();
    for card in [
        Card::Guard,
        Card::Baron,
        Card::Guard,
        Card::Handmaid,
        Card::King,
    ] {
        let index = remaining.iter().position(|&c| c == card).unwrap();
        deck.push(remaining.remove(index));
    }
    deck.extend(remaining);
    let state = model
        .init_with_deck(&p, 3, &deck.try_into().unwrap())
        .unwrap();
    let saved = tag_rust::replay::export_state(&model, &p, &state).unwrap();
    for ai in [false, true] {
        let mut engine = BrowserEngine::new();
        let controllers = if ai {
            json!([{"kind":"first_available","seed":"7"},{"kind":"human"},{"kind":"human"}])
        } else {
            json!([{"kind":"human"},{"kind":"human"},{"kind":"human"}])
        };
        let loaded = file(
            &mut engine,
            "0",
            json!({"type":"import","controllers":controllers}),
            &saved,
        );
        assert!(loaded["error"].is_null(), "{loaded}");
        let version = loaded["live_view"]["position_version"].as_str();
        let observer = usize::from(ai);
        let result = if ai {
            let mut r = send(&mut engine, "1", version, json!({"type":"prepare_ai"}));
            while r["proposal"]["pending"] == true {
                r = send(
                    &mut engine,
                    "1",
                    version,
                    json!({"type":"poll_ai","proposal":r["proposal"]["id"]}),
                );
            }
            send(
                &mut engine,
                "1",
                version,
                json!({"type":"commit_ai","proposal":r["proposal"]["id"],"observer":observer}),
            )
        } else {
            let invalid = send(
                &mut engine,
                "1",
                version,
                json!({"type":"human_move","actor":0,"action":{"target":{"card":"baron","target":1}},"observer":99}),
            );
            assert_eq!(invalid["error"]["code"], "invalid_observer");
            send(
                &mut engine,
                "1",
                version,
                json!({"type":"human_move","actor":0,"action":{"target":{"card":"baron","target":1}},"observer":observer}),
            )
        };
        assert!(result["error"].is_null(), "{result}");
        assert_eq!(result["view"]["observer"], observer);
        assert_eq!(result["view"]["next_player"], 2);
        let comparison = result["view"]["board"]["loveletter"]["private_events"]
            .as_array()
            .unwrap()
            .iter()
            .find_map(|e| e["information"].get("comparison"))
            .unwrap();
        assert_eq!(comparison["own_card"], if ai { "guard" } else { "king" });
        let fresh = send(&mut engine, "1", None, json!({"type":"view"}));
        assert_eq!(fresh["view"]["observer"], 2);
        assert!(
            !fresh["view"]["board"]["loveletter"]["private_events"]
                .to_string()
                .contains("comparison")
        );
    }
}

fn create(e: &mut BrowserEngine, n: usize, kind: &str) -> Value {
    let controllers:Vec<_>=(0..n).map(|s| match kind {
        "human"=>json!({"kind":"human"}),
        "mcts"=>json!({"kind":"mcts","seed":(s+7).to_string(),"configuration":{"schema_version":1,"rollout_length":0,"budgets":{"iterations":8}}}),
        "sampled_osla"=>json!({"kind":"sampled_osla","seed":(s+7).to_string(),"repetitions_per_action":2}),
        _=>json!({"kind":kind,"seed":(s+7).to_string()}),
    }).collect();
    let r = send(
        e,
        "0",
        None,
        json!({"type":"create","game":"loveletter","parameters":{"seed":"42","rules":{"winning_tokens":[1,1,1]}},"controllers":controllers}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
fn presentation(e: &mut BrowserEngine, observer: Value, public: bool, reveal: bool) -> Value {
    let r = send(
        e,
        "1",
        None,
        json!({"type":"presentation","observer":observer,"public_only":public,"reveal_hidden_information":reveal}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
fn step(e: &mut BrowserEngine, generation: &str, v: &Value) -> Value {
    let mut r = send(
        e,
        generation,
        v["position_version"].as_str(),
        json!({"type":"prepare_ai"}),
    );
    assert!(r["error"].is_null(), "{r}");
    while r["proposal"]["pending"] == true {
        r = send(
            e,
            generation,
            v["position_version"].as_str(),
            json!({"type":"poll_ai","proposal":r["proposal"]["id"]}),
        );
        assert!(r["error"].is_null(), "{r}");
    }
    let r = send(
        e,
        generation,
        v["position_version"].as_str(),
        json!({"type":"commit_ai","proposal":r["proposal"]["id"]}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
fn human(e: &mut BrowserEngine, v: &Value) -> Value {
    let r = send(
        e,
        "1",
        v["position_version"].as_str(),
        json!({"type":"human_move","actor":v["next_player"],"action":v["actions"][0]}),
    );
    assert!(r["error"].is_null(), "{r}");
    r["view"].clone()
}
#[test]
fn every_native_view_matches_rules_and_switches_directly_between_humans() {
    for n in 2..=4 {
        let mut e = BrowserEngine::new();
        let mut v = create(&mut e, n, "human");
        let model = LoveLetterForwardModel;
        let p = LoveLetterParameters {
            seed: 42,
            rules: LoveLetterRules {
                winning_tokens: [1; 3],
            },
            ..Default::default()
        };
        let mut state = model.init_state(&p, n).unwrap();
        while !v["next_player"].is_null() {
            let actor = v["next_player"].as_u64().unwrap() as usize;
            assert_eq!(v["observer"], actor);
            let observation = model.observe(&state, PlayerId::new(actor)).unwrap();
            assert_eq!(v["board"]["loveletter"]["hand"], json!(observation.hand()));
            assert_eq!(
                v["board"]["loveletter"]["private_events"],
                json!(observation.private_events())
            );
            assert!(v["board"]["loveletter"].get("privileged").is_none());
            let action: LoveLetterAction = serde_json::from_value(v["actions"][0].clone()).unwrap();
            model
                .next(&mut state, PlayerId::new(actor), &action)
                .unwrap();
            v = human(&mut e, &v);
        }
        for seat in 0..n {
            assert_eq!(
                v["scores"][seat],
                model.player_score(&state, PlayerId::new(seat)).unwrap()
            );
        }
    }
}
#[test]
fn reveal_preserves_prepared_search_rng_and_complete_recordings_for_all_native_policies() {
    for n in 2..=4 {
        for kind in ["first_available", "random", "sampled_osla", "mcts"] {
            let mut a = BrowserEngine::new();
            let mut b = BrowserEngine::new();
            let mut av = create(&mut a, n, kind);
            let mut bv = create(&mut b, n, kind);
            assert_eq!(av["actions"], json!([]));
            assert_eq!(av["board"]["loveletter"]["hand"], json!([]));
            let mut prepared = send(
                &mut a,
                "1",
                av["position_version"].as_str(),
                json!({"type":"prepare_ai"}),
            );
            assert!(prepared["error"].is_null(), "{prepared}");
            let revealed = presentation(&mut a, Value::Null, false, true);
            assert!(revealed["board"]["loveletter"]["privileged"]["draw_pile"].is_array());
            av = presentation(&mut a, Value::Null, false, false);
            while prepared["proposal"]["pending"] == true {
                prepared = send(
                    &mut a,
                    "1",
                    av["position_version"].as_str(),
                    json!({"type":"poll_ai","proposal":prepared["proposal"]["id"]}),
                );
            }
            assert!(prepared["proposal"]["action"].is_null());
            let committed = send(
                &mut a,
                "1",
                av["position_version"].as_str(),
                json!({"type":"commit_ai","proposal":prepared["proposal"]["id"]}),
            );
            assert!(committed["error"].is_null(), "{committed}");
            av = committed["view"].clone();
            bv = step(&mut b, "1", &bv);
            assert_eq!(av["board"], bv["board"]);
            let saved = file(
                &mut a,
                "1",
                json!({"type":"export","scope":"session","cursor":av["cursor"],"delay_ms":0}),
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
            let mut rv = loaded["live_view"].clone();
            for _ in 0..1000 {
                assert_eq!(av["board"], rv["board"]);
                if av["next_player"].is_null() {
                    break;
                }
                av = step(&mut a, "1", &av);
                rv = step(&mut restored, "1", &rv);
                bv = step(&mut b, "1", &bv);
                assert_eq!(av["board"], bv["board"]);
            }
            assert!(av["next_player"].is_null());
            let save_a = file(
                &mut a,
                "1",
                json!({"type":"export","scope":"session","cursor":av["cursor"],"delay_ms":0}),
                "",
            );
            let save_b = file(
                &mut b,
                "1",
                json!({"type":"export","scope":"session","cursor":bv["cursor"],"delay_ms":0}),
                "",
            );
            assert_eq!(
                save_a["text"], save_b["text"],
                "reveal changed saved RNG/controller progress"
            );
        }
    }
}
#[test]
fn private_history_spectator_forks_and_corrupt_files_keep_information_boundaries() {
    let mut e = BrowserEngine::new();
    let mut v = create(&mut e, 3, "human");
    let initial = v["board"].clone();
    for _ in 0..3 {
        v = human(&mut e, &v);
    }
    let old = send(
        &mut e,
        "1",
        None,
        json!({"type":"history","branch":0,"cursor":0,"observer":0}),
    );
    assert_eq!(old["view"]["board"], initial);
    assert_eq!(old["view"]["moves"], json!([]));
    let public = presentation(&mut e, Value::Null, true, false);
    assert_eq!(public["board"]["loveletter"]["private_events"], json!([]));
    assert_eq!(public["board"]["loveletter"]["hand"], json!([]));
    let fork = send(
        &mut e,
        "1",
        v["position_version"].as_str(),
        json!({"type":"fork","branch":0,"cursor":0,"replacements":[null,null,null]}),
    );
    assert!(fork["error"].is_null(), "{fork}");
    v = fork["view"].clone();
    assert_eq!(v["actions"], json!([]));
    v = presentation(&mut e, Value::Null, false, true);
    let save = file(
        &mut e,
        "1",
        json!({"type":"export","scope":"state","cursor":v["cursor"],"delay_ms":0}),
        "",
    );
    assert!(save["error"].is_null(), "{save}");
    let mut d: Value = serde_json::from_str(save["text"].as_str().unwrap()).unwrap();
    d["state"]["players"] = json!(99);
    let bad = file(
        &mut e,
        "1",
        json!({"type":"import","controllers":null}),
        &d.to_string(),
    );
    assert!(!bad["error"].is_null());
    let after = send(&mut e, "1", None, json!({"type":"view"}));
    assert_eq!(after["view"]["board"], v["board"]);
    let mut restored = BrowserEngine::new();
    let loaded = file(
        &mut restored,
        "0",
        json!({"type":"import","controllers":null}),
        save["text"].as_str().unwrap(),
    );
    assert!(loaded["error"].is_null(), "{loaded}");
    assert!(
        loaded["view"]["board"]["loveletter"]
            .get("privileged")
            .is_none()
    );
}
