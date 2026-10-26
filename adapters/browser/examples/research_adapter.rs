//! Native browser-adapter cost probes; this is not a browser rendering benchmark.
use serde_json::{Value, json};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use tag_browser::BrowserEngine;
struct Driver {
    engine: BrowserEngine,
    generation: String,
    position: Value,
}
impl Driver {
    fn new() -> Self {
        Self {
            engine: BrowserEngine::new(),
            generation: "0".into(),
            position: Value::Null,
        }
    }
    fn call(&mut self, command: Value) -> Value {
        let input = json!({"request_id":"1","generation":self.generation,"position_version":self.position,"command":command});
        let reply: Value = serde_json::from_str(&self.engine.handle(&input.to_string())).unwrap();
        assert!(reply["error"].is_null(), "{reply}");
        self.generation = reply["generation"].as_str().unwrap().into();
        if !reply["view"].is_null() {
            self.position = reply["view"]["position_version"].clone();
        }
        reply
    }
    fn file(&mut self, command: Value, text: &str) -> Value {
        let input = json!({"request_id":"1","generation":self.generation,"command":command});
        let reply: Value =
            serde_json::from_str(&self.engine.handle_file(&input.to_string(), text)).unwrap();
        assert!(reply["error"].is_null(), "{reply}");
        self.generation = reply["generation"].as_str().unwrap().into();
        if !reply["view"].is_null() {
            self.position = reply["view"]["position_version"].clone();
        }
        reply
    }
    fn decision(&mut self, commit: bool) -> Value {
        let mut reply = self.call(json!({"type":"prepare_ai"}));
        while reply["proposal"]["pending"] == true {
            reply = self.call(json!({"type":"poll_ai","proposal":reply["proposal"]["id"]}));
        }
        self.call(json!({"type":if commit {"commit_ai"} else {"discard_ai"},"proposal":reply["proposal"]["id"]}))
    }
}
fn measure(mut f: impl FnMut()) -> Vec<Value> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(150) {
        f();
    }
    (0..5)
        .map(|_| {
            let start = Instant::now();
            let mut calls = 0;
            while calls == 0 || start.elapsed() < Duration::from_millis(80) {
                f();
                calls += 1;
            }
            json!({"calls":calls,"elapsed_ns":start.elapsed().as_nanos()})
        })
        .collect()
}
fn measure_prepared(
    mut prepare: impl FnMut() -> Driver,
    mut work: impl FnMut(&mut Driver),
) -> Vec<Value> {
    work(&mut prepare());
    (0..5)
        .map(|_| {
            // Restore independent sessions before the timer; each is forked only once.
            let mut sessions: Vec<_> = (0..16).map(|_| prepare()).collect();
            let start = Instant::now();
            for session in &mut sessions {
                work(session);
            }
            let elapsed = start.elapsed().as_nanos();
            black_box(sessions);
            json!({"calls":16,"elapsed_ns":elapsed})
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let source = std::fs::read_to_string(&path)?;
        let document: Value = serde_json::from_str(&source)?;
        if document["format"] == "tag-rust-tictactoe-q" {
            let mut d = Driver::new();
            let loaded = d.file(json!({"type":"load_model"}), &source);
            let id = loaded["text"].clone();
            d.call(json!({"type":"create","game":"tictactoe","controllers":[{"kind":"q_learning","seed":"42","model_id":id},{"kind":"human"}]}));
            let samples = measure(|| {
                black_box(d.decision(false));
            });
            println!(
                "{}",
                json!({"game":"tictactoe","players":2,"operation":"local_decision_prepare_discard","kind":"q_learning","model_id":id,"samples":samples})
            );
            continue;
        }
        let game = document["contract"]["identity"]["game"].as_str().unwrap();
        let players = document["contract"]["identity"]["player_count"]
            .as_u64()
            .unwrap_or(2) as usize;
        let mut parameters = match game {
            "cantstop" => serde_json::to_value(tag_rust::games::cantstop::CantStopParameters {
                seed: 42,
                ..Default::default()
            })?,
            "loveletter" => {
                serde_json::to_value(tag_rust::games::loveletter::LoveLetterParameters {
                    seed: 42,
                    ..Default::default()
                })?
            }
            _ => Value::Null,
        };
        if matches!(game, "cantstop" | "loveletter") {
            parameters["seed"] = json!("42");
        }
        let mut model_driver = Driver::new();
        let loaded = model_driver.file(json!({"type":"load_model"}), &source);
        let model_id = loaded["text"].clone();
        let load_samples = measure(|| {
            let mut d = Driver::new();
            black_box(d.file(json!({"type":"load_model"}), &source));
        });
        println!(
            "{}",
            json!({"game":game,"players":players,"operation":"local_model_load","model_id":model_id,"model_file_bytes":source.len(),"samples":load_samples})
        );
        let kinds: Vec<_> = if ["tictactoe", "connect4"].contains(&game) {
            if document["contract"]["value"]["players"].is_null() {
                vec!["neural_osla"]
            } else {
                vec!["first_available", "random", "osla", "mcts", "neural_mcts"]
            }
        } else {
            vec![
                "first_available",
                "random",
                "sampled_osla",
                "mcts",
                "neural_mcts",
            ]
        };
        for kind in kinds {
            let mut d = Driver::new();
            d.file(json!({"type":"load_model"}), &source);
            let mut controller = json!({"kind":kind,"seed":"42"});
            if kind == "sampled_osla" {
                controller["repetitions_per_action"] = json!(1);
            }
            if ["neural_osla", "neural_mcts"].contains(&kind) {
                controller["model_id"] = model_id.clone();
            }
            if ["mcts", "neural_mcts"].contains(&kind) {
                let mut configuration =
                    json!({"schema_version":1,"rollout_length":0,"budgets":{"iterations":32}});
                if kind == "neural_mcts" {
                    configuration["tree_policy"] = json!("puct");
                    configuration["state_evaluator"] = json!({"name":"network_value"});
                    configuration["puct_preferences"] =
                        json!({"source":"probabilities","evaluator":{"name":"network_policy"}});
                }
                controller["configuration"] = configuration;
            }
            let mut controllers = vec![json!({"kind":"human"}); players];
            controllers[0] = controller;
            d.call(json!({"type":"create","game":game,"parameters":parameters,"controllers":controllers}));
            if game == "cantstop" {
                d.decision(true);
            }
            let samples = measure(|| {
                black_box(d.decision(false));
            });
            println!(
                "{}",
                json!({"game":game,"players":players,"operation":"local_decision_prepare_discard","kind":kind,"iterations":32,"model_id":model_id,"samples":samples})
            );
            if kind == "first_available" {
                for operation in ["view", "predict", "export_state", "export_session"] {
                    let samples = measure(|| {
                        black_box(match operation {
                        "view"=>d.call(json!({"type":"view","observer":0})),
                        "predict"=>d.file(json!({"type":"predict_model","model_id":model_id,"observer":0}),""),
                        _=>d.file(json!({"type":"export","scope":if operation=="export_state" {"state"} else {"session"},"cursor":if game=="cantstop" {1} else {0},"delay_ms":0}),""),
                    });
                    });
                    println!(
                        "{}",
                        json!({"game":game,"players":players,"operation":operation,"samples":samples})
                    );
                }
                let mut snapshots = Vec::new();
                let mut branch_source = String::new();
                for limit in [0, 8, 32] {
                    let mut view = d.call(json!({"type":"view","observer":0}));
                    let target = limit.max(if game == "cantstop" { 1 } else { 0 });
                    while view["view"]["cursor"].as_u64().unwrap() < target
                        && !view["view"]["next_player"].is_null()
                    {
                        let actor = view["view"]["next_player"].as_u64().unwrap();
                        if actor == 0 {
                            view = d.decision(true);
                        } else {
                            let moves = d.call(json!({"type":"view","observer":actor}));
                            view=d.call(json!({"type":"human_move","actor":actor,"action":moves["view"]["actions"][0]}));
                        }
                    }
                    let cursor = view["view"]["cursor"].clone();
                    let exported = d.file(
                        json!({"type":"export","scope":"session","cursor":cursor,"delay_ms":0}),
                        "",
                    );
                    let text = exported["text"].as_str().unwrap().to_string();
                    if limit == 0 {
                        branch_source = text.clone();
                    }
                    let samples = measure(|| {
                        let mut imported = Driver::new();
                        black_box(
                            imported.file(json!({"type":"import","controllers":null}), &text),
                        );
                    });
                    snapshots
                        .push(json!({"cursor":cursor,"file_bytes":text.len(),"samples":samples}));
                }
                println!(
                    "{}",
                    json!({"game":game,"players":players,"operation":"session_history_scaling","histories":snapshots})
                );
                for continuation in [false, true] {
                    let samples = measure_prepared(
                        || {
                            let mut imported = Driver::new();
                            imported
                                .file(json!({"type":"import","controllers":null}), &branch_source);
                            imported
                        },
                        |imported| {
                            black_box(imported.call(json!({"type":"fork","branch":0,"cursor":if game=="cantstop" {1} else {0},"replacements":vec![Value::Null;players]})));
                            if continuation {
                                black_box(imported.decision(true));
                            }
                        },
                    );
                    println!(
                        "{}",
                        json!({"game":game,"players":players,"operation":if continuation {"fork_and_continue"} else {"fork"},"preparation_excluded":true,"samples":samples})
                    );
                }
            }
        }
    }
    Ok(())
}
