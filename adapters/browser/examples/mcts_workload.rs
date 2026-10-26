//! Measure local inference requests and complete MCTS preparations in release mode.
//! Each supplied model determines its game. No move is committed during timing.
use serde_json::{Value, json};
use std::time::Instant;
use tag_browser::BrowserEngine;

fn request(engine: &mut BrowserEngine, command: Value) -> Value {
    let reply: Value = serde_json::from_str(
        &engine.handle(
            &json!({"request_id":"1",
        "generation":"1","position_version":"0","command":command})
            .to_string(),
        ),
    )
    .unwrap();
    assert!(reply["error"].is_null(), "{reply}");
    reply
}
fn file(engine: &mut BrowserEngine, generation: &str, command: Value, text: &str) -> Value {
    let reply: Value = serde_json::from_str(
        &engine.handle_file(
            &json!({"request_id":"1",
        "generation":generation,"command":command})
            .to_string(),
            text,
        ),
    )
    .unwrap();
    assert!(reply["error"].is_null(), "{reply}");
    reply
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path)?;
        let model: Value = serde_json::from_str(&text)?;
        let mut engine = BrowserEngine::new();
        let loaded = file(&mut engine, "0", json!({"type":"load_model"}), &text);
        for neural in [false, true] {
            let mut configuration =
                json!({"schema_version":1,"rollout_length":0,"budgets":{"iterations":256}});
            let mut seat = json!({"kind":"mcts","seed":"42"});
            if neural {
                seat["kind"] = json!("neural_mcts");
                seat["model_id"] = loaded["text"].clone();
                configuration["tree_policy"] = json!("puct");
                configuration["state_evaluator"] = json!({"name":"network_value"});
                configuration["puct_preferences"] =
                    json!({"source":"probabilities","evaluator":{"name":"network_policy"}});
            }
            seat["configuration"] = configuration;
            // Create a fresh lifecycle for each comparison, keeping identical initial positions.
            let mut trial = BrowserEngine::new();
            file(&mut trial, "0", json!({"type":"load_model"}), &text);
            let created: Value = serde_json::from_str(&trial.handle(&json!({"request_id":"1","generation":"0",
                "position_version":null,"command":{"type":"create","game":model["contract"]["identity"]["game"],
                    "controllers":[seat,{"kind":"human"}]}}).to_string()))?;
            assert!(created["error"].is_null(), "{created}");
            let start = Instant::now();
            for _ in 0..100 {
                file(
                    &mut trial,
                    "1",
                    json!({"type":"predict_model","model_id":loaded["text"],"observer":0}),
                    "",
                );
            }
            let inference_ms = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let mut choices = Vec::new();
            let mut chunks = 0;
            for _ in 0..20 {
                let mut reply = request(&mut trial, json!({"type":"prepare_ai"}));
                while reply["proposal"]["pending"] == true {
                    chunks += 1;
                    reply = request(
                        &mut trial,
                        json!({"type":"poll_ai","proposal":reply["proposal"]["id"]}),
                    );
                }
                choices.push(reply["proposal"]["action"].clone());
                request(
                    &mut trial,
                    json!({"type":"discard_ai","proposal":reply["proposal"]["id"]}),
                );
            }
            samples.push(
                json!({"game":model["contract"]["identity"]["game"],"model_id":loaded["text"],
                "source":path,"neural":neural,"iterations":256,"decisions":20,
                "decision_total_ms":start.elapsed().as_secs_f64()*1000.,"poll_chunks":chunks,
                "prediction_requests":100,"prediction_total_ms":inference_ms,"actions":choices}),
            );
        }
    }
    if samples.is_empty() {
        return Err("provide one or more self-play model JSON files".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"native browser adapter; prediction includes validation/JSON; decision includes polling/JSON; no worker transport",
        "samples":samples})
        )?
    );
    Ok(())
}
