//! Record Rust observations for Python's whole-game comparison tests.
#[path = "../src/engine.rs"]
mod engine;
#[path = "../src/execution.rs"]
mod execution;
use engine::Engine;
use serde_json::{Value, json};
use tag_rust::learning::{EncodedObservation, LearningLimits};
use tag_rust::{
    agents::{
        CandidateBatch, EvaluationLimits, EvaluatorIdentity, ExternalEvaluation, OslaPythonAgent,
        OslaPythonConfig,
    },
    core::checkpoint::Checkpoint,
    session::EvaluationStatus,
};

fn observation(value: EncodedObservation) -> Value {
    json!({
        "observer": value.observer.index(),
        "next_player": value.next_player.map(|p| p.index()),
        "progress": value.progress,
        "results": value.results.into_iter().map(|r| r as i8).collect::<Vec<_>>(),
        "action_mask": value.action_mask,
        "offered_action_ids": value.offered_action_ids,
        "payloads": value.tensors.into_iter().map(|t| (t.name.to_owned(), json!({"shape": t.shape, "values": t.values}))).collect::<serde_json::Map<_, _>>()
    })
}

fn main() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("../tests/cases.json")).unwrap();
    let mut results = Vec::new();
    for case in cases {
        let parameters = serde_json::from_value(case["parameters"].clone()).unwrap();
        let mut game = Engine::create(
            case["game"].as_str().unwrap(),
            parameters,
            2,
            LearningLimits::default(),
        )
        .unwrap();
        assert!(game.seed().is_none());
        assert_eq!(game.payload_names(), ["cells", "board"]);
        let spec = game.spec().unwrap();
        let mut positions = Vec::new();
        let moves: Vec<usize> = serde_json::from_value(case["moves"].clone()).unwrap();
        for step in 0..=moves.len() {
            positions.push(
                (0..2)
                    .map(|seat| observation(game.observe(seat, &["cells", "board"]).unwrap()))
                    .collect::<Vec<_>>(),
            );
            if let Some(&id) = moves.get(step) {
                let actor = game.next_player().unwrap();
                game.apply(id, Some(actor)).unwrap();
            }
        }
        // Exercise copying and reset here too; comparison output is the recorded line.
        let mut copy = game.snapshot().unwrap();
        copy.reset().unwrap();
        assert_eq!(copy.observe(0, &[]).unwrap().progress, [0; 4]);
        let mock = mock_evaluations(&case);
        results.push(
            json!({"case": case, "positions": positions, "mock_evaluations": mock,
            "owned_runs": [owned_run(&case,true),owned_run(&case,false)],
            "action_count": spec.actions.size, "window_length": spec.heuristic.as_ref().unwrap().window_length,
            "line_indices": spec.heuristic.as_ref().unwrap().indices}),
        );
    }
    println!("{}", serde_json::to_string(&results).unwrap());
}

fn owned_run(case: &Value, native: bool) -> Value {
    let mut game = Engine::create(
        case["game"].as_str().unwrap(),
        serde_json::from_value(case["parameters"].clone()).unwrap(),
        2,
        LearningLimits::default(),
    )
    .unwrap();
    for seat in 0..2 {
        let config = OslaPythonConfig {
            evaluator: EvaluatorIdentity {
                name: "rust-open-lines".into(),
                version: 1,
                configuration: "{}".into(),
            },
            payloads: vec!["cells".into()],
            limits: EvaluationLimits {
                max_rows: 3,
                ..Default::default()
            },
        };
        let agent = OslaPythonAgent::new(ExternalEvaluation, config, 7 + seat as u64).unwrap();
        game.set_controller_mode(
            seat,
            Some(agent.snapshot()),
            if native {
                "rust_open_lines"
            } else {
                "external"
            },
        )
        .unwrap();
    }
    let mut progress = game.advance(2);
    let mut moves = 0;
    loop {
        moves += progress.moves;
        progress = match progress.stop.unwrap() {
            execution::Stop::Finished => break,
            execution::Stop::MoveLimit(actor) => {
                assert_eq!(game.next_player(), Some(actor));
                game.advance(2)
            }
            execution::Stop::Evaluation(request) => {
                let values = vec![0.0; request.batch.action_ids.len()];
                game.resume(
                    &request.key,
                    &request.evaluator,
                    &request.batch.action_ids,
                    &values,
                    2,
                )
            }
            execution::Stop::Human(actor) => {
                panic!("both seats have AI controllers; unexpected human {actor}")
            }
        };
    }
    json!({"native":native,"moves":moves,"mode":game.execution_mode(0).unwrap(),
        "final":observation(game.observe(0,&["cells"]).unwrap())})
}

fn batch(value: &CandidateBatch) -> Value {
    json!({"actor": value.observer.index(), "offset": value.offset, "total": value.total,
        "action_ids": value.action_ids, "progress": value.progress, "results": value.results,
        "next_players": value.next_players,
        "payloads": value.payloads.iter().map(|t| (t.name.to_owned(), json!({"shape":t.shape, "values": t.values}))).collect::<serde_json::Map<_,_>>()})
}

/// A deliberately simple test evaluator: odd action IDs score one, even IDs zero.
fn mock_evaluations(case: &Value) -> Vec<Value> {
    let mut game = Engine::create(
        case["game"].as_str().unwrap(),
        serde_json::from_value(case["parameters"].clone()).unwrap(),
        2,
        LearningLimits::default(),
    )
    .unwrap();
    for seat in 0..2 {
        let config = OslaPythonConfig {
            evaluator: EvaluatorIdentity {
                name: "stage4_mock".into(),
                version: 1,
                configuration: "{\"pattern\":2}".into(),
            },
            payloads: vec!["cells".into(), "board".into()],
            limits: EvaluationLimits {
                max_rows: 2,
                ..Default::default()
            },
        };
        let agent = OslaPythonAgent::new(ExternalEvaluation, config, 7 + seat as u64).unwrap();
        game.set_controller(seat, Some(agent.snapshot())).unwrap();
    }
    let probe = game.prepare().unwrap();
    game.cancel(&probe.key).unwrap();
    let mut records = Vec::new();
    while game.next_player().is_some() {
        let mut request = game.prepare().unwrap();
        let mut chunks = Vec::new();
        let proposal = loop {
            chunks.push(batch(&request.batch));
            let values = request
                .batch
                .action_ids
                .iter()
                .map(|&id| (id % 2) as f64)
                .collect::<Vec<_>>();
            match game
                .finish(
                    &request.key,
                    &request.evaluator,
                    &request.batch.action_ids,
                    &values,
                )
                .unwrap()
            {
                EvaluationStatus::Request(next) => request = *next,
                EvaluationStatus::Ready(proposal) => break proposal,
            }
        };
        game.commit(&proposal.key).unwrap();
        let rng = (0..2)
            .map(|p| game.checkpoint(p).unwrap().unwrap().rng.word_position)
            .collect::<Vec<_>>();
        records.push(
            json!({"chunks": chunks, "selected": proposal.action_id, "rng_word_positions":rng,
            "after": observation(game.observe(0, &["cells", "board"]).unwrap())}),
        );
    }
    records
}
