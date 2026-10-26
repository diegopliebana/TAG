//! Release-mode OSLA timings and identical candidate positions for Python scoring.
use serde_json::{Value, json};
use std::{hint::black_box, time::Instant};
use tag_rust::agents::{
    Agent, BatchEvaluator, CandidateBatch, EvaluationLimits, EvaluatorIdentity,
    OpenLinesBatchEvaluator, OslaAgent, OslaPythonAgent, OslaPythonConfig,
};
use tag_rust::core::{Decision, PublicInformationModel, evaluation::Evaluator};
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{Float32Tensor, LearningGame, LearningLimits};

// Each sample lasts at least 500 ms to reduce scheduling noise on these small games.
fn measure(mut operation: impl FnMut()) -> Vec<f64> {
    operation();
    (0..7)
        .map(|_| {
            let start = Instant::now();
            let mut repeats = 0;
            while start.elapsed().as_secs_f64() < 0.5 {
                operation();
                repeats += 1;
            }
            start.elapsed().as_secs_f64() / repeats as f64
        })
        .collect()
}

// Both native agents use this identical complete-game loop.
fn play_with_agents<F, A>(
    model: &F,
    state: &mut F::State,
    agents: &mut [A; 2],
    record_moves: bool,
) -> Vec<usize>
where
    F: LearningGame,
    F::Error: std::fmt::Debug,
    A: Agent<F>,
    A::Error: std::fmt::Debug,
{
    let mut moves = Vec::new();
    while let Decision::Player(actor) = model.next_player(state) {
        let view = model.observe(state, actor).unwrap();
        let actions = model.offered_actions(&view);
        let selected = agents[actor.index()]
            .choose(model, actor, &view, &actions)
            .unwrap();
        if record_moves {
            moves.push(model.action_id(&view, &actions[selected]).unwrap());
        }
        model.next(state, actor, &actions[selected]).unwrap();
    }
    moves
}

fn benchmark<F, E>(
    name: &str,
    parameters_json: Value,
    model: F,
    parameters: F::Parameters,
    evaluator: E,
) -> Value
where
    F: LearningGame + PublicInformationModel,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Debug + std::fmt::Display,
    E: Evaluator<F::State> + Clone,
    E::Error: std::fmt::Debug,
{
    let play = |seed: u64, record_moves| {
        let mut state = model.init_state(&parameters, 2).unwrap();
        let mut agents = [
            OslaAgent::new(evaluator.clone(), seed),
            OslaAgent::new(evaluator.clone(), seed + 1),
        ];
        let moves = play_with_agents(&model, &mut state, &mut agents, record_moves);
        (moves, state)
    };
    let games = (0..16)
        .map(|i| play(100 + i * 2, true).0)
        .collect::<Vec<_>>();
    let full_seconds = measure(|| {
        for i in 0..16 {
            black_box(play(100 + i * 2, false));
        }
    })
    .into_iter()
    .map(|s| s / 16.0)
    .collect::<Vec<_>>();

    let play_batch = |seed: u64, rows, record_moves| {
        let spec = model
            .learning_spec(&parameters, LearningLimits::default())
            .unwrap();
        let backend = OpenLinesBatchEvaluator::new(&spec).unwrap();
        let config = OslaPythonConfig {
            evaluator: EvaluatorIdentity {
                name: "rust-open-lines".into(),
                version: 1,
                configuration: "{}".into(),
            },
            payloads: vec!["cells".into()],
            limits: EvaluationLimits {
                max_rows: rows,
                ..Default::default()
            },
        };
        let mut agents = [
            OslaPythonAgent::new(backend.clone(), config.clone(), seed).unwrap(),
            OslaPythonAgent::new(backend, config, seed + 1).unwrap(),
        ];
        let mut state = model.init_state(&parameters, 2).unwrap();
        let moves = play_with_agents(&model, &mut state, &mut agents, record_moves);
        (moves, state)
    };
    let mut rust_batch = serde_json::Map::new();
    for rows in [1, 3, 1024] {
        for (i, expected) in games.iter().enumerate() {
            assert_eq!(&play_batch(100 + i as u64 * 2, rows, true).0, expected);
        }
        let seconds = measure(|| {
            for i in 0..16 {
                black_box(play_batch(100 + i * 2, rows, false));
            }
        })
        .into_iter()
        .map(|s| s / 16.0)
        .collect::<Vec<_>>();
        rust_batch.insert(rows.to_string(), json!({"game_seconds":seconds}));
    }

    // Build the scoring corpus outside timed regions. Include every candidate
    // examined during these games, including terminal successors.
    let mut states = Vec::new();
    let mut batches = Vec::new();
    let mut native_batches = Vec::new();
    let spec = model
        .learning_spec(&parameters, LearningLimits::default())
        .unwrap();
    let mut batch_scoring = OpenLinesBatchEvaluator::new(&spec).unwrap();
    let mut scoring = evaluator;
    for moves in &games {
        let mut state = model.init_state(&parameters, 2).unwrap();
        for id in moves {
            let Decision::Player(actor) = model.next_player(&state) else {
                unreachable!()
            };
            let view = model.observe(&state, actor).unwrap();
            let mut cells = Vec::new();
            let mut results = Vec::new();
            let mut values = Vec::new();
            let mut ids = Vec::new();
            for action in model.offered_actions(&view) {
                let mut candidate = state.clone();
                model.next(&mut candidate, actor, &action).unwrap();
                let encoded = model
                    .encode(
                        &model.observe(&candidate, actor).unwrap(),
                        &["cells"],
                        LearningLimits::default(),
                    )
                    .unwrap();
                cells.push(encoded.tensors[0].values.clone());
                results.push(encoded.results.iter().map(|r| *r as i8).collect::<Vec<_>>());
                values.push(scoring.evaluate(&candidate, actor).unwrap());
                states.push((candidate, actor));
                ids.push(model.action_id(&view, &action).unwrap());
            }
            let native_batch = CandidateBatch {
                identity: spec.identity.clone(),
                observer: actor,
                offset: 0,
                total: ids.len(),
                action_ids: ids,
                progress: vec![],
                next_players: vec![],
                results: results.iter().flatten().copied().collect(),
                payloads: vec![Float32Tensor {
                    name: "cells",
                    shape: vec![cells.len(), cells[0].len()],
                    values: cells.iter().flatten().copied().collect(),
                }],
            };
            assert_eq!(batch_scoring.evaluate(&native_batch).unwrap(), values);
            native_batches.push(native_batch);
            batches.push(
                json!({"actor":actor.index(),"cells":cells,"results":results,"values":values}),
            );
            let action = model.action_from_id(&view, *id).unwrap();
            model.next(&mut state, actor, &action).unwrap();
        }
    }
    let scoring_seconds = measure(|| {
        for (state, actor) in &states {
            black_box(
                scoring
                    .evaluate(black_box(state), black_box(*actor))
                    .unwrap(),
            );
        }
    })
    .into_iter()
    .map(|s| s / states.len() as f64)
    .collect::<Vec<_>>();
    let batch_scoring_seconds = measure(|| {
        for batch in &native_batches {
            black_box(batch_scoring.evaluate(black_box(batch)).unwrap());
        }
    })
    .into_iter()
    .map(|s| s / states.len() as f64)
    .collect::<Vec<_>>();
    json!({"game":name,"parameters":parameters_json,"games":games,"batches":batches,
        "line_indices":spec.heuristic.as_ref().unwrap().indices,"window_length":spec.heuristic.as_ref().unwrap().window_length,
        "candidates":states.len(),"native_game_seconds":full_seconds,"native_score_seconds":scoring_seconds,
        "rust_batch":rust_batch,"rust_batch_score_seconds":batch_scoring_seconds})
}

fn main() {
    let mut records = Vec::new();
    for size in [3, 5] {
        records.push(benchmark(
            "tictactoe",
            json!({"grid_size":size}),
            ttt::TicTacToeForwardModel,
            ttt::TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            ttt::TicTacToeEvaluator,
        ));
    }
    for (width, height, winning_length) in [(7, 6, 4), (4, 3, 3)] {
        records.push(benchmark(
            "connect4",
            json!({"width":width,"height":height,"winning_length":winning_length}),
            c4::Connect4ForwardModel,
            c4::Connect4Parameters {
                width,
                height,
                winning_length,
                ..Default::default()
            },
            c4::Connect4Evaluator,
        ));
    }
    println!("{}", serde_json::to_string(&records).unwrap());
}
