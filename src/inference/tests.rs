//! PyTorch-produced fixtures check values, contexts and pre-mask logits.
use super::*;
use crate::core::{Decision, ForwardModel, PlayerId};
use crate::games::{connect4::Connect4ForwardModel, tictactoe::TicTacToeForwardModel};
use crate::learning::{LearningGame, LearningLimits};
fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("web/tests/fixtures/learned")
            .join(name),
    )
    .unwrap()
}
#[test]
fn borrowed_model_inspection_preserves_independent_owned_exports() {
    for name in ["q-table.json", "tictactoe-zero.json"] {
        let model = LearnedModel::parse(&fixture(name)).unwrap();
        let expected = model.document();
        let original_id = model.id().to_owned();
        let mut exported = model.document();
        exported["provenance"] = json!({"changed_only_in_export":true});
        assert_ne!(&exported, model.as_document());
        assert_eq!(model.as_document(), &expected);
        assert_eq!(model.id(), original_id);
        assert_eq!(model.as_document().to_string(), expected.to_string());
    }
}
#[test]
fn selfplay_scalar_and_reordered_player_vectors_match_pytorch() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web/tests/fixtures/mcts");
    let cases: Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("predictions.json")).unwrap())
            .unwrap();
    fn run<F: LearningGame>(game: F, case: &Value, model: LearnedModel)
    where
        F::Parameters: Default,
        F::Error: std::fmt::Debug,
    {
        let network = model.network().unwrap();
        model
            .validate_spec(
                &game
                    .learning_spec(&Default::default(), LearningLimits::default())
                    .unwrap(),
            )
            .unwrap();
        assert!(network.validate_osla().is_err());
        super::LocalMctsModel::new(std::sync::Arc::new(model.clone()), Default::default()).unwrap();
        for row in case["rows"].as_array().unwrap() {
            let mut state = game.init_state(&Default::default(), 2).unwrap();
            for id in row["prefix"].as_array().unwrap() {
                let Decision::Player(actor) = game.next_player(&state) else {
                    panic!()
                };
                let obs = game.observe(&state, actor).unwrap();
                let action = game
                    .action_from_id(&obs, id.as_u64().unwrap() as usize)
                    .unwrap();
                game.next(&mut state, actor, &action).unwrap();
            }
            let observation = game
                .observe(
                    &state,
                    PlayerId::new(row["observer"].as_u64().unwrap() as usize),
                )
                .unwrap();
            let encoded = game
                .encode(&observation, &network.payloads(), LearningLimits::default())
                .unwrap();
            let (logits, values) = network.predict_values(&encoded).unwrap();
            for (actual, expected) in logits.iter().zip(row["logits"].as_array().unwrap()) {
                close(*actual, expected.as_f64().unwrap());
            }
            assert_eq!(values.len(), row["values"].as_array().unwrap().len());
            for (actual, expected) in values.iter().zip(row["values"].as_array().unwrap()) {
                close(*actual, expected.as_f64().unwrap());
            }
            if network.value_players().is_some() {
                assert!(network.predict(&encoded).is_err());
            }
        }
        for context in [
            json!({"observers":[0],"turn_context":"any_turn"}),
            Value::Null,
        ] {
            let mut document = model.document();
            document["provenance"]["model_context"] = context;
            let altered = LearnedModel::parse(&document.to_string()).unwrap();
            assert!(
                super::LocalMctsModel::new(std::sync::Arc::new(altered), Default::default())
                    .is_err()
            );
        }
    }
    for case in cases.as_array().unwrap() {
        let model = LearnedModel::parse(
            &std::fs::read_to_string(directory.join(case["model_file"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        match case["game"].as_str().unwrap() {
            "tictactoe" => run(TicTacToeForwardModel, case, model),
            "connect4" => run(Connect4ForwardModel, case, model),
            _ => panic!(),
        }
    }
}

#[test]
fn local_search_backs_up_each_declared_player_value_without_negation() {
    use crate::core::PublicInformationModel;
    use crate::search::mcts::{MctsAgent, NativeEvaluators, Settings};
    fn verify<F>(game: F, text: &str)
    where
        F: LearningGame + PublicInformationModel + NativeEvaluators + 'static,
        F::Parameters: Default,
        F::State: Clone,
        F::Action: Clone + PartialEq,
        F::Error: std::fmt::Debug + std::fmt::Display,
    {
        let loaded = std::sync::Arc::new(LearnedModel::parse(text).unwrap());
        let local = super::LocalMctsModel::new(loaded.clone(), Default::default()).unwrap();
        let mut registry = F::native_evaluators();
        local.register(&mut registry, PlayerId::new(0)).unwrap();
        let mut settings = Settings {
            seed: Some(42),
            rollout_length: Some(0),
            state_evaluator: crate::search::mcts::EvaluatorSpec::named("network_value"),
            ..Default::default()
        };
        settings.budgets.iterations = Some(1);
        let mut agent = MctsAgent::new(settings.clone(), &registry).unwrap();
        let mut state = game.init_state(&Default::default(), 2).unwrap();
        let observation = game.observe(&state, PlayerId::new(0)).unwrap();
        let actions = game.offered_actions(&observation);
        let report = agent
            .search(&game, PlayerId::new(0), &observation, &actions)
            .unwrap();
        let visited = report.root_actions.iter().find(|r| r.visits == 1).unwrap();
        game.next(&mut state, PlayerId::new(0), &visited.action)
            .unwrap();
        let net = loaded.network().unwrap();
        if let Some(order) = net.value_players() {
            let encoded = game
                .encode(
                    &game.observe(&state, PlayerId::new(1)).unwrap(),
                    &net.payloads(),
                    LearningLimits::default(),
                )
                .unwrap();
            let (_, values) = net.predict_values(&encoded).unwrap();
            for (&player, value) in order.iter().zip(values) {
                assert_eq!(visited.mean_values[player], value as f64);
            }
        } else {
            for player in 0..2 {
                let encoded = game
                    .encode(
                        &game.observe(&state, PlayerId::new(player)).unwrap(),
                        &net.payloads(),
                        LearningLimits::default(),
                    )
                    .unwrap();
                assert_eq!(
                    visited.mean_values[player],
                    net.predict(&encoded).unwrap().1 as f64
                );
            }
            let binding = super::ModelBindings {
                value_observers: Some(super::ValuePerspective::DecidingPlayer),
                ..Default::default()
            };
            let single = super::LocalMctsModel::new(loaded, binding).unwrap();
            assert!(single.validate_search(&settings).is_err());
            settings.value_completion = crate::search::mcts::ValueCompletion::FillZeros;
            single.validate_search(&settings).unwrap();
            settings.value_completion = crate::search::mcts::ValueCompletion::Negation;
            single.validate_search(&settings).unwrap();
        }
    }
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web/tests/fixtures/mcts");
    for head in ["scalar", "vector"] {
        verify(
            TicTacToeForwardModel,
            &std::fs::read_to_string(directory.join(format!("tictactoe-{head}.json"))).unwrap(),
        );
        verify(
            Connect4ForwardModel,
            &std::fs::read_to_string(directory.join(format!("connect4-{head}.json"))).unwrap(),
        );
    }
}
fn close(a: f32, b: f64) {
    assert!(
        (a as f64 - b).abs() <= 2e-6 + 2e-5 * b.abs(),
        "prediction {a} differs from PyTorch {b}"
    );
}
fn check<F: LearningGame>(game: F, parameters: F::Parameters, data: &Value)
where
    F::Error: std::fmt::Debug,
{
    let model = LearnedModel::parse(&fixture(data["model_file"].as_str().unwrap())).unwrap();
    model
        .validate_spec(
            &game
                .learning_spec(&parameters, LearningLimits::default())
                .unwrap(),
        )
        .unwrap();
    let network = model.network().unwrap();
    for row in data["rows"].as_array().unwrap() {
        let mut state = game.init_state(&parameters, 2).unwrap();
        for id in row["prefix"].as_array().unwrap() {
            let Decision::Player(actor) = game.next_player(&state) else {
                panic!("prefix ended")
            };
            let obs = game.observe(&state, actor).unwrap();
            let action = game
                .action_from_id(&obs, id.as_u64().unwrap() as usize)
                .unwrap();
            game.next(&mut state, actor, &action).unwrap();
        }
        let obs = game
            .observe(
                &state,
                PlayerId::new(row["observer"].as_u64().unwrap() as usize),
            )
            .unwrap();
        let encoded = game
            .encode(&obs, &network.payloads(), LearningLimits::default())
            .unwrap();
        for (a, b) in network
            .inputs(&encoded)
            .unwrap()
            .iter()
            .zip(row["inputs"].as_array().unwrap())
        {
            assert_eq!(*a as f64, b.as_f64().unwrap());
        }
        let (logits, value) = network.predict(&encoded).unwrap();
        assert_eq!(logits.len(), row["logits"].as_array().unwrap().len());
        for (a, b) in logits.iter().zip(row["logits"].as_array().unwrap()) {
            close(*a, b.as_f64().unwrap());
        }
        close(value, row["value"].as_f64().unwrap());
    }
}
#[test]
fn trained_zero_and_combined_inputs_match_pytorch_in_both_seats() {
    let data: Value = serde_json::from_str(&fixture("fixtures.json")).unwrap();
    for model in data["models"].as_array().unwrap() {
        match model["game"].as_str().unwrap() {
            "tictactoe" => check(TicTacToeForwardModel, Default::default(), model),
            "connect4" => check(Connect4ForwardModel, Default::default(), model),
            _ => panic!(),
        }
    }
}
#[test]
fn q_scores_preserve_float64_rows_and_missing_rows_are_zero() {
    let document: Value = serde_json::from_str(&fixture("q-table.json")).unwrap();
    let model = LearnedModel::parse(&document.to_string()).unwrap();
    let game = TicTacToeForwardModel;
    model
        .validate_spec(
            &game
                .learning_spec(&Default::default(), LearningLimits::default())
                .unwrap(),
        )
        .unwrap();
    let state = game.init_state(&Default::default(), 2).unwrap();
    let mut encoded = game
        .encode(
            &game.observe(&state, PlayerId::new(0)).unwrap(),
            &["cells"],
            LearningLimits::default(),
        )
        .unwrap();
    for row in document["rows"].as_array().unwrap() {
        encoded.tensors[0].values = row["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as f32)
            .collect();
        let expected: Vec<_> = row["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(model.q_values(&encoded).unwrap(), expected);
    }
    // Synthetic unrecorded key checks defaults only, not game legality.
    encoded.tensors[0].values = vec![-1.; 9];
    assert_eq!(model.q_values(&encoded).unwrap(), vec![0.; 9]);
}
#[test]
fn invalid_documents_and_configuration_fail_before_use() {
    let original: Value = serde_json::from_str(&fixture("tictactoe-trained.json")).unwrap();
    for change in 0..5 {
        let mut doc = original.clone();
        match change {
            0 => doc["version"] = json!(999),
            1 => doc["model_identity"] = json!("bad"),
            2 => doc["weights"]["value.bias"]["shape"] = json!([100]),
            3 => doc["hidden_sizes"] = json!([999999]),
            _ => doc["contract"]["value"]["players"] = json!([0, 1]),
        }
        assert!(LearnedModel::parse(&doc.to_string()).is_err());
    }
    let model = LearnedModel::parse(&original.to_string()).unwrap();
    assert!(
        model
            .validate_spec(
                &Connect4ForwardModel
                    .learning_spec(&Default::default(), LearningLimits::default())
                    .unwrap()
            )
            .is_err()
    );
    assert!(LearnedModel::parse(&" ".repeat(MAX_MODEL_BYTES + 1)).is_err());
    let mut table: Value = serde_json::from_str(&fixture("q-table.json")).unwrap();
    let duplicate = table["rows"][0].clone();
    table["rows"].as_array_mut().unwrap().push(duplicate);
    assert!(LearnedModel::parse(&table.to_string()).is_err());
}
