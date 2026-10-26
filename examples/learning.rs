//! Inspect portable encodings without Python, browser or inference dependencies.
use tag_rust::core::PlayerId;
use tag_rust::games::{connect4 as c4, tictactoe as ttt};
use tag_rust::learning::{LearningGame, LearningLimits};
fn inspect<F: LearningGame>(model: F, parameters: F::Parameters)
where
    F::Error: std::fmt::Debug,
{
    let spec = model
        .learning_spec(&parameters, LearningLimits::default())
        .unwrap();
    let state = model.init_state(&parameters, 2).unwrap();
    let observation = model.observe(&state, PlayerId::new(0)).unwrap();
    // The encoder borrows observation; returned arrays own independent data.
    let encoded = model
        .encode(&observation, &["cells", "board"], LearningLimits::default())
        .unwrap();
    println!(
        "{}: {} policy IDs, {} heuristic windows",
        spec.identity.game,
        spec.actions.size,
        spec.heuristic.as_ref().unwrap().indices.len()
            / spec.heuristic.as_ref().unwrap().window_length
    );
    for tensor in encoded.tensors {
        println!("  {}: {:?}", tensor.name, tensor.shape);
    }
}
fn main() {
    // Fixed valid configuration; unwrap stops this example on unexpected failure.
    inspect(
        ttt::TicTacToeForwardModel,
        ttt::TicTacToeParameters::default(),
    );
    inspect(c4::Connect4ForwardModel, c4::Connect4Parameters::default());
}
