//! Compiled Python entry point. Rule execution remains in the Rust engine.
mod arrays;
mod engine;
mod episode;
mod episode_arrays;
mod episode_batch;
mod episode_binding;
mod evaluations;
mod execution;
mod game;
mod group_arrays;
mod group_limits;
mod grouping;
mod mcts;
mod opponents;
mod scoring;

use pyo3::prelude::*;

/// Initialize NumPy before any game calls so missing dependencies produce an import error.
#[pymodule]
fn _engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.py().import("numpy")?;
    module.add_class::<game::Game>()?;
    module.add_class::<mcts::MctsSearch>()?;
    module.add_class::<opponents::NativeOpponents>()?;
    module.add_class::<episode_binding::Episode>()?;
    module.add_class::<episode_batch::EpisodeBatch>()?;
    module.add_class::<episode_batch::BatchResetCandidate>()?;
    module.add_class::<episode_binding::ResetCandidate>()?;
    module.add_class::<grouping::GameBatch>()?;
    module.add_class::<grouping::EvaluationGroupHandle>()?;
    module.add_class::<game::Position>()?;
    module.add_class::<evaluations::EvaluationHandle>()?;
    module.add_class::<scoring::OpenLinesEvaluator>()?;
    module.add(
        "EvaluationError",
        module.py().get_type::<game::EvaluationError>(),
    )?;
    module.add(
        "StalePositionError",
        module.py().get_type::<game::StalePositionError>(),
    )?;
    module.add(
        "ClosedGameError",
        module.py().get_type::<game::ClosedGameError>(),
    )?;
    module.add_function(wrap_pyfunction!(game::create, module)?)?;
    module.add_function(wrap_pyfunction!(game::create_json, module)?)?;
    module.add_function(wrap_pyfunction!(game::registered_games, module)?)?;
    module.add_function(wrap_pyfunction!(mcts::parse_mcts_settings, module)?)?;
    module.add_function(wrap_pyfunction!(mcts::automatic_search_seed, module)?)?;
    module.add_function(wrap_pyfunction!(evaluations::create_controller, module)?)?;
    Ok(())
}
