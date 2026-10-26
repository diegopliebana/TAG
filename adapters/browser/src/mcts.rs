//! Browser clocks and independently restored native MCTS controllers.
use serde_json::Value;
use tag_rust::{
    core::{PublicInformationModel, random::AgentRng},
    inference::ModelError,
    search::mcts::{Clock, MctsAgent, NativeEvaluators, Settings, SystemClock},
};

#[derive(Default)]
pub(crate) struct BrowserClock(SystemClock);
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}
impl Clock for BrowserClock {
    fn now_ms(&self) -> f64 {
        #[cfg(target_arch = "wasm32")]
        {
            performance_now()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.0.now_ms()
        }
    }
}

pub(crate) fn agent<F>(
    configuration: &Value,
    rng: &AgentRng,
    model: Option<&tag_rust::inference::LocalMctsModel>,
    root: tag_rust::core::PlayerId,
) -> Result<MctsAgent<F>, ModelError>
where
    F: PublicInformationModel + NativeEvaluators + tag_rust::learning::LearningGame + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    let settings =
        Settings::from_json(&configuration.to_string()).map_err(|e| ModelError(e.to_string()))?;
    let registry = registry::<F>(model, root)?;
    tag_rust::inference::build_local_mcts(&settings, rng, model, &registry)
}

fn registry<F>(
    model: Option<&tag_rust::inference::LocalMctsModel>,
    root: tag_rust::core::PlayerId,
) -> Result<tag_rust::search::mcts::EvaluatorRegistry<F>, ModelError>
where
    F: NativeEvaluators + tag_rust::learning::LearningGame + 'static,
    F::State: Clone,
    F::Error: std::fmt::Display,
{
    let mut registry = F::native_evaluators();
    if let Some(model) = model {
        model
            .register(&mut registry, root)
            .map_err(|e| ModelError(e.to_string()))?;
    }
    Ok(registry)
}

pub(crate) fn prepare<'a, F>(
    settings: &Settings,
    rng: &'a mut AgentRng,
    model: Option<&tag_rust::inference::LocalMctsModel>,
    root: tag_rust::core::PlayerId,
) -> Result<tag_rust::inference::LocalMctsSearch<'a, F>, ModelError>
where
    F: PublicInformationModel + NativeEvaluators + tag_rust::learning::LearningGame + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    let registry = registry::<F>(model, root)?;
    tag_rust::inference::LocalMctsSearch::new(settings, rng, model, &registry)
}

/// Preserve the existing canonical JSON in snapshots and messages. Stored
/// settings have already passed validation when the controller was constructed.
pub(crate) fn configuration_value(settings: &Settings) -> Value {
    serde_json::from_str(&settings.to_json().expect("validated browser MCTS settings"))
        .expect("settings serializer produces valid JSON")
}
