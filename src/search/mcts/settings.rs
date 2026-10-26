//! Checked search settings, independent of file access and Python.
use super::{Error, Result};
use std::collections::BTreeMap;

macro_rules! choice {
    ($name:ident { $first:ident $(, $rest:ident)* }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
        #[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "mcts-json", serde(rename_all = "snake_case"))]
        pub enum $name { #[default] $first, $($rest),* }
    };
}
choice!(SearchMode { OpenLoop });
choice!(TreePolicy { Ucb1, Puct });
choice!(BiasMode {
    None,
    Additive,
    TagBlend
});
choice!(PreferenceSource {
    Uniform,
    NativeSoftmax,
    Probabilities
});
choice!(ExpansionOrder { Random, Heuristic });
choice!(RolloutPolicy {
    Random,
    Greedy,
    Softmax
});
choice!(FinalPolicy {
    MostVisits,
    HighestMean,
    VisitSampling
});
choice!(ValueCompletion {
    RequireFullVector,
    FillZeros,
    Negation
});

/// Portable custom evaluator parameters. Objects allow nested evaluator specifications.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(untagged))]
pub enum Parameter {
    Null,
    Bool(bool),
    Integer(i64),
    Number(f64),
    Text(String),
    Array(Vec<Parameter>),
    Object(BTreeMap<String, Parameter>),
}
pub type Parameters = BTreeMap<String, Parameter>;

/// A registry name and the parameters owned by that evaluator.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(deny_unknown_fields))]
pub struct EvaluatorSpec {
    pub name: String,
    #[cfg_attr(feature = "mcts-json", serde(default))]
    pub parameters: Parameters,
}
impl EvaluatorSpec {
    /// Construct an evaluator specification with no custom parameters.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: Parameters::new(),
        }
    }
}
impl Default for EvaluatorSpec {
    fn default() -> Self {
        Self::named("game_heuristic")
    }
}

/// Independent heuristic settings for tree bias.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct BiasSettings {
    pub mode: BiasMode,
    pub evaluator: Option<EvaluatorSpec>,
    pub bias_visits: Option<f64>,
}
/// Independent action preferences for PUCT.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct PreferenceSettings {
    pub source: PreferenceSource,
    pub evaluator: Option<EvaluatorSpec>,
    pub temperature: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct ExpansionSettings {
    pub order: ExpansionOrder,
    pub evaluator: Option<EvaluatorSpec>,
}
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct RolloutSettings {
    pub policy: RolloutPolicy,
    pub evaluator: Option<EvaluatorSpec>,
    pub temperature: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct FinalSettings {
    pub policy: FinalPolicy,
    pub temperature: Option<f64>,
}
/// Each optional limit is independent. At least one must remain enabled.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(deny_unknown_fields))]
pub struct Budgets {
    pub time_ms: Option<f64>,
    pub iterations: Option<u64>,
    pub forward_model_calls: Option<u64>,
}
impl Default for Budgets {
    fn default() -> Self {
        Self {
            time_ms: None,
            iterations: Some(1000),
            forward_model_calls: None,
        }
    }
}
impl Budgets {
    /// Check enabled limits without changing them. Zero means immediate exhaustion.
    /// `self` is read-only; all-disabled or invalid time values return an error.
    pub fn validate(&self) -> Result<()> {
        if self.time_ms.is_none() && self.iterations.is_none() && self.forward_model_calls.is_none()
        {
            return Err(Error::Invalid(
                "at least one decision budget must be enabled".into(),
            ));
        }
        if self.time_ms.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(Error::Invalid(
                "time_ms must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}
/// An omitted override retains the default; Disable and Replace are explicit choices.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Override<T> {
    #[default]
    Keep,
    Disable,
    Replace(T),
}
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BudgetOverrides {
    pub time_ms: Override<f64>,
    pub iterations: Override<u64>,
    pub forward_model_calls: Override<u64>,
}
impl BudgetOverrides {
    /// Merge this decision's overrides into an independent effective budget value.
    ///
    /// # Arguments
    /// `self` and `defaults` are read-only. Keep retains a default, Disable clears
    /// it, and Replace installs its value. The merged result is validated.
    pub fn apply(&self, defaults: &Budgets) -> Result<Budgets> {
        fn replace<T: Copy>(value: &Override<T>, default: Option<T>) -> Option<T> {
            match value {
                Override::Keep => default,
                Override::Disable => None,
                Override::Replace(v) => Some(*v),
            }
        }
        let result = Budgets {
            time_ms: replace(&self.time_ms, defaults.time_ms),
            iterations: replace(&self.iterations, defaults.iterations),
            forward_model_calls: replace(&self.forward_model_calls, defaults.forward_model_calls),
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct Limits {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_actions_per_decision: usize,
    pub max_request_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_edges: 1_000_000,
            max_actions_per_decision: 4096,
            max_request_bytes: 64 * 1024 * 1024,
        }
    }
}
/// All settings consumed by native MCTS. Null seeds are resolved once during setup.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mcts-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "mcts-json", serde(default, deny_unknown_fields))]
pub struct Settings {
    pub schema_version: u32,
    pub seed: Option<u64>,
    pub search_mode: SearchMode,
    pub tree_policy: TreePolicy,
    pub exploration_constant: f64,
    pub tree_depth: Option<u64>,
    pub rollout_length: Option<u64>,
    pub state_evaluator: EvaluatorSpec,
    pub value_completion: ValueCompletion,
    pub progressive_bias: BiasSettings,
    pub puct_preferences: PreferenceSettings,
    pub expansion: ExpansionSettings,
    pub rollout: RolloutSettings,
    pub final_selection: FinalSettings,
    pub budgets: Budgets,
    pub limits: Limits,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            schema_version: 1,
            seed: None,
            search_mode: SearchMode::default(),
            tree_policy: TreePolicy::default(),
            exploration_constant: std::f64::consts::SQRT_2,
            tree_depth: Some(100),
            rollout_length: Some(10),
            state_evaluator: EvaluatorSpec::default(),
            value_completion: ValueCompletion::default(),
            progressive_bias: BiasSettings::default(),
            puct_preferences: PreferenceSettings::default(),
            expansion: ExpansionSettings::default(),
            rollout: RolloutSettings::default(),
            final_selection: FinalSettings::default(),
            budgets: Budgets::default(),
            limits: Limits::default(),
        }
    }
}
impl Settings {
    /// Validate typed settings before constructing any evaluator or starting a decision.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.tree_depth == Some(0)
            || !self.exploration_constant.is_finite()
            || self.exploration_constant < 0.0
        {
            return Err(Error::Invalid(
                "unsupported schema, zero tree depth or invalid exploration constant".into(),
            ));
        }
        self.budgets.validate()?;
        if [
            self.limits.max_nodes,
            self.limits.max_edges,
            self.limits.max_actions_per_decision,
            self.limits.max_request_bytes,
        ]
        .contains(&0)
        {
            return Err(Error::Invalid("resource limits must be positive".into()));
        }
        let bias = &self.progressive_bias;
        require_evaluator(
            bias.mode != BiasMode::None,
            &bias.evaluator,
            "progressive bias",
        )?;
        if bias.mode != BiasMode::TagBlend && bias.bias_visits.is_some() {
            return irrelevant("bias_visits");
        }
        positive(bias.bias_visits, "bias_visits")?;
        let prior = &self.puct_preferences;
        if self.tree_policy != TreePolicy::Puct
            && (prior.source != PreferenceSource::Uniform
                || prior.evaluator.is_some()
                || prior.temperature.is_some())
        {
            return irrelevant("puct_preferences with UCB1");
        }
        require_evaluator(
            prior.source != PreferenceSource::Uniform,
            &prior.evaluator,
            "PUCT preferences",
        )?;
        if prior.source != PreferenceSource::NativeSoftmax && prior.temperature.is_some() {
            return irrelevant("PUCT temperature");
        }
        positive(prior.temperature, "PUCT temperature")?;
        require_evaluator(
            self.expansion.order == ExpansionOrder::Heuristic,
            &self.expansion.evaluator,
            "expansion",
        )?;
        require_evaluator(
            self.rollout.policy != RolloutPolicy::Random,
            &self.rollout.evaluator,
            "rollout",
        )?;
        if self.rollout.policy != RolloutPolicy::Softmax && self.rollout.temperature.is_some() {
            return irrelevant("rollout temperature");
        }
        positive(self.rollout.temperature, "rollout temperature")?;
        if self.final_selection.policy != FinalPolicy::VisitSampling
            && self.final_selection.temperature.is_some()
        {
            return irrelevant("final selection temperature");
        }
        positive(
            self.final_selection.temperature,
            "final selection temperature",
        )?;
        Ok(())
    }
}
fn irrelevant(name: &str) -> Result<()> {
    Err(Error::Invalid(format!(
        "{name} is irrelevant to the selected mode"
    )))
}
fn positive(value: Option<f64>, name: &str) -> Result<()> {
    if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
        return Err(Error::Invalid(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(())
}
fn require_evaluator(enabled: bool, spec: &Option<EvaluatorSpec>, name: &str) -> Result<()> {
    if enabled != spec.is_some() {
        return Err(Error::Invalid(format!(
            "{name} requires an evaluator exactly when enabled"
        )));
    }
    Ok(())
}
