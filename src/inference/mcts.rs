//! Local policy/value evaluation for MCTS, using only permitted observations.
use super::{LearnedModel, ModelError, invalid};
use crate::core::{PublicInformationModel, checkpoint::Checkpoint, random::AgentRng};
use crate::search::mcts::{Clock, MctsAgent, SearchReport, Settings};
use crate::{
    core::{Decision, PlayerId},
    learning::{LearningGame, LearningLimits},
    search::mcts::{
        ActionEvaluator, Error, EvaluatorRegistry, Parameters, SimulationContext, StateEvaluator,
        StateValues,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Construct one local search from typed settings and exact caller-owned RNG
/// progress. The registry already contains the selected native/model factories.
/// Evaluators are newly constructed: this restores RNG progress, not arbitrary
/// mutable evaluator memory. Retain an `MctsAgent` and its checkpoint when that
/// evaluator memory must persist across decisions.
pub fn build_local_mcts<F>(
    settings: &Settings,
    rng: &AgentRng,
    model: Option<&LocalMctsModel>,
    registry: &EvaluatorRegistry<F>,
) -> Result<MctsAgent<F>, ModelError>
where
    F: PublicInformationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    let mut settings = settings.clone();
    settings.seed = Some(rng.initial_seed());
    if let Some(model) = model {
        model.validate_search(&settings)?;
    }
    let mut agent = MctsAgent::new(settings, registry).map_err(|e| ModelError(e.to_string()))?;
    *crate::agents::Agent::<F>::rng_mut(&mut agent) =
        AgentRng::from_snapshot(&rng.snapshot()).map_err(|e| ModelError(e.to_string()))?;
    Ok(agent)
}

/// One decision's temporary local search. Both drivers return RNG progress to
/// their caller on completion or failure. Interactive callers must supply a
/// restored temporary RNG so discarding work preserves the live controller.
pub struct LocalMctsSearch<'a, F: crate::core::ForwardModel> {
    agent: MctsAgent<F>,
    rng: &'a mut AgentRng,
}
impl<'a, F> LocalMctsSearch<'a, F>
where
    F: PublicInformationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// Prepare an independent search without advancing the caller's RNG.
    pub fn new(
        settings: &Settings,
        rng: &'a mut AgentRng,
        model: Option<&LocalMctsModel>,
        registry: &EvaluatorRegistry<F>,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            agent: build_local_mcts(settings, rng, model, registry)?,
            rng,
        })
    }
    /// Complete one decision and return RNG progress even if evaluation fails.
    pub fn search(
        mut self,
        game: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        clock: &dyn Clock,
    ) -> Result<SearchReport<F::Action>, ModelError> {
        let result = self.agent.search_controlled(
            game,
            actor,
            observation,
            actions,
            &Default::default(),
            clock,
            &|| false,
        );
        self.finish(result)
    }
    /// Yield between computational work groups so a browser worker can cancel.
    /// Dropping this future before completion leaves the caller's RNG unchanged.
    pub async fn search_async(
        mut self,
        game: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
        clock: &dyn Clock,
    ) -> Result<SearchReport<F::Action>, ModelError> {
        let result = self
            .agent
            .search_controlled_async(
                game,
                actor,
                observation,
                actions,
                &Default::default(),
                clock,
                &|| false,
            )
            .await;
        self.finish(result)
    }
    fn finish(
        self,
        result: crate::search::mcts::Result<SearchReport<F::Action>>,
    ) -> Result<SearchReport<F::Action>, ModelError> {
        *self.rng = AgentRng::from_snapshot(&self.agent.rng().snapshot())
            .map_err(|e| ModelError(e.to_string()))?;
        result.map_err(|e| ModelError(e.to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValuePerspective {
    AllPlayers,
    RootPlayer,
    DecidingPlayer,
}

fn value_name() -> String {
    "network_value".into()
}
fn policy_name() -> String {
    "network_policy".into()
}
/// Bind model outputs to independent evaluator roles in the MCTS configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelBindings {
    pub value: String,
    pub policy: String,
    pub value_observers: Option<ValuePerspective>,
}
impl Default for ModelBindings {
    fn default() -> Self {
        Self {
            value: value_name(),
            policy: policy_name(),
            value_observers: None,
        }
    }
}

/// Immutable model and declared observation contexts shared by local evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalMctsModel {
    model: Arc<LearnedModel>,
    bindings: ModelBindings,
}
fn search_error(e: impl std::fmt::Display) -> Error {
    Error::Evaluation(e.to_string())
}
impl LocalMctsModel {
    /// Validate an immutable terminal-outcome model and its supported player views.
    /// This checks output meaning and contexts; callers also validate the actual
    /// game's LearningSpec before constructing an agent.
    pub fn new(model: Arc<LearnedModel>, bindings: ModelBindings) -> Result<Self, ModelError> {
        let network = model
            .network()
            .ok_or_else(|| invalid("neural MCTS requires a neural model"))?;
        let document = model.as_document();
        let value = &document["contract"]["value"];
        if value["gamma"] != 1.0 || value["discount_clock"] != "environment-action" {
            return Err(invalid(
                "neural MCTS requires gamma=1 terminal-outcome values",
            ));
        }
        if bindings.value.trim().is_empty()
            || bindings.policy.trim().is_empty()
            || bindings.value == bindings.policy
        {
            return Err(invalid(
                "model evaluator names must be nonempty and distinct",
            ));
        }
        let players = document["contract"]["identity"]["player_count"]
            .as_u64()
            .unwrap() as usize;
        let context = &document["provenance"]["model_context"];
        let declared = context["observers"].as_array().ok_or_else(||
            invalid("neural MCTS model must declare provenance.model_context observers and turn_context"))?;
        let mut observers = declared
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| invalid("invalid supported observer"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        observers.sort_unstable();
        if observers != (0..players).collect::<Vec<_>>() {
            return Err(invalid(
                "local neural MCTS requires declared support for every observer",
            ));
        }
        if network.value_players().is_some() {
            if context["turn_context"] != "own_turn"
                || !matches!(
                    bindings.value_observers,
                    None | Some(ValuePerspective::DecidingPlayer)
                )
            {
                return Err(invalid(
                    "player-vector self-play models require deciding-player own-turn observations",
                ));
            }
        } else if context["turn_context"] != "any_turn" {
            return Err(invalid(
                "scalar local neural MCTS requires declared any-turn support",
            ));
        }
        Ok(Self { model, bindings })
    }
    pub fn model(&self) -> &Arc<LearnedModel> {
        &self.model
    }
    pub fn bindings(&self) -> &ModelBindings {
        &self.bindings
    }
    /// Reject a scalar-only observation selection when the search requires every
    /// player's value. Fill-zeros and negation remain explicit search settings.
    pub fn validate_search(
        &self,
        settings: &crate::search::mcts::Settings,
    ) -> Result<(), ModelError> {
        use crate::search::mcts::{Parameter, ValueCompletion};
        let single = matches!(
            self.bindings.value_observers,
            Some(ValuePerspective::RootPlayer | ValuePerspective::DecidingPlayer)
        );
        let nested = settings.progressive_bias.evaluator.iter()
            .chain(settings.puct_preferences.evaluator.iter())
            .chain(settings.expansion.evaluator.iter()).chain(settings.rollout.evaluator.iter())
            .any(|spec| spec.name == "successor_state" &&
                matches!(spec.parameters.get("state_evaluator"), Some(Parameter::Object(inner))
                    if matches!(inner.get("name"), Some(Parameter::Text(name)) if name == &self.bindings.value)));
        if self.model.network().unwrap().value_players().is_none()
            && single
            && settings.value_completion == ValueCompletion::RequireFullVector
            && (settings.state_evaluator.name == self.bindings.value || nested)
        {
            return Err(invalid(
                "full player vector required: select all_players scalar observations, a full-vector model, or an explicit fill_zeros/negation completion policy",
            ));
        }
        Ok(())
    }
    /// Register stateless model evaluators. Native heuristic roles remain available.
    /// Root identity is supplied separately for each actual move.
    pub fn register<F>(
        &self,
        registry: &mut EvaluatorRegistry<F>,
        root: PlayerId,
    ) -> crate::search::mcts::Result<()>
    where
        F: LearningGame + 'static,
        F::State: Clone,
        F::Error: std::fmt::Display,
    {
        let values = self.clone();
        registry.register_state(&self.bindings.value, move |parameters| {
            empty(parameters)?;
            Ok(Box::new(LocalEvaluator {
                model: values.clone(),
                root,
            }))
        })?;
        let policy = self.clone();
        registry.register_action(&self.bindings.policy, move |parameters| {
            empty(parameters)?;
            Ok(Box::new(LocalEvaluator {
                model: policy.clone(),
                root,
            }))
        })
    }
}
fn empty(parameters: &Parameters) -> crate::search::mcts::Result<()> {
    if parameters.is_empty() {
        Ok(())
    } else {
        Err(search_error(
            "local model bindings take no evaluator parameters",
        ))
    }
}
struct LocalEvaluator {
    model: LocalMctsModel,
    root: PlayerId,
}
impl LocalEvaluator {
    fn encoded<F: LearningGame>(
        &self,
        game: &F,
        state: &F::State,
        observer: PlayerId,
    ) -> Result<crate::learning::EncodedObservation, ModelError>
    where
        F::Error: std::fmt::Display,
    {
        let observation = game
            .observe(state, observer)
            .map_err(|e| ModelError(e.to_string()))?;
        game.encode(
            &observation,
            &self.model.model.network().unwrap().payloads(),
            LearningLimits::default(),
        )
        .map_err(|e| ModelError(e.to_string()))
    }
}
impl<F: LearningGame> StateEvaluator<F> for LocalEvaluator
where
    F::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        state: &F::State,
        context: &mut SimulationContext<'_, F>,
    ) -> crate::search::mcts::Result<StateValues> {
        let game = context.model();
        let players = game.player_count(state);
        context.require_values(players)?;
        let actor = match game.next_player(state) {
            Decision::Player(p) => p,
            _ => {
                return Err(search_error(
                    "local model evaluation requires a player decision",
                ));
            }
        };
        let net = self.model.model.network().unwrap();
        if let Some(order) = net.value_players() {
            let encoded = self.encoded(game, state, actor).map_err(search_error)?;
            let (_, output) = net.predict_values(&encoded).map_err(search_error)?;
            let mut values = vec![0.; players];
            for (&player, value) in order.iter().zip(output) {
                values[player] = value as f64;
            }
            return Ok(StateValues::Full(values));
        }
        let selected = match self
            .model
            .bindings
            .value_observers
            .unwrap_or(ValuePerspective::AllPlayers)
        {
            ValuePerspective::AllPlayers => None,
            ValuePerspective::RootPlayer => Some(self.root),
            ValuePerspective::DecidingPlayer => Some(actor),
        };
        let predict = |p| {
            let encoded = self.encoded(game, state, p).map_err(search_error)?;
            net.predict(&encoded)
                .map(|(_, v)| v as f64)
                .map_err(search_error)
        };
        if let Some(player) = selected {
            Ok(StateValues::Scalar {
                player,
                value: predict(player)?,
            })
        } else {
            Ok(StateValues::Full(
                (0..players)
                    .map(|p| predict(PlayerId::new(p)))
                    .collect::<crate::search::mcts::Result<_>>()?,
            ))
        }
    }
    fn checkpoint(&self) -> crate::search::mcts::Result<Vec<u8>> {
        Ok(self.model.model.id().as_bytes().to_vec())
    }
    fn restore(&mut self, bytes: &[u8]) -> crate::search::mcts::Result<()> {
        if bytes == self.model.model.id().as_bytes() {
            Ok(())
        } else {
            Err(search_error("local model checkpoint identity mismatch"))
        }
    }
}
impl<F: LearningGame> ActionEvaluator<F> for LocalEvaluator
where
    F::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        state: &F::State,
        actor: PlayerId,
        actions: &[F::Action],
        context: &mut SimulationContext<'_, F>,
    ) -> crate::search::mcts::Result<Vec<f64>> {
        context.require_values(actions.len())?;
        let game = context.model();
        let observation = game.observe(state, actor).map_err(search_error)?;
        let encoded = self.encoded(game, state, actor).map_err(search_error)?;
        let (logits, _) = self
            .model
            .model
            .network()
            .unwrap()
            .predict_values(&encoded)
            .map_err(search_error)?;
        let logits = actions
            .iter()
            .map(|a| {
                let id = game.action_id(&observation, a).map_err(search_error)?;
                if !encoded.action_mask.get(id).copied().unwrap_or(false) {
                    return Err(search_error("model policy action is not legal"));
                }
                logits
                    .get(id)
                    .map(|v| *v as f64)
                    .ok_or_else(|| search_error("model policy action ID out of range"))
            })
            .collect::<crate::search::mcts::Result<Vec<_>>>()?;
        let best = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities: Vec<_> = logits.iter().map(|v| (v - best).exp()).collect();
        let sum: f64 = probabilities.iter().sum();
        if !sum.is_finite() || sum <= 0. {
            return Err(search_error("invalid legal policy probability mass"));
        }
        for p in &mut probabilities {
            *p /= sum;
        }
        Ok(probabilities)
    }
    fn checkpoint(&self) -> crate::search::mcts::Result<Vec<u8>> {
        Ok(self.model.model.id().as_bytes().to_vec())
    }
    fn restore(&mut self, bytes: &[u8]) -> crate::search::mcts::Result<()> {
        if bytes == self.model.model.id().as_bytes() {
            Ok(())
        } else {
            Err(search_error("local model checkpoint identity mismatch"))
        }
    }
}
