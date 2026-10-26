//! Browser learned-controller ownership, native OSLA scoring and replay conversion.
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};
use tag_rust::{
    agents::{
        Agent, BatchEvaluator, CandidateBatch, EvaluationLimits, EvaluatorIdentity,
        OslaPythonAgent, OslaPythonConfig, OslaPythonError, OslaPythonSnapshot,
    },
    core::{
        PlayerId, PublicInformationModel,
        checkpoint::Checkpoint,
        evaluation::Evaluator,
        random::{AgentRng, AgentRngSnapshot},
    },
    inference::{LearnedModel, ModelError},
    learning::{EncodedObservation, Float32Tensor, LearningGame, LearningLimits},
    recording::ControllerSnapshot,
    replay::{BuiltinCodec, PersistenceError, ReplayCodec, RngData, SeatData},
    session::{BuiltinAgent, BuiltinAgentSnapshot, Controller},
};

pub(crate) type Registry = BTreeMap<String, Arc<LearnedModel>>;
pub(crate) fn insert(models: &mut Registry, text: &str) -> Result<String, ModelError> {
    let model = LearnedModel::parse(text)?;
    let doc = model.as_document();
    let name = if model.is_neural() {
        doc["contract"]["identity"]["game"].as_str()
    } else {
        Some("tictactoe")
    };
    let limits = LearningLimits::default();
    let spec = match name {
        Some("tictactoe") => tag_rust::games::tictactoe::TicTacToeForwardModel
            .learning_spec(&Default::default(), limits),
        Some("connect4") => tag_rust::games::connect4::Connect4ForwardModel
            .learning_spec(&Default::default(), limits),
        Some("loveletter") => {
            let (parameters, players) = crate::loveletter::model_parameters(&model)?;
            tag_rust::games::loveletter::LoveLetterForwardModel.learning_spec_for_players(
                &parameters,
                players,
                limits,
            )
        }
        Some("cantstop") => {
            let (parameters, players) = crate::cantstop::model_parameters(&model)?;
            tag_rust::games::cantstop::CantStopForwardModel.learning_spec_for_players(
                &parameters,
                players,
                limits,
            )
        }
        _ => {
            return Err(error("unsupported browser learned-model game"));
        }
    }
    .map_err(error)?;
    model.validate_spec(&spec)?;
    let id = model.id().to_string();
    if models.contains_key(&id) {
        return Ok(id);
    }
    let bytes = models
        .values()
        .map(|m| m.as_document().to_string().len())
        .sum::<usize>()
        + text.len();
    if models.len() >= 16 || bytes > tag_rust::inference::MAX_MODEL_BYTES {
        return Err(error("model registry limit is 16 models and 10 MiB JSON"));
    }
    models.insert(id.clone(), Arc::new(model));
    Ok(id)
}
pub(crate) fn summaries(models: &Registry) -> Value {
    json!(models.values().map(|m|{let d=m.as_document();json!({"id":m.id(),"kind":if !m.is_neural(){"q_learning"}else if d["contract"]["value"]["discount_clock"] == "environment-action" {"neural_mcts"}else{"neural_osla"},"game":if m.is_neural(){&d["contract"]["identity"]["game"]}else{&d["task"]["game"]},"player_count":d["contract"]["identity"]["player_count"]})}).collect::<Vec<_>>())
}
fn error(e: impl std::fmt::Display) -> ModelError {
    ModelError(e.to_string())
}

#[derive(Debug)]
pub(crate) enum BrowserAgent<E> {
    Builtin(BuiltinAgent<E>),
    Mcts {
        configuration: Box<tag_rust::search::mcts::Settings>,
        rng: AgentRng,
        model: Option<tag_rust::inference::LocalMctsModel>,
    },
    Learned {
        model: Arc<LearnedModel>,
        rng: AgentRng,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Snapshot {
    Builtin(BuiltinAgentSnapshot<()>),
    Mcts {
        configuration: Value,
        rng: AgentRngSnapshot,
        model: Option<tag_rust::inference::LocalMctsModel>,
    },
    Learned {
        model: Arc<LearnedModel>,
        rng: AgentRngSnapshot,
    },
}
impl<E: Checkpoint<Snapshot = ()>> Checkpoint for BrowserAgent<E>
where
    E::RestoreError: std::fmt::Display,
{
    type Snapshot = Snapshot;
    type RestoreError = ModelError;
    fn snapshot(&self) -> Snapshot {
        match self {
            Self::Mcts {
                configuration,
                rng,
                model,
            } => Snapshot::Mcts {
                configuration: crate::mcts::configuration_value(configuration),
                rng: rng.snapshot(),
                model: model.clone(),
            },
            Self::Builtin(agent) => Snapshot::Builtin(agent.snapshot()),
            Self::Learned { model, rng } => Snapshot::Learned {
                model: model.clone(),
                rng: rng.snapshot(),
            },
        }
    }
    fn from_snapshot(s: &Snapshot) -> Result<Self, ModelError> {
        Ok(match s {
            Snapshot::Mcts {
                configuration,
                rng,
                model,
            } => {
                let settings =
                    tag_rust::search::mcts::Settings::from_json(&configuration.to_string())
                        .map_err(error)?;
                Self::Mcts {
                    configuration: Box::new(settings),
                    rng: AgentRng::from_snapshot(rng).map_err(error)?,
                    model: model.clone(),
                }
            }
            Snapshot::Builtin(s) => Self::Builtin(BuiltinAgent::from_snapshot(s).map_err(error)?),
            Snapshot::Learned { model, rng } => Self::Learned {
                model: model.clone(),
                rng: AgentRng::from_snapshot(rng).map_err(error)?,
            },
        })
    }
}

#[derive(Clone, Debug)]
struct ModelEvaluator(Arc<LearnedModel>);
impl Checkpoint for ModelEvaluator {
    type Snapshot = Arc<LearnedModel>;
    type RestoreError = ModelError;
    fn snapshot(&self) -> Self::Snapshot {
        self.0.clone()
    }
    fn from_snapshot(s: &Self::Snapshot) -> Result<Self, ModelError> {
        Ok(Self(s.clone()))
    }
}
impl BatchEvaluator for ModelEvaluator {
    fn evaluate(&mut self, batch: &CandidateBatch) -> Result<Vec<f64>, OslaPythonError> {
        let net = self
            .0
            .network()
            .ok_or_else(|| OslaPythonError::Invalid("neural model required".into()))?;
        let players = batch.identity.player_count;
        let mut values = Vec::new();
        for row in 0..batch.action_ids.len() {
            let code = batch.results[row * players + batch.observer.index()];
            if code != 0 {
                values.push(match code {
                    1 => 1.,
                    2 => 0.,
                    3 => -1.,
                    _ => return Err(OslaPythonError::Invalid("invalid result".into())),
                });
                continue;
            }
            let tensors = batch
                .payloads
                .iter()
                .map(|t| {
                    let count = t.values.len() / batch.action_ids.len();
                    Float32Tensor {
                        name: t.name,
                        shape: t.shape[1..].to_vec(),
                        values: t.values[row * count..(row + 1) * count].to_vec(),
                    }
                })
                .collect();
            let encoded = EncodedObservation {
                identity: batch.identity.clone(),
                observer: batch.observer,
                next_player: usize::try_from(batch.next_players[row])
                    .ok()
                    .map(PlayerId::new),
                tensors,
                progress: [0; 4],
                results: vec![],
                action_mask: vec![],
                offered_action_ids: vec![],
            };
            values.push(
                net.predict(&encoded)
                    .map_err(|e| OslaPythonError::Invalid(e.to_string()))?
                    .1 as f64,
            );
        }
        Ok(values)
    }
}

impl<F, E> Agent<F> for BrowserAgent<E>
where
    F: PublicInformationModel + LearningGame + tag_rust::search::mcts::NativeEvaluators + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
    E: Evaluator<F::State>,
    E::Error: std::fmt::Display,
{
    type Error = ModelError;
    fn rng(&self) -> &AgentRng {
        match self {
            Self::Mcts { rng, .. } => rng,
            Self::Builtin(a) => <BuiltinAgent<E> as Agent<F>>::rng(a),
            Self::Learned { rng, .. } => rng,
        }
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        match self {
            Self::Mcts { rng, .. } => rng,
            Self::Builtin(a) => <BuiltinAgent<E> as Agent<F>>::rng_mut(a),
            Self::Learned { rng, .. } => rng,
        }
    }
    fn choose(
        &mut self,
        game: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, ModelError> {
        if let Self::Mcts {
            configuration,
            rng,
            model,
        } = self
        {
            let search = crate::mcts::prepare::<F>(configuration, rng, model.as_ref(), actor)?;
            return search
                .search(
                    game,
                    actor,
                    observation,
                    actions,
                    &crate::mcts::BrowserClock::default(),
                )
                .map(|report| report.selected_index);
        }
        let Self::Learned { model, rng } = self else {
            if let Self::Builtin(a) = self {
                return a.choose(game, actor, observation, actions).map_err(error);
            }
            unreachable!()
        };
        if actions.is_empty() {
            return Err(error("actions must be nonempty"));
        }
        if let Some(network) = model.network() {
            let config = OslaPythonConfig {
                evaluator: EvaluatorIdentity {
                    name: "browser-neural-value".into(),
                    version: 1,
                    configuration: model.id().into(),
                },
                payloads: network.payloads().iter().map(|s| s.to_string()).collect(),
                limits: EvaluationLimits::default(),
            };
            let mut agent = OslaPythonAgent::<ModelEvaluator>::from_snapshot(&OslaPythonSnapshot {
                policy_version: 1,
                config,
                rng: rng.snapshot(),
                evaluator: model.clone(),
            })
            .map_err(error)?;
            let choice = agent
                .choose(game, actor, observation, actions)
                .map_err(error)?;
            *rng = AgentRng::from_snapshot(&agent.snapshot().rng).map_err(error)?;
            Ok(choice)
        } else {
            let encoded = game
                .encode(observation, &["cells"], LearningLimits::default())
                .map_err(error)?;
            if encoded.observer != actor || encoded.next_player != Some(actor) {
                return Err(error("Q-table requires the observer's decision"));
            }
            let values = model.q_values(&encoded)?;
            let scores = actions
                .iter()
                .map(|a| {
                    game.action_id(observation, a)
                        .map(|id| values[id])
                        .map_err(error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let choices: Vec<_> = scores
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (*v == best).then_some(i))
                .collect();
            Ok(choices[rng.index(choices.len()).unwrap()])
        }
    }
    fn choose_async<'a>(
        &'a mut self,
        game: &'a F,
        actor: PlayerId,
        observation: &'a F::Observation,
        actions: &'a [F::Action],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, ModelError>> + 'a>> {
        Box::pin(async move {
            if let Self::Mcts {
                configuration,
                rng,
                model,
            } = self
            {
                let search = crate::mcts::prepare::<F>(configuration, rng, model.as_ref(), actor)?;
                search
                    .search_async(
                        game,
                        actor,
                        observation,
                        actions,
                        &crate::mcts::BrowserClock::default(),
                    )
                    .await
                    .map(|report| report.selected_index)
            } else {
                self.choose(game, actor, observation, actions)
            }
        })
    }
}

pub(crate) struct Codec<'a>(pub &'a Registry);
impl<E: Checkpoint<Snapshot = ()>> ReplayCodec<BrowserAgent<E>> for Codec<'_>
where
    E::RestoreError: std::fmt::Display + std::fmt::Debug,
{
    fn recorded_choice(
        &self,
        before: &SeatData,
        after: &SeatData,
    ) -> Result<bool, PersistenceError> {
        let SeatData::Extension { rng, policy } = before else {
            return Ok(false);
        };
        if policy["kind"] != "mcts" && policy["kind"] != "neural_mcts" {
            return Ok(false);
        }
        let SeatData::Extension {
            rng: next_rng,
            policy: next_policy,
        } = after
        else {
            return Err(PersistenceError::Invalid(
                "recorded MCTS continuation changed controller type".into(),
            ));
        };
        let previous = rng.snapshot()?;
        let next = next_rng.snapshot()?;
        if policy != next_policy
            || previous.initial_seed != next.initial_seed
            || previous.stream != next.stream
            || next.word_position < previous.word_position
        {
            return Err(PersistenceError::Invalid(
                "recorded MCTS continuation changed policy or reversed RNG progress".into(),
            ));
        }
        <Self as ReplayCodec<BrowserAgent<E>>>::decode(self, after)?;
        Ok(true)
    }
    fn encode(&self, s: &ControllerSnapshot<Snapshot>) -> SeatData {
        match s {
            ControllerSnapshot::Ai(Snapshot::Mcts {
                configuration,
                rng,
                model,
            }) => SeatData::Extension {
                rng: RngData::from_snapshot(*rng),
                policy: if let Some(model) = model {
                    json!({"kind":"neural_mcts", "version":1,"configuration":configuration,
                        "model_id":model.model().id(),"bindings":model.bindings()})
                } else {
                    json!({"kind":"mcts", "version":1, "configuration":configuration})
                },
            },
            ControllerSnapshot::Human => SeatData::Human,
            ControllerSnapshot::Ai(Snapshot::Builtin(s)) => {
                <BuiltinCodec as ReplayCodec<BuiltinAgent<E>>>::encode(
                    &BuiltinCodec,
                    &ControllerSnapshot::Ai(s.clone()),
                )
            }
            ControllerSnapshot::Ai(Snapshot::Learned { model, rng }) => SeatData::Extension {
                rng: RngData::from_snapshot(*rng),
                policy: json!({"kind":if model.is_neural(){"neural_osla"}else{"q_learning"},"version":1,"model_id":model.id()}),
            },
        }
    }
    fn decode(&self, s: &SeatData) -> Result<Controller<BrowserAgent<E>>, PersistenceError> {
        if let SeatData::Extension { rng, policy } = s
            && (policy["kind"] == "mcts" || policy["kind"] == "neural_mcts")
        {
            let settings =
                tag_rust::search::mcts::Settings::from_json(&policy["configuration"].to_string())
                    .map_err(|e| PersistenceError::Invalid(e.to_string()))?;
            let configuration: Value = serde_json::from_str(
                &settings
                    .to_json()
                    .map_err(|e| PersistenceError::Invalid(e.to_string()))?,
            )
            .unwrap();
            let model = if policy["kind"] == "neural_mcts" {
                let loaded = policy["model_id"]
                    .as_str()
                    .and_then(|id| self.0.get(id))
                    .ok_or_else(|| PersistenceError::Invalid("missing neural MCTS model".into()))?;
                Some(
                    tag_rust::inference::LocalMctsModel::new(
                        loaded.clone(),
                        serde_json::from_value(policy["bindings"].clone())
                            .map_err(|e| PersistenceError::Invalid(e.to_string()))?,
                    )
                    .map_err(|e| PersistenceError::Invalid(e.to_string()))?,
                )
            } else {
                None
            };
            let expected = if let Some(model) = &model {
                json!({"kind":"neural_mcts","version":1,"configuration":configuration,
                    "model_id":model.model().id(),"bindings":model.bindings()})
            } else {
                json!({"kind":"mcts","version":1,"configuration":configuration})
            };
            if *policy != expected || settings.seed != Some(rng.snapshot()?.initial_seed) {
                return Err(PersistenceError::Invalid(
                    "incompatible MCTS controller configuration".into(),
                ));
            }
            return Ok(Controller::Ai(BrowserAgent::Mcts {
                configuration: Box::new(settings),
                model,
                rng: AgentRng::from_snapshot(&rng.snapshot()?)
                    .map_err(|e| PersistenceError::Invalid(e.to_string()))?,
            }));
        }
        if let SeatData::Extension { rng, policy } = s {
            let fail = || {
                PersistenceError::Invalid("missing/incompatible learned controller model".into())
            };
            let id = policy["model_id"].as_str().ok_or_else(fail)?;
            let model = self.0.get(id).ok_or_else(fail)?.clone();
            let expected = json!({"kind":if model.is_neural(){"neural_osla"}else{"q_learning"},"version":1,"model_id":id});
            if *policy != expected {
                return Err(fail());
            }
            return Ok(Controller::Ai(BrowserAgent::Learned {
                model,
                rng: AgentRng::from_snapshot(&rng.snapshot()?)
                    .map_err(|e| PersistenceError::Invalid(e.to_string()))?,
            }));
        }
        Ok(
            match <BuiltinCodec as ReplayCodec<BuiltinAgent<E>>>::decode(&BuiltinCodec, s)? {
                Controller::Human => Controller::Human,
                Controller::Ai(a) => Controller::Ai(BrowserAgent::Builtin(a)),
            },
        )
    }
    fn lookahead(&self, s: &SeatData) -> bool {
        matches!(s, SeatData::Osla(_))
            || matches!(s,SeatData::Extension {policy,..} if policy["kind"]=="neural_osla")
    }
    fn extra_work(&self, s: &SeatData) -> usize {
        if let SeatData::Extension { policy, .. } = s {
            if policy["kind"] == "mcts" || policy["kind"] == "neural_mcts" {
                return 0;
            }
            return policy["model_id"]
                .as_str()
                .and_then(|id| self.0.get(id))
                .and_then(|m| m.network())
                .map_or(0, |n| n.parameter_count() * n.action_count());
        }
        0
    }
}

pub(crate) fn documents(registry: &Registry) -> Value {
    json!(registry.values().map(|m| m.document()).collect::<Vec<_>>())
}

/// Return raw neural predictions or Q rows from a permitted game encoding.
/// Borrowed rules/state/settings/model remain unchanged; observer selects a player.
pub(crate) fn prediction<F: LearningGame>(
    model: &F,
    state: &F::State,
    parameters: &F::Parameters,
    loaded: &LearnedModel,
    observer: usize,
) -> Result<Value, crate::protocol::BridgeError>
where
    F::Error: std::fmt::Display,
{
    loaded
        .validate_spec(
            &model
                .learning_spec_for_players(
                    parameters,
                    model.player_count(state),
                    tag_rust::learning::LearningLimits::default(),
                )
                .map_err(bridge_error)?,
        )
        .map_err(bridge_error)?;
    let observation = model
        .observe(state, PlayerId::new(observer))
        .map_err(bridge_error)?;
    let payloads = loaded
        .network()
        .map(|n| n.payloads())
        .unwrap_or_else(|| vec!["cells"]);
    let encoded = model
        .encode(
            &observation,
            &payloads,
            tag_rust::learning::LearningLimits::default(),
        )
        .map_err(bridge_error)?;
    if let Some(network) = loaded.network() {
        let (logits, values) = network.predict_values(&encoded).map_err(bridge_error)?;
        Ok(
            serde_json::json!({"model_id":loaded.id(),"inputs":network.inputs(&encoded).map_err(bridge_error)?,"logits":logits,"value":if network.value_players().is_none(){Some(values[0])}else{None},"values":values,"value_players":network.value_players()}),
        )
    } else {
        Ok(
            serde_json::json!({"model_id":loaded.id(),"q_values":loaded.q_values(&encoded).map_err(bridge_error)?}),
        )
    }
}
fn bridge_error(error: impl std::fmt::Display) -> crate::protocol::BridgeError {
    crate::protocol::BridgeError::new("invalid_model", error.to_string())
}
/// Collect exact learned model references from replay controller entries.
pub(crate) fn references(
    text: &str,
) -> Result<std::collections::BTreeSet<String>, crate::protocol::BridgeError> {
    fn visit(value: &Value, ids: &mut std::collections::BTreeSet<String>) {
        if value["kind"] == "extension"
            && let Some(id) = value["rng"]["policy"]["model_id"].as_str()
        {
            ids.insert(id.to_string());
        }
        match value {
            Value::Object(fields) => {
                for v in fields.values() {
                    visit(v, ids)
                }
            }
            Value::Array(values) => {
                for v in values {
                    visit(v, ids)
                }
            }
            _ => {}
        }
    }
    let document: Value = serde_json::from_str(text).map_err(bridge_error)?;
    let mut ids = std::collections::BTreeSet::new();
    visit(&document, &mut ids);
    Ok(ids)
}
/// Validate each replay's referenced model against the actual game description.
pub(crate) fn validate_models<F>(
    model: &F,
    parameters: &F::Parameters,
    models: &Registry,
    text: &str,
) -> Result<(), crate::protocol::BridgeError>
where
    F: LearningGame + PublicInformationModel + tag_rust::search::mcts::NativeEvaluators + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    let spec = model
        .learning_spec(parameters, LearningLimits::default())
        .map_err(bridge_error)?;
    for id in references(text)? {
        models
            .get(&id)
            .ok_or_else(|| bridge_error("missing referenced model"))?
            .validate_spec(&spec)
            .map_err(bridge_error)?;
    }
    fn policies(value: &Value, found: &mut std::collections::BTreeSet<String>) {
        if value["kind"] == "extension" {
            found.insert(value.to_string());
        }
        match value {
            Value::Array(items) => {
                for item in items {
                    policies(item, found);
                }
            }
            Value::Object(fields) => {
                for item in fields.values() {
                    policies(item, found);
                }
            }
            _ => {}
        }
    }
    let data: Value = serde_json::from_str(text).map_err(bridge_error)?;
    let mut entries = std::collections::BTreeSet::new();
    policies(&data, &mut entries);
    for entry in entries {
        let seat: SeatData = serde_json::from_str(&entry).map_err(bridge_error)?;
        if let SeatData::Extension { rng, policy } = seat {
            if policy["kind"] == "neural_osla" {
                models
                    .get(policy["model_id"].as_str().unwrap_or_default())
                    .and_then(|m| m.network())
                    .ok_or_else(|| bridge_error("missing neural OSLA model"))?
                    .validate_osla()
                    .map_err(bridge_error)?;
            }
            if policy["kind"] != "mcts" && policy["kind"] != "neural_mcts" {
                continue;
            }
            let binding = if policy["kind"] == "neural_mcts" {
                let loaded = models
                    .get(policy["model_id"].as_str().unwrap_or_default())
                    .ok_or_else(|| bridge_error("missing neural MCTS model"))?;
                Some(
                    tag_rust::inference::LocalMctsModel::new(
                        loaded.clone(),
                        serde_json::from_value(policy["bindings"].clone()).map_err(bridge_error)?,
                    )
                    .map_err(bridge_error)?,
                )
            } else {
                None
            };
            let rng = AgentRng::from_snapshot(&rng.snapshot().map_err(bridge_error)?)
                .map_err(bridge_error)?;
            crate::mcts::agent::<F>(
                &policy["configuration"],
                &rng,
                binding.as_ref(),
                PlayerId::new(0),
            )
            .map_err(bridge_error)?;
        }
    }
    Ok(())
}
