//! Native match capabilities are optional: Random needs neither search nor replay.
use super::{
    Result,
    config::{AgentConfig, ResolvedPlayer},
};
use serde::{Serialize, de::DeserializeOwned};
use tag_rust::agents::{
    Agent, FirstAvailableAgent, MctsAgent, OslaAgent, RandomAgent, SampledOslaAgent,
};
use tag_rust::core::{ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::core::{evaluation::Evaluator, random::AgentRng, simulation::SimulationModel};
use tag_rust::replay::ReplayGame;
use tag_rust::search::mcts::{Counters, EvaluatorRegistry, NativeEvaluators, Settings};
use tag_rust::session::BuiltinAgent;

/// A game declares only the optional operations this application can provide.
/// The default supports First Available and Random without a heuristic, learning
/// encoder, simulation fork or browser replay implementation.
pub trait MatchGame:
    ForwardModel<
        State: Clone + PartialEq,
        Action: Clone + PartialEq + Serialize,
        Parameters: Default + Clone + Serialize + DeserializeOwned,
        Error: std::error::Error + 'static,
    > + Clone
    + 'static
{
    const RULES_VERSION: u32 = 1;
    const STOCHASTIC: bool = false;
    const BROWSER_REPLAY: bool = false;
    fn registry() -> EvaluatorRegistry<Self>
    where
        Self: Sized,
    {
        EvaluatorRegistry::default()
    }
    fn osla(_seed: u64, _repetitions: std::num::NonZeroU32) -> Result<MatchAgent<Self>>
    where
        Self: Sized,
    {
        Err("this game has no OSLA capability".into())
    }
    fn mcts(_settings: Settings, _registry: &EvaluatorRegistry<Self>) -> Result<MatchAgent<Self>>
    where
        Self: Sized,
    {
        Err("this game has no MCTS capability".into())
    }
    fn validate_ranking(rule: Option<&str>) -> Result<()> {
        if rule.is_none() || rule == Some("game") {
            Ok(())
        } else {
            Err("this game does not support the requested final ranking".into())
        }
    }
    fn final_ranks(&self, _state: &Self::State, _rule: Option<&str>) -> Result<Option<Vec<usize>>> {
        Ok(None)
    }
    fn action_replay(
        _parameters: &serde_json::Value,
        _outcome: &super::report::GameReport,
    ) -> Result<Option<String>> {
        Ok(None)
    }
    fn replay_dimensions(_parameters: &Self::Parameters) -> Option<(usize, usize)> {
        None
    }
    fn game_seed(_parameters: &Self::Parameters) -> Option<u64> {
        None
    }
    fn with_game_seed(parameters: &Self::Parameters, _seed: u64) -> Self::Parameters {
        parameters.clone()
    }
    fn chance_event(_state: &Self::State, _action: &Self::Action) -> Option<serde_json::Value> {
        None
    }
    fn recorded(
        _settings: &super::config::ResolvedMatch,
        _game: usize,
        _seeds: Vec<u64>,
    ) -> Result<super::execution::CompletedGame>
    where
        Self: Sized,
    {
        Err("this game has no browser replay capability".into())
    }
    #[cfg(feature = "inference")]
    fn neural(_player: &ResolvedPlayer, _seed: u64) -> Result<MatchAgent<Self>>
    where
        Self: Sized,
    {
        Err("this game has no local neural MCTS integration".into())
    }
    #[cfg(feature = "inference")]
    fn validate_neural_model(
        &self,
        _parameters: &Self::Parameters,
        _model: &tag_rust::inference::LocalMctsModel,
    ) -> Result<()> {
        Err("this game has no local neural MCTS integration".into())
    }
}

// Shared forwarding for the two deterministic placement games. Their ordinary
// OSLA path and replay controller settings retain the original seeded behaviour.
macro_rules! placement_game {
    ($model:ty, $evaluator:ty, $dimensions:expr) => {
        impl MatchGame for $model {
            const BROWSER_REPLAY: bool = true;
            fn registry() -> EvaluatorRegistry<Self> {
                Self::native_evaluators()
            }
            fn osla(seed: u64, repetitions: std::num::NonZeroU32) -> Result<MatchAgent<Self>> {
                if repetitions.get() == 1 {
                    Ok(native(OslaAgent::new(<$evaluator>::default(), seed)))
                } else {
                    Ok(native(SampledOslaAgent::new(
                        <$evaluator>::default(),
                        seed,
                        repetitions,
                    )))
                }
            }
            fn mcts(
                settings: Settings,
                registry: &EvaluatorRegistry<Self>,
            ) -> Result<MatchAgent<Self>> {
                native_mcts(settings, registry)
            }
            fn final_ranks(
                &self,
                state: &Self::State,
                _: Option<&str>,
            ) -> Result<Option<Vec<usize>>> {
                Ok(Some(
                    (0..self.player_count(state))
                        .map(|seat| {
                            Ok(match self.player_result(state, PlayerId::new(seat))? {
                                PlayerResult::Win | PlayerResult::Draw => 1,
                                PlayerResult::Loss => 2,
                                PlayerResult::Ongoing => {
                                    return Err("final ranks require a terminal game".into());
                                }
                            })
                        })
                        .collect::<Result<_>>()?,
                ))
            }
            fn action_replay(
                parameters: &serde_json::Value,
                outcome: &super::report::GameReport,
            ) -> Result<Option<String>> {
                super::execution::action_replay(Self, parameters, outcome).map(Some)
            }
            fn replay_dimensions(p: &Self::Parameters) -> Option<(usize, usize)> {
                Some(($dimensions)(p))
            }
            fn recorded(
                settings: &super::config::ResolvedMatch,
                game: usize,
                seeds: Vec<u64>,
            ) -> Result<super::execution::CompletedGame> {
                super::execution::play_recorded(Self, settings, game, seeds)
            }
            #[cfg(feature = "inference")]
            fn neural(player: &ResolvedPlayer, seed: u64) -> Result<MatchAgent<Self>> {
                neural::<Self>(player, seed)
            }
            #[cfg(feature = "inference")]
            fn validate_neural_model(
                &self,
                p: &Self::Parameters,
                model: &tag_rust::inference::LocalMctsModel,
            ) -> Result<()> {
                use tag_rust::learning::LearningGame;
                model
                    .model()
                    .validate_spec(&self.learning_spec(p, Default::default())?)?;
                Ok(())
            }
        }
    };
}
placement_game!(
    tag_rust::games::tictactoe::TicTacToeForwardModel,
    tag_rust::games::tictactoe::TicTacToeEvaluator,
    |p: &tag_rust::games::tictactoe::TicTacToeParameters| (p.grid_size, p.grid_size)
);
placement_game!(
    tag_rust::games::connect4::Connect4ForwardModel,
    tag_rust::games::connect4::Connect4Evaluator,
    |p: &tag_rust::games::connect4::Connect4Parameters| (p.width, p.height)
);

impl MatchGame for tag_rust::games::cantstop::CantStopForwardModel {
    const STOCHASTIC: bool = true;
    fn registry() -> EvaluatorRegistry<Self> {
        Self::native_evaluators()
    }
    fn osla(seed: u64, repetitions: std::num::NonZeroU32) -> Result<MatchAgent<Self>> {
        Ok(native(SampledOslaAgent::new(
            tag_rust::games::cantstop::CantStopProgress::default(),
            seed,
            repetitions,
        )))
    }
    fn mcts(settings: Settings, registry: &EvaluatorRegistry<Self>) -> Result<MatchAgent<Self>> {
        native_mcts(settings, registry)
    }
    fn validate_ranking(rule: Option<&str>) -> Result<()> {
        match rule {
            None | Some("game" | "score" | "claimed_columns") => Ok(()),
            _ => Err(
                "Can't Stop ranking must be game or score (claimed_columns is a legacy alias)"
                    .into(),
            ),
        }
    }
    fn final_ranks(&self, state: &Self::State, rule: Option<&str>) -> Result<Option<Vec<usize>>> {
        if !matches!(rule, Some("score" | "claimed_columns")) {
            return Ok(None);
        }
        let scores = (0..self.player_count(state))
            .map(|seat| self.player_score(state, PlayerId::new(seat)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(
            scores
                .iter()
                .map(|score| 1 + scores.iter().filter(|other| *other > score).count())
                .collect(),
        ))
    }
    fn game_seed(p: &Self::Parameters) -> Option<u64> {
        Some(p.seed)
    }
    fn with_game_seed(p: &Self::Parameters, seed: u64) -> Self::Parameters {
        tag_rust::games::cantstop::CantStopParameters { seed, ..p.clone() }
    }
    fn chance_event(state: &Self::State, action: &Self::Action) -> Option<serde_json::Value> {
        matches!(action, tag_rust::games::cantstop::CantStopAction::Roll)
            .then(|| serde_json::json!({"dice":state.position().dice().expect("roll stores dice")}))
    }
}

impl MatchGame for tag_rust::games::loveletter::LoveLetterForwardModel {
    const STOCHASTIC: bool = true;
    fn registry() -> EvaluatorRegistry<Self> {
        Self::native_evaluators()
    }
    fn osla(seed: u64, repetitions: std::num::NonZeroU32) -> Result<MatchAgent<Self>> {
        Ok(native(SampledOslaAgent::new(
            tag_rust::games::loveletter::LoveLetterHeuristic,
            seed,
            repetitions,
        )))
    }
    fn mcts(settings: Settings, registry: &EvaluatorRegistry<Self>) -> Result<MatchAgent<Self>> {
        native_mcts(settings, registry)
    }
    fn game_seed(p: &Self::Parameters) -> Option<u64> {
        Some(p.seed)
    }
    fn with_game_seed(p: &Self::Parameters, seed: u64) -> Self::Parameters {
        tag_rust::games::loveletter::LoveLetterParameters { seed, ..p.clone() }
    }
}

/// The wrapper owns a concrete agent and translates its errors for the CLI.
/// Dynamic dispatch here permits optional game capabilities without requiring
/// every game to implement a heuristic or a replay controller enum.
trait NativeAgent<F: ForwardModel>: Agent<F, Error = String> {
    fn counters(&self) -> Option<&Counters> {
        None
    }
}
struct Wrapped<A> {
    agent: A,
    counters: fn(&A) -> Option<&Counters>,
}
impl<F: ForwardModel, A: Agent<F, Error: std::fmt::Display>> Agent<F> for Wrapped<A> {
    type Error = String;
    fn rng(&self) -> &AgentRng {
        self.agent.rng()
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        self.agent.rng_mut()
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> std::result::Result<usize, String> {
        self.agent
            .choose(model, actor, observation, actions)
            .map_err(|e| e.to_string())
    }
}
impl<F: ForwardModel, A: Agent<F, Error: std::fmt::Display>> NativeAgent<F> for Wrapped<A> {
    fn counters(&self) -> Option<&Counters> {
        (self.counters)(&self.agent)
    }
}
pub struct MatchAgent<F: ForwardModel>(Box<dyn NativeAgent<F>>);
fn native<F: ForwardModel, A: Agent<F, Error: std::fmt::Display> + 'static>(
    agent: A,
) -> MatchAgent<F> {
    MatchAgent(Box::new(Wrapped {
        agent,
        counters: |_| None,
    }))
}
fn native_mcts<F: MatchGame + SimulationModel>(
    settings: Settings,
    registry: &EvaluatorRegistry<F>,
) -> Result<MatchAgent<F>> {
    Ok(MatchAgent(Box::new(Wrapped {
        agent: MctsAgent::new(settings, registry)?,
        counters: |a: &MctsAgent<F>| a.last_report().map(|r| &r.counters),
    })))
}

pub fn basic<F: MatchGame + PublicInformationModel + ReplayGame>(
    player: &ResolvedPlayer,
    seed: u64,
) -> Result<BuiltinAgent<F::SavedEvaluator>>
where
    F::SavedEvaluator: Default,
    <F::SavedEvaluator as Evaluator<F::State>>::Error: std::error::Error + 'static,
{
    Ok(match player.kind {
        AgentConfig::FirstAvailable {} => {
            BuiltinAgent::FirstAvailable(FirstAvailableAgent::new(seed))
        }
        AgentConfig::Random {} => BuiltinAgent::Random(RandomAgent::new(seed)),
        AgentConfig::Osla {
            repetitions_per_action,
        } if repetitions_per_action.get() == 1 => {
            BuiltinAgent::Osla(OslaAgent::new(F::SavedEvaluator::default(), seed))
        }
        _ => return Err("this controller has no resumable basic replay representation".into()),
    })
}
pub fn build<F: MatchGame>(
    player: &ResolvedPlayer,
    seed: u64,
    registry: &EvaluatorRegistry<F>,
) -> Result<MatchAgent<F>> {
    #[cfg(feature = "inference")]
    if matches!(player.kind, AgentConfig::NeuralMcts { .. }) {
        return F::neural(player, seed);
    }
    if let Some(settings) = &player.mcts {
        let mut settings = settings.clone();
        settings.seed = Some(seed);
        return F::mcts(settings, registry);
    }
    match player.kind {
        AgentConfig::FirstAvailable {} => Ok(native(FirstAvailableAgent::new(seed))),
        AgentConfig::Random {} => Ok(native(RandomAgent::new(seed))),
        AgentConfig::Osla {
            repetitions_per_action,
        } => F::osla(seed, repetitions_per_action),
        _ => Err("missing search configuration".into()),
    }
}
impl<F: ForwardModel> Agent<F> for MatchAgent<F> {
    type Error = String;
    fn rng(&self) -> &AgentRng {
        self.0.rng()
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        self.0.rng_mut()
    }
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> std::result::Result<usize, String> {
        self.0.choose(model, actor, observation, actions)
    }
}
impl<F: ForwardModel> MatchAgent<F> {
    pub fn last_counters(&self) -> Option<&Counters> {
        self.0.counters()
    }
}

#[cfg(feature = "inference")]
struct NeuralAgent<F: ForwardModel> {
    settings: Settings,
    rng: AgentRng,
    model: tag_rust::inference::LocalMctsModel,
    counters: Option<Counters>,
    marker: std::marker::PhantomData<F>,
}
#[cfg(feature = "inference")]
fn neural<F: MatchGame + PublicInformationModel + tag_rust::learning::LearningGame>(
    player: &ResolvedPlayer,
    seed: u64,
) -> Result<MatchAgent<F>> {
    let model = player
        .selected_model
        .as_ref()
        .ok_or("select a model for the current game; ordinary matches require model_file")?
        .clone();
    let mut settings = player.mcts.clone().ok_or("missing neural MCTS settings")?;
    settings.seed = Some(seed);
    let mut registry = F::registry();
    model.register(&mut registry, PlayerId::new(0))?;
    tag_rust::inference::build_local_mcts(
        &settings,
        &AgentRng::new(seed),
        Some(&model),
        &registry,
    )?;
    Ok(MatchAgent(Box::new(NeuralAgent::<F> {
        settings,
        rng: AgentRng::new(seed),
        model,
        counters: None,
        marker: std::marker::PhantomData,
    })))
}
#[cfg(feature = "inference")]
impl<F: MatchGame + PublicInformationModel + tag_rust::learning::LearningGame> Agent<F>
    for NeuralAgent<F>
{
    type Error = String;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        game: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> std::result::Result<usize, String> {
        self.counters = None;
        let mut registry = F::registry();
        self.model
            .register(&mut registry, actor)
            .map_err(|e| e.to_string())?;
        tag_rust::inference::LocalMctsSearch::new(
            &self.settings,
            &mut self.rng,
            Some(&self.model),
            &registry,
        )
        .and_then(|search| {
            search.search(
                game,
                actor,
                observation,
                actions,
                &tag_rust::search::mcts::SystemClock::default(),
            )
        })
        .map(|report| {
            self.counters = Some(report.counters);
            report.selected_index
        })
        .map_err(|e| e.to_string())
    }
}
#[cfg(feature = "inference")]
impl<F: MatchGame + PublicInformationModel + tag_rust::learning::LearningGame> NativeAgent<F>
    for NeuralAgent<F>
{
    fn counters(&self) -> Option<&Counters> {
        self.counters.as_ref()
    }
}
