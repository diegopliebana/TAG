//! Shared native opponent execution for observation-derived simulation games.
//! Owned agent checkpoints persist between detached calls; evaluators stay local.
use crate::{
    engine::Engine,
    game::{Game, Position, StalePositionError},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::{Value, json};
use std::num::NonZeroU32;
use tag_rust::{
    agents::{Agent, RandomAgent, SampledOslaAgent, SampledOslaSnapshot},
    core::{ForwardModel, PlayerId, checkpoint::Checkpoint, random::AgentRngSnapshot},
    games::{
        cantstop::{CantStopForwardModel, CantStopProgress},
        loveletter::{LoveLetterForwardModel, LoveLetterHeuristic},
    },
    search::mcts::{MctsAgent, MctsSnapshot, Settings},
};

// Compile the same runtime for each game; only model, heuristic and public
// event formatting differ. The outer enum prevents restoring a different game's
// typed controller checkpoints.
macro_rules! backend {
    ($module:ident, $variant:ident, $model:ty, $heuristic:ty, $snapshot:ty, $event:expr) => {
        mod $module {
            use super::*;
            enum Saved {
                Random(AgentRngSnapshot),
                Osla(SampledOslaSnapshot<$snapshot>),
                Mcts(Box<MctsSnapshot>),
            }
            enum Active {
                Random(Box<RandomAgent>),
                Osla(Box<SampledOslaAgent<$heuristic>>),
                Mcts(Box<MctsAgent<$model>>),
            }
            type NativeGame = crate::engine::Game<$model>;
            type Result<T> = std::result::Result<T, String>;
            fn error(e: impl std::fmt::Display) -> String {
                e.to_string()
            }

            impl Active {
                fn create(value: &Value) -> Result<Self> {
                    let object = value.as_object().ok_or("native agent must be an object")?;
                    let kind = value["type"].as_str().ok_or("native agent needs type")?;
                    let seed = value["seed"]
                        .as_u64()
                        .ok_or("native agent needs resolved u64 seed")?;
                    let allowed: &[&str] = match kind {
                        "random" => &["type", "seed"],
                        "osla" => &["type", "seed", "repetitions_per_action"],
                        "mcts" => &["type", "seed", "config"],
                        _ => return Err("native opponents support random, osla and mcts".into()),
                    };
                    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
                        return Err("unknown native opponent setting".into());
                    }
                    Ok(match kind {
                        "random" => Self::Random(Box::new(RandomAgent::new(seed))),
                        "osla" => {
                            let repetitions = object.get("repetitions_per_action").map_or(Ok(1), |v| {
                                v.as_u64()
                                    .ok_or("repetitions_per_action must be positive u32")
                            })?;
                            let repetitions = u32::try_from(repetitions)
                                .ok()
                                .and_then(NonZeroU32::new)
                                .ok_or("repetitions_per_action must be positive u32")?;
                            Self::Osla(Box::new(SampledOslaAgent::new(
                                <$heuristic>::default(),
                                seed,
                                repetitions,
                            )))
                        }
                        _ => {
                            let configuration = object
                                .get("config")
                                .cloned()
                                .unwrap_or(json!({"schema_version":1}));
                            let mut settings =
                                Settings::from_json(&configuration.to_string()).map_err(error)?;
                            settings.seed = Some(seed);
                            Self::Mcts(Box::new(
                                MctsAgent::new(
                                    settings,
                                    &<$model as tag_rust::search::mcts::NativeEvaluators>::native_evaluators(),
                                )
                                .map_err(error)?,
                            ))
                        }
                    })
                }
                fn restore(saved: &Saved) -> Result<Self> {
                    Ok(match saved {
                        Saved::Random(s) => {
                            Self::Random(Box::new(RandomAgent::from_snapshot(s).map_err(error)?))
                        }
                        Saved::Osla(s) => {
                            Self::Osla(Box::new(SampledOslaAgent::from_snapshot(s).map_err(error)?))
                        }
                        Saved::Mcts(s) => Self::Mcts(Box::new(
                            MctsAgent::from_snapshot(s, &<$model as tag_rust::search::mcts::NativeEvaluators>::native_evaluators())
                                .map_err(error)?,
                        )),
                    })
                }
                fn save(&self) -> Result<Saved> {
                    Ok(match self {
                        Self::Random(agent) => Saved::Random(agent.snapshot()),
                        Self::Osla(agent) => Saved::Osla(agent.snapshot()),
                        Self::Mcts(agent) => Saved::Mcts(Box::new(agent.snapshot().map_err(error)?)),
                    })
                }
                fn choose(&mut self, game: &NativeGame, actor: usize) -> Result<usize> {
                    let model = game.session.model();
                    let player = PlayerId::new(actor);
                    let observation = model.observe(game.session.state(), player).map_err(error)?;
                    let actions = model.available_actions(game.session.state(), player).map_err(error)?;
                    let index = match self {
                        Self::Random(agent) => agent
                            .choose(model, player, &observation, &actions)
                            .map_err(error)?,
                        Self::Osla(agent) => agent
                            .choose(model, player, &observation, &actions)
                            .map_err(error)?,
                        Self::Mcts(agent) => agent
                            .choose(model, player, &observation, &actions)
                            .map_err(error)?,
                    };
                    use tag_rust::learning::LearningGame;
                    let action = actions
                        .get(index)
                        .ok_or("native opponent returned an invalid action index")?;
                    model.action_id(&observation, action).map_err(error)
                }
            }

            /// An evaluation lineup bound to one game position. This is separate from
            /// attached OSLA controllers and does not promise reset/copy or file persistence.
            pub struct NativeOpponents {
                candidate: usize,
                position: Position,
                seats: Vec<Option<Saved>>,
                failed: bool,
            }
            fn native_game(game: &Game) -> PyResult<&NativeGame> {
                match game.engine()? {
                    Engine::$variant(game) => Ok(game),
                    _ => Err(PyValueError::new_err(
                        "native lineup game mismatch",
                    )),
                }
            }
            fn event(game: &NativeGame, actor: usize, id: usize) -> Value {
                $event(game, actor, id)
            }
            impl NativeOpponents {
                fn current(&self, game: &Game) -> PyResult<()> {
                    game.engine()?;
                    if game.token() != self.position {
                        return Err(StalePositionError::new_err(
                            "native lineup source game changed",
                        ));
                    }
                    if self.failed {
                        return Err(PyValueError::new_err("native lineup has failed"));
                    }
                    Ok(())
                }
                /// Restore local agents, execute real native moves, and save their continuing
                /// RNG/evaluator state. Failed decisions retain preceding accepted moves.
                fn run(&mut self, game: &mut NativeGame, limit: Option<u64>, trace: bool) -> (u64, Value) {
                    let mut moves = 0;
                    let mut events = Vec::new();
                    let mut run = || -> Result<(Vec<Option<Saved>>, &'static str)> {
                        let mut agents = self
                            .seats
                            .iter()
                            .map(|s| s.as_ref().map(Active::restore).transpose())
                            .collect::<Result<Vec<_>>>()?;
                        let reason = loop {
                            let actor = match game.session.next_player() {
                                tag_rust::core::Decision::Finished => break "finished",
                                tag_rust::core::Decision::Player(player) => player.index(),
                            };
                            if limit.is_some_and(|limit| moves >= limit) {
                                break "action_limit";
                            }
                            if actor == self.candidate {
                                break "candidate";
                            }
                            if trace && events.len() >= 100_000 {
                                return Err("native action trace exceeds 100000 entries".into());
                            }
                            let id = agents[actor]
                                .as_mut()
                                .ok_or("missing native opponent")?
                                .choose(game, actor)?;
                            game.apply(id, Some(actor)).map_err(error)?;
                            moves += 1;
                            if trace {
                                events.push(event(game, actor, id));
                            }
                        };
                        Ok((
                            agents
                                .iter()
                                .map(|a| a.as_ref().map(Active::save).transpose())
                                .collect::<Result<_>>()?,
                            reason,
                        ))
                    };
                    match run() {
                        Ok((seats, reason)) => {
                            self.seats = seats;
                            (
                                moves,
                                json!({"reason":reason,"moves":moves,"actions":events}),
                            )
                        }
                        Err(error) => {
                            self.failed = true;
                            (
                                moves,
                                json!({"reason":"failure","moves":moves,"actions":events,"error":error}),
                            )
                        }
                    }
                }
            }

            impl NativeOpponents {
                pub(super) fn new(py: Python<'_>, game: &Game, candidate: usize, agents_json: &str) -> PyResult<Self> {
                    let count = native_game(game)?.description.identity.player_count;
                    let values: Vec<Option<Value>> =
                        serde_json::from_str(agents_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
                    if candidate >= count
                        || values.len() != count
                        || values
                            .iter()
                            .enumerate()
                            .any(|(seat, v)| v.is_none() != (seat == candidate))
                    {
                        return Err(PyValueError::new_err(
                            "supply exactly one native agent per non-candidate seat",
                        ));
                    }
                    let seats = py
                        .detach(|| {
                            values
                                .iter()
                                .map(|v| {
                                    v.as_ref()
                                        .map(|v| Active::create(v).and_then(|a| a.save()))
                                        .transpose()
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .map_err(PyValueError::new_err)?;
                    Ok(Self {
                        candidate,
                        position: game.token(),
                        seats,
                        failed: false,
                    })
                }
                /// Run native opponents while Python releases its interpreter lock. A zero
                /// remaining-action limit reports truncation; terminal completion wins ties.
                pub(super) fn advance(
                    &mut self,
                    py: Python<'_>,
                    game: &mut Game,
                    max_actions: Option<u64>,
                    trace: bool,
                ) -> PyResult<String> {
                    self.current(game)?;
                    let available = u64::MAX - game.revision;
                    if max_actions.is_some_and(|n| n > available) {
                        return Err(PyValueError::new_err(
                            "action limit exceeds remaining position revisions",
                        ));
                    }
                    let limit = Some(max_actions.unwrap_or(available));
                    let Engine::$variant(native) = game.engine.as_mut().expect("checked") else {
                        unreachable!()
                    };
                    let (moves, report) = py.detach(|| self.run(native, limit, trace));
                    game.revision += moves;
                    self.position = game.token();
                    Ok(report.to_string())
                }
                /// Apply only the candidate's recommendation against the retained root token.
                pub(super) fn apply_candidate(
                    &mut self,
                    py: Python<'_>,
                    game: &mut Game,
                    id: usize,
                    position: &Position,
                ) -> PyResult<String> {
                    self.current(game)?;
                    if *position != self.position {
                        return Err(StalePositionError::new_err("candidate position changed"));
                    }
                    if game.engine()?.next_player() != Some(self.candidate) {
                        return Err(PyValueError::new_err("candidate does not act next"));
                    }
                    let next = game
                        .revision
                        .checked_add(1)
                        .ok_or_else(|| PyValueError::new_err("position revisions exhausted"))?;
                    let Engine::$variant(native) = game.engine.as_mut().expect("checked") else {
                        unreachable!()
                    };
                    let result = py
                        .detach(|| {
                            native.apply(id, Some(self.candidate)).map_err(error)?;
                            Ok::<_, String>(event(native, self.candidate, id))
                        })
                        .map_err(PyValueError::new_err)?;
                    game.revision = next;
                    self.position = game.token();
                    Ok(result.to_string())
                }
                /// Read current public scores and result labels without advancing dice.
                pub(super) fn state(&self, py: Python<'_>, game: &Game) -> PyResult<String> {
                    // Failure diagnostics may still inspect the last accepted position.
                    game.engine()?;
                    if game.token() != self.position {
                        return Err(StalePositionError::new_err("native lineup source changed"));
                    }
                    let native = native_game(game)?;
                    py.detach(|| -> Result<String> {
                        let model = native.session.model();
                        let scores = (0..model.player_count(native.session.state())).map(|seat|model.player_score(native.session.state(),PlayerId::new(seat)).map_err(error)).collect::<Result<Vec<_>>>()?;
                        let result = native.observe(0,&[]).map_err(error)?;
                        Ok(json!({"scores":scores,"outcomes":result.results.iter().map(|r|*r as i8).collect::<Vec<_>>()}).to_string())
                    }).map_err(PyValueError::new_err)
                }
            }
        }
    };
}
backend!(
    cantstop,
    CantStop,
    CantStopForwardModel,
    CantStopProgress,
    (f64, f64),
    |game: &crate::engine::Game<CantStopForwardModel>, actor: usize, id: usize| {
        json!({"actor":actor,"action_id":id,
            "dice":if id == 0 {game.session.state().position().dice()} else {None}})
    }
);
backend!(
    loveletter,
    LoveLetter,
    LoveLetterForwardModel,
    LoveLetterHeuristic,
    LoveLetterHeuristic,
    |_game: &crate::engine::Game<LoveLetterForwardModel>, actor: usize, id: usize| {
        json!({"actor":actor,"action_id":id})
    }
);

enum Backend {
    CantStop(cantstop::NativeOpponents),
    LoveLetter(loveletter::NativeOpponents),
}
/// Persistent native opponents, bound to a single live game and its position.
#[pyclass(module = "tag_rust._engine")]
pub struct NativeOpponents {
    inner: Backend,
}
macro_rules! dispatch {
    ($this:expr, $method:ident $(, $arg:expr)*) => {
        match &mut $this.inner {
            Backend::CantStop(runtime) => runtime.$method($($arg),*),
            Backend::LoveLetter(runtime) => runtime.$method($($arg),*),
        }
    };
}
#[pymethods]
impl NativeOpponents {
    #[new]
    fn new(py: Python<'_>, game: &Game, candidate: usize, agents_json: &str) -> PyResult<Self> {
        let inner = match game.engine()? {
            Engine::CantStop(_) => Backend::CantStop(cantstop::NativeOpponents::new(
                py,
                game,
                candidate,
                agents_json,
            )?),
            Engine::LoveLetter(_) => Backend::LoveLetter(loveletter::NativeOpponents::new(
                py,
                game,
                candidate,
                agents_json,
            )?),
            _ => {
                return Err(PyValueError::new_err(
                    "native lineups support Can't Stop and Love Letter",
                ));
            }
        };
        Ok(Self { inner })
    }
    #[pyo3(signature = (game, max_actions=None, trace=false))]
    fn advance(
        &mut self,
        py: Python<'_>,
        game: &mut Game,
        max_actions: Option<u64>,
        trace: bool,
    ) -> PyResult<String> {
        dispatch!(self, advance, py, game, max_actions, trace)
    }
    fn apply_candidate(
        &mut self,
        py: Python<'_>,
        game: &mut Game,
        id: usize,
        position: &Position,
    ) -> PyResult<String> {
        dispatch!(self, apply_candidate, py, game, id, position)
    }
    fn state(&self, py: Python<'_>, game: &Game) -> PyResult<String> {
        match &self.inner {
            Backend::CantStop(runtime) => runtime.state(py, game),
            Backend::LoveLetter(runtime) => runtime.state(py, game),
        }
    }
}
