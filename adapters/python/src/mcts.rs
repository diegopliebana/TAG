//! A native-only worker owns each search future. No Python objects enter the worker.
//!
//! Native evaluator traits allow thread-local custom implementations. Constructing
//! and polling the future on this worker preserves that API while Python releases
//! its interpreter lock during search. The core itself never creates threads.
use crate::arrays;
use numpy::PyReadonlyArray1;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::task::{Context, Poll, Waker};
use tag_rust::core::{ForwardModel, PlayerId, simulation::SimulationModel};
use tag_rust::learning::{EncodedObservation, LearningGame, LearningLimits};
use tag_rust::search::mcts::{
    BudgetOverrides, EvaluationKind, EvaluatorRegistry, ExternalEvaluations, ExternalResponse,
    MctsAgent, RequestId, Settings, StateValues, SystemClock, ValueObservers,
};

/// Reuse the core parser before Python inspects mode-specific configuration.
/// This returns defaults without resolving a seed or constructing an agent.
#[pyfunction]
pub(super) fn parse_mcts_settings(text: &str) -> PyResult<String> {
    Settings::from_json(text)
        .and_then(|settings| settings.to_json())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
pub(super) fn automatic_search_seed() -> PyResult<u64> {
    tag_rust::search::mcts::automatic_seed()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) struct SearchOptions {
    pub settings: Settings,
    pub observers: ValueObservers,
    pub payloads: Vec<String>,
    pub values_name: String,
    pub policy_name: String,
    pub max_rows: usize,
    pub max_bytes: usize,
}
struct Request {
    id: RequestId,
    kind: EvaluationKind,
    policy_node: Option<usize>,
    root: usize,
    actor: usize,
    rows: Vec<EncodedObservation>,
    actions: Vec<usize>,
}
enum Status {
    Waiting(Option<Request>),
    Yielded,
    Finished(serde_json::Value),
}
enum Command {
    Advance,
    Validate(RequestId, ExternalResponse),
    Respond(RequestId, ExternalResponse),
    Close,
}
enum Event {
    Status(Result<Status, String>),
    Validated(Result<(), String>),
}
struct Worker {
    commands: mpsc::Sender<Command>,
    events: mpsc::Receiver<Event>,
}

#[pyclass(module = "tag_rust._engine")]
pub struct MctsSearch {
    worker: Mutex<Worker>,
    cancelled: Arc<AtomicBool>,
    settings: String,
}

pub(super) fn spawn<F>(
    model: F,
    observation: F::Observation,
    actor: PlayerId,
    actions: Vec<F::Action>,
    mut limits: LearningLimits,
    mut options: SearchOptions,
    registry: impl FnOnce() -> EvaluatorRegistry<F> + Send + 'static,
) -> Result<MctsSearch, String>
where
    F: LearningGame + SimulationModel + Send + 'static,
    F::State: Clone,
    F::Observation: Send + 'static,
    F::Action: Clone + PartialEq + Send + 'static,
    F::Error: std::fmt::Display,
{
    options.max_bytes = options
        .max_bytes
        .min(options.settings.limits.max_request_bytes);
    limits.max_bytes = limits.max_bytes.min(options.max_bytes);
    let (commands, receiver) = mpsc::channel();
    let (sender, events) = mpsc::channel();
    let (ready_sender, ready) = mpsc::sync_channel(1);
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = Arc::clone(&cancelled);
    std::thread::Builder::new()
        .name("tag-mcts".into())
        .spawn(move || {
            let initialize = || -> Result<_, String> {
                let exchange = ExternalEvaluations::new(actor, options.observers)
                    .map_err(|e| e.to_string())?;
                let mut registry = registry();
                exchange
                    .register(&mut registry, &options.values_name, &options.policy_name)
                    .map_err(|e| e.to_string())?;
                let agent =
                    MctsAgent::new(options.settings, &registry).map_err(|e| e.to_string())?;
                let settings = agent.settings().to_json().map_err(|e| e.to_string())?;
                Ok((exchange, agent, settings))
            };
            let (exchange, mut agent, settings) = match initialize() {
                Ok(value) => value,
                Err(error) => {
                    let _ = ready_sender.send(Err(error));
                    return;
                }
            };
            if ready_sender.send(Ok(settings)).is_err() {
                return;
            }
            let clock = SystemClock::default();
            let probe = || cancellation.load(Ordering::Relaxed);
            let overrides = BudgetOverrides::default();
            let mut future = Box::pin(agent.search_controlled_async(
                &model,
                actor,
                &observation,
                &actions,
                &overrides,
                &clock,
                &probe,
            ));
            let mut finished = false;
            while let Ok(command) = receiver.recv() {
                let response = match command {
                    Command::Close => {
                        exchange.cancel();
                        let _ = sender.send(Event::Validated(Ok(())));
                        break;
                    }
                    Command::Validate(id, response) => Event::Validated(
                        exchange.validate(id, &response).map_err(|e| e.to_string()),
                    ),
                    command => {
                        let deliver = match command {
                            Command::Respond(id, response) => {
                                exchange.respond(id, response).map_err(|e| e.to_string())
                            }
                            _ => Ok(()),
                        };
                        let status = deliver.and_then(|()| {
                            if finished {
                                return Err("search has already finished".into());
                            }
                            let mut task = Context::from_waker(Waker::noop());
                            match future.as_mut().poll(&mut task) {
                                Poll::Ready(result) => {
                                    finished = true;
                                    Ok(Status::Finished(search_report(
                                        &model,
                                        &observation,
                                        &actions,
                                        result.map_err(|e| e.to_string())?,
                                    )?))
                                }
                                Poll::Pending => {
                                    if !exchange.is_waiting() {
                                        return Ok(Status::Yielded);
                                    }
                                    let request = exchange
                                        .take_request()
                                        .map(|request| {
                                            encode_request(
                                                &model,
                                                request,
                                                &options.payloads,
                                                limits,
                                                options.max_rows,
                                                options.max_bytes,
                                            )
                                        })
                                        .transpose()?;
                                    Ok(Status::Waiting(request))
                                }
                            }
                        });
                        Event::Status(status)
                    }
                };
                if sender.send(response).is_err() {
                    break;
                }
            }
        })
        .map_err(|e| e.to_string())?;
    let settings = ready.recv().map_err(|e| e.to_string())??;
    Ok(MctsSearch {
        worker: Mutex::new(Worker { commands, events }),
        cancelled,
        settings,
    })
}

impl MctsSearch {
    fn command(&self, command: Command) -> PyResult<Event> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| PyRuntimeError::new_err("search worker poisoned"))?;
        worker
            .commands
            .send(command)
            .map_err(|_| PyRuntimeError::new_err("search worker closed"))?;
        worker
            .events
            .recv()
            .map_err(|_| PyRuntimeError::new_err("search worker closed"))
    }
    fn status<'py>(&self, py: Python<'py>, event: Event) -> PyResult<Bound<'py, PyDict>> {
        let Event::Status(status) = event else {
            return Err(PyRuntimeError::new_err("unexpected search event"));
        };
        let out = PyDict::new(py);
        match status.map_err(PyValueError::new_err)? {
            Status::Yielded => {
                out.set_item("reason", "yielded")?;
            }
            Status::Finished(report) => {
                out.set_item("reason", "finished")?;
                out.set_item("report_json", report.to_string())?;
            }
            Status::Waiting(request) => {
                out.set_item("reason", "evaluation")?;
                if let Some(request) = request {
                    let r = PyDict::new(py);
                    r.set_item("search_id", request.id.search)?;
                    r.set_item("sequence", request.id.sequence)?;
                    r.set_item(
                        "kind",
                        if request.kind == EvaluationKind::Values {
                            "values"
                        } else {
                            "policy"
                        },
                    )?;
                    r.set_item("root_player", request.root)?;
                    r.set_item("policy_node", request.policy_node)?;
                    r.set_item("deciding_player", request.actor)?;
                    r.set_item("action_ids", request.actions)?;
                    let rows = request
                        .rows
                        .into_iter()
                        .map(|row| arrays::observation(py, row).map(Bound::unbind))
                        .collect::<PyResult<Vec<_>>>()?;
                    r.set_item("observations", rows)?;
                    out.set_item("request", r)?;
                }
            }
        }
        Ok(out)
    }
}

/// Convert a simulation's permitted observations before transferring them to
/// Python. Byte limits cover numeric arrays, rather than evaluator-private memory.
fn encode_request<F: LearningGame>(
    model: &F,
    request: tag_rust::search::mcts::ExternalRequest<F>,
    payloads: &[String],
    limits: LearningLimits,
    max_rows: usize,
    max_bytes: usize,
) -> Result<Request, String> {
    if request.observations.is_empty() || request.observations.len() > max_rows {
        return Err("MCTS request exceeds max_rows or has no observations".into());
    }
    let names: Vec<_> = payloads.iter().map(String::as_str).collect();
    let actions = request
        .actions
        .iter()
        .map(|action| {
            model
                .action_id(&request.observations[0].1, action)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::new();
    let mut bytes = 0usize;
    for (_, observation) in request.observations {
        let row = model
            .encode(&observation, &names, limits)
            .map_err(|e| e.to_string())?;
        let count = row.tensors.iter().try_fold(0usize, |n, tensor| {
            n.checked_add(tensor.values.len())
                .ok_or("request size overflow")
        })?;
        // Masks and action IDs need at most nine bytes per possible action.
        // The fixed allowance covers progress and player-context numeric fields.
        bytes = bytes
            .checked_add(count.checked_mul(4).ok_or("request size overflow")?)
            .and_then(|n| n.checked_add(row.action_mask.len().checked_mul(9)?))
            .and_then(|n| n.checked_add(128usize.checked_add(row.results.len())?))
            .ok_or("request size overflow")?;
        if bytes > max_bytes {
            return Err("MCTS request exceeds max_bytes".into());
        }
        rows.push(row);
    }
    Ok(Request {
        id: request.id,
        kind: request.kind,
        policy_node: request.policy_node,
        root: request.root_player.index(),
        actor: request.deciding_player.index(),
        rows,
        actions,
    })
}

/// Report permanent action IDs so a caller can apply the recommendation through
/// the ordinary version-checked game API. Search itself never commits that move.
fn search_report<F: LearningGame>(
    model: &F,
    observation: &F::Observation,
    actions: &[F::Action],
    report: tag_rust::search::mcts::SearchReport<F::Action>,
) -> Result<serde_json::Value, String> {
    let rows = report.root_actions.iter().map(|row| {
        let id = model.action_id(observation, &row.action).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"action_id":id,"visits":row.visits,"mean_values":row.mean_values}))
    }).collect::<Result<Vec<_>, String>>()?;
    let settings = serde_json::from_str::<serde_json::Value>(
        &report
            .effective_settings
            .to_json()
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "schema_version":1,"algorithm_version":1,
        "selected_index":report.selected_index,
        "action_id":model.action_id(observation, &actions[report.selected_index]).map_err(|e| e.to_string())?,
        "root_actions":rows,"fallback":report.fallback,"elapsed_ms":report.elapsed_ms,
        "nodes":report.nodes,"edges":report.edges,"settings":settings,
        "stop_reasons":report.stop_reasons.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>(),
        "counters":{"completed_iterations":report.counters.completed_iterations,
            "forward_model_calls":report.counters.forward_model_calls,
            "state_copies":report.counters.state_copies,
            "state_evaluations":report.counters.state_evaluations,
            "action_evaluations":report.counters.action_evaluations}
    }))
}

#[pymethods]
impl MctsSearch {
    #[getter]
    fn settings_json(&self) -> &str {
        &self.settings
    }
    fn advance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let event = py.detach(|| self.command(Command::Advance))?;
        self.status(py, event)
    }
    #[pyo3(signature = (search, sequence, kind, values, scalar_player=None, validate_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn respond<'py>(
        &self,
        py: Python<'py>,
        search: u64,
        sequence: u64,
        kind: &str,
        values: PyReadonlyArray1<'py, f64>,
        scalar_player: Option<usize>,
        validate_only: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let values = values.as_array().iter().copied().collect::<Vec<_>>();
        let response = match (kind, scalar_player) {
            ("values", None) => ExternalResponse::Values(StateValues::Full(values)),
            ("values", Some(player)) if values.len() == 1 => {
                ExternalResponse::Values(StateValues::Scalar {
                    player: PlayerId::new(player),
                    value: values[0],
                })
            }
            ("policy", None) => ExternalResponse::Policy(values),
            _ => {
                return Err(PyValueError::new_err(
                    "invalid response kind or scalar layout",
                ));
            }
        };
        let id = RequestId { search, sequence };
        let command = if validate_only {
            Command::Validate(id, response)
        } else {
            Command::Respond(id, response)
        };
        let event = py.detach(|| self.command(command))?;
        if validate_only {
            let Event::Validated(result) = event else {
                return Err(PyRuntimeError::new_err("unexpected validation event"));
            };
            result.map_err(PyValueError::new_err)?;
            Ok(PyDict::new(py))
        } else {
            self.status(py, event)
        }
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.cancelled.store(true, Ordering::Relaxed);
        py.detach(|| {
            let worker = self
                .worker
                .lock()
                .map_err(|_| PyRuntimeError::new_err("search worker poisoned"))?;
            if worker.commands.send(Command::Close).is_ok() {
                // A disconnected worker is already closed. A live worker sends
                // its acknowledgement before releasing the search continuation.
                let _ = worker.events.recv();
            }
            Ok(())
        })
    }
}
impl Drop for MctsSearch {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Ok(worker) = self.worker.lock() {
            let _ = worker.commands.send(Command::Close);
        }
    }
}

impl crate::engine::Engine {
    pub(super) fn start_mcts(
        &self,
        options: crate::mcts::SearchOptions,
    ) -> Result<crate::mcts::MctsSearch, String> {
        macro_rules! start {
            ($game:expr, $model:ty) => {{
                let game = $game;
                let actor = match game.session.next_player() {
                    tag_rust::core::Decision::Player(actor) => actor,
                    tag_rust::core::Decision::Finished => {
                        return Err("cannot search a finished game".into());
                    }
                };
                let model = game.session.model().clone();
                let observation = model
                    .observe(game.session.state(), actor)
                    .map_err(|e| e.to_string())?;
                let actions = model
                    .available_actions(game.session.state(), actor)
                    .map_err(|e| e.to_string())?;
                crate::mcts::spawn(
                    model,
                    observation,
                    actor,
                    actions,
                    game.limits,
                    options,
                    <$model as tag_rust::search::mcts::NativeEvaluators>::native_evaluators,
                )
            }};
        }
        match self {
            Self::LoveLetter(game) => {
                if options
                    .payloads
                    .iter()
                    .any(|name| name == "last_private_history")
                {
                    return Err("Love Letter MCTS cannot supply previous-round private history for hypothetical opponents".into());
                }
                if options.observers != ValueObservers::DecidingPlayer
                    || options.settings.value_completion
                        != tag_rust::search::mcts::ValueCompletion::RequireFullVector
                {
                    return Err("Love Letter MCTS requires deciding_player observations and require_full_vector completion".into());
                }
                start!(game, tag_rust::games::loveletter::LoveLetterForwardModel)
            }
            Self::TicTacToe(game) => {
                start!(game, tag_rust::games::tictactoe::TicTacToeForwardModel)
            }
            Self::Connect4(game) => start!(game, tag_rust::games::connect4::Connect4ForwardModel),
            Self::CantStop(game) => {
                if options.settings.state_evaluator.name == options.values_name
                    && (options.observers != ValueObservers::DecidingPlayer
                        || options.settings.value_completion
                            != tag_rust::search::mcts::ValueCompletion::RequireFullVector)
                {
                    return Err("Can't Stop external values require deciding_player observations and require_full_vector completion".into());
                }
                start!(game, tag_rust::games::cantstop::CantStopForwardModel)
            }
        }
    }
}
