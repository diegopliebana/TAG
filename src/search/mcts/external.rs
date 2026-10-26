//! External evaluation retains the search future while an application owns its request.
use super::*;
use crate::core::{Decision, ForwardModel, PlayerId};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::task::{Context, Poll, Waker};
use std::{future::Future, pin::Pin};

static NEXT_SEARCH: AtomicU64 = AtomicU64::new(1);

/// Select whose observation supplies a scalar value. AllPlayers requests one row
/// per player; RootPlayer and DecidingPlayer require the configured completion rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueObservers {
    AllPlayers,
    RootPlayer,
    DecidingPlayer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationKind {
    Values,
    Policy,
}

/// Identities must match exactly on response. A fresh exchange identifies one decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestId {
    pub search: u64,
    pub sequence: u64,
}

/// Owned player-visible observations, with explicit action order for policy requests.
/// Encoding and model identity belong to the adapter; no full state crosses this API.
pub struct ExternalRequest<F: ForwardModel> {
    pub id: RequestId,
    pub kind: EvaluationKind,
    /// Present for tree-policy requests; node zero is the real decision's root.
    pub policy_node: Option<usize>,
    pub root_player: PlayerId,
    pub deciding_player: PlayerId,
    pub player_count: usize,
    pub observations: Vec<(PlayerId, F::Observation)>,
    pub actions: Vec<F::Action>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExternalResponse {
    Values(StateValues),
    Policy(Vec<f64>),
}

struct Waiting {
    id: RequestId,
    kind: EvaluationKind,
    players: usize,
    actions: usize,
    completion: ValueCompletion,
    scalar_player: Option<PlayerId>,
    response: Option<ExternalResponse>,
}
struct Exchange<F: ForwardModel> {
    search: u64,
    sequence: u64,
    request: Option<ExternalRequest<F>>,
    waiting: Option<Waiting>,
    cancelled: bool,
    waker: Option<Waker>,
}

/// One decision's request/response exchange. Clones share identity and pending work.
/// It does not own a tree, spawn a thread, or change the real game.
pub struct ExternalEvaluations<F: ForwardModel> {
    shared: Arc<Mutex<Exchange<F>>>,
    root: PlayerId,
    observers: ValueObservers,
}
impl<F: ForwardModel> Clone for ExternalEvaluations<F> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            root: self.root,
            observers: self.observers,
        }
    }
}
impl<F> ExternalEvaluations<F>
where
    F: ForwardModel + 'static,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    pub fn new(root: PlayerId, observers: ValueObservers) -> Result<Self> {
        let search = NEXT_SEARCH
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Overflow)?;
        Ok(Self {
            shared: Arc::new(Mutex::new(Exchange {
                search,
                sequence: 0,
                request: None,
                waiting: None,
                cancelled: false,
                waker: None,
            })),
            root,
            observers,
        })
    }

    /// Register explicit names for state values and policy probabilities. Settings
    /// select them independently; native evaluator registrations remain available.
    pub fn register(
        &self,
        registry: &mut EvaluatorRegistry<F>,
        values: &str,
        policy: &str,
    ) -> Result<()> {
        let state = self.clone();
        registry.register_state(values, move |parameters| {
            if !parameters.is_empty() {
                return Err(Error::Invalid(
                    "external bindings take no core parameters".into(),
                ));
            }
            Ok(Box::new(state.clone()))
        })?;
        let action = self.clone();
        registry.register_action(policy, move |parameters| {
            if !parameters.is_empty() {
                return Err(Error::Invalid(
                    "external bindings take no core parameters".into(),
                ));
            }
            Ok(Box::new(action.clone()))
        })
    }

    /// Transfer an owned request to its caller, at most once. Its response remains
    /// pending until validated or cancelled, even after the observation is taken.
    pub fn take_request(&self) -> Option<ExternalRequest<F>> {
        self.shared
            .lock()
            .expect("evaluation exchange poisoned")
            .request
            .take()
    }
    pub fn is_waiting(&self) -> bool {
        self.shared
            .lock()
            .expect("evaluation exchange poisoned")
            .waiting
            .is_some()
    }

    pub fn validate(&self, id: RequestId, response: &ExternalResponse) -> Result<()> {
        let exchange = self.shared.lock().expect("evaluation exchange poisoned");
        let waiting = exchange
            .waiting
            .as_ref()
            .ok_or_else(|| Error::Evaluation("no pending request".into()))?;
        if exchange.cancelled || waiting.id != id || waiting.response.is_some() {
            return Err(Error::Evaluation(
                "stale, duplicate or foreign request identity".into(),
            ));
        }
        match (waiting.kind, response) {
            (EvaluationKind::Values, ExternalResponse::Values(values)) => {
                if let StateValues::Scalar { player, .. } = values
                    && waiting.scalar_player != Some(*player)
                {
                    return Err(Error::Evaluation(
                        "scalar perspective differs from the requested observer".into(),
                    ));
                }
                complete_values(values.clone(), waiting.players, waiting.completion)?;
            }
            (EvaluationKind::Policy, ExternalResponse::Policy(values)) => {
                if values.len() != waiting.actions {
                    return Err(Error::Evaluation("policy action count mismatch".into()));
                }
                super::selection::probabilities(values)?;
            }
            _ => return Err(Error::Evaluation("response kind mismatch".into())),
        }
        Ok(())
    }

    /// A rejected response leaves the request available for correction. Applications
    /// must serialize access to one exchange while validating and submitting a group.
    pub fn respond(&self, id: RequestId, response: ExternalResponse) -> Result<()> {
        self.validate(id, &response)?;
        let waker = {
            let mut exchange = self.shared.lock().expect("evaluation exchange poisoned");
            let waiting = exchange
                .waiting
                .as_mut()
                .ok_or_else(|| Error::Evaluation("request was cancelled".into()))?;
            if waiting.id != id || waiting.response.is_some() {
                return Err(Error::Evaluation("request changed during response".into()));
            }
            waiting.response = Some(response);
            exchange.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    /// Invalidate the exchange and wake a waiting evaluation. To cancel a search
    /// suspended between iterations, also drop its future or set its cancellation probe.
    pub fn cancel(&self) {
        let waker = {
            let mut exchange = self.shared.lock().expect("evaluation exchange poisoned");
            exchange.cancelled = true;
            exchange.request = None;
            exchange.waiting = None;
            exchange.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn request<'a, 'model>(
        &self,
        state: &F::State,
        actor: PlayerId,
        actions: &[F::Action],
        kind: EvaluationKind,
        context: &'a SimulationContext<'model, F>,
    ) -> Result<Reply<'a, 'model, F>> {
        context.check_interrupt()?;
        let players = context.model().player_count(state);
        let observers: Vec<_> = match (kind, self.observers) {
            (EvaluationKind::Policy, _) => vec![actor],
            (_, ValueObservers::AllPlayers) => (0..players).map(PlayerId::new).collect(),
            (_, ValueObservers::RootPlayer) => vec![self.root],
            (_, ValueObservers::DecidingPlayer) => vec![actor],
        };
        let observations = observers
            .into_iter()
            .map(|player| {
                Ok((
                    player,
                    context
                        .model()
                        .observe(state, player)
                        .map_err(|e| Error::Game(e.to_string()))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut exchange = self.shared.lock().expect("evaluation exchange poisoned");
        if exchange.cancelled {
            return Err(Error::Cancelled);
        }
        if exchange.waiting.is_some() {
            return Err(Error::Evaluation(
                "tree already has a pending evaluation".into(),
            ));
        }
        exchange.sequence = exchange.sequence.checked_add(1).ok_or(Error::Overflow)?;
        let id = RequestId {
            search: exchange.search,
            sequence: exchange.sequence,
        };
        exchange.waiting = Some(Waiting {
            id,
            kind,
            players,
            actions: actions.len(),
            completion: context.value_completion(),
            scalar_player: if observations.len() == 1 {
                Some(observations[0].0)
            } else {
                None
            },
            response: None,
        });
        exchange.request = Some(ExternalRequest {
            id,
            kind,
            policy_node: if kind == EvaluationKind::Policy {
                context.policy_node
            } else {
                None
            },
            root_player: self.root,
            deciding_player: actor,
            player_count: players,
            observations,
            actions: actions.to_vec(),
        });
        Ok(Reply {
            shared: Arc::clone(&self.shared),
            id,
            context,
        })
    }
}

struct Reply<'a, 'model, F: ForwardModel> {
    shared: Arc<Mutex<Exchange<F>>>,
    id: RequestId,
    context: &'a SimulationContext<'model, F>,
}
impl<F: ForwardModel> Future for Reply<'_, '_, F> {
    type Output = Result<ExternalResponse>;
    fn poll(self: Pin<&mut Self>, task: &mut Context<'_>) -> Poll<Self::Output> {
        if let Err(error) = self.context.check_interrupt() {
            return Poll::Ready(Err(error));
        }
        let mut exchange = self.shared.lock().expect("evaluation exchange poisoned");
        if exchange.cancelled {
            return Poll::Ready(Err(Error::Cancelled));
        }
        let Some(waiting) = exchange.waiting.as_mut() else {
            return Poll::Ready(Err(Error::Evaluation("pending request disappeared".into())));
        };
        if waiting.id != self.id {
            return Poll::Ready(Err(Error::Evaluation("pending request changed".into())));
        }
        if let Some(response) = waiting.response.take() {
            exchange.waiting = None;
            exchange.request = None;
            Poll::Ready(Ok(response))
        } else {
            exchange.waker = Some(task.waker().clone());
            Poll::Pending
        }
    }
}
impl<F: ForwardModel> Drop for Reply<'_, '_, F> {
    fn drop(&mut self) {
        let mut exchange = self.shared.lock().expect("evaluation exchange poisoned");
        if exchange
            .waiting
            .as_ref()
            .is_some_and(|waiting| waiting.id == self.id)
        {
            exchange.waiting = None;
            exchange.request = None;
            exchange.waker = None;
        }
    }
}

impl<F> StateEvaluator<F> for ExternalEvaluations<F>
where
    F: ForwardModel + 'static,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    fn evaluate(&mut self, _: &F::State, _: &mut SimulationContext<'_, F>) -> Result<StateValues> {
        Err(Error::EvaluationRequired)
    }
    fn evaluate_async<'a, 'model>(
        &'a mut self,
        state: &'a F::State,
        context: &'a mut SimulationContext<'model, F>,
    ) -> EvaluationFuture<'a, StateValues> {
        Box::pin(async move {
            let actor = match context.model().next_player(state) {
                Decision::Player(actor) => actor,
                Decision::Finished => {
                    return Err(Error::Evaluation(
                        "terminal state does not need external evaluation".into(),
                    ));
                }
            };
            match self
                .request(state, actor, &[], EvaluationKind::Values, context)?
                .await?
            {
                ExternalResponse::Values(values) => Ok(values),
                _ => Err(Error::Evaluation("expected state values".into())),
            }
        })
    }
}
impl<F> ActionEvaluator<F> for ExternalEvaluations<F>
where
    F: ForwardModel + 'static,
    F::State: Clone,
    F::Action: Clone,
    F::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        _: &F::State,
        _: PlayerId,
        _: &[F::Action],
        _: &mut SimulationContext<'_, F>,
    ) -> Result<Vec<f64>> {
        Err(Error::EvaluationRequired)
    }
    fn evaluate_async<'a, 'model>(
        &'a mut self,
        state: &'a F::State,
        actor: PlayerId,
        actions: &'a [F::Action],
        context: &'a mut SimulationContext<'model, F>,
    ) -> EvaluationFuture<'a, Vec<f64>> {
        Box::pin(async move {
            match self
                .request(state, actor, actions, EvaluationKind::Policy, context)?
                .await?
            {
                ExternalResponse::Policy(values) => super::selection::probabilities(&values),
                _ => Err(Error::Evaluation("expected action probabilities".into())),
            }
        })
    }
}
