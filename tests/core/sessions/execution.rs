use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use tag_rust::agents::{Agent, FirstAvailableAgent};
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::runner::run_game;
use tag_rust::session::{BuiltinAgent, Controller, Session};

#[derive(Debug, PartialEq)] // Debug: diagnose differences; PartialEq: compare each decision's inputs and selection.
struct RecordedDecision<O, A> {
    actor: PlayerId,
    observation: O,
    actions: Vec<A>,
    selected: usize,
}

struct DecisionLog<O, A> {
    records: Vec<RecordedDecision<O, A>>,
}

struct RecordingAgent<A, O, M> {
    inner: A,
    log: Rc<RefCell<DecisionLog<O, M>>>,
}

impl<F, A> Agent<F> for RecordingAgent<A, F::Observation, F::Action>
where
    F: ForwardModel,
    A: Agent<F>,
    F::Observation: Clone,
    F::Action: Clone,
{
    type Error = A::Error;
    fn rng(&self) -> &AgentRng {
        self.inner.rng()
    } // Shared RNG borrow; observation recording makes no draws.
    fn rng_mut(&mut self) -> &mut AgentRng {
        self.inner.rng_mut()
    } // Exclusive RNG borrow.
    fn choose(
        &mut self,
        model: &F,
        actor: PlayerId,
        observation: &F::Observation,
        actions: &[F::Action],
    ) -> Result<usize, Self::Error> {
        let selected = self.inner.choose(model, actor, observation, actions)?;
        // Record the live runner decision after selection; OSLA's internal branches stay outside this log.
        self.log.borrow_mut().records.push(RecordedDecision {
            actor,
            observation: observation.clone(),
            actions: actions.to_vec(),
            selected,
        });
        Ok(selected)
    }
}

fn matches<F, E>(parameters: &F::Parameters, make_model: impl Fn() -> F, evaluator: E)
where
    F: PublicInformationModel,
    F::State: Clone + Debug + PartialEq,
    F::Action: Clone + Debug + PartialEq,
    F::Observation: Clone + Debug + PartialEq,
    F::Error: Debug,
    E: Copy + Evaluator<F::State>,
    E::Error: Debug,
{
    for configuration in 0..4 {
        let seats = || match configuration {
            0 => vec![
                Controller::first_available(100),
                Controller::first_available(200),
            ],
            1 => vec![
                Controller::osla(evaluator, 100),
                Controller::osla(evaluator, 200),
            ],
            2 => vec![
                Controller::first_available(100),
                Controller::osla(evaluator, 200),
            ],
            _ => vec![
                Controller::osla(evaluator, 100),
                Controller::first_available(200),
            ],
        };
        let mut session = Session::create(make_model(), parameters, 2, seats()).unwrap();
        let mut replay = Session::create(make_model(), parameters, 2, seats()).unwrap();
        let mut trace = Vec::new();
        let mut decision_trace = Vec::new();
        let mut states_before = Vec::new();
        while let Decision::Player(actor) = session.next_player() {
            let position_version = session.position_version();
            let view = session.decision_view(actor).unwrap();
            states_before.push(session.state().clone());
            let record = session.advance_ai(position_version).unwrap();
            let selected = view
                .actions
                .iter()
                .position(|action| *action == record.action)
                .unwrap(); // Locate the accepted move in its original ordered offer.
            decision_trace.push(RecordedDecision {
                actor,
                observation: view.observation,
                actions: view.actions,
                selected,
            });
            assert_eq!(record.actor, actor);
            assert_eq!(record.position_version, position_version + 1);
            assert_eq!(record.decision, session.next_player());
            let repeated = replay.advance_ai(position_version).unwrap();
            assert_eq!(record.action, repeated.action);
            assert_eq!(session.state(), replay.state());
            trace.push(record.action); // Own each accepted move for the test's replay trace.
            assert!(
                trace.len() <= 400,
                "selected finite grid games must terminate"
            );
        }
        let model = make_model();
        let mut state = model.init_state(parameters, 2).unwrap();
        let log = Rc::new(RefCell::new(DecisionLog {
            records: Vec::new(),
        }));
        let mut agents = seats()
            .into_iter()
            .map(|seat| match seat {
                Controller::Ai(agent) => RecordingAgent {
                    inner: agent,
                    log: Rc::clone(&log),
                },
                Controller::Human => unreachable!(),
            })
            .collect::<Vec<_>>();
        run_game(&model, &mut state, &mut agents).unwrap();
        let log = log.borrow();
        assert_eq!(log.records, decision_trace); // Compare every actor, player view, action ordering and selected move.
        let mut recorded_replay = model.init_state(parameters, 2).unwrap();
        for (decision, expected_state) in log.records.iter().zip(&states_before) {
            assert_eq!(&recorded_replay, expected_state);
            assert_eq!(
                model.simulation_state(&decision.observation).unwrap(),
                *expected_state
            );
            // Replay the recorded action through the rules; graphical playback will use this same path.
            model
                .next(
                    &mut recorded_replay,
                    decision.actor,
                    &decision.actions[decision.selected],
                )
                .unwrap();
        }
        assert_eq!(&recorded_replay, session.state());
        assert_eq!(&state, session.state());
        for (seat, agent) in agents.iter().enumerate() {
            let Some(Controller::Ai(interactive)) = session.controller(PlayerId::new(seat)) else {
                panic!("AI seat")
            };
            assert_eq!(
                <BuiltinAgent<E> as Agent<F>>::rng(&agent.inner).initial_seed(),
                <BuiltinAgent<E> as Agent<F>>::rng(interactive).initial_seed()
            );
        }
        for observer in 0..2 {
            let view = session.decision_view(PlayerId::new(observer)).unwrap();
            assert_eq!(view.position_version, trace.len() as u64);
            assert!(view.actions.is_empty());
        }
    }
}

#[test]
fn seeded_mixed_ai_sessions_match_headless_execution_in_both_games() {
    for size in [1, 3, 5] {
        matches(
            &TicTacToeParameters {
                grid_size: size,
                ..Default::default()
            },
            || TicTacToeForwardModel,
            TicTacToeEvaluator,
        );
    }
    for (width, height, winning_length) in [(7, 6, 4), (5, 4, 3), (4, 3, 1)] {
        matches(
            &Connect4Parameters {
                width,
                height,
                winning_length,
                ..Default::default()
            },
            || Connect4ForwardModel,
            Connect4Evaluator,
        );
    }
}

fn human_matches<F>(make_model: impl Fn() -> F, parameters: &F::Parameters)
where
    F: ForwardModel,
    F::State: Debug + PartialEq,
    F::Action: PartialEq,
    F::Error: Debug,
{
    for human_seat in 0..3 {
        // Index two means both seats are human; other cases put a human in either seat.
        let seats = (0..2)
            .map(|seat| {
                if seat == human_seat || human_seat == 2 {
                    Controller::Human
                } else {
                    Controller::Ai(FirstAvailableAgent::new(100 + seat as u64))
                }
            })
            .collect();
        let model = make_model();
        let mut reference = model.init_state(parameters, 2).unwrap();
        let mut session = Session::create(make_model(), parameters, 2, seats).unwrap();
        while let Decision::Player(actor) = session.next_player() {
            let view = session.decision_view(actor).unwrap();
            let action = view.actions.into_iter().next().unwrap(); // Take ownership of the first offered move.
            model.next(&mut reference, actor, &action).unwrap();
            let record = if actor.index() == human_seat || human_seat == 2 {
                session
                    .submit_human(view.position_version, actor, action)
                    .unwrap()
            } else {
                session.advance_ai(view.position_version).unwrap()
            };
            assert_eq!(record.actor, actor);
            assert_eq!(session.state(), &reference);
        }
    }
}

#[test]
fn human_seats_and_ai_seats_advance_one_move_in_both_games() {
    human_matches(|| TicTacToeForwardModel, &TicTacToeParameters::default());
    human_matches(|| Connect4ForwardModel, &Connect4Parameters::default());
}
