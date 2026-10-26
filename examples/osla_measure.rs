//! Count planning work separately from release-mode full-match timings.
use std::cell::Cell;
use std::rc::Rc;
use std::time::Instant;
use tag_rust::agents::OslaAgent;
use tag_rust::core::evaluation::Evaluator;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult, PublicInformationModel};
use tag_rust::games::connect4::{Connect4Evaluator, Connect4ForwardModel, Connect4Parameters};
use tag_rust::games::tictactoe::{TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::runner::run_game;

type Failure = Box<dyn std::error::Error>;
const SEEDS: [u64; 2] = [100, 200];
const BATCHES: usize = 5;
const WARMUP: usize = 50;

#[derive(Debug, Default, PartialEq, Eq)] // Default: zero counts; PartialEq/Eq: exact work-count comparison.
struct Work {
    views: Cell<usize>,
    decisions: Cell<usize>,
    candidates: Cell<usize>,
    branches: Cell<usize>,
    simulated_moves: Cell<usize>,
    evaluations: Cell<usize>,
    real_moves: Cell<usize>,
}
struct CountedState<S> {
    state: S,
    work: Rc<Work>,
    branch: bool,
}
impl<S: Clone> Clone for CountedState<S> {
    fn clone(&self) -> Self {
        // Clone the owned game state and record this candidate's independent copy.
        self.work.branches.set(self.work.branches.get() + 1);
        Self {
            state: self.state.clone(),
            work: Rc::clone(&self.work),
            branch: true,
        }
    }
}
struct CountedModel<F> {
    model: F,
    work: Rc<Work>,
}
impl<F: ForwardModel> ForwardModel for CountedModel<F> {
    type State = CountedState<F::State>;
    type Action = F::Action;
    type Observation = F::Observation;
    type Parameters = F::Parameters;
    type Error = F::Error;
    fn init_state(
        &self,
        parameters: &F::Parameters,
        count: usize,
    ) -> Result<Self::State, Self::Error> {
        Ok(CountedState {
            state: self.model.init_state(parameters, count)?,
            work: Rc::clone(&self.work),
            branch: false,
        })
    }
    fn player_count(&self, state: &Self::State) -> usize {
        self.model.player_count(&state.state)
    }
    fn next_player(&self, state: &Self::State) -> Decision {
        self.model.next_player(&state.state)
    }
    fn observe(
        &self,
        state: &Self::State,
        player: PlayerId,
    ) -> Result<Self::Observation, Self::Error> {
        let view = self.model.observe(&state.state, player)?;
        self.work.views.set(self.work.views.get() + 1);
        Ok(view)
    }
    fn offered_actions(&self, view: &Self::Observation) -> Vec<Self::Action> {
        let actions = self.model.offered_actions(view);
        self.work
            .candidates
            .set(self.work.candidates.get() + actions.len());
        actions
    }
    fn available_actions(
        &self,
        state: &Self::State,
        player: PlayerId,
    ) -> Result<Vec<Self::Action>, Self::Error> {
        self.model.available_actions(&state.state, player)
    }
    fn next(
        &self,
        state: &mut Self::State,
        player: PlayerId,
        action: &Self::Action,
    ) -> Result<(), Self::Error> {
        self.model.next(&mut state.state, player, action)?;
        let count = if state.branch {
            &self.work.simulated_moves
        } else {
            &self.work.real_moves
        };
        count.set(count.get() + 1);
        Ok(())
    }
    fn player_result(
        &self,
        state: &Self::State,
        player: PlayerId,
    ) -> Result<PlayerResult, Self::Error> {
        self.model.player_result(&state.state, player)
    }
    fn player_score(&self, state: &Self::State, player: PlayerId) -> Result<f64, Self::Error> {
        self.model.player_score(&state.state, player)
    }
}
impl<F: PublicInformationModel> PublicInformationModel for CountedModel<F> {
    fn simulation_state(&self, view: &Self::Observation) -> Result<Self::State, Self::Error> {
        let state = self.model.simulation_state(view)?;
        self.work.decisions.set(self.work.decisions.get() + 1);
        Ok(CountedState {
            state,
            work: Rc::clone(&self.work),
            branch: false,
        })
    }
}
struct CountedEvaluator<E>(E);
impl<S, E: Evaluator<S>> Evaluator<CountedState<S>> for CountedEvaluator<E> {
    type Error = E::Error;
    fn evaluate(&mut self, state: &CountedState<S>, player: PlayerId) -> Result<f64, Self::Error> {
        state.work.evaluations.set(state.work.evaluations.get() + 1);
        self.0.evaluate(&state.state, player)
    }
}

/// Count one completed match using wrappers around game rules.
///
/// # Arguments
/// The name/configuration are borrowed; model and evaluator factory are consumed.
///
/// # Returns
/// `Ok(())` after checking and printing exact operation counts.
///
/// # Errors
/// Game, planning or evaluator errors are returned to the caller.
fn count<F, E>(
    name: &str,
    model: F,
    parameters: &F::Parameters,
    evaluator: impl Fn() -> E,
) -> Result<(), Failure>
where
    F: PublicInformationModel,
    F::State: Clone,
    F::Error: std::error::Error + 'static,
    E: Evaluator<F::State>,
    E::Error: std::error::Error + 'static,
{
    let model = CountedModel {
        model,
        work: Rc::new(Work::default()),
    };
    let mut state = model.init_state(parameters, 2)?;
    let mut agents = [
        OslaAgent::new(CountedEvaluator(evaluator()), SEEDS[0]),
        OslaAgent::new(CountedEvaluator(evaluator()), SEEDS[1]),
    ];
    run_game(&model, &mut state, &mut agents)?;
    let work = &model.work;
    let decisions = work.decisions.get();
    let candidates = work.candidates.get();
    assert_eq!(decisions, work.real_moves.get());
    assert_eq!(decisions, work.views.get());
    assert_eq!(candidates, work.branches.get());
    assert_eq!(candidates, work.simulated_moves.get());
    assert_eq!(candidates, work.evaluations.get());
    assert_eq!(model.next_player(&state), Decision::Finished);
    println!(
        "{name}: decisions={decisions}, candidates={candidates}, observations={}, base_copies={decisions}, branch_copies={}, simulated_transitions={}, evaluations={}, real_moves={}",
        work.views.get(),
        work.branches.get(),
        work.simulated_moves.get(),
        work.evaluations.get(),
        work.real_moves.get()
    );
    // Both current games clone one board per observation and per simulation state.
    println!(
        "  board_snapshots={}, result={:?}",
        2 * decisions + candidates,
        model.player_result(&state, PlayerId::new(0))?
    );
    Ok(())
}

/// Time warmed-up full matches through the ordinary model and runner.
///
/// # Arguments
/// Model/configuration/name are passed by a shared (read-only) reference, factory is consumed, repetitions copied.
///
/// # Returns
/// Prints each batch and minimum/median/maximum microseconds per match.
/// The timed region includes construction, initialization, execution and destruction.
///
/// # Errors
/// Initialization or runner failures are returned to the caller.
fn measure<F, E>(
    name: &str,
    model: &F,
    parameters: &F::Parameters,
    evaluator: impl Fn() -> E,
    repetitions: usize,
) -> Result<(), Failure>
where
    F: PublicInformationModel,
    F::State: Clone,
    F::Error: std::error::Error + 'static,
    E: Evaluator<F::State>,
    E::Error: std::error::Error + 'static,
{
    let play = || -> Result<(), Failure> {
        let mut state = model.init_state(parameters, 2)?;
        let mut agents = [
            OslaAgent::new(evaluator(), SEEDS[0]),
            OslaAgent::new(evaluator(), SEEDS[1]),
        ];
        run_game(model, &mut state, &mut agents)?;
        std::hint::black_box(state);
        std::hint::black_box(agents);
        Ok(())
    };
    for _ in 0..WARMUP {
        play()?;
    }
    let mut samples = Vec::with_capacity(BATCHES);
    for batch in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..repetitions {
            play()?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let per_match = elapsed * 1_000_000.0 / repetitions as f64;
        println!(
            "  {name} batch {}: {:.3} ms total, {:.3} us/match",
            batch + 1,
            elapsed * 1000.0,
            per_match
        );
        samples.push(per_match);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "  {name} us/match min={:.3}, median={:.3}, max={:.3}",
        samples[0],
        samples[BATCHES / 2],
        samples[BATCHES - 1]
    );
    Ok(())
}

fn main() -> Result<(), Failure> {
    let repetitions = std::env::args()
        .nth(1)
        .map(|text| {
            // Parse the optional number of matches in each batch.
            text.parse::<usize>()
        })
        .transpose()?
        .unwrap_or(500);
    if !(1..=10_000).contains(&repetitions) {
        return Err("repetitions must be between 1 and 10000".into());
    }
    println!(
        "OSLA seeds={SEEDS:?}, batches={BATCHES}, repetitions={repetitions}, warmup={WARMUP}, debug_assertions={}",
        cfg!(debug_assertions)
    );
    for size in [3, 5] {
        let name = format!("TicTacToe {size}x{size}");
        let parameters = TicTacToeParameters {
            grid_size: size,
            ..Default::default()
        };
        count(&name, TicTacToeForwardModel, &parameters, || {
            TicTacToeEvaluator
        })?;
        measure(
            &name,
            &TicTacToeForwardModel,
            &parameters,
            || TicTacToeEvaluator,
            repetitions,
        )?;
    }
    for (width, height, target) in [(7, 6, 4), (10, 8, 5)] {
        let name = format!("Connect4 {width}x{height} target={target}");
        let parameters = Connect4Parameters {
            width,
            height,
            winning_length: target,
            ..Default::default()
        };
        count(&name, Connect4ForwardModel, &parameters, || {
            Connect4Evaluator
        })?;
        measure(
            &name,
            &Connect4ForwardModel,
            &parameters,
            || Connect4Evaluator,
            repetitions,
        )?;
    }
    Ok(())
}
