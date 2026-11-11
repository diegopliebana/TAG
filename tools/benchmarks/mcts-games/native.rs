//! Full MCTS matches with tool-only API sampling; no rule implementation is replaced.
use serde_json::json;
use std::{cell::RefCell, io::Write, marker::PhantomData, rc::Rc, time::Instant};
use tag_rust::{
    core::{
        Decision, ForwardModel, PlayerId, PlayerResult, evaluation::Evaluator,
        random::SimulationRng, simulation::SimulationModel,
    },
    games::*,
    search::mcts::{
        EvaluatorRegistry, MctsAgent, Settings, SimulationContext, StateEvaluator, StateValues,
    },
};

const OPS: [&str; 8] = [
    "init",
    "actions",
    "copy",
    "observe",
    "copy_apply",
    "apply",
    "fork",
    "actual_copy",
];
#[derive(Clone, Copy, Default)]
struct Metric {
    calls: u64,
    samples: u64,
    ns: u64,
}
struct Meter {
    scope: usize,
    stride: u64,
    data: [[Metric; 8]; 2],
    probe_ns: u64,
}
type Shared = Rc<RefCell<Meter>>;
fn begin(m: &Shared, op: usize) -> (usize, Option<Instant>) {
    let mut m = m.borrow_mut();
    let scope = m.scope;
    let stride = m.stride;
    if stride == 0 {
        return (scope, None);
    }
    let row = &mut m.data[scope][op];
    row.calls += 1;
    (
        scope,
        (scope == 0 || row.calls <= 16 || row.calls.is_multiple_of(stride)).then(Instant::now),
    )
}
fn end(m: &Shared, op: usize, start: (usize, Option<Instant>)) {
    if let (scope, Some(t)) = start {
        let ns = t.elapsed().as_nanos() as u64;
        let mut m = m.borrow_mut();
        let r = &mut m.data[scope][op];
        r.samples += 1;
        r.ns += ns;
    }
}
struct State<S> {
    inner: S,
    meter: Shared,
}
impl<S: Clone> Clone for State<S> {
    fn clone(&self) -> Self {
        let t = begin(&self.meter, 7);
        let inner = self.inner.clone();
        end(&self.meter, 7, t);
        Self {
            inner,
            meter: self.meter.clone(),
        }
    }
}
struct Model<F> {
    inner: F,
    meter: Shared,
}
impl<F: ForwardModel> ForwardModel for Model<F>
where
    F::State: Clone,
{
    type State = State<F::State>;
    type Action = F::Action;
    type Observation = F::Observation;
    type Parameters = F::Parameters;
    type Error = F::Error;
    fn init_state(&self, p: &F::Parameters, n: usize) -> Result<Self::State, F::Error> {
        let t = begin(&self.meter, 0);
        let result = self.inner.init_state(p, n);
        end(&self.meter, 0, t);
        result.map(|inner| State {
            inner,
            meter: self.meter.clone(),
        })
    }
    fn player_count(&self, s: &Self::State) -> usize {
        self.inner.player_count(&s.inner)
    }
    fn next_player(&self, s: &Self::State) -> Decision {
        self.inner.next_player(&s.inner)
    }
    fn observe(&self, s: &Self::State, p: PlayerId) -> Result<F::Observation, F::Error> {
        let t = begin(&self.meter, 3);
        let result = self.inner.observe(&s.inner, p);
        end(&self.meter, 3, t);
        result
    }
    fn offered_actions(&self, o: &F::Observation) -> Vec<F::Action> {
        let t = begin(&self.meter, 1);
        let result = self.inner.offered_actions(o);
        end(&self.meter, 1, t);
        result
    }
    fn available_actions(&self, s: &Self::State, p: PlayerId) -> Result<Vec<F::Action>, F::Error> {
        let t = begin(&self.meter, 1);
        let result = self.inner.available_actions(&s.inner, p);
        end(&self.meter, 1, t);
        result
    }
    fn next(&self, s: &mut Self::State, p: PlayerId, a: &F::Action) -> Result<(), F::Error> {
        let sample = {
            let m = self.meter.borrow();
            let c = m.data[m.scope][5].calls + 1;
            m.stride > 0 && (m.scope == 0 || c <= 16 || c.is_multiple_of(m.stride))
        };
        if sample {
            let overall = Instant::now();
            let copy = Instant::now();
            let mut branch = s.inner.clone();
            let copy_ns = copy.elapsed().as_nanos() as u64;
            self.inner.next(&mut branch, p, a)?;
            let pair_ns = overall.elapsed().as_nanos() as u64;
            std::hint::black_box(&branch);
            drop(branch);
            let overhead = overall.elapsed().as_nanos() as u64;
            let mut m = self.meter.borrow_mut();
            let scope = m.scope;
            // Controlled probes use the same pre-transition state/action, independently of play.
            for (op, ns) in [(2, copy_ns), (4, pair_ns)] {
                let r = &mut m.data[scope][op];
                r.samples += 1;
                r.ns += ns;
            }
            m.probe_ns += overhead;
        }
        let t = begin(&self.meter, 5);
        let result = self.inner.next(&mut s.inner, p, a);
        end(&self.meter, 5, t);
        result
    }
    fn player_result(&self, s: &Self::State, p: PlayerId) -> Result<PlayerResult, F::Error> {
        self.inner.player_result(&s.inner, p)
    }
    fn player_score(&self, s: &Self::State, p: PlayerId) -> Result<f64, F::Error> {
        self.inner.player_score(&s.inner, p)
    }
}
impl<F: SimulationModel> SimulationModel for Model<F>
where
    F::State: Clone,
{
    fn simulation_fork(
        &self,
        o: &F::Observation,
        r: &mut SimulationRng,
    ) -> Result<Self::State, F::Error> {
        let t = begin(&self.meter, 6);
        let result = self.inner.simulation_fork(o, r);
        end(&self.meter, 6, t);
        result.map(|inner| State {
            inner,
            meter: self.meter.clone(),
        })
    }
}
trait Profile: SimulationModel + Default + 'static {
    fn parameters(seed: u64) -> Self::Parameters;
    fn value(model: &Model<Self>, s: &State<Self::State>, p: PlayerId) -> Result<f64, String>
    where
        Self: Sized,
        Self::State: Clone;
}
macro_rules! profile {
    ($model:ty,$params:expr,$eval:expr) => {
        impl Profile for $model {
            fn parameters(seed: u64) -> Self::Parameters {
                ($params)(seed)
            }
            fn value(_: &Model<Self>, s: &State<Self::State>, p: PlayerId) -> Result<f64, String> {
                $eval.evaluate(&s.inner, p).map_err(|e| e.to_string())
            }
        }
    };
}
profile!(
    tictactoe::TicTacToeForwardModel,
    |_| tictactoe::TicTacToeParameters::default(),
    tictactoe::TicTacToeEvaluator
);
profile!(
    connect4::Connect4ForwardModel,
    |_| connect4::Connect4Parameters::default(),
    connect4::Connect4Evaluator
);
profile!(
    cantstop::CantStopForwardModel,
    |seed| cantstop::CantStopParameters {
        seed,
        ..Default::default()
    },
    cantstop::CantStopProgress::default()
);
impl Profile for loveletter::LoveLetterForwardModel {
    fn parameters(seed: u64) -> Self::Parameters {
        loveletter::LoveLetterParameters {
            seed,
            ..Default::default()
        }
    }
    fn value(model: &Model<Self>, s: &State<Self::State>, p: PlayerId) -> Result<f64, String> {
        let view = model.observe(s, p).map_err(|e| e.to_string())?;
        loveletter::LoveLetterHeuristic
            .evaluate_observation(&view)
            .map_err(|e| e.to_string())
    }
}
struct Value<F>(PhantomData<F>);
impl<F: Profile> StateEvaluator<Model<F>> for Value<F>
where
    F::State: Clone,
    F::Error: std::fmt::Display,
{
    fn evaluate(
        &mut self,
        s: &State<F::State>,
        ctx: &mut SimulationContext<'_, Model<F>>,
    ) -> tag_rust::search::mcts::Result<StateValues> {
        let values = (0..ctx.model().player_count(s))
            .map(|p| F::value(ctx.model(), s, PlayerId::new(p)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(tag_rust::search::mcts::Error::Evaluation)?;
        Ok(StateValues::Full(values))
    }
}
#[allow(clippy::too_many_arguments)] // Explicit low-level measurement CLI settings.
fn run<F: Profile>(
    name: &str,
    players: usize,
    games: usize,
    iterations: u64,
    seed: u64,
    stride: u64,
    aligned: bool,
    skip_forced: bool,
    warmup: usize,
) where
    F::State: Clone,
    F::Action: Clone + PartialEq + std::fmt::Debug,
    F::Error: std::fmt::Display,
{
    let meter = Rc::new(RefCell::new(Meter {
        scope: 0,
        stride,
        data: [[Metric::default(); 8]; 2],
        probe_ns: 0,
    }));
    let model = Model {
        inner: F::default(),
        meter: meter.clone(),
    };
    let mut registry = EvaluatorRegistry::<Model<F>>::default();
    registry
        .register_state("game_heuristic", |_| Ok(Box::new(Value::<F>(PhantomData))))
        .unwrap();
    for i in 0..warmup {
        let mut state = model
            .init_state(&F::parameters(seed + i as u64), players)
            .unwrap_or_else(|e| panic!("warmup: {e}"));
        let (actor, actions) = loop {
            let Decision::Player(actor) = model.next_player(&state) else {
                panic!("warmup ended before a choice")
            };
            let actions = model
                .available_actions(&state, actor)
                .unwrap_or_else(|e| panic!("warmup actions: {e}"));
            if actions.len() > 1 {
                break (actor, actions);
            }
            model
                .next(&mut state, actor, &actions[0])
                .unwrap_or_else(|e| panic!("warmup next: {e}"));
        };
        let config=Settings::from_json(&json!({"schema_version":1,"seed":seed+100000+i as u64,"tree_depth":if aligned{1000}else{100},
            "rollout_length":10,"budgets":{"iterations":iterations}}).to_string()).unwrap();
        let view = model
            .observe(&state, actor)
            .unwrap_or_else(|e| panic!("warmup view: {e}"));
        let mut agent = MctsAgent::new(config, &registry).unwrap();
        agent.search(&model, actor, &view, &actions).unwrap();
    }
    let timer_control_ns = (0..10000)
        .map(|_| Instant::now().elapsed().as_nanos() as u64)
        .sum::<u64>() as f64
        / 10000.;
    println!(
        "{}",
        json!({"phase":"ready","engine":"rust","game":name,"players":players,"timer_control_ns":timer_control_ns})
    );
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    for game_id in 0..games {
        meter.borrow_mut().data = [[Metric::default(); 8]; 2];
        meter.borrow_mut().probe_ns = 0;
        let clock = Instant::now();
        let mut state = model
            .init_state(&F::parameters(seed + game_id as u64), players)
            .unwrap_or_else(|e| panic!("init: {e}"));
        let mut agents: Vec<_> = (0..players)
            .map(|seat| {
                let mut config = Settings::from_json(
                    &json!({"schema_version":1,"seed":(seed+game_id as u64).wrapping_mul(4).wrapping_add(100000+seat as u64),
                "rollout_length":10,"budgets":{"iterations":iterations}})
                    .to_string(),
                )
                .unwrap();
                if aligned {
                    config.tree_depth = Some(1000);
                }
                MctsAgent::new(config, &registry).unwrap()
            })
            .collect();
        let mut decisions = 0;
        let mut searched_decisions = 0;
        let mut search_ns = 0u64;
        let mut trace_hash = 14695981039346656037u64;
        while let Decision::Player(actor) = model.next_player(&state) {
            let actions = model
                .available_actions(&state, actor)
                .unwrap_or_else(|e| panic!("actions: {e}"));
            let view = model
                .observe(&state, actor)
                .unwrap_or_else(|e| panic!("observe: {e}"));
            let selected = if skip_forced && actions.len() == 1 {
                0
            } else {
                meter.borrow_mut().scope = 1;
                let start = Instant::now();
                let report = agents[actor.index()]
                    .search(&model, actor, &view, &actions)
                    .unwrap();
                search_ns += start.elapsed().as_nanos() as u64;
                meter.borrow_mut().scope = 0;
                assert_eq!(
                    report.counters.completed_iterations, iterations,
                    "iteration budget not completed"
                );
                searched_decisions += 1;
                report.selected_index
            };
            for byte in format!("{:?}", actions[selected]).bytes() {
                trace_hash = (trace_hash ^ u64::from(byte)).wrapping_mul(1099511628211);
            }
            model
                .next(&mut state, actor, &actions[selected])
                .unwrap_or_else(|e| panic!("next: {e}"));
            decisions += 1;
            assert!(decisions < 100000, "game failed to terminate");
        }
        let elapsed_ns = clock.elapsed().as_nanos() as u64;
        let results: Vec<_> = (0..players)
            .map(|p| {
                format!(
                    "{:?}",
                    model
                        .player_result(&state, PlayerId::new(p))
                        .unwrap_or_else(|e| panic!("result: {e}"))
                )
            })
            .collect();
        let m = meter.borrow();
        let mut metrics = Vec::new();
        for scope in 0..2 {
            for (op, label) in OPS.iter().enumerate() {
                let r = m.data[scope][op];
                metrics.push(json!({"scope":if scope==0{"real"}else{"search"},"operation":label,"calls":r.calls,"samples":r.samples,"ns":r.ns}));
            }
        }
        println!(
            "{}",
            json!({"phase":"game","engine":"rust","game":name,"players":players,"id":game_id,
            "seed":seed+game_id as u64,"decisions":decisions,"searched_decisions":searched_decisions,"elapsed_ns":elapsed_ns,"search_ns":search_ns,
            "probe_ns":m.probe_ns,"trace_hash":trace_hash.to_string(),"results":results,"metrics":metrics})
        );
        std::io::stdout().flush().unwrap();
        drop(m);
        line.clear();
        std::io::stdin().read_line(&mut line).unwrap();
    }
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(
        a.len(),
        10,
        "game players games iterations seed sampling_stride aligned skip_forced warmup"
    );
    let name = &a[1];
    let p = a[2].parse().unwrap();
    let games = a[3].parse().unwrap();
    let it = a[4].parse().unwrap();
    let seed = a[5].parse().unwrap();
    let stride = a[6].parse().unwrap();
    let aligned = a[7] == "true";
    let skip = a[8] == "true";
    let warmup = a[9].parse().unwrap();
    match name.as_str() {
        "tictactoe" => run::<tictactoe::TicTacToeForwardModel>(
            name, p, games, it, seed, stride, aligned, skip, warmup,
        ),
        "connect4" => run::<connect4::Connect4ForwardModel>(
            name, p, games, it, seed, stride, aligned, skip, warmup,
        ),
        "cantstop" => run::<cantstop::CantStopForwardModel>(
            name, p, games, it, seed, stride, aligned, skip, warmup,
        ),
        "loveletter" => run::<loveletter::LoveLetterForwardModel>(
            name, p, games, it, seed, stride, aligned, skip, warmup,
        ),
        _ => panic!("unknown game"),
    }
}
