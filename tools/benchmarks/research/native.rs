//! Reproducible research workloads. Rules remain unchanged; instrumentation is tool-local.
#![allow(unexpected_cfgs)]
use serde_json::json;
use std::{
    cell::Cell,
    hint::black_box,
    io::{self, Write},
    num::NonZeroU32,
    time::{Duration, Instant},
};
use tag_rust::{
    agents::{Agent, FirstAvailableAgent, OslaAgent, RandomAgent, SampledOslaAgent},
    core::{Decision, ForwardModel, PlayerId, random::SimulationRng, simulation::SimulationModel},
    learning::{LearningGame, LearningLimits},
    replay::PersistableGame,
    search::mcts::{MctsAgent, NativeEvaluators, Settings},
};

#[cfg(benchmark_alloc)]
mod allocations {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicU64, Ordering::Relaxed},
    };
    pub static BYTES: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub static LIVE: AtomicU64 = AtomicU64::new(0);
    pub static PEAK: AtomicU64 = AtomicU64::new(0);
    pub struct Counting;
    // The wrapper delegates unchanged layouts/pointers to System. Counters use no allocation.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                BYTES.fetch_add(layout.size() as u64, Relaxed);
                CALLS.fetch_add(1, Relaxed);
                let live = LIVE.fetch_add(layout.size() as u64, Relaxed) + layout.size() as u64;
                PEAK.fetch_max(live, Relaxed);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            LIVE.fetch_sub(layout.size() as u64, Relaxed);
            unsafe { System.dealloc(ptr, layout) };
        }
    }
    pub fn read() -> [u64; 4] {
        [
            BYTES.load(Relaxed),
            CALLS.load(Relaxed),
            LIVE.load(Relaxed),
            PEAK.load(Relaxed),
        ]
    }
    pub fn reset_peak() {
        PEAK.store(LIVE.load(Relaxed), Relaxed);
    }
}
#[cfg(benchmark_alloc)]
#[global_allocator]
static ALLOCATOR: allocations::Counting = allocations::Counting;
fn memory() -> [u64; 4] {
    #[cfg(benchmark_alloc)]
    {
        allocations::read()
    }
    #[cfg(not(benchmark_alloc))]
    {
        [0; 4]
    }
}
trait Bench:
    SimulationModel + NativeEvaluators + LearningGame + PersistableGame + Sized + 'static
{
    type Histories: Clone + serde::Serialize;
    fn histories(&self, state: &Self::State) -> Self::Histories;
    fn progress(&self, state: &Self::State) -> tag_rust::core::progress::TurnCounters;
    fn osla(
        &self,
        actor: PlayerId,
        obs: &Self::Observation,
        actions: &[Self::Action],
        seed: u64,
    ) -> usize;
}
macro_rules! public_game {
    ($model:ty, $eval:expr) => {
        impl Bench for $model {
            type Histories = ();
            fn histories(&self, _: &Self::State) {}
            fn progress(&self, state: &Self::State) -> tag_rust::core::progress::TurnCounters {
                state.counters()
            }
            fn osla(
                &self,
                p: PlayerId,
                o: &Self::Observation,
                a: &[Self::Action],
                seed: u64,
            ) -> usize {
                OslaAgent::new($eval, seed).choose(self, p, o, a).unwrap()
            }
        }
    };
}
macro_rules! sampled_game {
    ($model:ty, $eval:expr, $progress:expr, $histories:ty, $history:expr) => {
        impl Bench for $model {
            type Histories = $histories;
            fn histories(&self, state: &Self::State) -> Self::Histories {
                ($history)(state)
            }
            fn progress(&self, state: &Self::State) -> tag_rust::core::progress::TurnCounters {
                ($progress)(state)
            }
            fn osla(
                &self,
                p: PlayerId,
                o: &Self::Observation,
                a: &[Self::Action],
                seed: u64,
            ) -> usize {
                SampledOslaAgent::new($eval, seed, NonZeroU32::new(1).unwrap())
                    .choose(self, p, o, a)
                    .unwrap()
            }
        }
    };
}
public_game!(
    tag_rust::games::tictactoe::TicTacToeForwardModel,
    tag_rust::games::tictactoe::TicTacToeEvaluator
);
public_game!(
    tag_rust::games::connect4::Connect4ForwardModel,
    tag_rust::games::connect4::Connect4Evaluator
);
sampled_game!(
    tag_rust::games::cantstop::CantStopForwardModel,
    tag_rust::games::cantstop::CantStopBaseline,
    |s: &tag_rust::games::cantstop::CantStopState| s.position().counters(),
    (),
    |_: &tag_rust::games::cantstop::CantStopState| ()
);
type LoveLetterHistories = (
    Vec<tag_rust::games::loveletter::PublicEvent>,
    Option<tag_rust::games::loveletter::PublicRound>,
    Vec<Vec<tag_rust::games::loveletter::PrivateEvent>>,
    Vec<Vec<tag_rust::games::loveletter::PrivateEvent>>,
);
sampled_game!(
    tag_rust::games::loveletter::LoveLetterForwardModel,
    tag_rust::games::loveletter::LoveLetterHeuristic,
    |s: &tag_rust::games::loveletter::LoveLetterState| s.public().counters,
    LoveLetterHistories,
    |s: &tag_rust::games::loveletter::LoveLetterState| {
        let views: Vec<_> = (0..s.public().player_count())
            .map(|i| {
                tag_rust::games::loveletter::LoveLetterForwardModel
                    .observe(s, PlayerId::new(i))
                    .unwrap()
            })
            .collect();
        (
            s.public().events.clone(),
            s.public().last_round.clone(),
            views.iter().map(|o| o.private_events().to_vec()).collect(),
            views
                .iter()
                .map(|o| o.last_round_private_events().to_vec())
                .collect(),
        )
    }
);

#[allow(clippy::too_many_arguments)] // Keep the low-level CLI settings explicit in this measurement tool.
fn run<F>(
    model: F,
    params: F::Parameters,
    players: usize,
    op: &str,
    count: usize,
    seed: u64,
    iterations: u64,
    warm_ms: u64,
    sample_ms: u64,
) where
    F: Bench,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Debug + std::fmt::Display,
{
    let (op, prefix) = op.split_once('@').map_or((op, 0), |(op, prefix)| {
        (op, prefix.parse::<usize>().unwrap())
    });
    let mut root = model.init_state(&params, players).unwrap();
    if op != "init" && <F as PersistableGame>::GAME_ID == "cantstop" {
        let p = match model.next_player(&root) {
            Decision::Player(p) => p,
            _ => unreachable!(),
        };
        let a = model.available_actions(&root, p).unwrap();
        model.next(&mut root, p, &a[0]).unwrap();
    }
    let mut preparation_rng = tag_rust::core::random::AgentRng::new(seed.wrapping_add(91));
    let mut prepared = 0;
    for _ in 0..prefix {
        let Decision::Player(actor) = model.next_player(&root) else {
            break;
        };
        let legal = model.available_actions(&root, actor).unwrap();
        let selected = preparation_rng.index(legal.len()).unwrap();
        // Fixture construction is outside timing. Keep the last nonterminal root.
        let mut successor = root.clone();
        model.next(&mut successor, actor, &legal[selected]).unwrap();
        if model.next_player(&successor) == Decision::Finished {
            break;
        }
        root = successor;
        prepared += 1;
    }
    let actor = match model.next_player(&root) {
        Decision::Player(p) => p,
        _ => panic!("nonterminal root required"),
    };
    let observation = model.observe(&root, actor).unwrap();
    let actions = model.available_actions(&root, actor).unwrap();
    assert!(!actions.is_empty());
    let saved = model.state_data(&root);
    let serialized = serde_json::to_vec(&saved).unwrap();
    let histories = model.histories(&root);
    let metadata = json!({"prepared_actions":prepared,"root_actions":actions.len(),"root_save_bytes":serialized.len(),"retained_history_json_bytes":serde_json::to_vec(&histories).unwrap().len(),"seed":seed,"iterations":iterations});
    if op == "apply" || op == "apply_control" {
        // Successor inputs are cloned before each timed block, not in the timer.
        let block = || {
            let mut states: Vec<_> = (0..32).map(|_| black_box(&root).clone()).collect();
            let before = memory();
            let start = Instant::now();
            for state in &mut states {
                if op == "apply" {
                    model
                        .next(black_box(state), black_box(actor), black_box(&actions[0]))
                        .unwrap();
                } else {
                    black_box(state);
                }
            }
            let elapsed = start.elapsed().as_nanos();
            let after = memory();
            black_box(&states);
            (elapsed, after[0] - before[0], after[1] - before[1])
        };
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(warm_ms) {
            black_box(block());
        }
        for batch in 0..5 {
            let (mut calls, mut elapsed, mut bytes, mut allocations) = (0, 0, 0, 0);
            while calls < count || elapsed < u128::from(sample_ms) * 1_000_000 {
                let sample = block();
                calls += 32;
                elapsed += sample.0;
                bytes += sample.1;
                allocations += sample.2;
            }
            println!(
                "{}",
                json!({"batch":batch,"calls":calls,"elapsed_ns":elapsed,"allocated_bytes":bytes,"allocation_calls":allocations,"instrumented":cfg!(benchmark_alloc)})
            );
        }
        println!("{metadata}");
        return;
    }
    let settings = Settings {
        seed: Some(seed),
        rollout_length: Some(4),
        budgets: tag_rust::search::mcts::Budgets {
            iterations: Some(iterations),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut mcts = MctsAgent::<F>::new(settings, &F::native_evaluators()).unwrap();
    let mut rng = SimulationRng::new(seed);
    let mut random = RandomAgent::new(seed);
    let mut first = FirstAvailableAgent::new(seed);
    // Each measured block covers the same 32 agent seeds, in every workflow.
    let game_ordinal = Cell::new(0_u64);
    // games, actions, logical turns (including terminal), completed rounds, observations.
    let game_work = Cell::new([0_u64; 5]);
    if op == "retain_states" {
        println!("{}", json!({"phase":"idle","alloc":memory()}));
        io::stdout().flush().unwrap();
        let mut line = String::new();
        io::stdin().read_line(&mut line).unwrap();
        let before = memory();
        let states: Vec<_> = (0..count).map(|_| root.clone()).collect();
        println!(
            "{}",
            json!({"phase":"loaded","count":states.len(),"alloc":memory(),"before":before})
        );
        io::stdout().flush().unwrap();
        line.clear();
        io::stdin().read_line(&mut line).unwrap();
        black_box(states);
        return;
    }
    let mut work = 0u64;
    let mut checksum = 0u64;
    let mut operation = || match op {
        "control" => {
            black_box(0usize);
        }
        "init" => {
            black_box(model.init_state(black_box(&params), players).unwrap());
        }
        "copy" => {
            black_box(black_box(&root).clone());
        }
        "observe" => {
            black_box(model.observe(black_box(&root), black_box(actor)).unwrap());
        }
        "actions" => {
            checksum += black_box(
                model
                    .available_actions(black_box(&root), black_box(actor))
                    .unwrap(),
            )
            .len() as u64;
        }
        "validate" => {
            black_box(
                model
                    .available_actions(black_box(&root), black_box(actor))
                    .unwrap()
                    .contains(black_box(&actions[0])),
            );
        }
        "history_copy" => {
            black_box(black_box(&histories).clone());
        }
        "offered" => {
            black_box(model.offered_actions(black_box(&observation)));
        }
        "transition" => {
            let mut s = black_box(&root).clone();
            model.next(&mut s, actor, &actions[0]).unwrap();
            black_box(s);
        }
        "query" => {
            black_box((
                model.next_player(black_box(&root)),
                model
                    .player_result(black_box(&root), black_box(actor))
                    .unwrap(),
                model
                    .player_score(black_box(&root), black_box(actor))
                    .unwrap(),
            ));
        }
        "fork" => {
            black_box(
                model
                    .simulation_fork(black_box(&observation), &mut rng)
                    .unwrap(),
            );
        }
        "encode" => {
            black_box(
                model
                    .encode(black_box(&observation), &[], LearningLimits::default())
                    .unwrap(),
            );
        }
        "save" => {
            black_box(serde_json::to_vec(&model.state_data(black_box(&root))).unwrap());
        }
        "load" => {
            let data: F::SavedState = serde_json::from_slice(&serialized).unwrap();
            black_box(model.restore_state(&params, &data).unwrap());
        }
        "first" => {
            checksum += first.choose(&model, actor, &observation, &actions).unwrap() as u64;
        }
        "random" => {
            checksum += random
                .choose(&model, actor, &observation, &actions)
                .unwrap() as u64;
        }
        "osla" => {
            checksum += model.osla(actor, &observation, &actions, seed) as u64;
        }
        "mcts" => {
            let r = mcts.search(&model, actor, &observation, &actions).unwrap();
            assert_eq!(r.counters.completed_iterations, iterations);
            work += r.counters.forward_model_calls;
            black_box(r);
        }
        "game" | "game_direct" | "game_aligned" | "game_aligned_direct" => {
            let observe = op == "game" || op == "game_aligned";
            let mut s = model.init_state(black_box(&params), players).unwrap();
            let game_seed = seed.wrapping_add(game_ordinal.get() % 32);
            game_ordinal.set(game_ordinal.get() + 1);
            let mut agent = RandomAgent::new(game_seed);
            let mut steps = 0;
            let mut turns = 0;
            while let Decision::Player(p) = model.next_player(&s) {
                let a = model.available_actions(&s, p).unwrap();
                let i = if observe {
                    let o = model.observe(&s, p).unwrap();
                    agent.choose(&model, p, &o, &a).unwrap()
                } else {
                    <RandomAgent as Agent<F>>::rng_mut(&mut agent)
                        .index(a.len())
                        .unwrap()
                };
                model.next(&mut s, p, &a[i]).unwrap();
                steps += 1;
                if <F as PersistableGame>::GAME_ID != "cantstop"
                    || model.next_player(&s) != Decision::Player(p)
                {
                    turns += 1;
                }
                assert!(steps < 100_000, "game did not terminate");
            }
            let mut counts = game_work.get();
            counts[0] += 1;
            counts[1] += steps;
            counts[2] += turns;
            counts[3] += model.progress(&s).round_count() as u64;
            counts[4] += if observe { steps } else { 0 };
            game_work.set(counts);
            work += steps;
            black_box(s);
        }
        _ => panic!("unknown operation {op}"),
    };
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(warm_ms) {
        operation();
    }
    for batch in 0..5 {
        #[cfg(benchmark_alloc)]
        allocations::reset_peak();
        let before = memory();
        let work_before = game_work.get();
        let start = Instant::now();
        let mut calls = 0;
        while calls < count || start.elapsed() < Duration::from_millis(sample_ms) {
            for _ in 0..32 {
                operation();
                calls += 1;
            }
        }
        let elapsed = start.elapsed().as_nanos();
        let after = memory();
        let measured_work = std::array::from_fn::<_, 5, _>(|i| game_work.get()[i] - work_before[i]);
        println!(
            "{}",
            json!({"batch":batch,"calls":calls,"elapsed_ns":elapsed,"allocated_bytes":after[0]-before[0],"allocation_calls":after[1]-before[1],"live_before":before[2],"peak_live_bytes":after[3],"instrumented":cfg!(benchmark_alloc),"game_work":measured_work})
        );
    }
    println!(
        "{}",
        json!({"checksum":checksum,"transitions":work,"root_actions":actions.len(),"root_save_bytes":serialized.len(),"iterations":iterations,"seed":seed,"prepared_actions":prepared,"retained_history_json_bytes":serde_json::to_vec(&histories).unwrap().len()})
    );
}
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        a.len(),
        8,
        "game players operation count seed iterations warm_ms sample_ms"
    );
    let p = a[1].parse().unwrap();
    let count = a[3].parse().unwrap();
    let seed = a[4].parse().unwrap();
    let iterations = a[5].parse().unwrap();
    let warm = a[6].parse().unwrap();
    let sample = a[7].parse().unwrap();
    macro_rules! invoke {
        ($f:expr,$param:expr) => {
            run($f, $param, p, &a[2], count, seed, iterations, warm, sample)
        };
    }
    match a[0].as_str() {
        "tictactoe" => invoke!(
            tag_rust::games::tictactoe::TicTacToeForwardModel,
            tag_rust::games::tictactoe::TicTacToeParameters::default()
        ),
        "connect4" => invoke!(
            tag_rust::games::connect4::Connect4ForwardModel,
            tag_rust::games::connect4::Connect4Parameters::default()
        ),
        "cantstop" => invoke!(
            tag_rust::games::cantstop::CantStopForwardModel,
            tag_rust::games::cantstop::CantStopParameters {
                seed,
                ..Default::default()
            }
        ),
        "loveletter" => invoke!(
            tag_rust::games::loveletter::LoveLetterForwardModel,
            tag_rust::games::loveletter::LoveLetterParameters {
                seed,
                ..Default::default()
            }
        ),
        _ => panic!("unknown game"),
    }
}
