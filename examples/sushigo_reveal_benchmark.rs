//! Narrow first-reveal comparison with the public Java implementation.
use std::time::Instant;
use tag_rust::{
    core::{ForwardModel, PlayerId, joint},
    games::sushigo::{Action, Parameters, SushiGoForwardModel},
};

fn run(
    model: &SushiGoForwardModel,
    initial: &tag_rust::games::sushigo::State,
    choices: &[(PlayerId, Action)],
    iterations: usize,
) -> u128 {
    let started = Instant::now();
    let mut checksum = 0;
    for _ in 0..iterations {
        let mut state = initial.clone();
        joint::apply(model, &mut state, choices).expect("legal first reveal");
        checksum += state.public.played[0].len();
    }
    std::hint::black_box(checksum);
    started.elapsed().as_nanos()
}
fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .expect("iterations")
        .parse()
        .expect("positive integer");
    let model = SushiGoForwardModel;
    let initial = model
        .init_state(&Parameters { seed: 12345 }, 3)
        .expect("three players");
    let choices: Vec<_> = (0..3)
        .map(|seat| {
            (
                PlayerId::new(seat),
                Action::Pick(initial.privileged_hand(PlayerId::new(seat)).expect("seat")[0]),
            )
        })
        .collect();
    run(&model, &initial, &choices, iterations);
    for trial in 0..5 {
        println!(
            "rust\t{trial}\t{iterations}\t{}",
            run(&model, &initial, &choices, iterations)
        );
    }
}
