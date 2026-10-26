//! Compare initialized process footprints with optional features disabled/enabled.
use std::{
    hint::black_box,
    io::{self, Write},
};
use tag_rust::core::ForwardModel;
fn hold<F: ForwardModel>(model: F, params: F::Parameters, players: usize)
where
    F::Error: std::fmt::Debug,
{
    let state = model.init_state(&params, players).unwrap();
    println!("{{\"phase\":\"idle\"}}");
    io::stdout().flush().unwrap();
    let mut line = String::new();
    io::stdin().read_line(&mut line).unwrap();
    println!("{{\"phase\":\"loaded\",\"count\":1}}");
    io::stdout().flush().unwrap();
    line.clear();
    io::stdin().read_line(&mut line).unwrap();
    black_box(state);
}
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 3, "game players seed");
    let players = a[1].parse().unwrap();
    let seed = a[2].parse().unwrap();
    match a[0].as_str() {
        "tictactoe" => hold(
            tag_rust::games::tictactoe::TicTacToeForwardModel,
            Default::default(),
            players,
        ),
        "connect4" => hold(
            tag_rust::games::connect4::Connect4ForwardModel,
            Default::default(),
            players,
        ),
        "cantstop" => hold(
            tag_rust::games::cantstop::CantStopForwardModel,
            tag_rust::games::cantstop::CantStopParameters {
                seed,
                ..Default::default()
            },
            players,
        ),
        "loveletter" => hold(
            tag_rust::games::loveletter::LoveLetterForwardModel,
            tag_rust::games::loveletter::LoveLetterParameters {
                seed,
                ..Default::default()
            },
            players,
        ),
        _ => panic!("unknown game"),
    }
}
