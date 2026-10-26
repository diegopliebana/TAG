//! Matched public-API workloads over the saved Java/Rust replay corpus.
use std::hint::black_box;
use std::time::{Duration, Instant};
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{
    Cell as C4Cell, Connect4ForwardModel, Connect4Parameters, DropPiece,
};
use tag_rust::games::tictactoe::{
    Cell as TttCell, PlaceMark, TicTacToeForwardModel, TicTacToeParameters,
};

type Error = Box<dyn std::error::Error>;
const MODES: [&str; 5] = ["init", "actions", "copy", "transition", "replay"];
const BATCHES: usize = 5;
struct Replay {
    moves: Vec<usize>,
    board: String,
    result: String,
}
struct Frame<S, A> {
    state: S,
    action: A,
    actor: PlayerId,
}

/// Load the selected game histories and expected final board/result.
/// The corpus is borrowed; returned histories own their parsed fields.
fn load(corpus: &str, game: &str) -> Vec<Replay> {
    corpus
        .lines()
        .filter(|line| {
            // Select data rows for this game.
            line.starts_with(&format!("{game}\t"))
        })
        .map(|line| {
            // Parse the recorded moves and terminal checks for this history.
            let fields = line.split('\t').collect::<Vec<_>>();
            let expected = if game == "ttt" {
                ["3", "3", "3"]
            } else {
                ["7", "6", "4"]
            };
            assert_eq!(&fields[1..4], &expected);
            Replay {
                moves: fields[4]
                    .split(',')
                    .map(|value| {
                        // Parse one recorded cell index or selected column.
                        value.parse().expect("valid recorded move")
                    })
                    .collect(),
                board: fields[5].to_owned(),
                result: fields[6].to_owned(),
            }
        })
        .collect()
}
fn actor<F: ForwardModel>(model: &F, state: &F::State) -> PlayerId {
    match model.next_player(state) {
        Decision::Player(player) => player,
        Decision::Finished => panic!("replay already finished"),
    }
}

/// Prepare matching prefixes, validate final outcomes, then measure five public APIs.
/// Rules/configuration, corpus and interpretation closures are borrowed by the workload.
/// Errors from initialization or transitions are returned to the caller.
fn benchmark<F, P, B>(
    game: &str,
    model: &F,
    parameters: &F::Parameters,
    replays: &[Replay],
    matches: P,
    board: B,
    repetitions: usize,
) -> Result<(), Error>
where
    F: ForwardModel,
    F::State: Clone,
    F::Error: std::error::Error + 'static,
    P: Fn(&F::Action, usize) -> bool,
    B: Fn(&F::State) -> String,
{
    let mut frames = Vec::new();
    let mut action_count = 0usize;
    for replay in replays {
        let mut state = model.init_state(parameters, 2)?;
        for &code in &replay.moves {
            let player = actor(model, &state);
            let actions = model.available_actions(&state, player)?;
            action_count += actions.len();
            let action = actions
                .into_iter()
                .find(|action| {
                    // Find the recorded move in the generated action list.
                    matches(action, code)
                })
                .expect("recorded move available");
            let prefix = state.clone();
            model.next(&mut state, player, &action)?;
            frames.push(Frame {
                state: prefix,
                action,
                actor: player,
            });
        }
        assert_eq!(board(&state), replay.board);
        let result = match model.player_result(&state, PlayerId::new(0))? {
            PlayerResult::Win => "WIN_GAME",
            PlayerResult::Loss => "LOSE_GAME",
            PlayerResult::Draw => "DRAW_GAME",
            PlayerResult::Ongoing => panic!("expected terminal replay"),
        };
        assert_eq!(result, replay.result);
    }
    println!(
        "corpus\t{game}\tgames={}\tprefixes={}\tavailable_actions={action_count}",
        replays.len(),
        frames.len()
    );
    for mode in MODES {
        let operations = if mode == "init" || mode == "replay" {
            replays.len()
        } else {
            frames.len()
        };
        let run = |passes: usize| -> Result<(Duration, usize), Error> {
            let mut elapsed = Duration::ZERO;
            let mut observed = 0usize;
            for _ in 0..passes {
                // Prepare independent mutable successors before transition timing.
                let mut states = if mode == "transition" {
                    frames
                        .iter()
                        .map(|frame| {
                            // Give this transition its original prefix state.
                            frame.state.clone()
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let start = Instant::now();
                match mode {
                    "init" => {
                        for _ in replays {
                            black_box(model.init_state(parameters, 2)?);
                        }
                    }
                    "actions" => {
                        for frame in &frames {
                            let actions = model.available_actions(&frame.state, frame.actor)?;
                            observed += actions.len();
                            black_box(actions);
                        }
                    }
                    "copy" => {
                        for frame in &frames {
                            black_box(frame.state.clone());
                        }
                    }
                    "transition" => {
                        for (state, frame) in states.iter_mut().zip(&frames) {
                            model.next(state, frame.actor, &frame.action)?;
                            black_box(&*state);
                        }
                    }
                    "replay" => {
                        for replay in replays {
                            let mut state = model.init_state(parameters, 2)?;
                            for &code in &replay.moves {
                                let player = actor(model, &state);
                                let actions = model.available_actions(&state, player)?;
                                let action = actions
                                    .into_iter()
                                    .find(|action| {
                                        // Select this history's next recorded move.
                                        matches(action, code)
                                    })
                                    .expect("recorded move available");
                                model.next(&mut state, player, &action)?;
                            }
                            black_box(state);
                        }
                    }
                    _ => unreachable!(),
                }
                elapsed += start.elapsed();
            }
            if mode == "actions" {
                assert_eq!(observed, passes * action_count);
            }
            Ok((elapsed, passes * operations))
        };
        let warmup = Instant::now();
        let mut warmup_passes = 0;
        while warmup.elapsed() < Duration::from_secs(1) || warmup_passes < 200 {
            run(10)?;
            warmup_passes += 10;
        }
        println!("warmup\t{game}\t{mode}\tpasses={warmup_passes}");
        for batch in 1..=BATCHES {
            let (elapsed, operations) = run(repetitions)?;
            println!(
                "sample\trust\t{game}\t{mode}\t{batch}\t{operations}\t{:.3}",
                elapsed.as_secs_f64() * 1e9 / operations as f64
            );
        }
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let repetitions = std::env::args()
        .nth(1)
        .map(|value| {
            // Parse the requested passes per timing batch.
            value.parse::<usize>()
        })
        .transpose()?
        .unwrap_or(500);
    if !(1..=10_000).contains(&repetitions) {
        return Err("passes must be between 1 and 10000".into());
    }
    println!(
        "rust\tdebug_assertions={}\tpasses={repetitions}",
        cfg!(debug_assertions)
    );
    let corpus = include_str!("../tools/java-comparison/benchmark-corpus.tsv");
    benchmark(
        "ttt",
        &TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        &load(corpus, "ttt"),
        |action: &PlaceMark, code| {
            // Identify a recorded x-major cell index.
            action.x * 3 + action.y == code
        },
        |state| {
            // Encode the final board using the fixture's symbols.
            state
                .board()
                .iter()
                .map(|cell| match cell {
                    TttCell::Empty => '.',
                    TttCell::X => 'x',
                    TttCell::O => 'o',
                })
                .collect()
        },
        repetitions,
    )?;
    benchmark(
        "c4",
        &Connect4ForwardModel,
        &Connect4Parameters::default(),
        &load(corpus, "c4"),
        |action: &DropPiece, code| {
            // Identify the recorded selected column.
            action.column == code
        },
        |state| {
            // Encode the final board using the fixture's symbols.
            state
                .board()
                .iter()
                .map(|cell| match cell {
                    C4Cell::Empty => '.',
                    C4Cell::Piece(player) if player.index() == 0 => 'x',
                    C4Cell::Piece(_) => 'o',
                })
                .collect()
        },
        repetitions,
    )?;
    Ok(())
}
