//! Native Random Sushi Go! games with rotating agent seats.
use std::error::Error;
use tag_rust::agents::RandomAgent;
use tag_rust::core::random::automatic_seed;
use tag_rust::core::{ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::sushigo::{Parameters, SushiGoForwardModel};
use tag_rust::runner::run_joint_game;

fn main() -> Result<(), Box<dyn Error>> {
    let mut players = 3usize;
    let mut games = 3usize;
    let mut seed = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("expected value after option")?;
        match flag.as_str() {
            "--players" => players = value.parse()?,
            "--games" => games = value.parse()?,
            "--seed" => seed = Some(value.parse()?),
            _ => return Err(format!("unknown option: {flag}").into()),
        }
    }
    if !(2..=5).contains(&players) || games == 0 {
        return Err("players must be 2..5 and games must be positive".into());
    }
    let resolved_seed = match seed {
        Some(value) => value,
        None => automatic_seed()?,
    };
    println!("seed={resolved_seed} players={players} games={games}");
    let model = SushiGoForwardModel;
    // Each entry is [wins, draws, losses] for one agent identity and seat.
    let mut seat_outcomes = vec![vec![[0usize; 3]; players]; players];
    for game in 0..games {
        let game_seed = resolved_seed.wrapping_add(game as u64);
        let mut state = model.init_state(&Parameters { seed: game_seed }, players)?;
        let mut agents = (0..players)
            .map(|seat| {
                let identity = (seat + players - game % players) % players;
                RandomAgent::new(
                    resolved_seed
                        .wrapping_add(1_000_003u64.wrapping_mul(game as u64))
                        .wrapping_add(identity as u64),
                )
            })
            .collect::<Vec<_>>();
        run_joint_game(&model, &mut state, &mut agents, |_, _, _| {})?;
        for (seat, _) in state.public.scores.iter().enumerate() {
            let identity = (seat + players - game % players) % players;
            let outcome = match model.player_result(&state, PlayerId::new(seat))? {
                PlayerResult::Win => 0,
                PlayerResult::Draw => 1,
                PlayerResult::Loss => 2,
                PlayerResult::Ongoing => return Err("finished game has an ongoing result".into()),
            };
            seat_outcomes[identity][seat][outcome] += 1;
        }
        println!(
            "game={} seed={} scores={:?} pudding={:?}",
            game + 1,
            game_seed,
            state.public.scores,
            state.public.pudding
        );
    }
    for (identity, seats) in seat_outcomes.iter().enumerate() {
        let totals = seats.iter().fold([0usize; 3], |mut total, seat| {
            for (sum, &count) in total.iter_mut().zip(seat) {
                *sum += count;
            }
            total
        });
        println!("agent={identity} totals={totals:?} seat_outcomes={seats:?}");
    }
    Ok(())
}
