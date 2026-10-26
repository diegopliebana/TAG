//! Run complete Can't Stop matches through the generic runner.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::RandomAgent;
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::cantstop::{CantStopForwardModel, CantStopParameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let players: usize = arguments
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2);
    let seed = seeds::resolve_seed(arguments.next().as_deref())?;
    if arguments.next().is_some() {
        return Err("usage: cantstop [players: 2-4] [game seed]".into());
    }
    let model = CantStopForwardModel;
    let parameters = CantStopParameters {
        seed,
        ..Default::default()
    };
    let mut state = model.init_state(&parameters, players)?;
    let mut agents: Vec<_> = (0..players)
        .map(|seat| RandomAgent::new(seed.wrapping_add(seat as u64)))
        .collect();
    println!("Can't Stop: {players} players, game seed {seed}");
    println!(
        "Random agent seeds: {:?}",
        (0..players)
            .map(|seat| seed.wrapping_add(seat as u64))
            .collect::<Vec<_>>()
    );
    tag_rust::runner::run_game(&model, &mut state, &mut agents)?;
    let position = state.position();
    println!(
        "Actions: {}; completed turns: {}",
        position.counters().move_count(),
        position.counters().completed_turn_count()
    );
    for seat in 0..players {
        let player = PlayerId::new(seat);
        println!(
            "Player {seat}: {:?}, {} claimed columns",
            position.player_result(player)?,
            position.player_score(player)?
        );
    }
    Ok(())
}
