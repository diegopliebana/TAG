//! Run a seeded native cooperative match with Random agents.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::RandomAgent;
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::thegame::{CommunicationMode, Parameters, TeamOutcome, TheGameForwardModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let players: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(2);
    let game_seed = seeds::resolve_seed(args.next().as_deref())?;
    let agent_seed = seeds::resolve_seed(args.next().as_deref())?;
    let first_player: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0);
    if args.next().is_some() {
        return Err(
            "usage: thegame [players: 1-5] [game seed or auto] [agent seed or auto] [first seat]"
                .into(),
        );
    }
    let model = TheGameForwardModel;
    let parameters = Parameters {
        seed: game_seed,
        first_player: PlayerId::new(first_player),
        communication: CommunicationMode::LimitedChoices,
    };
    let mut state = model.init_state(&parameters, players)?;
    let agent_seeds = (0..players)
        .map(|seat| agent_seed.wrapping_add(seat as u64))
        .collect::<Vec<_>>();
    let mut agents = agent_seeds
        .iter()
        .copied()
        .map(RandomAgent::new)
        .collect::<Vec<_>>();
    println!(
        "The Game: {players} {}; first seat {first_player}",
        if players == 1 { "player" } else { "players" }
    );
    println!("Game seed: {game_seed}; agent seed base: {agent_seed}; agent seeds: {agent_seeds:?}");
    tag_rust::runner::run_game(&model, &mut state, &mut agents)?;
    let result = model.game_result(&state)?;
    let cards_placed = result
        .score
        .as_ref()
        .expect("The Game has a team score")
        .values[0];
    println!(
        "Team: {}; score: {cards_placed}/98 cards placed; cards remaining: {}; completed turns: {}",
        match state.public.outcome {
            TeamOutcome::Won => "win",
            TeamOutcome::Lost => "loss",
            TeamOutcome::Ongoing => "ongoing",
        },
        state.cards_remaining(),
        state.public.turns_completed
    );
    for seat in 0..players {
        println!(
            "Seat {seat}: {:?}",
            model.player_result(&state, PlayerId::new(seat))?
        );
    }
    Ok(())
}
