//! Complete native classic games through the unchanged generic runner.
#[path = "support/seeds.rs"]
mod seeds;
use tag_rust::agents::RandomAgent;
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::loveletter::{LoveLetterForwardModel, LoveLetterParameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let players: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(2);
    let seed = seeds::resolve_seed(args.next().as_deref())?;
    let agent_seed = seeds::resolve_seed(args.next().as_deref())?;
    if args.next().is_some() {
        return Err(
            "usage: loveletter [players: 2-4] [game seed or auto] [agent seed or auto]".into(),
        );
    }
    let model = LoveLetterForwardModel;
    let parameters = LoveLetterParameters {
        seed,
        ..Default::default()
    };
    let mut state = model.init_state(&parameters, players)?;
    // Independent game/agent setup prevents an agent's seed revealing the shuffle
    // seed. Print both resolved sources so the entire run can be repeated.
    let agent_seeds: Vec<_> = (0..players)
        .map(|seat| agent_seed.wrapping_add(seat as u64))
        .collect();
    let mut agents: Vec<_> = agent_seeds.iter().copied().map(RandomAgent::new).collect();
    println!("Love Letter classic: {players} players; game seed {seed}");
    println!("Agent seed base: {agent_seed}");
    println!("Random agent seeds: {agent_seeds:?}");
    tag_rust::runner::run_game(&model, &mut state, &mut agents)?;
    let public = state.public();
    println!(
        "Actions: {}; completed turns: {}; completed rounds: {}",
        public.counters.move_count(),
        public.counters.completed_turn_count(),
        public.counters.round_count()
    );
    for seat in 0..players {
        let player = PlayerId::new(seat);
        println!(
            "Player {seat}: {:?}, {} favor tokens",
            model.player_result(&state, player)?,
            model.player_score(&state, player)?
        );
    }
    println!(
        "Winners: {:?}",
        public
            .outcome
            .winners()
            .iter()
            .map(|p| p.index())
            .collect::<Vec<_>>()
    );
    Ok(())
}
