//! Run a round-robin tournament from JSON and save its rankings and all game results.
#[allow(dead_code, unused_imports)] // Shared support also contains standalone match entry points.
#[path = "support/matches/mod.rs"]
mod matches;
#[path = "support/tournaments/mod.rs"]
mod tournaments;

fn main() {
    if let Err(error) = execute() {
        eprintln!("tournament: {error}");
        std::process::exit(1);
    }
}
fn execute() -> matches::Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: tournament <tournament.json>".into());
    }
    let settings = tournaments::TournamentConfig::load(std::path::Path::new(&arguments[0]))?;
    let report: tournaments::TournamentReport = tournaments::run(&settings)?;
    print!("{}", report.summary());
    matches::write_report(
        &settings.config.results_file,
        &serde_json::to_string_pretty(&report)?,
    )
    .map_err(|error| {
        format!(
            "could not save tournament results '{}': {error}",
            settings.config.results_file.display()
        )
    })?;
    println!("Results: {}", settings.config.results_file.display());
    if report.failure.is_some() {
        return Err("tournament stopped; partial results were saved".into());
    }
    Ok(())
}
