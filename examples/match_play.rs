//! Run any supported native game/agent combination from one match configuration.
#[path = "support/matches/mod.rs"]
mod matches;
use std::path::Path;

fn main() {
    if let Err(error) = execute() {
        eprintln!("match_play: {error}");
        std::process::exit(1);
    }
}
fn execute() -> matches::Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: match_play <match.json>".into());
    }
    let settings = matches::MatchConfig::load(Path::new(&arguments[0]))?;
    let report: matches::MatchReport = matches::run(&settings)?;
    let json = serde_json::to_string_pretty(&report)?;
    match settings.config.reporting.format {
        matches::Format::Summary => print!("{}", report.summary()),
        matches::Format::Json => println!("{json}"),
    }
    if let Some(path) = &settings.config.reporting.path {
        // Stdout has already received the report even if saving it fails.
        matches::write_report(path, &json)?;
    }
    if report.failure.is_some() {
        return Err(
            "experiment stopped; see the report for completed games and diagnostics".into(),
        );
    }
    Ok(())
}
