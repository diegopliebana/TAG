//! Prepare the whole tournament before play and retain every completed or failed attempt.
use super::Result;
use super::config::{ResolvedTournament, TournamentGame};
use super::ranking::{self, Rankings};
use super::schedule::{self, ScheduleSettings, ScheduledGame};
use crate::matches::config::{MatchConfig, Positions, Reporting, ResolvedMatch};
use crate::matches::derive_seed;
use serde::Serialize;
use serde_json::Value;

// Fixed tags distinguish schedule seeds from agent-instance seeds. Their values
// are part of the version-1 seed recipe and must remain unchanged for replay.
const SCHEDULE_SEED_TAG: u64 = 0x5343484544554c45;
const AGENT_SEED_TAG: u64 = 0x4147454e54;

#[derive(Debug, Serialize, PartialEq)]
pub struct AgentManifest {
    pub name: String,
    pub algorithm: String,
    pub base_seed: u64,
    pub settings: Value,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct GameManifest {
    pub id: String,
    pub name: String,
    pub player_count: usize,
    pub ranking: Option<String>,
    pub parameters: Value,
    pub rules_version: u32,
    pub schedule_seed: u64,
    pub planned_pairings: usize,
    pub planned_games: usize,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct PlayedGame {
    pub index: usize,
    pub entry_index: usize,
    pub game_id: String,
    pub pairing: usize,
    pub allocation: usize,
    pub repetition: usize,
    /// Global pool indices and names in seat order; names may repeat in self-play.
    pub agents_by_seat: Vec<usize>,
    pub agent_names: Vec<String>,
    pub agent_seeds: Vec<u64>,
    pub outcome: Option<crate::matches::report::GameReport>,
    pub points: Option<Vec<f64>>,
    pub rank_points: Option<Vec<f64>>,
    pub error: Option<String>,
    pub game_log: Option<std::path::PathBuf>,
    pub replay_file: Option<std::path::PathBuf>,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct TournamentReport {
    pub schema_version: u32,
    pub engine_version: &'static str,
    pub rng_algorithm: &'static str,
    pub rng_version: u32,
    pub rank_points_rule: &'static str,
    pub points_rule: &'static str,
    pub seed_derivation_version: u32,
    pub master_seed: u64,
    pub mode: super::config::Mode,
    pub configured_matchups: Option<usize>,
    pub seat_allocation: super::config::SeatAllocation,
    pub repetitions: usize,
    pub allow_self_play: bool,
    pub max_games: usize,
    pub include_actions: bool,
    pub game_log_mode: super::config::LogMode,
    pub tournament_id: Option<String>,
    pub agents: Vec<AgentManifest>,
    pub game_entries: Vec<GameManifest>,
    pub planned_games: usize,
    pub completed_games: usize,
    pub truncated_games: usize,
    pub failure: Option<String>,
    pub games: Vec<PlayedGame>,
    pub rankings: Rankings,
}

pub fn run(settings: &ResolvedTournament) -> Result<TournamentReport> {
    validate_destination(&settings.config.results_file)?;
    let mut entries = Vec::new();
    let mut schedules = Vec::new();
    let mut total = 0usize;
    for game in &settings.config.games {
        let pool = settings
            .agents
            .iter()
            .map(|agent| agent.for_game(&game.id))
            .collect::<Result<Vec<_>>>()?;
        let validated =
            crate::matches::execution::validate_pool(&game.game_config(), game.player_count, &pool)
                .map_err(|e| format!("game entry '{}': {e}", game.id))?;
        if settings.log_mode() == super::config::LogMode::Separate
            && let Some((width, height)) = validated.replay_dimensions
            && (width > 128
                || height > 128
                || width.checked_mul(height).is_none_or(|cells| cells > 4096))
        {
            return Err(format!("game entry '{}': separate browser replays require at most 128 cells per dimension and 4096 cells total", game.id).into());
        }
        // Reuse the pairing/seat schedule for game entries with the same player
        // count. This gives comparable opponents and equal actual schedule sizes,
        // including when self-play changes the number of distinct permutations.
        let schedule_seed = derive_seed(
            settings.master_seed,
            game.player_count as u64,
            SCHEDULE_SEED_TAG,
        );
        let schedule = schedule::build(&ScheduleSettings {
            agent_count: settings.agents.len(),
            player_count: game.player_count,
            mode: settings.config.mode,
            matchups: settings.config.matchups,
            seat_allocation: settings.config.seat_allocation,
            allow_self_play: settings.config.allow_self_play,
            repetitions: settings.config.repetitions,
            seed: schedule_seed,
            max_games: settings
                .config
                .max_games
                .checked_sub(total)
                .ok_or("tournament exceeds max_games")?,
        })
        .map_err(|e| format!("game entry '{}': {e}", game.id))?;
        total = total
            .checked_add(schedule.len())
            .ok_or("tournament game count overflow")?;
        entries.push(GameManifest {
            id: game.id.clone(),
            name: game.name.clone(),
            player_count: game.player_count,
            ranking: game.ranking.clone(),
            parameters: validated.parameters,
            rules_version: validated.rules_version,
            schedule_seed,
            planned_pairings: schedule.last().map_or(0, |r| r.pairing + 1),
            planned_games: schedule.len(),
        });
        schedules.push(schedule);
    }
    let agents = settings
        .agents
        .iter()
        .map(|agent| {
            let (algorithm, effective) = crate::matches::report::describe_agent(agent)?;
            Ok(AgentManifest {
                name: agent.name.clone(),
                algorithm: algorithm.into(),
                base_seed: agent.base_seed,
                settings: effective,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let separate = settings.log_mode() == super::config::LogMode::Separate;
    let tournament_id = if separate {
        let directory = settings
            .config
            .game_logs
            .as_ref()
            .and_then(|logs| logs.directory.as_ref())
            .expect("validated directory");
        let result = super::logs::normal(&settings.config.results_file)?;
        let logs = super::logs::normal(directory)?;
        if result.starts_with(&logs) {
            return Err("results_file must be outside the game log directory".into());
        }
        if let Some(parent) = directory.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Reserve a fresh directory, so a previous tournament is never mixed in.
        std::fs::create_dir(directory).map_err(|e| {
            format!(
                "could not create new game log directory '{}': {e}",
                directory.display()
            )
        })?;
        Some(format!(
            "{:016x}",
            tag_rust::search::mcts::automatic_seed()?
        ))
    } else {
        None
    };
    let mut records = Vec::new();
    let mut failure = None;
    'entries: for (entry, schedule) in schedules.iter().enumerate() {
        for row in schedule {
            let mut record = play(
                settings,
                &settings.config.games[entry],
                entry,
                row,
                records.len(),
            );
            if separate
                && record.outcome.is_some()
                && let Err(error) = super::logs::save(
                    settings,
                    tournament_id.as_deref().expect("separate logs have an ID"),
                    &mut record,
                    &entries[entry],
                )
            {
                record.error = Some(format!(
                    "game {} completed but saving game logs failed: {error}",
                    record.index
                ));
                record.outcome.as_mut().expect("completed game").actions = None;
            }
            failure = record.error.clone();
            records.push(record);
            if failure.is_some() {
                break 'entries;
            }
        }
    }
    let names: Vec<_> = settings.agents.iter().map(|a| a.name.clone()).collect();
    let ids: Vec<_> = settings.config.games.iter().map(|g| g.id.clone()).collect();
    let rankings = ranking::calculate(&names, &ids, &records, settings.config.ranking_metric)?;
    Ok(TournamentReport {
        schema_version: 1,
        engine_version: env!("CARGO_PKG_VERSION"),
        rng_algorithm: "ChaCha8",
        rng_version: 1,
        rank_points_rule: "N,N-1,...,1; ties average occupied places; normalized rates divide by N; unsupported/truncated excluded",
        points_rule: "win=1; tied winners share 1 equally; loss=0; truncated games excluded",
        seed_derivation_version: 1,
        master_seed: settings.master_seed,
        mode: settings.config.mode,
        configured_matchups: settings.config.matchups,
        seat_allocation: settings.config.seat_allocation,
        repetitions: settings.config.repetitions,
        allow_self_play: settings.config.allow_self_play,
        agents,
        game_entries: entries,
        max_games: settings.config.max_games,
        include_actions: settings.config.include_actions,
        game_log_mode: settings.log_mode(),
        tournament_id,
        planned_games: total,
        completed_games: records
            .iter()
            .filter(|r| r.outcome.as_ref().is_some_and(|o| !o.truncated))
            .count(),
        truncated_games: records
            .iter()
            .filter(|r| r.outcome.as_ref().is_some_and(|o| o.truncated))
            .count(),
        failure,
        games: records,
        rankings,
    })
}

/// Agent seeds depend on the global pool identity and actual game index. A second
/// instance of an agent in self-play receives another independent per-seat seed.
fn play(
    settings: &ResolvedTournament,
    game: &TournamentGame,
    entry: usize,
    row: &ScheduledGame,
    index: usize,
) -> PlayedGame {
    let lineup: Vec<_> = row
        .agents_by_seat
        .iter()
        .map(|agent| {
            settings.agents[*agent]
                .for_game(&game.id)
                .expect("every game-entry model binding was validated before scheduling")
        })
        .collect();
    let seeds: Vec<_> = row
        .agents_by_seat
        .iter()
        .enumerate()
        .map(|(seat, agent)| {
            let base = derive_seed(
                settings.agents[*agent].base_seed,
                *agent as u64,
                AGENT_SEED_TAG,
            );
            derive_seed(base, index as u64, seat as u64 + 1)
        })
        .collect();
    let resolved = ResolvedMatch {
        game_index_offset: index,
        config: MatchConfig {
            schema_version: 1,
            game: game.game_config(),
            players: row
                .agents_by_seat
                .iter()
                .map(|i| settings.config.agents[*i].clone())
                .collect(),
            games: 1,
            seed: Some(settings.master_seed),
            positions: Positions::Fixed,
            max_actions: game.max_actions,
            reporting: Reporting {
                include_actions: settings.log_mode() != super::config::LogMode::Off,
                ..Reporting::default()
            },
        },
        master_seed: settings.master_seed,
        players: lineup,
    };
    let mut record = PlayedGame {
        index,
        entry_index: entry,
        game_id: game.id.clone(),
        pairing: row.pairing,
        allocation: row.allocation,
        repetition: row.repetition,
        agents_by_seat: row.agents_by_seat.clone(),
        agent_names: resolved.players.iter().map(|a| a.name.clone()).collect(),
        agent_seeds: seeds.clone(),
        outcome: None,
        points: None,
        rank_points: None,
        error: None,
        game_log: None,
        replay_file: None,
    };
    match crate::matches::execution::run_scheduled(&resolved, &seeds) {
        Ok(mut outcome) => {
            outcome.index = index;
            if !outcome.truncated {
                let winners = outcome
                    .results
                    .iter()
                    .filter(|r| r.as_str() == "win")
                    .count();
                let ties = outcome
                    .results
                    .iter()
                    .filter(|r| r.as_str() == "draw")
                    .count();
                record.points = Some(
                    outcome
                        .results
                        .iter()
                        .map(|result| match result.as_str() {
                            "win" => 1.0 / winners as f64,
                            "draw" => 1.0 / ties as f64,
                            "loss" => 0.0,
                            _ => unreachable!("native runner returns terminal result labels"),
                        })
                        .collect(),
                );
            }
            if let Some(ranks) = &outcome.ranks {
                match ranking::rank_points(ranks) {
                    Ok(points) => record.rank_points = Some(points),
                    Err(error) => {
                        record.error = Some(error.to_string());
                        return record;
                    }
                }
            }
            record.outcome = Some(outcome);
        }
        Err(error) => {
            record.error = Some(format!(
                "game '{}' at tournament index {index}: {error}",
                game.id
            ))
        }
    }
    record
}

fn validate_destination(path: &std::path::Path) -> Result<()> {
    if path.exists() {
        return Err(format!("results file already exists: {}", path.display()).into());
    }
    for parent in path
        .ancestors()
        .skip(1)
        .filter(|p| !p.as_os_str().is_empty())
    {
        if parent.exists() && !parent.is_dir() {
            return Err(format!("results parent is not a directory: {}", parent.display()).into());
        }
    }
    Ok(())
}

impl TournamentReport {
    pub fn summary(&self) -> String {
        let metric = match self.rankings.sort_by {
            super::config::Metric::WinRate => "win rate",
            super::config::Metric::PointsRate => "outcome points rate",
            super::config::Metric::RankPointsRate => "finishing-place points rate (legacy)",
            super::config::Metric::AverageFinishingPosition => {
                "average finishing position, lower is better"
            }
        };
        let truncated = if self.truncated_games > 0 {
            format!("; truncated={}", self.truncated_games)
        } else {
            String::new()
        };
        let mut text = format!(
            "Tournament: {}/{} games completed{truncated}; master_seed={}\n",
            self.completed_games, self.planned_games, self.master_seed
        );
        text.push_str("Average finishing position: 1 is first; lower is better. Ties average occupied positions. Games without final places and truncations are excluded.\n");
        text.push_str("Win rate counts wins only. Outcome points: win=1, tied winners share 1 equally, loss=0; a two-player draw gives 0.5 points.\n");
        text.push_str("Both statistics use the same completed games within each entry; the sorting metric only orders the table. Games played counts completed games; truncations are reported separately when present.\n");
        if self.game_entries.len() > 1 {
            text.push_str("Each game entry is a separately played configuration identified by its JSON id. Combined results average each entry's available statistics equally.\n");
        }
        for (index, (label, rows)) in self
            .rankings
            .per_game
            .iter()
            .map(|r| (r.game_id.as_str(), &r.rows))
            .chain(
                (self.game_entries.len() > 1)
                    .then_some(("Combined results", &self.rankings.global)),
            )
            .enumerate()
        {
            if let Some(entry) = self.game_entries.get(index) {
                text.push_str(&format!(
                    "\nGame entry '{label}': {}, {} players (sorted by {metric})\n",
                    entry.name, entry.player_count
                ));
                let ranking = match entry.ranking.as_deref() {
                    Some("score" | "claimed_columns") => "game score (opt-in)",
                    None | Some("game") => "game rules",
                    Some(rule) => rule,
                };
                let availability = if rows
                    .iter()
                    .any(|row| row.average_finishing_position.is_some())
                {
                    "finishing-place results available"
                } else {
                    "no finishing-place results available"
                };
                text.push_str(&format!("  Final ranking: {ranking}; {availability}.\n"));
            } else {
                text.push_str(&format!(
                    "\nCombined results across {} game entries (sorted by {metric})\n",
                    self.game_entries.len()
                ));
            }
            let show_truncated = rows.iter().any(|row| row.truncated > 0);
            for row in rows {
                let rates = match (row.win_rate, row.points_rate) {
                    (Some(wins), Some(points)) => {
                        format!(
                            "Win rate={:.2}%; Outcome points={:.2}%",
                            wins * 100.0,
                            points * 100.0
                        )
                    }
                    _ => "unranked".into(),
                };
                let position = row
                    .average_finishing_position
                    .map_or("unavailable".into(), |position| format!("{position:.2}"));
                let truncated = if show_truncated {
                    format!("; truncated={}", row.truncated)
                } else {
                    String::new()
                };
                text.push_str(&format!(
                    "  {} {}: {rates}; Average finishing position={position}; games played={}{truncated}\n",
                    row.rank.map_or("-".into(), |r| r.to_string()), row.agent, row.appearances,
                ));
            }
        }
        if let Some(error) = &self.failure {
            text.push_str(&format!("Stopped: {error}\n"));
        }
        text
    }
}
