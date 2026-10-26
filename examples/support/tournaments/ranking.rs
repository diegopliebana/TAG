//! Rankings use explicit per-appearance points, distinct from game scores or MCTS rewards.
use super::Result;
use super::config::Metric;
use super::execution::PlayedGame;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RankingRow {
    pub rank: Option<usize>,
    pub agent: String,
    pub appearances: usize,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub truncated: usize,
    pub win_rate: Option<f64>,
    pub points_rate: Option<f64>,
    pub rank_points_rate: Option<f64>,
    pub average_finishing_position: Option<f64>,
    pub ranked_appearances: usize,
    pub rank_games_covered: usize,
    /// Number of game entries with observations, out of entries in this ranking.
    pub games_covered: usize,
    pub game_entries: usize,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct GameRanking {
    pub game_id: String,
    pub rows: Vec<RankingRow>,
}
#[derive(Debug, Serialize, PartialEq)]
pub struct Rankings {
    pub sort_by: Metric,
    pub global_weighting: &'static str,
    pub per_game: Vec<GameRanking>,
    pub global: Vec<RankingRow>,
}

pub fn calculate(
    agents: &[String],
    game_ids: &[String],
    records: &[PlayedGame],
    metric: Metric,
) -> Result<Rankings> {
    let mut per_game = Vec::new();
    for (entry, id) in game_ids.iter().enumerate() {
        let mut rows: Vec<_> = agents
            .iter()
            .map(|agent| RankingRow {
                rank: None,
                agent: agent.clone(),
                appearances: 0,
                wins: 0,
                draws: 0,
                losses: 0,
                truncated: 0,
                win_rate: None,
                points_rate: None,
                rank_points_rate: None,
                average_finishing_position: None,
                ranked_appearances: 0,
                rank_games_covered: 0,
                games_covered: 0,
                game_entries: 1,
            })
            .collect();
        let mut position_totals = vec![0.0; agents.len()];
        let mut rank_totals = vec![0.0; agents.len()];
        let mut points = vec![0.0; agents.len()];
        for record in records.iter().filter(|r| r.entry_index == entry) {
            let Some(outcome) = &record.outcome else {
                continue;
            };
            if outcome.truncated {
                for agent in &record.agents_by_seat {
                    rows.get_mut(*agent)
                        .ok_or("unknown tournament agent index")?
                        .truncated += 1;
                }
                continue;
            }
            let credits = record
                .points
                .as_ref()
                .ok_or("completed tournament result has no points")?;
            if outcome.results.len() != record.agents_by_seat.len()
                || credits.len() != outcome.results.len()
            {
                return Err("tournament participant/result lengths differ".into());
            }
            if let Some(rank_points) = &record.rank_points
                && rank_points.len() != outcome.results.len()
            {
                return Err("rank-points participant lengths differ".into());
            }
            for (seat, agent) in record.agents_by_seat.iter().enumerate() {
                let row = rows
                    .get_mut(*agent)
                    .ok_or("unknown tournament agent index")?;
                let credit = credits[seat];
                if !credit.is_finite() || !(0.0..=1.0).contains(&credit) {
                    return Err("invalid tournament points".into());
                }
                row.appearances += 1;
                match outcome.results[seat].as_str() {
                    "win" => row.wins += 1,
                    "draw" => row.draws += 1,
                    "loss" => row.losses += 1,
                    _ => return Err("nonterminal result in tournament ranking".into()),
                }
                points[*agent] += credit;
                if let Some(rank_points) = &record.rank_points {
                    let value = rank_points[seat] / outcome.results.len() as f64;
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err("invalid rank points".into());
                    }
                    let position = outcome.results.len() as f64 + 1.0 - rank_points[seat];
                    if !(1.0..=outcome.results.len() as f64).contains(&position) {
                        return Err("invalid finishing position".into());
                    }
                    position_totals[*agent] += position;
                    rank_totals[*agent] += value;
                    row.ranked_appearances += 1;
                }
            }
        }
        for (((row, points), rank_total), position_total) in rows
            .iter_mut()
            .zip(points)
            .zip(rank_totals)
            .zip(position_totals)
        {
            if row.ranked_appearances > 0 {
                row.rank_points_rate = Some(rank_total / row.ranked_appearances as f64);
                row.average_finishing_position =
                    Some(position_total / row.ranked_appearances as f64);
                row.rank_games_covered = 1;
            }
            if row.appearances > 0 {
                row.win_rate = Some(row.wins as f64 / row.appearances as f64);
                row.points_rate = Some(points / row.appearances as f64);
                row.games_covered = 1;
            }
        }
        sort(&mut rows, metric);
        per_game.push(GameRanking {
            game_id: id.clone(),
            rows,
        });
    }
    let mut global = Vec::new();
    for agent in agents {
        let mut row = RankingRow {
            rank: None,
            agent: agent.clone(),
            appearances: 0,
            wins: 0,
            draws: 0,
            losses: 0,
            truncated: 0,
            win_rate: None,
            points_rate: None,
            rank_points_rate: None,
            average_finishing_position: None,
            ranked_appearances: 0,
            rank_games_covered: 0,
            games_covered: 0,
            game_entries: game_ids.len(),
        };
        let (mut win_sum, mut points_sum, mut rank_sum, mut position_sum) = (0.0, 0.0, 0.0, 0.0);
        for game in &per_game {
            let local = game
                .rows
                .iter()
                .find(|r| &r.agent == agent)
                .expect("each game lists every agent");
            row.appearances += local.appearances;
            row.wins += local.wins;
            row.draws += local.draws;
            row.losses += local.losses;
            row.truncated += local.truncated;
            row.ranked_appearances += local.ranked_appearances;
            if let Some(value) = local.rank_points_rate {
                row.rank_games_covered += 1;
                rank_sum += value;
                position_sum += local
                    .average_finishing_position
                    .ok_or("ranked entry has no average finishing position")?;
            }
            if let (Some(wins), Some(points)) = (local.win_rate, local.points_rate) {
                row.games_covered += 1;
                win_sum += wins;
                points_sum += points;
            }
        }
        // Missing observations are not losses. Report coverage so a rate based on
        // one game entry is distinguishable from one measured across every entry.
        if row.games_covered > 0 {
            row.win_rate = Some(win_sum / row.games_covered as f64);
            row.points_rate = Some(points_sum / row.games_covered as f64);
        }
        if row.rank_games_covered > 0 {
            row.rank_points_rate = Some(rank_sum / row.rank_games_covered as f64);
            row.average_finishing_position = Some(position_sum / row.rank_games_covered as f64);
        }
        global.push(row);
    }
    sort(&mut global, metric);
    Ok(Rankings {
        sort_by: metric,
        global_weighting: "equal_game_weight_over_observed_entries",
        per_game,
        global,
    })
}

fn sort(rows: &mut [RankingRow], metric: Metric) {
    let score = |row: &RankingRow| match metric {
        Metric::WinRate => row.win_rate,
        Metric::PointsRate => row.points_rate,
        Metric::RankPointsRate => row.rank_points_rate,
        Metric::AverageFinishingPosition => {
            row.average_finishing_position.map(|position| -position)
        }
    };
    rows.sort_by(|a, b| match (score(a), score(b)) {
        (Some(a_score), Some(b_score)) => b_score
            .total_cmp(&a_score)
            .then_with(|| a.agent.cmp(&b.agent)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.agent.cmp(&b.agent),
    });
    let mut previous = None;
    let mut rank = 0;
    for (index, row) in rows.iter_mut().enumerate() {
        if let Some(value) = score(row) {
            if previous != Some(value) {
                rank = index + 1;
            }
            row.rank = Some(rank);
            previous = Some(value);
        }
    }
}

/// Competition ranks use gaps after ties: [1,2,2,4], not [1,2,2,3].
/// Equal ranks receive the mean of the points for their occupied places.
pub fn rank_points(ranks: &[usize]) -> Result<Vec<f64>> {
    if ranks.is_empty() {
        return Err("final ranks must be nonempty".into());
    }
    let mut sorted = ranks.to_vec();
    sorted.sort_unstable();
    let mut offset = 0;
    while offset < sorted.len() {
        if sorted[offset] != offset + 1 {
            return Err("final ranks must use competition ranking with gaps after ties".into());
        }
        let count = sorted[offset..]
            .iter()
            .take_while(|&&r| r == sorted[offset])
            .count();
        offset += count;
    }
    Ok(ranks
        .iter()
        .map(|&rank| {
            let tied = ranks.iter().filter(|&&r| r == rank).count();
            ranks.len() as f64 - (rank - 1) as f64 - (tied - 1) as f64 / 2.0
        })
        .collect())
}
