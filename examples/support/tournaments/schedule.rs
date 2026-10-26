//! Agent groups are unordered; seat allocations turn them into ordered lineups.
use super::Result;
use super::config::{Mode, SeatAllocation};
use serde::Serialize;
use std::collections::BTreeSet;
use tag_rust::core::random::AgentRng;

// A fixed tag gives group sampling a seed stream separate from seat shuffling.
const GROUP_SEED_TAG: u64 = 0x47524f555053;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ScheduledGame {
    pub pairing: usize,
    pub allocation: usize,
    pub repetition: usize,
    /// Global agent-pool indices in seat order. Repeated indices mean self-play.
    pub agents_by_seat: Vec<usize>,
}
pub struct ScheduleSettings {
    pub agent_count: usize,
    pub player_count: usize,
    pub mode: Mode,
    pub matchups: Option<usize>,
    pub seat_allocation: SeatAllocation,
    pub allow_self_play: bool,
    pub repetitions: usize,
    pub seed: u64,
    pub max_games: usize,
}

/// Enumerate distinct groups or sample uniformly from them with replacement.
/// With self-play, groups are multisets: AA, AB and BB are three distinct groups.
pub fn build(settings: &ScheduleSettings) -> Result<Vec<ScheduledGame>> {
    let n = settings.agent_count;
    let p = settings.player_count;
    if n == 0
        || p == 0
        || p > 64
        || settings.repetitions == 0
        || (!settings.allow_self_play && p > n)
    {
        return Err("invalid schedule dimensions".into());
    }
    let mut rng = AgentRng::new(settings.seed);
    let mut output = Vec::new();
    let mut pairing = 0;
    let mut emit = |group: &[usize]| -> Result<()> {
        let mut lineup = group.to_vec();
        if settings.seat_allocation == SeatAllocation::Random {
            shuffle(&mut lineup, &mut rng);
        }
        let mut allocation = 0;
        loop {
            for repetition in 0..settings.repetitions {
                if output.len() >= settings.max_games {
                    return Err("tournament schedule exceeds max_games; increase the limit or reduce the schedule".into());
                }
                output.push(ScheduledGame {
                    pairing,
                    allocation,
                    repetition,
                    agents_by_seat: lineup.clone(),
                });
            }
            if settings.seat_allocation == SeatAllocation::Random || !next_permutation(&mut lineup)
            {
                break;
            }
            allocation += 1;
        }
        pairing += 1;
        Ok(())
    };
    match settings.mode {
        Mode::Exhaustive => groups(
            n,
            p,
            settings.allow_self_play,
            0,
            &mut Vec::new(),
            &mut emit,
        )?,
        Mode::Random => {
            let count = settings.matchups.ok_or("random schedule needs matchups")?;
            if count == 0 || count > settings.max_games {
                return Err("random matchups exceeds max_games or is zero".into());
            }
            // Group sampling has its own RNG. Seat shuffling cannot alter which
            // groups are selected, and agent simulations cannot alter either stream.
            let mut group_rng = AgentRng::new(crate::matches::derive_seed(
                settings.seed,
                0,
                GROUP_SEED_TAG,
            ));
            for _ in 0..count {
                emit(&sample_group(
                    n,
                    p,
                    settings.allow_self_play,
                    &mut group_rng,
                )?)?;
            }
        }
    }
    Ok(output)
}

fn groups(
    n: usize,
    left: usize,
    self_play: bool,
    start: usize,
    group: &mut Vec<usize>,
    emit: &mut impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    if left == 0 {
        return emit(group);
    }
    for agent in start..n {
        if !self_play && n - agent < left {
            break;
        }
        group.push(agent);
        groups(
            n,
            left - 1,
            self_play,
            if self_play { agent } else { agent + 1 },
            group,
            emit,
        )?;
        group.pop();
    }
    Ok(())
}

fn sample_group(n: usize, p: usize, self_play: bool, rng: &mut AgentRng) -> Result<Vec<usize>> {
    // Multisets map to subsets of n+p-1 positions: sorted_position[i] - i
    // recovers each agent index. This samples multisets uniformly, including AA.
    let total = if self_play {
        n.checked_add(p - 1).ok_or("group size overflow")?
    } else {
        n
    };
    let mut chosen = BTreeSet::new();
    for upper in total - p..total {
        let candidate = rng.index(upper + 1).ok_or("empty sampling range")?;
        if !chosen.insert(candidate) {
            chosen.insert(upper);
        }
    }
    Ok(chosen
        .into_iter()
        .enumerate()
        .map(|(i, x)| if self_play { x - i } else { x })
        .collect())
}
fn shuffle(values: &mut [usize], rng: &mut AgentRng) {
    for end in (1..values.len()).rev() {
        values.swap(end, rng.index(end + 1).expect("positive shuffle range"));
    }
}
/// Generate each distinct lexicographic permutation once, including repeated IDs.
fn next_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|i| values[*i] < values[*i + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|i| values[*i] > values[pivot])
        .expect("increasing suffix exists");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}
