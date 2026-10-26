# Run round-robin tournaments

For operating instructions, use the [short manual](../../guides/running/tournaments.md). This page contains
the detailed reference.

A tournament selects agent groups, assigns them to seats and executes matches
across one or more game configurations. It produces a ranking for each game entry
and one global ranking, together with the results of every played game.

The application reuses the [match runner](matches.md). Available agents are First
Available, Random, native OSLA and native MCTS. Current playable games are
Tic-Tac-Toe, Connect 4, Can't Stop and Love Letter (the latter two support 2–4 players).
Love Letter reuses the same registry, schedule, seed handling and reports;
its [reference](../games/loveletter.md) defines sampling/evaluation capabilities and limits.
Each shared winner retains a `win`; tournament outcome points divide one point
equally among winners. Research logs include seeds that reproduce private deals
and must stay separate from ordinary player observations. The native tournament
CLI does not register local neural agents or action-replay export for Love Letter;
its browser Neural MCTS/session saves and Python evaluation are separate supported
workflows. Native Love Letter finishing places are unavailable: no score-ranking
convention is registered here, so finishing-place statistics remain null. Python
frozen-model evaluation separately supports token-score ranking.

The optional `inference` feature adds [neural MCTS](../learning/local-mcts.md), with
model files explicitly selected for each game-entry ID. Scheduling, ranking and
separate game logs retain the settings described on this page.

## Run from the repository root

```powershell
cargo run --offline --release --features mcts-json,replay --example tournament -- examples/tournaments/exhaustive.json
cargo run --offline --release --features mcts-json,replay --example tournament -- examples/tournaments/random.json
```

The first example runs three agents in both games. There are three distinct
two-agent groups and two seat allocations per group: six games per entry, twelve
in total. The second samples five groups per entry, with replacement, and plays
both seat allocations: ten games per entry, twenty in total.

Rankings are printed to stdout. Full JSON results are required and saved to
`examples/tournaments/results/exhaustive.json` or `random.json`, respectively.
Existing files are preserved; choose a new `results_file` for another run.
`exhaustive.json` disables move logs. `random.json` saves separate game metadata
and browser replay files under `results/random-game-logs`; choose a new log
directory as well when repeating that run.

## Configuration

```json
{
  "schema_version": 1,
  "agents": [
    {"name": "random", "agent": {"type": "random"}},
    {"name": "osla", "agent": {"type": "osla"}},
    {"name": "ucb1", "agent": {"type": "mcts", "config_file": "agents/ucb1.json"}}
  ],
  "games": [
    {"id": "ttt", "name": "tictactoe", "player_count": 2},
    {"id": "c4", "name": "connect4", "parameters": {"width": 7, "height": 6, "winning_length": 4}}
  ],
  "mode": "random",
  "matchups": 5,
  "seat_allocation": "permutations",
  "allow_self_play": false,
  "repetitions": 1,
  "ranking_metric": "win_rate",
  "seed": null,
  "results_file": "results/tournament.json",
  "game_logs": {"mode": "off"},
  "max_games": 100000
}
```

The illustrative `agents/ucb1.json` must exist and use the existing
[MCTS configuration schema](../agents/mcts-configuration.md). The runnable files
above reference the supplied MCTS examples. All relative agent and result paths
are resolved from the tournament JSON file's directory.

| Field | Meaning and default |
|---|---|
| `schema_version` | Required integer `1`. Unknown fields are rejected. |
| `agents` | Required nonempty agent pool. Names are unique identities. Agent types, optional base seeds and inline/file MCTS settings use the match schema. |
| `games` | Required nonempty list of game entries. Each entry has a unique `id`, registered game `name`, optional rule `parameters`, and `player_count` (default 2). Rule defaults come from the game. Different parameter sets can have separate IDs and rankings. |
| `games[].max_actions` | Positive accepted-action limit per played game; null/omitted disables it. Truncated games receive no outcomes or points. |
| `games[].ranking` | Null/omitted or `game` uses declared finishing places. Can't Stop needs opt-in `score` for its research ranking. |
| `mode` | Required `exhaustive` or `random`. |
| `matchups` | Positive sampled-group count per game entry, required in random mode. Omit in exhaustive mode. It is multiplied by seat allocations and repetitions. |
| `seat_allocation` | `permutations` by default: play every distinct seat allocation for a group. `random`: choose one allocation uniformly at random for each group. |
| `allow_self_play` | Default false. If true, a group may contain independent instances of the same named agent. Distinct names for the same algorithm remain distinct pool entries in either mode. |
| `repetitions` | Positive games per chosen seat allocation, default 1. Each repetition has fresh agents and seeds. |
| `ranking_metric` | `win_rate` by default, `points_rate` (outcome points), or `average_finishing_position` (lower is better). `rank_points_rate` remains a legacy JSON sorting option. This selects the sorting rule for all rankings. |
| `seed` | Unsigned 64-bit master seed; omitted/null uses operating-system randomness once and records the result. |
| `results_file` | Required nonempty JSON output path. Existing files are never overwritten. |
| `game_logs` | Optional object. `mode` is `off` (no move traces), `inline` (moves in the main report), or `separate` (per-match metadata and browser replay files). `directory` is required only for `separate`, relative to the tournament JSON file. |
| `include_actions` | Older setting, default false. True is equivalent to `game_logs.mode=inline`. Do not combine true with `game_logs`. |
| `max_games` | Positive schedule-size limit across all game entries, default 100,000 actual games after all multipliers. Exceeding it rejects the whole configuration before play. |

Configuration files and referenced MCTS files are limited to 1 MiB each. The
scheduler accepts player counts up to 64, subject to the game's own supported
counts. The schedule limit does not bound every allocation in simulations, action
traces or the final JSON file; including move traces increases memory and file size.

## Pairings, seats and self-play

A **group** identifies participating agents without assigning seats. A **seat
allocation** orders that group for actual play. For two distinct agents A and B,
the allocations are `[A,B]` and `[B,A]`. For three distinct agents there are six.

In exhaustive mode, every valid unordered group is selected once. Without
self-play, each agent occurs at most once in a group. With self-play, groups can
include repeated identities. For a pool `[A,B]` and two players, the groups are
`AA`, `AB` and `BB`; their distinct seat allocations are `AA`, `AB`, `BA` and `BB`.
Swapping indistinguishable A instances does not produce an extra configuration.

Random mode samples valid groups **uniformly with replacement**. The same group
can recur even after all groups have appeared. With self-play enabled, each valid
multiset is equally likely; for pool `[A,B]`, `AA`, `AB` and `BB` each have probability
one third. Seat allocation is a separate step. `random` chooses one allocation;
`permutations` plays all distinct allocations, without truncation.

For distinct two-agent groups, `matchups=5`, permutations enabled and
`repetitions=1` therefore produces **5 × 2 = 10 actual games per game entry**.
Changing repetitions to 3 produces 30. With random seat allocation, the same
counts produce 5 or 15. A self-play group such as AA has only one distinct
allocation, so the actual multiplier can vary; the report states the final count.

Game entries with the same player count reuse the same group and seat schedule.
They therefore have equal actual game counts and comparable opponent exposure,
including in random mode with self-play. Agent simulations still have independent
seeds. Games with different player counts can have different exhaustive schedule
sizes; global rankings use equal game weighting.

## Ranking rules

For each game entry:

- **Win rate:** wins divided by completed seat appearances. Draws contribute zero wins.
- **Outcome points rate (`points_rate`):** outcome points divided by completed
  seat appearances. Win=1; K tied winners each receive 1/K; loss=0. Two-player
  draws therefore retain 0.5 points per player.
- **Finishing-place points (`rank_points`):** an N-player game awards N, N-1, ...,
  1 for first through last. Equal ranks average their occupied places' points:
  ranks [1,2,2,4] give [4,2.5,2.5,1]. Games declare competition ranks with gaps
  after ties. Scores supply ranks only under an explicitly declared convention.
- **Average finishing position (`average_finishing_position`):** mean occupied
  place over completed appearances with final ranks. First=1, second=2, etc.
  Tied second/third places each contribute 2.5. Lower values are better; missing
  ranks and truncations are excluded. Positions are not normalized for player count.
  Combined results average per-entry means equally, including across different
  player counts. For example, entry means 1.5 and 3.0 give 2.25.
- **Legacy finishing-place rate (`rank_points_rate`):** retained in JSON for
  compatibility, along with raw `rank_points`. It is not shown in console rows.

Console rows show win rate, outcome points, average finishing position and
`games played` (completed games). Entry-count fields are omitted. Per-player
truncations are shown only if at least one player in that table has a nonzero
count; the tournament header also omits a zero truncation count.

Placement games declare their win/loss/draw finishing places. Can't Stop's rules
identify a winner but no full ordering of losing players. Its default therefore
has null `ranks` and `rank_points`. Set the entry's `ranking` to `score`
to rank by claimed columns, with ties sharing occupied places. This research
convention leaves game outcomes unchanged.

Truncated games have null points of both types. They are excluded from rate
denominators and counted in `truncated_games` and per-agent `truncated` counts.
Rank coverage is separate: `rank_games_covered` counts entries and
`ranked_appearances` counts appearances with declared finishing places.
Missing rank support does not become a zero-point result.

Tournament points serve ranking. They do not change game scores, learning rewards
or MCTS terminal values. Each self-play seat counts separately; an agent occupying
both seats receives both results. Every ranking includes all configured agents.
An agent with no completed appearances has null rates and rank, displayed as unranked.

The global rates average each agent's per-game rates with equal weight. For example,
one win in one Tic-Tac-Toe appearance and zero wins in nine Connect 4 appearances
give `(1 + 0) / 2 = 0.5`, rather than the pooled `1 / 10`. Raw counts are also reported.
If a game entry has no observations for an agent, its rate is missing rather than
zero. Average over observed entries and report `games_covered` out of `game_entries`;
coverage is particularly important for partial tournaments and multiplayer games.

The console labels the statistics **Outcome points** and **Average finishing
position**. JSON retains contributing-entry counts for research inspection;
these counts are not displayed in console rows.

Sort ascending for average finishing position and descending for win/outcome points. Equal numeric rates share rank positions
such as `1,1,3`; agent names give deterministic display order within ties. Unranked
agents appear last. These are empirical results for the sampled opponents and game
configurations; small tournaments do not establish playing strength reliably.

## Raw results, reproducibility and failures

The required file contains agent settings and base seeds, resolved game parameters
and rule versions, RNG/seed-derivation versions, requested scheduling settings,
planned and completed game counts, raw game records and **N+1 rankings**.

Each record identifies the game entry, pairing, seat allocation and repetition.
`agents_by_seat` indexes the saved agent pool, `agent_names` gives those names
directly, and `agent_seeds` records actual seeds. A completed record's `outcome`
contains per-seat results, move counts, MCTS counters and optional actions; `points`
contains outcome credit per seat; `rank_points` contains optional raw finishing-place credit. Seat-allocation statistics can be calculated
from these records without separate aggregate tables.

Fresh agents are constructed for each actual game. Scheduling uses separate RNG
streams for groups and seat choices, independently of agent simulations. Recorded
master/base seeds and realised allocations reproduce fixed-work runs under the
same code, evaluator, rules and RNG versions. Time budgets retain their existing
reproducibility limits. Placement game seeds are null. Can't Stop records each
actual environment seed; the [match reference](matches.md#random-seeds-and-fresh-agents)
defines its separate derivation. The [source module](../../../examples/support/tournaments/execution.rs)
defines the tournament's version-1 schedule and per-instance seed derivation.

All game/agent bindings and the complete bounded schedule are validated before
play. A runtime failure stops the tournament and saves partial results: completed
games retain their outcomes; the failed attempt records its lineup, seeds and
error with null outcome/points. It does not become a loss or enter rate denominators.
Partial failed-game states and action traces are not exported. A nonzero process
exit distinguishes failure from complete execution.

The main results file is written at completion or a handled runtime failure. This is not an
incremental checkpoint/resume format and does not guarantee saving after process
termination or power loss. Saving errors are reported with a nonzero exit. Tournament
results are research records. Separate logging, described below, also produces
browser-compatible action recordings with human seats. They do not restore the
original MCTS controllers; browser-created session saves can retain their own
controller checkpoints through a different export path.

## Separate game logs and browser replay

To keep moves outside the main report, configure:

```json
"game_logs": {
  "mode": "separate",
  "directory": "results/game-logs"
}
```

Use the supplied `examples/tournaments/random.json` for a complete example
(its log directory is `results/random-game-logs`).
The directory must be new: an existing directory is rejected before play to avoid
mixing tournaments. The main results file must be outside this directory. Both
paths are relative to the tournament configuration file. Choose fresh destinations
when repeating a run; files are never overwritten.

Each completed or truncated match writes research metadata immediately. Games
with browser replay support also write a companion recording:

- `game-000000.json`: research metadata, including tournament ID, match index,
  pairing, seat allocation, repetition, actual seeds, resolved game parameters,
  rules version, agent settings, outcome and a reference to the replay.
- `game-000000.tag-replay.json`: the existing portable browser replay format,
  including the initial state and every accepted action in order.

The main report retains the outcome and relative `game_log`/`replay_file` references,
but no move array. The metadata file identifies the main report and shares its
`tournament_id` and match `index`. The replay filename is relative to its metadata
file; main-report references are relative to the main report. Keep this folder
structure when moving results. Paths on different Windows drives are absolute.
For placement games, move sequences are stored once in the replay. Can't Stop
stores action/dice records in metadata `actions`, with effective rules and the
actual game seed; `replay_file` is null. These files support native reproduction,
not browser import or arbitrary-state restoration. In both cases, separate mode retains at most one
match's move sequence in memory. Outcome records still grow with match count.
The tournament ID uses operating-system randomness independently of experiment
seeds and does not change scheduling or agent decisions.

Open the **`.tag-replay.json`** file through the browser's existing file-load
control, rather than opening the metadata or main results file. Playback and
navigation use recorded moves, including moves originally selected by MCTS.
The replay uses human-controlled seats, so you can explore manual continuations;
it does not reconstruct the original agents or resume their search/RNG state.
Agent identities, configurations and actual seeds remain in the research metadata.

Separate mode checks the browser board limits before play: at most 128 cells per
dimension and 4096 cells total. Existing replay import size/work limits still
apply. The tournament action-log exporter produces browser recordings for the placement
games. Can't Stop and Love Letter retain native research action logs here; their
browser's own save/replay workflows are separate capabilities.

A game that fails during play has no completed replay. Previously saved logs
remain available if a later match fails. If saving a completed game's logs fails,
the tournament stops, preserves that game's outcome and points, and reports the
save error in the main results. A replay already saved before its metadata fails
retains a reference in that report. Writing completed-game logs immediately does
not add tournament resume support or guarantee saving the main report after a
crash.

See [browser files](../persistence/formats.md) for import limits and validation.

## Understand the result headings

Each per-entry heading identifies an `id` from the configuration, the game and
player count. IDs are user-chosen labels for independently played configurations;
they are not names of scoring formulas. The Can't Stop example has one entry
named `cantstop`, with all six seat permutations repeated twice (12 games).
Its `ranking: "score"` supplies finishing places alongside game outcomes. Both
statistics use completed games in that same set; `ranking_metric`
selects the table's ordering. Changing the metric does not change scheduling,
seeds or either points calculation. Automatic seeds still select a new experiment
on each invocation, so use an explicit seed when reproducing a run.

Additional entries are for different games or configurations that should play
separate matches. They are unnecessary for alternative scoring methods.
**Combined results across game entries** appears only with multiple entries,
replacing the earlier `Global` heading. It averages each entry's rates equally.
Finishing-place results are averaged only over entries with declared ranks.
JSON records how many entries supplied final places; these counts are omitted
from the console.

`ranking: "score"` is the game-independent name for an opt-in score-based final
ranking. Each game must explicitly support and define it; a numeric score alone
does not establish an ordering. For Can't Stop, higher claimed-column counts
rank first, with equal counts tied. The earlier `claimed_columns` value remains
accepted as a compatibility alias. `game` or omission uses the game's declared
natural finishing order, which may be incomplete or unavailable.
