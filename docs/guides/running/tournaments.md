# Run tournaments

A tournament compares an agent pool over one or more game configurations.
Start with a supplied example:

```powershell
cargo run --release --features mcts-json,replay --example tournament -- examples/tournaments/exhaustive.json
```

## Select games and competitors

Save this as `examples/tournaments/my-tournament.json`:

```json
{
  "schema_version": 1,
  "agents": [
    {"name": "random", "agent": {"type": "random"}},
    {"name": "osla", "agent": {"type": "osla"}},
    {"name": "ucb1", "agent": {"type": "mcts", "config_file": "../mcts/basic.json"}}
  ],
  "games": [
    {"id": "ttt", "name": "tictactoe"},
    {"id": "c4", "name": "connect4", "parameters": {"width": 7, "height": 6, "winning_length": 4}}
  ],
  "mode": "random",
  "matchups": 5,
  "seat_allocation": "permutations",
  "allow_self_play": false,
  "ranking_metric": "win_rate",
  "seed": 42,
  "results_file": "results/my-tournament.json",
  "game_logs": {"mode": "separate", "directory": "results/my-tournament-games"}
}
```

Run it by replacing the filename in the command above. All relative paths start
at the tournament JSON's directory. Use fresh output names/directories for each run.

## Choose how many matches to play

| Setting | Meaning |
|---|---|
| `mode: "exhaustive"` | Play every agent group; omit `matchups` |
| `mode: "random"` | Sample `matchups` groups per game entry; groups may repeat |
| `seat_allocation: "permutations"` | Play every seat allocation for each group |
| `seat_allocation: "random"` | Pick one seat allocation for each group |
| `repetitions` | Repeat each chosen allocation this many times; default 1 |
| `allow_self_play` | Allow one named agent to occupy multiple seats; default false |

For two-player games, 5 groups × 2 seat permutations = **10 games per game
entry**, or 20 across the example's two entries. Distinct IDs distinguish separately played rule configurations of the same game.
Choose player counts/parameters from its [game manual](../README.md#game-manuals).
Each entry accepts optional `max_actions` (null disables it). Keep hidden-game
research reports and seeds separate from ordinary player observations.

Agent types/settings are the same as in [matches](matches.md). For neural agents,
train a model for each game/configuration and use `model_files` keyed by entry ID;
see [loading models into tournaments](../agents/models.md).

```powershell
cargo run --release --features mcts-json,replay,inference --example tournament -- examples/tournaments/neural.json
```

## Results and browser replay

`results_file` contains every attempted game's result and summary statistics.
`ranking_metric` chooses `win_rate`, `points_rate` (outcome points), or
`average_finishing_position` (lower is better). Outcome points give 1 to a winner,
share 1 among tied winners, and give 0 to losses. Win rate excludes draw credit;
one win, one two-player draw and one loss give 33.33% wins and 50% outcome points.

Average finishing position reports places directly: first=1, second=2, and so on.
Ties average occupied positions: two players tied for second/third each receive
2.5. A game must supply a finishing-place convention; `ranking: "score"` is an
opt-in where that game supports public-score ranking.
Missing finishing places and truncated games are excluded from these averages.
Combined statistics give equal weight to each entry with available results.
Positions are not normalized for player count.

`games played` counts completed games. Truncation counts appear separately only
when at least one player in the table has a truncated game. Both statistics are
calculated from the same games; changing the sorting metric only orders rows.

`game_logs.mode` is `off`, `inline` or `separate`. Separate mode writes individual
game metadata and, for supported games, browser replay files, cross-referenced from
the main results. Games without this action-replay exporter retain native
research traces instead. This does not imply that their browser sessions lack
save/load support: the two exporters have different capabilities.
Load a replay in the [browser](browser.md) to review its moves. These logs replay
accepted actions with human controllers; they do not resume the original agents.

Details: [complete tournament configuration, counts and scoring](../../reference/experiments/tournaments.md).

Result headings show the configured game-entry ID, game name and player count.
IDs label separately played configurations; **Combined results across game
entries** averages their rates when there is more than one entry. A single game entry produces one set of tables; adding rule/game entries creates
separately played configurations. Changing the ranking metric changes only table
ordering, not which games are played.
`ranking: "score"` asks the game to supply finishing places from its score;
games without an explicitly supported score ranking reject that option.

Seed choices apply here as in other runners; see [reproducibility](README.md#reproducibility).
