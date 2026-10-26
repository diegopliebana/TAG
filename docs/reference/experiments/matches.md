# Configure and run matches

For operating instructions, use the [short manual](../../guides/running/matches.md). This page contains
the detailed reference.

To compare an agent pool across multiple games, use
[round-robin tournaments](tournaments.md). This page covers one configured matchup.

Use one match configuration to choose the game, rule parameters and an agent for
each player. This application supports Tic-Tac-Toe, Connect 4, Can't Stop and Love Letter with First
Available, Random, native OSLA and native MCTS. Each agent owns its evaluator and
random-number generator. The Rust runner executes the game rules.

With the optional `inference` feature, [local neural MCTS](../learning/local-mcts.md)
adds model-backed agents using the same match settings and seat rotation for
Tic-Tac-Toe and Connect 4. Local neural native match registration is absent for
Can't Stop and Love Letter, independently of their browser/Python support.

## Run an example

From the repository root:

```powershell
cargo run --offline --release --features mcts-json,replay --example match_play -- examples/matches/mcts-vs-osla.json
cargo run --offline --release --features mcts-json,replay --example match_play -- examples/matches/mcts-vs-mcts.json
cargo run --offline --release --features mcts-json,replay --example match_play -- examples/matches/random-vs-osla.json
```

The first command plays two Connect 4 games using PUCT against OSLA, alternating
their seats. The second compares two MCTS configurations in Tic-Tac-Toe. The third
prints a JSON report for five Random-versus-OSLA games, including accepted actions.
These examples establish execution; two or five games do not measure strength reliably.

The existing `--example mcts` remains a focused MCTS-versus-Random demonstration.
Use `match_play` when choosing opponents or running experiments.

## Match JSON

For example, save this beside your experiment files:

```json
{
  "schema_version": 1,
  "game": {"name": "connect4", "parameters": {"width": 7, "height": 6, "winning_length": 4}},
  "players": [
    {"name": "ucb1", "agent": {"type": "mcts", "config_file": "agents/ucb1.json"}},
    {"name": "osla", "seed": 17, "agent": {"type": "osla"}}
  ],
  "games": 20,
  "seed": null,
  "positions": "rotate",
  "reporting": {"format": "summary", "path": "results/match.json", "include_actions": true}
}
```

The agent file uses the existing [MCTS schema](../agents/mcts-configuration.md).
Every relative file or directory path is resolved from the **match file's directory**.
Launching the executable from another working directory preserves those references.

| Field | Accepted values and defaults |
|---|---|
| `schema_version` | Required integer `1`. Unknown fields are rejected. |
| `game.name` | Required `tictactoe`, `connect4`, `cantstop` or `loveletter`. |
| `game.parameters` | Object; omitted fields use the game's defaults. Tic-Tac-Toe accepts `grid_size`; Connect 4 accepts `width`, `height`, `winning_length`. Can't Stop accepts `rules` and an optional `seed` base; see its [reference](../games/cantstop.md). Love Letter accepts `rules.winning_tokens`, `first_player` and an optional `seed` base; see its [reference](../games/loveletter.md). The placement games also accept `limits: {"max_cells": N}`. Unknown parameters and invalid rule configurations fail before play. |
| `players` | Required ordered list. Each entry has a unique nonempty `name`, an `agent`, and an optional `seed`. The placement games require two players; Can't Stop supports 2-4. |
| `game.ranking` | Null/omitted or `game` uses the game's declared final ranking. Can't Stop supplies none by default; `score` explicitly opts into its research ranking. Equal scores tie. |
| `max_actions` | Positive accepted-player-action limit; null/omitted disables it. Terminal completion takes precedence at the limit. |
| `games` | Required positive integer, at most 10,000. Exactly this many games run unless an execution or output error stops the experiment. |
| `seed` | Unsigned 64-bit experiment seed. Omitted or `null` obtains one seed from operating-system randomness and reports it. |
| `positions` | `rotate` by default; `fixed` keeps configuration order as seat order. |
| `reporting.format` | `summary` by default, or `json`, selecting stdout output. |
| `reporting.path` | Optional JSON result-file path; omitted/`null` writes no report file. The file always contains the full JSON report regardless of stdout format. |
| `reporting.recordings` | Optional directory for existing-format replay files named `game-000000.json`, etc. Default disabled. Supported for the placement games with First Available, Random and ordinary one-repetition OSLA. |
| `reporting.include_actions` | Boolean, default false. Include each completed game's accepted actions in order, with the acting seat. These traces aid reproducibility checks; replay files additionally preserve controller progress. |

`reporting` can be omitted entirely. Existing output files are never overwritten.
Configuration files, including referenced agent files, are limited to 1 MiB each.
Known destination conflicts are checked before games start; parent directories
are created when writing outputs. Use a new report path and recording directory
for another experiment.

## Select each agent independently

| `agent` | Behaviour and settings |
|---|---|
| `{"type":"first_available"}` | Choose the first available action. |
| `{"type":"random"}` | Sample available actions uniformly. |
| `{"type":"osla"}` | Placement games retain the native open-line heuristic and original 0.001 tolerance. Can't Stop uses the progress heuristic and exact equal-mean ties. Optional `repetitions_per_action` is a positive integer (default 1). Repeated OSLA averages independently simulated successor values and selects a highest mean. Ties use the agent RNG. |
| `{"type":"mcts","config_file":"agents/search.json"}` | Load a complete existing MCTS JSON configuration. |
| `{"type":"mcts","config":{"schema_version":1,"budgets":{"iterations":1000}}}` | Use inline MCTS settings instead. Exactly one of `config` and `config_file` is required. |

To compare two MCTS agents, give them different names and independently select
their files or inline settings. Configuration order identifies agents throughout
the report, even when they change seats. In a three-game two-player experiment,
the first agent occupies seats `[0,1,0]` and the second `[1,0,1]`. Reports show
combined results and results for each seat; an odd game count is intentionally
unbalanced by one game. `fixed` instead keeps the first agent in seat 0.

The `inference` feature additionally registers local `neural_mcts` agents using
portable self-play models. [Saved models](../../guides/agents/models.md) shows their match
and tournament configuration. Arbitrary Python-backed agents, Q-table agents,
Neural OSLA and Rust-scored OSLAPython are not registered in this native CLI.
Native MCTS needs no Python dependency. Browser native/neural MCTS is implemented.

## Random seeds and fresh agents

The experiment resolves its master seed once. Each agent's **base seed** is chosen
from its player `seed`, otherwise its MCTS file's explicit `seed`, otherwise the
master seed. Specifying both a player seed and a non-null MCTS seed is an error.
Null MCTS seeds therefore inherit the experiment seed instead of independently
requesting randomness for each construction.

Every game creates fresh agents. Derive an actual per-game seed from the base
seed, zero-based game index and agent identity in configuration order. The seed
stays attached to that identity when positions rotate. An explicit seed in an
agent file is a **base** in this match workflow; direct MCTS construction retains
its existing meaning as the actual initial seed.

Version 1 derivation is defined by `derive_seed` in
[the application module](../../../examples/support/matches/mod.rs). It uses SplitMix64's
finaliser and explicit wrapping 64-bit arithmetic, without depending on how many
random numbers an earlier game consumed. Role 0 is reserved for game randomness;
agent k uses role k+1. Placement games retain `game_seed: null`.
Can't Stop derives its actual game seed using the same finaliser with role tag
`0x47414d45`, its base seed and game index. The base is `game.parameters.seed`,
or the resolved master seed when that field is null/omitted. In tournaments the
index is the global actual-game index, keeping environment streams distinct
across repetitions and entries. Results record the actual integer supplied to
`CantStopParameters`, sufficient to reproduce dice independently of agents.
Existing deterministic seed recipes are preserved.

The JSON report records the resolved master seed, each base seed, fully defaulted
MCTS settings and each game's actual agent seeds. Replace an automatic master seed
with its reported integer to repeat a fixed-work experiment. Time-limited MCTS
can complete different numbers of iterations between runs. Reproducibility also
requires the same rules, evaluator behaviour and RNG/software versions.

## Results and failures

The full report contains effective game parameters, agent names and algorithms,
seed derivation version, position schedule, requested/completed game counts,
per-agent/per-seat outcomes and per-game results. `seats`, `agent_seeds`, `results`
optional final `ranks` and MCTS counter totals use **agent configuration order**. Action traces identify
the acting seat. Search totals count completed iterations and forward-model calls
from accepted real decisions; basic agents have zero MCTS counters.

Complete configuration validation happens before gameplay: this includes native
evaluator construction, game parameter checks, unsupported recording requests,
unknown fields and all-disabled MCTS budgets. These errors produce a diagnostic
and a nonzero process exit without creating report or replay files.

An execution error stops at the first failed game. Earlier completed games and
recordings remain available; their totals are retained. The failed game has no
win/draw/loss entry. The report's `failure` identifies its zero-based game index
and diagnostic. Partial failed-game states and move traces are not exported.
A recording error after game termination retains that game's outcomes and stops
further games. Its recording path remains null if saving failed.

Stdout receives the selected report format before the optional report-file write.
For a runtime failure, the application still attempts to save the report, then
exits nonzero. If saving the report fails, stdout still contains the report and
stderr explains the file error. No automatic retries or experiment continuation
format are introduced.

## Read the implementation

[match_play.rs](../../../examples/match_play.rs) handles CLI arguments and output.
Its [support module](../../../examples/support/matches/mod.rs) separates configuration,
agent construction, execution and reporting. It is application code outside core
rules. Game registration dispatches to typed `MatchGame` implementations; agent
construction reuses existing `Agent` implementations. The application loop uses the shared ForwardModel/Agent operations and enforces
optional action limits. Recording uses existing `BranchSession` and replay export.

The registration table dispatches optional search, ranking and browser replay
capabilities. A minimal game can run Random without simulation reconstruction,
a heuristic, learning or persistence. Unsupported agent/output combinations fail
before play.

Each played game records `truncated` and `max_actions`. A truncated entry has
`results: ["truncated", ...]` and null ranks, while the engine state remains
ongoing. `completed_games` counts terminal games; `truncated_games` counts action
limits. Per-agent/per-seat counts retain truncations separately. Accepted action
traces include optional `chance: {"dice": [d0,d1,d2,d3]}` on Can't Stop Roll actions.
This is native research data, not a browser replay/state import format.

See the per-game references for rules, parameter domains and capability limits.
