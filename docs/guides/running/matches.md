# Run matches

Use a match JSON to select one game, one agent per player and a number of games.
First try the supplied configuration:

```powershell
cargo run --release --features mcts-json,replay --example match_play -- examples/matches/mcts-vs-osla.json
```

## Make your own configuration

Save this as `examples/matches/my-match.json`:

```json
{
  "schema_version": 1,
  "game": {"name": "connect4"},
  "players": [
    {"name": "search", "agent": {"type": "mcts", "config_file": "../mcts/basic.json"}},
    {"name": "opponent", "agent": {"type": "osla"}}
  ],
  "games": 10,
  "positions": "rotate",
  "seed": 42,
  "reporting": {"path": "results/my-match.json", "include_actions": true}
}
```

Run it by replacing the filename in the command above. Paths inside the JSON
resolve from that JSON's directory. Existing result files are not overwritten;
choose another output name for a new experiment.

## Select the game and agents

| Setting | Choices |
|---|---|
| `game.name` | `tictactoe`, `connect4`, `cantstop`, `loveletter` |
| `game.parameters` | Game-specific setup/rules from the [game manuals](../README.md#game-manuals) |
| `agent.type` | `first_available`, `random`, `osla`, `mcts`, `neural_mcts` |
| `positions` | `rotate` cycles agents through every seat; `fixed` keeps configuration order |
| `max_actions` | Positive accepted-action limit, or null/omitted to disable; truncations have no win/draw/loss |
| `seed` | Integer for reproducibility; `null` selects and records an automatic seed |

Agent names identify competitors in results, independently of their seat.
Two agents can use the same algorithm with different names/configurations.
An agent type must be registered for the selected game's capabilities; see
[simple agents](../agents/basic-agents.md), [model compatibility](../agents/models.md)
and the [game manuals](../README.md#game-manuals). Stochastic lookahead uses
`repetitions_per_action` to average independent successor estimates; default 1.
Use an inline `config` object **or** `config_file` for each MCTS agent.
Native experiment registration is independent of browser/Python registration.
Research reports may contain seeds that reconstruct hidden state; do not expose
them to acting policies or ordinary player displays.
See [MCTS settings](../agents/mcts.md) for budgets, tree policies and native heuristics.

For a neural agent, use the supplied
[neural matchup](../../../examples/matches/neural-mcts-connect4.json) after training
its model, and enable `inference`:

```powershell
cargo run --release --features mcts-json,replay,inference --example match_play -- examples/matches/neural-mcts-connect4.json
```

[Saved models](../agents/models.md) explains the model file and agent bindings. Q-learning,
Neural OSLA and arbitrary Python-backed agents are not registered in this native CLI.

## Read/save the results

The console reports combined and per-seat outcomes. `reporting.path` saves full
JSON results; `include_actions: true` includes accepted moves. Each game has fresh
agents and separately derived seeds, which are recorded in its result.

`reporting.recordings` supports resumable recordings for basic native agents;
it supports the two placement games and rejects MCTS or repeated OSLA. For browser-playable MCTS action logs, use
[tournaments with separate logs](tournaments.md).

Details: [all match parameters and result fields](../../reference/experiments/matches.md).

For seeds shared across execution modes, see [reproducibility](README.md#reproducibility).
