# Configure MCTS agents

MCTS simulates possible moves and uses visit/value statistics to choose an action.
Use it in [matches](../running/matches.md), [tournaments](../running/tournaments.md) or the [browser](../running/browser.md).
For individual native/neural decisions and independent-tree batching from Python,
use [Run Rust MCTS from Python](rust-mcts-from-python.md).

## A native configuration

```json
{
  "schema_version": 1,
  "seed": null,
  "tree_policy": "ucb1",
  "tree_depth": 100,
  "rollout_length": 0,
  "budgets": {"time_ms": null, "iterations": 1000, "forward_model_calls": null},
  "state_evaluator": {"name": "game_heuristic"}
}
```

Save it as your agent's `config_file`, or paste it into the browser's MCTS JSON.
Native Rust heuristics need no neural model or Python process.

| Change | Setting |
|---|---|
| Time per decision | `budgets.time_ms`, in milliseconds |
| Iterations per decision | `budgets.iterations` |
| Simulated transitions per decision | `budgets.forward_model_calls` |
| Tree policy | `tree_policy`: `ucb1` or `puct` |
| Maximum tree depth | `tree_depth`: positive integer or `null` |
| Maximum rollout length | `rollout_length`: nonnegative integer or `null` |
| Automatic/fixed randomness | `seed`: `null` or an integer |

Enable several budgets to stop when any is reached; disable each independently
with `null`. At least one budget must remain enabled. Match/tournament per-game
seeds and browser seat seeds override the search template's seed.

`rollout_length: 0` evaluates the expanded leaf. Reaching `tree_depth` evaluates
the selected node immediately, without expansion or rollout. A disabled rollout
limit continues until terminal or another computational budget stops simulation.

## Heuristics and neural PUCT

The [native PUCT example](../../../examples/mcts/puct.json) demonstrates independent
progressive-bias, expansion and rollout heuristics. Configure each role separately.
Progressive bias can be disabled or use either supported formula; rollout policy
can be random, greedy or softmax.

For neural PUCT, use [neural.json](../../../examples/mcts/neural.json) and a
[self-play model](../learning/mcts-selfplay.md). The agent's model bindings provide
`network_value` and `network_policy`; [saved models](models.md) shows how to attach
the file. Models must match the game, rules, observations and value convention.

Details: [every JSON parameter and formula](../../reference/agents/mcts-configuration.md),
[native search API](../../reference/agents/mcts-api.md),
[Python batched neural search](../../reference/learning/neural-mcts.md).
