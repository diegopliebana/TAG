# Save, load and use trained models

A trained model stores learned action-value estimates or the parameters of a
policy/value network. An agent loads it to obtain predictions when choosing moves.
Use this guide after training to select the right file, attach it to a controller
and run it in the browser or a native experiment. For measuring strength separately from play, use [frozen-model evaluation](../learning/evaluate-mcts-models.md).

The **evaluation JSON** contains the table or neural weights plus their game/input/
output contract. A neural **`.pt` checkpoint** also stores training state and is
used by the training runner to resume learning. Loading evaluation JSON for play
does not update its weights. Training chooses the output paths below.

| Training method | Example evaluation file | Compatible agents |
|---|---|---|
| [Q-learning](../learning/tabular-learning.md) | `target/models/tictactoe-q.json` | Browser Q-learning and Python table evaluation |
| [Teacher/actor-critic](../learning/neural-training.md) | `target/models/connect4-osla.json` | Browser Neural OSLA; Python policy/Neural OSLA evaluation |
| [MCTS self-play](../learning/mcts-selfplay.md) | `examples/selfplay/results/connect4-model.json` | Browser Neural MCTS; native `neural_mcts` matches/tournaments for Tic-Tac-Toe and Connect 4 |

These are different model contracts. Match the game's rules, observation encoding,
action mapping and value convention. Changing a filename does not convert a model.
Q-learning and Neural OSLA browser examples use their supported default placement
game configurations. Browser Neural MCTS supports compatible self-play models for
all four games, including the correct multiplayer player count and rules. Native
neural match/tournament registration supports only Tic-Tac-Toe and Connect 4.
Python MCTS and frozen-model evaluation support the documented stochastic/private
model contracts as well. The [game manuals](../README.md#game-manuals) describe the
appropriate export and setup for each game.

## Browser

Choose the matching game and controller, upload the `.json`, then **select that
game/model in the picker**. Uploading another model preserves an existing valid
selection, so check it when changing games. Start a new game or fork to change a
running agent. Browser session/branch saves include the referenced models once;
loading those saves does not require locating the original model files again.

## Native match: attach a self-play model

In a [match configuration](../running/matches.md), use:

```json
{
  "name": "neural-puct",
  "agent": {
    "type": "neural_mcts",
    "config_file": "../mcts/neural.json",
    "model_file": "../selfplay/results/connect4-model.json"
  }
}
```

This entry is relative to `examples/matches/`. After training Connect 4, run
the supplied complete configuration:

```powershell
cargo run --release --features mcts-json,replay,inference --example match_play -- examples/matches/neural-mcts-connect4.json
```

## Tournament: select a model for every game entry

Use `model_files` instead of `model_file`:

```json
"model_files": {
  "ttt": "../selfplay/results/tictactoe-model.json",
  "c4": "../selfplay/results/connect4-model.json"
}
```

Keys match `games[].id` in the tournament. Each agent needs compatible weights
for every entry in which it participates. Unknown/missing IDs and incompatible
models are rejected before play. A single `model_file` is allowed when it matches
every entry; supplying both forms is an error.

After training/exporting both games, run:

```powershell
cargo run --release --features mcts-json,replay,inference --example tournament -- examples/tournaments/neural.json
```

All paths resolve from the experiment JSON's directory. Models are loaded for
evaluation; weights are not updated while a match or tournament runs. Each agent
instance has its own per-game RNG seed. Results record model identities and settings.

## Neural evaluator names

The default search file uses `network_value` and `network_policy`. If you rename
these roles in your search JSON, attach an optional `bindings` object to the agent:

```json
"bindings": {"value": "my_value", "policy": "my_policy", "value_observers": null}
```

Omit it for the supplied examples. Advanced observer selection, local model-size
limits and scalar-to-multiplayer completion rules are in the
[model binding reference](../../reference/learning/local-mcts.md).
