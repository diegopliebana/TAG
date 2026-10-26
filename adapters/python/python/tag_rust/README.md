# Python package map

The implementations are grouped by the work they perform:

| Folder | Responsibility |
|---|---|
| `engine/` | Native game access and descriptions of observations/actions. |
| `search/` | Pending OSLA evaluations, resumable MCTS and neural evaluation across independent trees, grouping independent games, and native scoring. |
| `episodes/` | Optional Gymnasium learner environments, batches, and owned transitions. |
| `learning/` | Tabular learning; optional neural inputs, networks, inference, collection, training, evaluation, and files. |

For example, import `GameEnv` from `tag_rust.episodes.gym` and
`TrainingSession` from `tag_rust.learning.training`. The root `api.py`, `gym.py`,
`neural.py` and other historical module names are small compatibility imports.
They resolve to the same module/classes, including private helper identities.
New code uses the topic paths. The compiled `_engine` extension and its type
description remain at the package root, as required by the current build.

Importing `tag_rust` or `tag_rust.learning` does not import PyTorch or Gymnasium.
Love Letter is available through `engine` for game operations/permitted arrays
and `search` for existing native/neural MCTS. `learning/selfplay` supports
complete-game training and checkpoints. Frozen-model opponent evaluation is supported. Attached evaluation controllers
and learner episodes remain unsupported; see the
[game reference](../../../../docs/reference/games/loveletter.md).
Import the relevant optional module when it is needed. `learning.tabular` contains
the reusable `QTable`; `examples/tabular_q.py` retains the original command.

The [training guide](../../../../docs/reference/learning/neural-training.md) explains
teacher/actor-critic training and the division between portable evaluation JSON and resumable
Python training checkpoints. The [neural MCTS guide](../../../../docs/reference/learning/neural-mcts.md)
describes search/model bindings and cross-search batching. The separate
[local MCTS guide](../../../../docs/reference/learning/local-mcts.md) covers browser
and native experiment execution of exported self-play models.

`learning/selfplay/` contains MCTS collection, root noise, bounded training data,
losses, evaluation, training and checkpoint files. The
[self-play guide](../../../../docs/reference/learning/mcts-selfplay.md) owns this
workflow and its complete parameter reference.
