# Execution and extension workflow map

Choose a task here or use the [manual index](../../guides/README.md). Each row
connects an implemented entry point to a short guide and a complete contract.
Capability restrictions are intentional: support for basic game access does not
enable every controller, encoder or training algorithm.

| Task / entry points | Operating guide | Complete options and semantics |
|---|---|---|
| Native game examples: `cantstop`, `loveletter`; standalone placement `random`, `osla`, `mcts` | [Games](../../guides/README.md#game-manuals), [simple agents](../../guides/agents/basic-agents.md) | [Native arguments/features](command-line.md), [game contracts](../games/README.md) |
| Configured lineups: `match_play` | [Matches](../../guides/running/matches.md) | [All JSON fields, seeds and outputs](../experiments/matches.md) |
| Agent pools: `tournament` | [Tournaments](../../guides/running/tournaments.md) | [Scheduling, ranking, logs and limits](../experiments/tournaments.md) |
| Basic Python game operations: `play.py`, `cantstop.py`, `Game` | [Game operations](../../guides/running/game-operations-from-python.md) and game-specific examples | [Game API](../python/game-api.md), [CLI flags](../python/command-line.md) |
| Rust-owned native/external decisions: `run_agents.py`, `play_osla_python.py`, native `osla_python` | [External evaluation](../../guides/running/external-evaluation.md) | [Request/commit/checkpoint API](../python/external-osla.md), [Rust-owned loop](../python/rust-owned-execution.md), CLI references |
| Independent-game candidate groups: `run_grouped.py`, `GameBatch` | [Grouped execution](../../guides/running/external-evaluation.md#group-independent-games) | [Row identities, limits and member errors](../python/grouped-evaluation.md) |
| Learner steps: `play_episode.py`, `GameEnv` | [Episodes](../../guides/learning/episodes-and-collection.md) | [Constructor/reset/reward API](../python/gym-episodes.md) |
| Bulk learner steps: `play_episode_batch.py`, `GameEnvBatch` | [Bulk episodes](../../guides/learning/episodes-and-collection.md#run-environments-in-bulk) | [Validity masks, selected reset and lifecycle](../python/batched-episodes.md) |
| Teacher/policy/neural-OSLA trajectories: `neural_collect.py` | [Collection](../../guides/learning/episodes-and-collection.md#collect-trajectories-for-learning) | [Owned trajectories/intermediate views](../learning/neural-play.md), [CLI](../python/command-line.md) |
| Tabular learning/reload: `tabular_q.py`, `evaluate_tabular_q.py` | [Q-learning](../../guides/learning/tabular-learning.md) | [Algorithm, transitions and model JSON](../learning/tabular-learning.md), [CLI](../python/command-line.md) |
| Teacher/actor-critic training/resume: `neural_train.py`, `evaluate_neural.py` | [Neural training](../../guides/learning/neural-training.md) | [Settings/loss/checkpoints](../learning/neural-training.md), [CLI](../python/command-line.md) |
| Native/neural Python search: `MctsSearch`, `MctsBatch`, `neural_mcts.py`, `sampled_mcts.py`, compatibility game wrappers | [Run Rust MCTS](../../guides/agents/rust-mcts-from-python.md) | [Requests, perspectives and batching](../learning/neural-mcts.md), [CLI](../python/command-line.md) |
| Whole-game self-play/checkpoints: `selfplay_train.py` | [Self-play](../../guides/learning/mcts-selfplay.md), including a small complete workflow | [Every configuration field](../learning/mcts-selfplay.md), [CLI](../python/command-line.md) |
| Frozen-model opponents: `selfplay_evaluate.py` | [Evaluation](../../guides/learning/evaluate-mcts-models.md) | [Legacy placement versus named multiplayer lineup formats](../learning/model-evaluation.md), [CLI](../python/command-line.md) |
| Browser play, local models, branches, saves and imports | [Browser](../../guides/running/browser.md), [models](../../guides/agents/models.md), game controls in game manuals | [Protocol/limits](../browser/protocol.md), [local model contracts](../learning/local-mcts.md), [file formats](../persistence/formats.md) |
| Native model/file/encoding inspection: `learning`, `model_inference`, `replay` | [Native command examples](command-line.md#inspect-models-encodings-and-saved-games) | [LearningGame](../learning/learning-game.md), [persistence integration](../persistence/recordings.md) |
| Add a game | [Game implementation task](../../guides/games/new-game.md) | [Rules/information/integration checklist](implementing-games.md), [encoding support](../learning/adding-support.md) |
| Add an agent or evaluator | [Agent](../../guides/agents/new-agent.md), [evaluator](../../guides/agents/new-evaluator.md) | [Agent/checkpoint/adapter contracts](../agents/implementing-agents.md), [MCTS registry](../agents/mcts-api.md), [external evaluators](../learning/adding-evaluator.md) |
| Reproduce existing measurements | [Benchmark collection](../../benchmarks/README.md) | Workload-specific methods, samples and tool READMEs; fixed-work Rust and warmed-JVM comparisons are distinct from simple command execution |

## Limits and unsupported combinations

- Learner-step Gym/teacher/actor-critic examples support the two placement games.
  Whole-game MCTS self-play and Python frozen-model evaluation support all four
  games within their model contracts.
- Browser Neural MCTS supports all four games. Local neural agents in the native
  match/tournament CLI are registered only for Tic-Tac-Toe and Connect 4.
- Python attached OSLAPython controllers and independent-game groups are distinct
  from the resumable MCTS API. Native JSON matches do not load arbitrary Python
  callbacks, Q-tables or Neural OSLA agents.
- Simultaneous decisions, IS-MCTS, a GPU inference backend, remote authentication,
  passive agent lifecycle hooks, generic metric/listener registries and
  search-wide action-filter decorators are not implemented interfaces.
- Fixture generators (`save_files`, Python/Rust comparison exporters) deliberately
  replace recorded fixtures. They are developer maintenance tools, not general
  experiments. Their source/manifests declare required features and destinations.
- There is no bundled general framework-size/test-census command. Existing
  benchmark tools do not imply a complete all-game speed/memory study.

## Output, seeds and failures

Follow the shared [reproducibility instructions](../../guides/running/README.md#reproducibility).
Keep game/player identity, seat allocation, model/value convention and actual
resolved seeds with results. A fixed work budget is needed for reproducible
search; equal elapsed-time budgets need not yield equal trajectories.

Training outputs can replace their configured model/checkpoint/report paths;
native experiment writers reject collisions. In either case, use a separate
configuration and destinations for another experiment. A failure or external
truncation is not a game-defined loss. Check completed counts, failures and
per-seat denominators before interpreting a summary. Hidden-game reports/files
can contain privileged data even when acting policies receive only observations.
