# Python example command-line options

Run from the repository root after [setup](../../guides/setup.md). On Windows use
`.\.venv\Scripts\python.exe adapters/python/examples/NAME.py ...`; on POSIX use
`.venv/bin/python adapters/python/examples/NAME.py ...`. Every parser provides
`--help` and exits before running an experiment. Tables list all other parser
options, their effective fresh-run defaults and permitted choices. Cross-field,
resource and model validation is performed by the underlying API.

A seed accepts a u64 integer or `auto`/`null`; omission resolves OS entropy.
`None`/unset means the command's documented default, not a string to type.
Training and evaluation support fixed seats or alternating logical episodes.
Game identifiers alone do not enable an unsupported learner/search integration.

## Choose a workflow

[Game operations](../../guides/running/game-operations-from-python.md),
[external evaluation](../../guides/running/external-evaluation.md),
[episodes/collection](../../guides/learning/episodes-and-collection.md),
[training](../../guides/learning/README.md), [Rust MCTS](../../guides/agents/rust-mcts-from-python.md).

## cantstop.py

Game-specific First Available demonstration; see the game manual for phase/action meanings.

[Parser source](../../../adapters/python/examples/cantstop.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--players` | 2, choices 2, 3, 4 | Number of seats; the rules/model must support this count. |
| `--seed` | automatic, seed_value | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--max-actions` | unset, positive | Positive accepted-action cap; unset disables it. Reaching the cap without termination is truncation, not a loss. |

## evaluate_neural.py

Evaluate a teacher/actor-critic model, using policy actions or neural-value OSLA. Policy evaluation defaults to Random; OSLA evaluation defaults to native-scored osla_python and rejects other opponent choices.

[Parser source](../../../adapters/python/examples/evaluate_neural.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--model` | required, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |
| `--episodes` | 100, int | Positive exact number of complete or truncated learner episodes requested. |
| `--batch-size` | 16, int | 1-256 independent episodes per group. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--mode` | "policy", choices policy, osla | Execution/evaluation algorithm; meanings are described below. |
| `--opponent` | unset, choices first_available, random, osla, osla_python | Policy mode defaults to Random and accepts all listed opponents. OSLA mode requires osla_python, which is its default. |
| `--learner` | unset, choices 0, 1 | Fixed learner seat; unset alternates logical episodes across seats 0 and 1. |

## evaluate_tabular_q.py

Evaluate a default Tic-Tac-Toe Q-table against a registered opponent.

[Parser source](../../../adapters/python/examples/evaluate_tabular_q.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--model` | required, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |
| `--games` | 1000, int | Positive number of evaluation games requested. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--learner` | unset, choices 0, 1 | Fixed learner seat; unset alternates logical episodes across seats 0 and 1. |
| `--opponent` | unset, choices first_available, random, osla, osla_python | Default to the table's recorded training opponent; an explicit value changes only this evaluation. |

## neural_collect.py

Collect untrained policy/teacher trajectories or run neural-value OSLA. all runs teacher, policy and osla. OSLA collection requires the osla_python opponent; teacher/policy modes accept any listed opponent.

[Parser source](../../../adapters/python/examples/neural_collect.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--game` | "tictactoe" | Registered game ID; learner-step training/collection currently supports tictactoe and connect4. MCTS has its own game capabilities. |
| `--mode` | "all", choices all, teacher, policy, osla | Execution/evaluation algorithm; meanings are described below. |
| `--episodes` | 6, int | Positive exact number of complete or truncated learner episodes requested. |
| `--batch-size` | 4, int | 1-256 independent episodes per group. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--learner` | unset, choices 0, 1 | Fixed learner seat; unset alternates logical episodes across seats 0 and 1. |
| `--opponent` | unset, choices first_available, random, osla, osla_python | Defaults by mode: teacher=osla, policy=random, osla=osla_python. A supplied opponent must be valid for every selected mode. |
| `--max-steps` | unset, int | Positive learner-step cap per episode; unset disables external truncation. |

## neural_mcts.py

Calculate independent placement-game neural search recommendations; untrained weights are used when --model is absent.

[Parser source](../../../adapters/python/examples/neural_mcts.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--game` | "tictactoe" | Registered game ID; learner-step training/collection currently supports tictactoe and connect4. MCTS has its own game capabilities. |
| `--config` | examples/mcts/neural.json, Path | Search or experiment JSON file, interpreted by this command. Relative paths inside experiment files resolve from that file's directory. |
| `--model` | unset, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |
| `--searches` | 4, int | 1-64 independent root searches to calculate. Demonstrations do not apply moves. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--max-group-rows` | 4096, int | Positive bound on compatible rows in one external evaluation group. |

## neural_train.py

Teacher/actor-critic CPU training. Defaults below are effective fresh-run TrainingSettings defaults, not the parser's unset sentinels. Resume uses saved settings and rejects supplied training-setting overrides; --output and --checkpoint remain required destinations.

[Parser source](../../../adapters/python/examples/neural_train.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--game` | "tictactoe" | Registered game ID; learner-step training/collection currently supports tictactoe and connect4. MCTS has its own game capabilities. |
| `--teacher-episodes` | 256, int | Nonnegative OSLA-demonstration episode count; total teacher+actor episodes must be positive. |
| `--actor-episodes` | 1024, int | Nonnegative actor-critic episode count. |
| `--batch-size` | 16, int | 1-256 independent episodes per group. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--learner` | unset, choices 0, 1 | Fixed learner seat; unset alternates logical episodes across seats 0 and 1. |
| `--teacher-opponent` | "osla", choices first_available, random, osla, osla_python | Opponent for OSLA-demonstration collection. |
| `--opponent` | "random", choices first_available, random, osla, osla_python | Actor-critic opponent; teacher collection uses the independent --teacher-opponent setting. |
| `--max-steps` | unset, int | Positive learner-step cap per episode; unset disables external truncation. |
| `--learning-rate` | 0.001, float | Finite positive Adam learning rate. |
| `--hidden-sizes` | [64, 64], int; one or more values | One or more positive hidden-layer widths. |
| `--decision-value-weight` | 0.5, float | Finite nonnegative decision-row value-loss coefficient. |
| `--intermediate-value-weight` | 0.5, float | Finite nonnegative nonterminal intermediate-row value-loss coefficient. |
| `--entropy-weight` | 0.01, float | Finite nonnegative policy-entropy coefficient. |
| `--threads` | 1, int | Positive PyTorch CPU computation-thread count. |
| `--resume` | unset, Path | Training checkpoint to restore, including saved settings, schedule, weights, optimizer and RNG progress. |
| `--max-groups` | unset, int | Positive additional complete trajectory groups to run before saving/stopping. |
| `--output` | required, Path | Portable evaluation JSON destination; saving replaces it. Choose a distinct path for another experiment. |
| `--checkpoint` | required, Path | Resumable training .pt destination; must differ from --output. |

## play_episode.py

One masked Random learner episode against a native opponent; no model training.

[Parser source](../../../adapters/python/examples/play_episode.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |

## play_episode_batch.py

Fixed four-member bulk episode example; --seed is its only configurable flag.

[Parser source](../../../adapters/python/examples/play_episode_batch.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |

## play_osla_python.py

Both placement games through the candidate preparation/evaluation/commit interface.

[Parser source](../../../adapters/python/examples/play_osla_python.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--scoring` | "rust", choices python, rust | Choose Python open-line scoring or the Rust-scored external-evaluation interface. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |

## run_agents.py

Both placement games with Rust owning the loop and either native or external scoring.

[Parser source](../../../adapters/python/examples/run_agents.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--evaluation` | "rust", choices rust, python | Choose native Rust scoring or Python evaluation requests while Rust owns the game loop. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |

## run_grouped.py

Fixed four-game example grouping compatible evaluation rows across games.

[Parser source](../../../adapters/python/examples/run_grouped.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |

## sampled_mcts.py

Observation-derived native/neural search for stochastic/private games. The native flag disables the neural evaluator; --model and --native cannot be combined.

[Parser source](../../../adapters/python/examples/sampled_mcts.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--game` | "cantstop", choices cantstop, loveletter | Registered game ID; learner-step training/collection currently supports tictactoe and connect4. MCTS has its own game capabilities. |
| `--players` | 2, choices 2, 3, 4 | Number of seats; the rules/model must support this count. |
| `--searches` | 4, int | 1-64 independent root searches to calculate. Demonstrations do not apply moves. |
| `--iterations` | 32, int | Nonnegative completed MCTS iterations per decision; zero permits a fallback without completing an iteration. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--native` | false | Use native Rust evaluators instead of a frozen PyTorch evaluator; no neural weights are loaded. |
| `--model` | unset, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |

## selfplay_evaluate.py

Frozen whole-game MCTS evaluation. Inputs must declare compatible rules; evaluation.enabled must be true. Only output.report_file may be written. Print master/model identity, then the result object; process exit is nonzero on evaluation failure.

[Parser source](../../../adapters/python/examples/selfplay_evaluate.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--model` | required, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |
| `--config` | required, Path | Search or experiment JSON file, interpreted by this command. Relative paths inside experiment files resolve from that file's directory. |

## selfplay_train.py

Whole-game self-play/training. Default config is examples/selfplay/tictactoe.json. Resume must omit --config and --model; destinations/settings come from the checkpoint. --model initializes weights with a fresh optimizer/buffer rather than resuming training.

[Parser source](../../../adapters/python/examples/selfplay_train.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--config` | unset, Path | Fresh-run self-play JSON; unset selects examples/selfplay/tictactoe.json. Schema and defaults: [self-play reference](../learning/mcts-selfplay.md). |
| `--resume` | unset, Path | Training checkpoint to restore, including saved settings, schedule, weights, optimizer and RNG progress. |
| `--model` | unset, Path | Portable evaluation JSON, not a training .pt checkpoint. Game/rules/input/value contracts must match. |
| `--max-cycles` | unset, int | Positive additional completed self-play cycles to run; unset continues to the saved/configured cycle count. |
| `--checkpoint-limit-mib` | 512, int | Positive maximum checkpoint read size in MiB; applies to self-play resume. |

## tabular_q.py

Default Tic-Tac-Toe Q-learning; 128 rounds times 8 members gives 1,024 training episodes. --output is required and retains the table for later evaluation.

[Parser source](../../../adapters/python/examples/tabular_q.py).

| Option | Default / type or choices | Meaning |
|---|---|---|
| `--rounds` | 128, int | Positive training episodes per member. |
| `--members` | 8, int | 1-256 independent learner environments. |
| `--eval-rounds` | 16, int | Positive evaluation episodes per member. |
| `--learner` | unset, choices 0, 1 | Fixed learner seat; unset alternates logical episodes across seats 0 and 1. |
| `--opponent` | "random", choices first_available, random, osla, osla_python | Training opponent bound into the saved table task. Use the evaluation CLI for independent opponents. |
| `--seed` | automatic, seed_argument | Resolved experiment/master seed: unsigned 64-bit integer, auto or null. Omission resolves OS randomness once; retain the printed value. |
| `--alpha` | 0.2, float | Finite Q-learning step size in (0,1]. |
| `--gamma` | 0.99, float | Finite discount in [0,1]. |
| `--epsilon` | 0.2, float | Finite exploration probability in [0,1]. |
| `--max-episode-steps` | unset, int | Positive learner-step cap per episode; unset disables external truncation. |
| `--output` | required, Path | Portable evaluation JSON destination; saving replaces it. Choose a distinct path for another experiment. |

## Compatibility entry points and fixed demonstrations

`cantstop_mcts.py` and `loveletter_mcts.py` forward to `sampled_mcts.py` with their
respective default game; all listed sampled-search options remain available.
`tabular_q.py` forwards to `tag_rust.learning.tabular.main` and retains its imports.

`play.py` has no options and plays both default placement games plus a small custom placement board by
choosing the first offered action through the basic API. `open_lines.py` is an importable
heuristic helper, not a configurable experiment. `export_mcts_fixtures.py` is a
developer fixture generator that replaces tracked numerical fixtures; it is not
a training or evaluation runner and should be used only when intentionally
regenerating those fixtures.

## Configuration files and APIs

[Self-play fields and limits](../learning/mcts-selfplay.md),
[model evaluation configuration](../learning/model-evaluation.md),
[teacher/actor-critic settings](../learning/neural-training.md),
[game API](game-api.md), [episode lifecycle](gym-episodes.md),
[MCTS settings](../agents/mcts-configuration.md).
