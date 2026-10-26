# Train a policy/value model from MCTS self-play

For operating instructions, use the [short manual](../../guides/learning/mcts-selfplay.md). This page contains
the detailed reference.

Self-play uses the current network to guide MCTS for every player in a game.
After the game finishes, each decision provides two training targets: the search's
distribution of root visits and the actual terminal outcome for each player.
PyTorch learns to predict those targets. The updated network then generates the
next group of games. Evaluation is separate and does not decide model promotion.

Rules, simulated transitions, tree selection, expansion, rollouts, computational
budgets, MCTS back-propagation and final move selection run in Rust. Python
coordinates independent searches, batches frozen CPU model inference, stores
training examples and performs gradient updates. MCTS back-propagation updates
tree statistics; PyTorch's gradient back-propagation updates network parameters.

This workflow supports Tic-Tac-Toe, Connect 4 and 2-4-player Can't Stop and
Love Letter. Love Letter uses complete-game collection/training; its
[game-specific contract](../games/loveletter.md#complete-game-self-play-and-training)
defines private inputs, rotating starters, shared wins and replay-validated
checkpoints. [Opponent evaluation](model-evaluation.md) and
[browser play](../browser/integration.md) are also supported.
A session trains one game/ruleset and one model
input/output specification. [Local inference](local-mcts.md) uses compatible
exports in browser MCTS for all four games. Native neural match/tournament
registration currently supports Tic-Tac-Toe and Connect 4 only.
[Opponent-lineup evaluation](model-evaluation.md) tests frozen models independently.

## Run, save and resume

Use the [Python setup](../python/setup.md) and optional [PyTorch setup](neural-inference.md).
Rebuild the adapter after Rust changes, including the policy-node metadata added
for root exploration:

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/tictactoe.json --max-cycles 1
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --resume examples/selfplay/results/tictactoe-training.pt --max-cycles 1
```

The supplied [Tic-Tac-Toe](../../../examples/selfplay/tictactoe.json) and
[Connect 4](../../../examples/selfplay/connect4.json) configurations each request
ten cycles, 16 games per cycle, four active games, 64 MCTS iterations per decision,
64 optimization updates per cycle and 20 evaluation games. `--max-cycles` limits
the number of additional cycles for this invocation without changing the saved
schedule. Omit it to complete that schedule. Zero saves/inspects the initial or
loaded state without another cycle.

The JSON specifies three distinct output paths:

- `model_file`: portable evaluation-model JSON containing weights and their contract.
- `checkpoint_file`: complete self-play training state, including Adam and examples.
- `report_file`: resolved settings, seed and cycle histories, including outcomes,
  losses, sample indices, model identities, buffer statistics and timings.

Each file is replaced atomically after it has been completely written. These are
three independent writes, not a multi-file transaction. Choose destinations for
each experiment; running a fresh session again replaces those selected outputs.
Resume uses settings and destinations saved in its checkpoint, not current copies
of referenced configuration files. Omit `--config` when using `--resume`.

`--model path/to/model.json` initializes a fresh session from a compatible model
with a fresh optimizer/buffer. It must match the configured architecture, game,
inputs, output player order and gamma=1 value convention. Learner-step gamma=0.99 models
and teacher/actor-critic checkpoints are not silently converted.

## One training cycle

1. Make an independent frozen copy of the current model and record its identity.
2. Generate complete self-play games through [resumable MCTS](neural-mcts.md).
   A game always uses that fixed model while its group is being collected.
3. Add actual terminal labels to the retained decisions. Insert positions in
   logical game/ply order, independently of which active game finishes first.
4. Add the examples to the replay buffer, or replace the newest-group training set.
5. Sample position minibatches and perform the configured number of Adam updates.
6. When enabled, evaluate the updated model with exploration disabled. Record results, save
   completed-cycle state and use the updated model for the next cycle.

An incomplete or failed game aborts collection before any examples from that group
enter the buffer. A move guard is a failure, not a terminal draw or loss. A failed
collection/update blocks further cycles and checkpoint replacement on that session;
restart from the last successful checkpoint. Evaluation failures are reported
separately and do not undo valid updates or create losses for unfinished games.

## Training targets and loss

The policy target is `visits(a) / sum(visits)` over the real decision's legal
actions. Every legal action retains its ID, including actions with zero visits.
Illegal actions have zero target probability. A search with no completed root
visits cannot provide this target, so collection rejects it. Ordinary zero-budget
MCTS fallback behavior remains available outside training collection.

For the current games, terminal rewards are +1 for a win, 0 for a draw and -1 for
a loss. The value target at every collected position is the eventual reward for
that explicitly identified player. This is undiscounted outcome regression:
gamma=1, with the contract's clock recorded as `environment-action`. The model
does not learn actor-critic advantages or learner-step discounted returns.

Two model output forms are supported:

| `model.value_output` | Stored observations and value labels |
|---|---|
| `observer` (default) | One observation for every player at each decision, with that observer's own terminal reward. Only the acting row contributes a policy loss. Non-acting rows train the contexts needed for strict all-player scalar evaluation. |
| `players` | One acting-player observation per decision, with the complete terminal vector in the declared output-player order. Inference uses `deciding_player` and declares `own_turn` context. |

Never infer another player's outcome by negating the first outcome. Training
labels use actual per-player results. The existing search-time scalar completion
policies remain explicit inference assumptions. Synthetic three-player/non-zero-sum
fixtures test target ordering and loss arithmetic. Can't Stop supplies actual
2-4-player trajectories with complete absolute-seat vectors. Its configurations
require `model.value_output="players"`, absolute `value_players` ordering (or
omission) and deciding-player observations. Gamma=1 and the environment-action
clock are retained. Evaluation supports the [native lineups](model-evaluation.md)
for every player count.
The [game contract](../games/cantstop.md#python-stochastic-mcts) defines its model inputs.
Love Letter also uses real complete-game trajectories and absolute-seat vectors,
including multiple positive shared-win components. Its [opponent evaluation](model-evaluation.md) uses native lineups; the [Love Letter contract](../games/loveletter.md#complete-game-self-play-and-training)
defines input restrictions and checkpoint replay checks.

The objective is:

`policy_weight * cross_entropy(visit_target, legal_policy) + value_weight * mean_squared_error`

Policy cross-entropy averages actual decisions. Value error averages players
within each position, then positions. This gives consistent weighting for scalar
observer rows and full vectors. Targets remain fixed during each optimization
update. Loss reduction on a controlled batch tests learning arithmetic; it is not
evidence of improved playing strength.

## Move sampling and root noise

The played move and training policy target have different roles. By default, Rust
samples moves proportionally to visits for the first ten accepted actions of a
game, then chooses most visits. The schedule counts all actual actions, including
consecutive actions by the same player. It does not count simulated transitions.
`sampling_moves: null` keeps sampling throughout the game. A positive temperature
changes visit-sampling weights to `visits^(1/temperature)`. Select `most_visits` to
disable sampling; do not pass zero to the positive-temperature sampler.

Root noise mixes legal neural probabilities P with a Dirichlet probability vector
η: `(1 - noise_fraction) * P + noise_fraction * η`. Concentration `noise_alpha`
controls the variability of η: smaller values tend to put more mass on a few
actions; larger values tend toward more even vectors. The starting defaults,
fraction 0.25 and concentration 0.3, are untuned and configurable.

Each real decision has its own recorded noise seed. Draw η once when that root
first needs policy probabilities; keep its mapping by action ID for subsequent
root requests. The Rust request identifies the actual policy node: node zero is
the root. The same player appearing at a deeper node does not identify a root.
Noise changes PUCT priors only, preserving expansion order, rollout configuration
and state values. MCTS still tries unvisited actions first, so very small search
budgets may finish before those priors are used.

Sampling and noise are independent. Native-only profiles can train from native
MCTS search targets, with root noise disabled. Root noise currently requires the
declared external neural PUCT probability binding. The evaluator identity records
the post-processing algorithm/settings separately from its frozen base model;
each position retains the actual decision/noise seeds.

## Replay storage

`replay_buffer` retains recent positions across cycles. New examples evict the
oldest until both position capacity and numeric-byte bounds are met. Each position
keeps all applicable observer rows together. `newest_group` discards earlier
cycles before adding the new group, and enforces the same bounds. One oversized
position is an error. Eviction counts include replaced groups and capacity/byte
evictions. This buffer contains training examples, not browser replay files.

Training samples positions uniformly with replacement. For example, a first group
of 20 positions can still supply a minibatch of 64, with repeated samples. Buffer
capacity, games generated and optimization updates are independent settings.
Samples and public buffer views own their numeric arrays. Packing for PyTorch
creates further copies; no zero-copy claim is made.

Collection also has position/byte limits for the incomplete group awaiting final
outcomes. Exceeding them aborts collection; it does not silently omit required
labels. Numeric limits do not bound all Python object overhead, model-private
allocations or total process memory.

## JSON parameter reference

All fields below are optional except `schema_version`. Unknown fields are rejected.
Supply one of inline `search` and `search_config_file`. Files are limited to 1 MiB;
relative references and output paths resolve from the self-play configuration's
directory. Saved settings retain resolved contents, not live file dependencies.

| Field | Default and meaning |
|---|---|
| `schema_version` | Required integer 1. |
| `game`, `parameters`, `players` | `tictactoe`, `{}`, 2. Registry lookup and the game validate supported rules/player count. No duplicate Python game-name list. |
| `seed` | Null/omitted resolves one u64 from OS randomness; record the actual integer. Explicit u64 repeats the stream derivation. |
| `cycles` | 10 complete generation/training cycles. Positive. |
| `games_per_cycle` | 16 complete self-play games. Positive. |
| `parallel_games` | 4 active games; positive, at most 64. Fewer slots are used for a smaller group. |
| `search` / `search_config_file` | Native MCTS schema. Default is neural PUCT, zero rollout length, 64 iterations and `network_value`/`network_policy` bindings. |
| `model.payloads` | `["cells"]`; distinct named arrays supported by the game. |
| `model.hidden_sizes` | `[64,64]`; positive reference-network layer widths. |
| `model.value_output` | `observer` or `players`, as described above. |
| `model.value_players` | Null/omitted uses absolute player order for vector output. Placement games also accept a permutation covering every player; Can't Stop and Love Letter require absolute order. Only valid for `players`. |
| `model.value_binding`, `model.policy_binding` | `network_value`, `network_policy`; distinct external evaluator names selected by the search configuration. |
| `model.value_observers` | Null selects `all_players` for scalar output and `deciding_player` for full vectors. Scalar output may explicitly choose root/deciding observers with a compatible completion policy. Full-vector training supports deciding-player inference because its training observations are acting-player views. |
| `exploration.move_policy` | `visit_sampling` or `most_visits`. |
| `exploration.temperature` | 1; finite positive sampling temperature. |
| `exploration.sampling_moves` | 10; nonnegative count, or null for the entire game. Zero immediately uses most visits. |
| `exploration.root_noise` | True; independently enable/disable root noise. Requires compatible neural PUCT probabilities. |
| `exploration.noise_fraction` | 0.25; finite number in [0,1]. |
| `exploration.noise_alpha` | 0.3; finite positive symmetric Dirichlet concentration. |
| `data.mode` | `replay_buffer` (default) or `newest_group`. |
| `data.capacity` | 10,000 positions, including all player rows belonging to each position. Positive. |
| `data.max_bytes` | 268,435,456 numeric bytes (256 MiB). Positive. |
| `optimization.updates_per_cycle` | 64 sampled Adam updates. Positive. |
| `optimization.batch_size` | 64 sampled positions per update, with replacement. Positive. |
| `optimization.learning_rate` | 0.001; finite positive Adam learning rate. |
| `optimization.policy_weight`, `optimization.value_weight` | 1 each; finite nonnegative. At least one must be positive. |
| `evaluation.enabled` | True; evaluate updated weights separately. Can't Stop supports the [multiplayer lineup fields](model-evaluation.md#configuration-and-compatibility); its short example disables evaluation. Love Letter supports the same lineup evaluation fields. |
| `evaluation.games` | 20; positive exact requested count unless an error stops evaluation. |
| `evaluation.positions` | `rotate` default, or `fixed`. |
| `evaluation.candidate_seat` | Required valid seat for `fixed`; omit/null with rotation. |
| `evaluation.opponent_search` / `opponent_config_file` | Inline or referenced native-only MCTS profile; default UCB1, zero rollout length, 64 iterations. No general tournament-agent selection is introduced. |
| `evaluation.opponents`, `evaluation.opponent` | Null by default; named N-1 lineup or homogeneous shorthand for supported lineup games. Mutually exclusive with each other and legacy opponent-search fields. |
| `evaluation.ranking` | Null by default; `score` enables public-score finishing places for lineup games (`claimed_columns` is an alias). |
| `evaluation.max_actions` | Null by default; positive accepted-action cap for lineup evaluation. Placement evaluation instead uses `limits.max_moves`. |
| `evaluation.include_actions` | False; include accepted public action/chance traces for lineup games. |

| `limits.max_moves` | 512 real actions per game, or null to disable. If the last permitted action ends the game, accept the terminal outcome. |
| `limits.max_cells` | 4096; encoding/game resource limit. |
| `limits.max_collection_positions` | 100,000 retained positions awaiting/holding outcomes in the current collection group. |
| `limits.max_collection_bytes` | 256 MiB of numeric collection data. |
| `limits.max_group_rows` | 4096 observation rows per inference group and per request. Required player perspectives cannot be dropped to fit. |
| `limits.max_request_bytes` | 64 MiB of numeric request/training-batch arrays. Existing MCTS/encoding limits also apply. |
| `limits.max_model_parameters` | 10,000,000; may be lowered, and cannot exceed the portable model reader's ceiling. The existing 64 MiB portable-model JSON file limit also applies. |
| `limits.max_checkpoint_bytes` | 512 MiB; final checkpoint file limit. Larger configured files require a corresponding reader size limit. |
| `limits.threads` | 1 PyTorch CPU thread; positive. Rust workers batch independent searches. |
| `output.model_file`, `output.checkpoint_file`, `output.report_file` | Null by default; nonempty distinct paths enable saving those outputs. Supplied examples set all three. |

Self-play overrides the MCTS template's `seed` and `final_selection` for every
real decision. Evaluation overrides those fields for both candidate and opponent,
uses most visits, and disables root noise. Reports preserve the templates and
declare those overrides; position records contain actual effective search settings.
Other MCTS parameters, including depth, rollout policies, budgets and native
heuristic roles, retain the [MCTS parameter reference](../agents/mcts-configuration.md).

## Reproducibility and checkpoints

The seed derivation uses the first eight SHA-256 digest bytes, interpreted as an
unsigned little-endian integer, over compact UTF-8 JSON:

`["tag-m8-selfplay-seed",1,master_seed,phase,logical_game_id,ply,purpose]`

Purposes separate initialization, search, root noise, replay sampling and native
evaluation opponents. Root noise and sampling use NumPy PCG64; search uses Rust's
existing ChaCha8 generator. Game IDs and real action indices do not depend on
worker slot or completion order. Learner-step and tournament seed recipes remain unchanged.
Reports retain model/rules/encoding identities and actual settings. Fixed-work
zero-model comparisons match individual and grouped generation. General neural
floating-point results can still depend on batch shapes, hardware and versions;
time budgets additionally depend on scheduling and waits.

### Stochastic self-play seed recipe

Can't Stop and Love Letter select seed derivation version 2. Hash compact UTF-8 JSON with
`ensure_ascii=True` and separators `(',', ':')`:

`["tag-mcts-stochastic-selfplay-seed",2,master_seed,phase,logical_game_id,ply,purpose]`

Interpret the first eight SHA-256 digest bytes as a little-endian u64. Phases and
purposes are:

| Phase | Game/action indices | Purpose |
|---|---|---|
| `initialization` | 0, 0 | `model`: initial PyTorch weights |
| `training` | 0, 0 | `minibatches`: initial NumPy PCG64 sampling state |
| `selfplay` | logical game ID, 0 | `game-chance`: real game's Rust chance seed |
| `selfplay` | logical game ID, real action index | `search`: Rust search and visit-sampling seed |
| `selfplay` | logical game ID, real action index | `root-noise`: NumPy PCG64 Dirichlet seed |

For master seed 42 and logical game/action 0, chance/search/noise seeds are
16813839899447826188, 13558718246091080344 and 4033141263566526617.
Search separates its agent/simulation chance streams internally. Starting a game
never resolves fresh OS randomness: `parameters.seed` is rejected in self-play
configuration, and the derived game seed is supplied explicitly. The master seed
alone offers explicit/automatic selection. Rule parameters and game seed are kept
separate when describing model identity.

Each completed game's report records `game_seed`, actual outcomes and action count.
Each position records its actual actor, search/noise seeds and search settings.
Replaying recorded actions from the same seed/rules reproduces actual chance outcomes within
the pinned engine/RNG versions. Slot count does not enter seed derivation. Fixed
zero-model comparisons establish trajectory equality across collection schedules;
arbitrary neural floating-point calculations retain the batch-shape limitations
described above.

Every seat uses the same frozen model in self-play, including the starting seat.
There is no distinct candidate/opponent assignment to rotate. Love Letter rotates
the initial starter by logical game ID unless `parameters.first_player` is fixed;
subsequent rounds follow the game's winner/starter rules. Per-seat summaries
count every seat's actual results; combined summaries count player appearances,
so N players and G games give N*G combined entries. These are training-game
diagnostics, separate from [candidate-versus-opponents evaluation](model-evaluation.md).

`save_checkpoint(path, session)` saves only successful complete cycles, including
the initial state before cycle one. It checks the model/buffer against the recorded
history and rejects changed session settings/runtime. The separate
`tag-mcts-selfplay-training/1` format for placement games retains weights, Adam, copied training rows,
buffer order, sample RNG state, PyTorch RNG state, counters and provenance.
`load_checkpoint(path)` validates format, CPU/software runtime, model contract,
history, array shapes, outcomes, seeds, buffer bounds/digest and optimizer state.
It uses restricted weights-only loading and does not restore pending searches or
partly played games. `--checkpoint-limit-mib` raises/lowers the CLI reader's
default 512 MiB ceiling when required; the API accepts `max_bytes`.

Can't Stop and Love Letter use `tag-mcts-selfplay-training/2` with seed-derivation provenance 2
and required per-game seed/seat summaries. Loading checks the format against the
game's recipe, verifies game/search/noise seeds, checks summaries against the raw
outcomes, and applies the existing model/buffer/optimizer/RNG validation. Existing
placement-game version-1 checkpoints and seed recipes retain their semantics.
Love Letter additionally validates recorded real-game traces and retained player
inputs; its [reference](../games/loveletter.md#completed-cycle-checkpoints-and-information-boundaries)
owns these game-specific checks and research access markers.

At a saved boundary every game in the cycle has ended, so the next game's chance
stream is determined by the next logical game ID. Resume restores saved PyTorch
and minibatch RNG progress and counters. It neither restarts an unfinished chance
stream nor requires serializing a live game/search. Failed collection adds no
examples, and failed updates cannot replace the completed-cycle recovery file.

The portable model container is the existing evaluation format, with an explicit
gamma=1 contract and self-play provenance. Consumers must accept that contract;
existing learner-step workflows are not automatically interchangeable. When
binding a full-vector self-play model outside this workflow, use its recorded
`own_turn` context and deciding-player observations.

Python entry points live in `tag_rust.learning.selfplay`: `SelfPlaySettings`,
`SelfPlaySession`, `save_checkpoint`, `load_checkpoint`, and `save_outputs`.
Call `session.run_cycle()` or iterate `session.run(max_cycles=...)`. Returned
reports/history views are copies. Newest-group mode and model-head choices are
configured before constructing the session; change experiments by starting a new
session or use a checkpoint's saved settings to continue it.

See [frozen-model evaluation](model-evaluation.md) for how to assess the exported
model against independent opponents.
