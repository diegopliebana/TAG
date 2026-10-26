# Use native and neural MCTS in the browser and experiments

For operating instructions, use the [short manual](../../guides/agents/models.md). This page contains
the detailed reference.

MCTS now runs in the browser and in general match/tournament applications.
The tree search, forward model and local neural evaluation run in Rust. Browser
execution uses Wasm inside the existing Web Worker. Python/PyTorch trains a
model before play; no Python process or inference server is required to use it.
The separate [Python neural-search interface](neural-mcts.md) remains available
for batched PyTorch evaluation across independent trees and broader model code.
Browser native/neural MCTS supports all four games. Local neural agents in the
native match/tournament CLI are registered for Tic-Tac-Toe and Connect 4 only;
Can't Stop and Love Letter use Python frozen-model lineup evaluation instead.

## Play in the browser

Build the engine and refresh the app as described in [web/README](../../../web/README.md):

```powershell
cargo build --offline --release -p tag-browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tag_browser.wasm
.\serve-browser.ps1
```

Open `http://127.0.0.1:8080`. In any player seat, choose **MCTS** for native
heuristics or **Neural MCTS** for a local policy/value model. Either can face a
human, another MCTS agent or any existing controller. Select the game and its
parameters before starting.

Can't Stop supports 2–4 players with independently retained action/simulation RNG
streams. Its local neural model must match rules/player count and use complete
absolute-seat terminal-reward vectors. See the
[game contract](../games/cantstop.md#browser-presentation-and-continuation) for formats,
controller/import limits and stochastic continuation. General operation belongs
in the [browser manual](../../guides/running/browser.md).

Each MCTS seat has a JSON editor and configuration file picker. The editor uses
the existing [MCTS schema](../agents/mcts-configuration.md), including UCB1/PUCT,
independently configured heuristic roles, progressive bias, expansion and rollout
policies, tree depth and rollout length. `null` disables an optional depth/length
or computational limit as defined there. At least one computational budget must
remain enabled. The browser's starting profile uses 256 iterations and evaluates
expanded leaves immediately. These starting settings are not tuned for strength.

The seat seed overrides the seed inside the search JSON. Its decimal string keeps
the exact integer; displayed search JSON uses `seed: null` to avoid rounding large
integers in JavaScript. Effective settings and RNG remain in saved controllers. Browser
seed controls retain their existing generated/entered-seed behaviour; native
experiment seeds retain operating-system randomness when their JSON seed is null.

For **Neural MCTS**, load the portable evaluation `.json` produced by
[self-play training](mcts-selfplay.md), then select it in the seat's model list.
The `.pt` checkpoint resumes Python training and cannot be loaded for play.
Neural MCTS initially selects PUCT, the model's value evaluator and legal policy
probabilities. Discounted learner-step models belong to **Neural OSLA**;
their value contract is not silently converted into a self-play model.

The separate **neural MCTS bindings JSON** names the model's evaluators:

```json
{
  "value": "network_value",
  "policy": "network_policy",
  "value_observers": null
}
```

`value` and `policy` default to the names above. Search roles select those names
independently, just as they select native heuristic names. Their evaluator
`parameters` must be empty. The local policy evaluator returns softmax probabilities
over the supplied legal actions, mapped through the game's policy action IDs.
Illegal actions receive no probability. A role using these scores as an action
heuristic receives those same probabilities.

`value_observers` selects which observations supply state values:

| Value | State evaluation |
|---|---|
| `null` or omitted | Scalar model: evaluate once for every player. Vector model: evaluate the deciding player's observation once. |
| `all_players` | Evaluate each player's scalar separately. |
| `root_player` | Return one scalar labelled with the real move's root player. |
| `deciding_player` | Use the simulated position's acting player. A scalar remains labelled with that player; a vector identifies every output player. |

Full-vector self-play models use deciding-player observations, matching their
own-turn training data. Their declared output order is reordered into absolute
player order for MCTS back-propagation. Scalar single-observer evaluation needs
an explicit `fill_zeros` or `negation` completion policy when its values are used;
`require_full_vector` rejects an incomplete estimate. No implicit negation occurs.
For multiplayer games, the design must explicitly define how those estimates
supply the required player values. The browser supports 2-4-player Can't Stop and Love Letter through their
game-specific observation and model contracts.

Press **AI step** or **Play live**. Pause discards unfinished calculations and
prepared moves before commitment. Reset and replacement invalidate obsolete work.
Search processes at most sixteen iterations per request and stops the chunk after
about four milliseconds at an iteration boundary. One long iteration can exceed
that interval. The main rendering thread stays separate, but cancellation is
cooperative rather than an immediate interruption of arbitrary model arithmetic.
Timed budgets use the browser's monotonic `performance.now()` clock, including
time between preparation requests.

## Run a neural match

Train/export the matching model first. The supplied configuration expects the
Connect 4 output of `examples/selfplay/connect4.json`:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/connect4.json --max-cycles 1
cargo run --offline --release --features mcts-json,replay,inference --example match_play -- examples/matches/neural-mcts-connect4.json
```

Add `inference` to the existing native runner's features. A neural agent selects
exactly one search configuration and its model:

```json
{
  "name": "neural-puct",
  "agent": {
    "type": "neural_mcts",
    "config_file": "../mcts/neural.json",
    "model_file": "../selfplay/results/connect4-model.json",
    "bindings": {
      "value": "network_value",
      "policy": "network_policy",
      "value_observers": null
    }
  }
}
```

Inline `config` replaces `config_file`; supplying both is an error. `bindings` is
optional and uses the defaults above. Paths resolve against the match JSON file.
Player seeds, rotated/fixed seats, results and failure handling follow the
[match guide](../experiments/matches.md). Each game gets a fresh agent and separately derived
seed. Loading weights does not share RNG state between agents.

## Run a neural tournament

Train/export both games' models, then run:

```powershell
cargo run --offline --release --features mcts-json,replay,inference --example tournament -- examples/tournaments/neural.json
```

A multi-game neural agent uses `model_files` in place of `model_file`:

```json
"model_files": {
  "ttt": "../selfplay/results/tictactoe-model.json",
  "c4": "../selfplay/results/connect4-model.json"
}
```

Keys are the tournament's **game-entry IDs**, not inferred game names or tensor
dimensions. This allows separate models for different configurations of one game.
Missing/unknown entries or mismatched game/configuration/observation/action
contracts fail before tournament play. A single `model_file` can be used when it
is compatible with every entry. Ordinary matches use `model_file`.

The [tournament guide](../experiments/tournaments.md) owns scheduling and ranking settings.
Neural agents use the existing seed derivation, multiplicative seat permutations,
outcome accounting and equal-game-weighted global ranking. Result manifests retain
model identities, input/value contracts, supported contexts, search settings and
bindings; individual games retain actual agent seeds and search work totals.
General native applications evaluate locally and sequentially. They do not batch
PyTorch requests or train while playing a tournament.

Keep `game_logs.mode` as `separate` to save per-game metadata and browser-compatible
action recordings. These preserve the existing main-results/log cross references.
Recordings load as human-controlled games and reproduce accepted actions; they do
not embed neural weights or restore the original tournament agents. The standalone
match runner's older `reporting.recordings` mode still rejects MCTS controllers;
use tournament separate logs for action recordings. This restriction is explicit,
rather than writing an incomplete resumable-agent file.

## Model compatibility and numeric limits

Local execution supports the existing portable fully connected float32 reference
network, fixed-action policies and format-2 observation context. Self-play values
must declare gamma=1 per environment action and win/draw/loss units +1/0/-1.
The artifact must explicitly declare `provenance.model_context.observers` covering
all players and `turn_context`: `any_turn` for scalar models, `own_turn` for
full-vector models. Missing context is rejected rather than inferred from shapes.
Older exports missing this declaration must be exported again through the
current training/checkpoint API; do not invent a declaration for unknown models.

Existing local limits remain: 10 MiB model JSON, one million parameters, 1–8 hidden
layers of at most 512 units, up to 16,384 input features, 4096 policy outputs and at most
256 player-context entries. These are narrower than the Python trainer's maximum
model size. Unsupported operators, candidate-action feature policies and larger
models require a separately selected backend. Each native agent can load at most
16 entry models with a combined 10 MiB source JSON. Browser registry limits remain
16 models and 10 MiB. Object overhead and total process memory are additional.
The browser's ordinary command limit is 32 KiB; a configuration file picker accepts
at most 16 KiB, and combined seat settings must fit the command limit.

Rust float32 predictions are compared with PyTorch using absolute tolerance
2e-6 plus relative tolerance 2e-5. Different arithmetic or near ties can change
search trajectories; matching seeds do not promise identical moves across runtimes.
This implementation makes no GPU, zero-copy, speedup or playing-strength claim.

## Save, load and fork

Browser session/branch saves embed referenced models once, alongside MCTS settings
and each agent's RNG continuation. They resume supported controllers without
external model files. State-only saves retain the position and ask for fresh
controllers as before. Trees are rebuilt for each real decision, and pending
searches are not saved. Historical forks restore the recorded controller state;
discarding a calculation does not advance that state.

MCTS recordings apply the recorded moves and restore the recorded RNG progress.
They validate available actions, resulting state/turn, unchanged policy/model/
binding identity and nondecreasing draw position for the same RNG seed/stream.
This does not prove that rerunning MCTS would choose the recorded move. Repeating
a time-limited search would not be a reliable verification method. Existing
deterministic agents keep their independent selection verification.

See [browser integration](../browser/integration.md) for worker/controller ownership
and [model evaluation](model-evaluation.md) for empirical comparisons.
