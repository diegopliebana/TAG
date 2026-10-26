# Play learned models in the browser

For operating instructions, use the [short manual](../../guides/running/browser.md). This page contains
the detailed reference.

For native MCTS and self-play policy/value models, follow
[local browser MCTS](../learning/local-mcts.md). This page explains the existing Q-learning
and Neural OSLA controllers, which retain their own model contracts.

The browser can load the evaluation JSON files produced by Q-learning and neural training. Rust/Wasm runs the game and controller inside the existing Web
Worker, which keeps calculation away from the rendering thread. Python trains
the models before play; no Python model process or inference server is needed
while you play. The local HTTP server only serves app files.

The supported model contracts are checked before use. Test the native model
executor and actual Wasm separately from graphical file-picker and play controls.

## Train a model

From the repository root, produce a Q-table for default Tic-Tac-Toe:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/tabular_q.py --output target/models/tictactoe-q.json
```

For neural-value OSLA, train an evaluation model:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --game tictactoe --output target/models/tictactoe.json --checkpoint target/models/tictactoe.pt
.\.venv\Scripts\python.exe adapters/python/examples/neural_train.py --game connect4 --output target/models/connect4.json --checkpoint target/models/connect4.pt
```

Choose the `.json` evaluation file in the browser. The `.pt` checkpoint is for
resuming Python training. The [Q-learning guide](../learning/tabular-learning.md) and
[neural training guide](../learning/neural-training.md) explain opponents, budgets and results.
Both training workflows alternate seats by default. Learning strength depends
on the training data; successful loading does not establish playing strength.

## Open and play

Build the updated Rust engine and JavaScript bindings once:

```powershell
cargo build --offline --release -p tag-browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tag_browser.wasm
.\serve-browser.ps1
```

Open `http://127.0.0.1:8080` in Chrome or Edge. Refresh with Ctrl+F5 if an older
version is cached. The [browser README](../../../web/README.md) owns setup/build
details and the complete controls.

1. Select the matching game and keep its default parameters for a learned seat.
2. Select **Q-learning** or **Neural OSLA** for either player.
3. Use that seat's model file picker to load the local `.json` file. The adjacent
   selector lists loaded models with their game and the first twelve identity
   characters. Select a model; a single compatible controller-type entry is
   selected automatically. Rust checks the complete game/configuration contract
   when starting the game or creating a fork.
4. Choose the other player's controller and seeds. Human, First Available, Random
   and heuristic OSLA remain available. Learned players can also face each other.
5. Start the game. Human/AI games enable automatic AI responses. AI-only games
   start paused; use **Play live** or **AI step**.

An invalid file reports its error beside the file picker and leaves existing
loaded models/game progress intact. Model loading never changes the weights of
an already-running seat. Load/select another model when creating a new game or
replacing a controller at a fork. The current controller/seed remains visible in
the player cards, with the model retained in its saved settings.

## What the two players do

| Controller | Calculation and selection | Supported task |
|---|---|---|
| Q-learning | Look up Q(s,a) for each currently legal action. Select a greatest value; use the seat's Rust RNG to resolve exact ties. Missing rows have zero estimates. | Version 1/2 tables for default 3x3 Tic-Tac-Toe. |
| Neural OSLA | Rust simulates each legal action, encodes the resulting position for the original observing player, and evaluates its neural value. Select within the existing 0.001 near-best tolerance using Rust RNG. Terminal candidates use exact +1/0/-1 outcomes. | Version 1 reference evaluation models for default Tic-Tac-Toe and Connect 4. |

The Q-table estimates values of actions at a learner decision; it is used directly
as the tabular policy. The neural value estimates a position's future return,
including the opponent-turn positions OSLA produces. The browser preserves both
meanings and displays distinct controller names. It does not convert the Q-table
into an opponent-turn state evaluator.

Neural encoding keeps the trained named payload order and public format-2 context:
observer indicators, next-player indicators, and whether the observer acts next.
The value always belongs to that observer. The supported reference model uses
float32 fully connected layers, tanh hidden activations and a bounded observer
value. Trained fixed-policy weights are retained and checked in full even though
Neural OSLA selects moves using its value output. Direct neural-policy sampling
and GPU inference remain separate capabilities.

## Pause, history, forks and files

Preparation computes a proposal using a restored controller; it does not advance
live RNG. Pause/discard leaves the authoritative controller unchanged. Committing
a valid proposal records the move and its exact RNG progress. Reviewing old moves
uses player-visible observations and does not ask agents for new decisions.

Reset retains original models/seeds. A fork with unchanged settings retains the
exact model and RNG progress at that point. Changing the selected model, policy
or seed creates a fresh controller at the fork. Other branches remain available.

**Save whole session** and **Save current branch** embed each required model once
in a version-1 `tag-learned-browser` bundle when learned controllers are present.
The bundle contains model documents and the normal verified replay structure.
Loading validates every model and reproduces recorded actions/controller progress
before replacing live play. A missing model fails even if the worker previously
loaded that model: a learned bundle must be self-contained. Existing files with
only built-in controllers retain their normal state/replay format.

**Save current state** stores a position without controllers. Loading it asks for
fresh controllers and models, using a candidate worker until the user accepts
replacement. Cancelling keeps the original session. Model files alone are not
game saves and should be loaded through the seat model picker.

## Limits, comparison checks and source map

The browser accepts at most 16 loaded model identities and 10 MiB of model JSON.
Individual files/bundles also have the existing 10 MiB file limit. The reference
reader limits input features to 16,384, named inputs to 16, tensor rank to 8, hidden
layers to 8 with at most 512 units each, fixed policy outputs to 4096, and weights
to one million parameters. Q-table files have at most 19683 distinct nine-cell
keys. Parsed JSON and temporary arrays add memory beyond file/numeric limits.
Replay validation also charges bounded game and inference work.

Python model digests cover the declared implementation, model contract, shapes,
types and exact float32 tensor bytes. Rust reproduces that framing and verifies
the digest. `serde_json`'s float-roundtrip feature preserves the Python float32
numbers represented by JSON. Q-table identities hash canonical parsed JSON;
they bind its task, float64 values and settings. No arbitrary model code loads.

Numerical tests compare packed inputs exactly and pre-mask logits/values within
absolute tolerance 2e-6 plus relative tolerance 2e-5. Native and actual Wasm tests
cover both seats, initial/ongoing/opponent-turn/terminal positions, trained and
zero models, default (64,64) widths and combined board/cells inputs. Full trained
and controlled-zero game fixtures also compare actions with Python neural OSLA.
Floating-point reduction/tanh differences may change choices near a tie on other
models. Rust and NumPy use different RNG algorithms; equal seeds do not promise
equal Q-learning tie samples across languages.

| Source | Responsibility |
|---|---|
| [src/inference](../../../src/inference/mod.rs) | Optional model parsing, bounded reference arithmetic and content identities. |
| [browser learned controllers](../../../adapters/browser/src/learned.rs) | Immutable model registry, Q choice, native batched neural OSLA and replay codec. |
| [replay codec](../../../src/replay/controllers.rs) | Explicit conversion/restoration interface for adapter policies, preserving built-in entry points. |
| [seat controls](../../../web/seat-fields.js) | Model picker, identity choices, labels and loading errors. |
| [fixture generator](../../../tools/learning/browser-fixtures.py) | Reproducible Python models, predictions and complete-game references. |

Controller snapshots share immutable model allocations through Rust `Arc`, like
Java references to immutable objects or C++ `shared_ptr<const Model>`. Each seat
still owns its independent RNG; snapshots restore that progress separately. The
`BrowserAgent` enum stores either a built-in policy or a learned policy/model,
comparable to Java sealed variants or C++ `std::variant`.

Run native/Wasm/Node checks and the
graphical proof:

```powershell
cargo test --offline --workspace --all-features
node --test web/tests/*.test.js
node tools/browser/test-browser.mjs
```

The graphical test exercises real file pickers and complete UI/worker games,
including learned seats and mixed opposition. Run it in an environment that can
launch the configured browser, and report launch failures separately from
rule/protocol failures. CPU inference, request limits and model checks are
covered by the native and actual-Wasm suites.

GPU execution and training in the browser are not implemented. Native and neural
MCTS use the separate [local MCTS contract](../learning/local-mcts.md).
