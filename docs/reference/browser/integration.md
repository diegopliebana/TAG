# Add a game to the browser bridge

For operating instructions, use the [short manual](../../guides/games/new-game.md). This page contains
the detailed reference.

The adapter/worker provides configured play, pacing, history, forks and portable
files for Tic-Tac-Toe, Connect 4, Can't Stop and Love Letter.
[Implementing games](../framework/implementing-games.md) explains native integration;
[protocol and limits](protocol.md) defines browser messages; [web/README](../../../web/README.md)
provides build and launch commands.

If Rust is new to you, start with [Reading this framework from Java or C++](../framework/rust-for-java-cpp.md)
for ownership, references, traits, associated types, enums and error handling.

## How browser play is connected to Rust

A click becomes an action request. BrowserClient sends it to a Web Worker (a separate
browser execution thread), which calls the compiled Rust engine through WebAssembly
(Wasm). Rust checks the move and updates the game. JavaScript receives the resulting
observation and redraws the board. It must not implement the rules a second time.

The **adapter** is the conversion layer between Rust game types and browser messages.
Its **protocol** is the agreed set of request/reply fields. A request ID matches a
reply to its request. A session generation distinguishes old and new games. A position
version changes when the game moves or switches continuation. These checks prevent
a delayed click or reply from changing a game position it was not intended for.

## 1. Verify the required capabilities

Confirm native rules, observations and game tests first. Runtime is shared command-handling code for different games. It needs independent
copies of state and rules, actions that can be written to messages without references
to old state objects, parameters with defaults and input parsing, and a game-selected
agent whose settings and RNG progress can be saved and restored.
BrowserGame says how to construct controllers and convert a player's observation
into display values. Those values must not expose information that player cannot see.
Human play can use a simple controller type independently of evaluators, public
simulation reconstruction or file persistence. Each selected AI declares its own
simulation/evaluation requirements. Placement-game OSLA uses deterministic
reconstruction; sampled OSLA and MCTS use `SimulationModel` for stochastic games.
Its BrowserGame trait documents these bounds in
[games.rs](../../../adapters/browser/src/games.rs).

Runtime creation uses the supplied controller count, defaulting to two human
seats. Setup, player cards, state imports and forks support Can't Stop's 2–4
players. Its `BoardView.cantstop` optional object carries public track/phase/dice
facts while existing grid views retain their fields. `parse_parameters`,
`parameters_view` and `validate_seats` let each game validate setup and preserve
wide integers at the JavaScript boundary. Love Letter supplies its permitted
private/reveal conversion and observation-derived simulation design.

## 2. Register game selection and setup

Add the supported game identifier to GameKind in
[protocol.rs](../../../adapters/browser/src/protocol.rs), a typed Runtime variant to
[engine.rs](../../../adapters/browser/src/engine.rs), and its creation/view/command
dispatch arms. The generic runtime handles sessions and agents once for all games.
Extend the UI's game selector when human play is selected for this game.

Implement BrowserGame on the existing forward-model type in the adapter. Supply
its identifier, checked dimensions, controller type/factory/settings and mappings
for the observation's board and player result/score values. Use the existing
shared agent types where suitable. Keep rule logic in src/games/<game>.
Game parameter controls/defaults are frontend work following the supported schema.

## 3. Define permitted display data and input

BoardView is the structure of board data sent to JavaScript. It describes the
current public grids. It is a starting message format,
not a required shape for every game. Terraforming Mars will need data for its board,
player resources, public cards, phase and permitted choices. Extend/replace this
adapter schema when implementing it; preserve information boundaries.

Map from Observation. Include only facts permitted to that observer; private data
must remain hidden. Results/scores retain explicit player IDs identifying whose values or observations are being described. Convert
portable offered actions into input controls, preserving their identity/order.
Use text nodes for game/error strings; keep identifiers too large for JavaScript to represent exactly as Number/seeds as strings.
Read permitted result/score values directly from the observation. Display initial
seeds through Agent::rng().initial_seed(); snapshots serve exact restoration.

Add the game's renderer/input mapping under web/. Its cells/components are display
facts; JavaScript sends chosen actions to Rust. Gravity, legality, scoring, counters,
termination and automatic consequences remain in the forward model.

## 4. Respect lifecycle and asynchronous work

Use BrowserClient for worker readiness, request IDs, generations and obsolete replies.
A command based on a displayed board must include that board's position_version.
Keep its game generation and the user's current intention with calculated AI moves.
If reset, replacement or pause makes that work obsolete, discard it instead of
applying it later. The synchronous calculation
may finish inside its worker before cancellation can be processed.

The worker owns the Wasm runtime; main-thread rendering remains separate.
Bounded file import/export uses a separate file channel and a temporary worker
for loading. It validates a file before replacing the current game worker.
Playback cursors and branches use portable history. Follow the persistence
reference for sufficient state and file validation.

## 5. Build and verify the actual browser

Follow web/README for prerequisites and reproducible bindings/server commands.
Extend native adapter tests for setup/actions/observers/errors, and Node tests for
new frontend message logic, including actual Wasm/client rejection paths. Build
web/pkg before running those connected Node tests. Add a native workload in
[native_workload.rs](../../../adapters/browser/examples/native_workload.rs) and exercise
its decisions in [browser cases](../../../web/tests/browser-cases.js).

Compare ordered actions, chosen moves, states/counters/outcomes with native execution
at every transition. Include configurations, controller mixtures, bad input, stale
requests, cancellation and real UI clicks. Compile checks and Node mocks alone do
not establish browser execution. Record who ran real browser tests and its version.
Retain board elements for unchanged game/dimensions and index offered actions once
per view. Test control identity, updated input, busy states and keyboard focus.

## 6. Extend presentation

Use licensed assets with recorded attribution, retain accessible controls and
respect reduced motion and sound settings. Use the separate [presentation preview](../../../web/presentation.html).
Use [board.js](../../../web/board.js)'s optional sprite layers for stock artwork;
keep animation and sound in the [presentation modules](../../../web/presentation/app.js).
Follow the [asset README](../../../web/assets/README.md) for unchanged files, stable
asset IDs and licence/hash evidence. Samples execute through Rust and display
views containing only information the player is allowed to see. Samples highlight effects from Rust-produced views. Graphical configured human play is now integrated in [app.js](../../../web/app.js),
with defaults/parameter fields in [setup.js](../../../web/setup.js). Register a new
selected game's setup fields and validation there; Rust remains the authority.
Provide winning coordinates from the observation, and map accepted-view changes
in animation.js when the game needs another movement style. Reuse explicit
mute/volume and reduced-motion controls. Section 7 covers
history/fork controls; file integration is described below.
AI setup/pacing uses
[ai-playback.js](../../../web/ai-playback.js), the existing prepare/commit/discard protocol
and per-agent u64 seeds. Register a new selected agent's controller factory/settings
in Rust and its UI choice and label in seat-fields.js, with seed/configuration validation
in setup.js/readControllers. Follow the [agent workflow](../agents/implementing-agents.md) for all
integration points; app.js coordinates the resulting controls.
Extend [AI UX checks](../../../web/tests/ai-play.js), including cancellation and seeded
step/automatic equivalence. Preserve separately owned RNG and inputs the player is allowed to see. Extend [human-play checks](../../../web/tests/human-play.js) with configured
play, terminal feedback and cancelled/rejected input for the selected game.

Update this guide and the workflow as those integrations become real. State the
supported player counts, observation visibility, presentation and performance limits.

## 7. Supply historical views and review controls

The [history adapter](../../../adapters/browser/src/runtime/history.rs) maps
historical state through the same observations containing only information each player is allowed to know and reconstructs controller
labels from checkpoints. Generic BranchSession::recorded_prefix supplies chronological
borrowed ancestor/local records. Only actor/action/controller-label metadata crosses
the bridge. Complete controller snapshots remain in Rust. Love Letter restricts
move lists to the selected historical prefix and projects private facts from that
position. Each new hidden game must define its own permitted historical metadata
before enabling these views.

Register the new game's readable action labels in [review-panel.js](../../../web/review-panel.js)
when its selected view/actions require another shape. Live/display identities stay
separate; recorded positions are read-only, and recording playback never runs agents.
Preserve the current branch/cursor during failed operations and pause/cancel outstanding
AI work before navigation, selection or forks. For each seat in a fork request, None means keep that controller with its exact
settings and random progress; new settings mean create a separate replacement.

Extend [review tests](../../../web/tests/review.js) and native browser workloads with
historical observations/results, source preservation, exact AI continuation, inherited
controller changes, takeover, terminal navigation and rapid requests. The
[Protocol limits](protocol.md#limits) define resource bounds;
[measurement report](../../benchmarks/browser-history/README.md) owns navigation timings.

## 8. Integrate browser file operations

State/replay conversion remains optional and independent of human-only BrowserGame
support. The current Runtime file implementation additionally requires BrowserReplay,
which supplies ReplayGame, a matching ReplayCodec and serializable parameters/actions. When a new
selected game supports files, add its GameSession file dispatch and retain parameter
ownership in Runtime. Reuse portable validation instead of reconstructing private
state in JavaScript. Check game-specific storage/verification estimates before load.

A temporary worker checks the whole file while the current session remains intact.
For a state-only file the UI asks for new controllers; for a replay it restores the
saved controller settings and random progress. Preserve exact agent identity, complete
configuration, evaluator/model identity and persistent state; future configurable
agents need explicit versioned schemas. Do not deserialize arbitrary unsupported
agent implementations or silently substitute a new policy.

Keep ordinary and file transport byte limits separate. Test native-produced files,
historical state export, flattened inherited lines/controller boundaries, cancellation,
invalid files leaving the current session unchanged and exact continuation, through real upload/download controls.
[File formats](../persistence/formats.md) defines schemas and verification limits; [persistence](../persistence/recordings.md) owns reusable native steps.

## Learned browser controllers

The [learned-player guide](learned-controllers.md) documents current model
loading and execution. The supported BrowserGame implementations wrap built-ins
and learned policies in BrowserAgent; they validate model contracts against the
game's LearningSpec. Neural OSLA uses the existing native candidate/evaluator
path, and Q-learning preserves current-decision action values. Model support is
optional: ordinary BrowserGame observations/state files need no encoder.

BrowserReplay supplies a controller codec for file restoration. Its optional
model-validation/prediction methods are implemented by games with learning support.
Keep model IDs and complete settings in snapshots/files, share only immutable
weights, and stage registry/file replacements. Extend native/Wasm numerical and
lifecycle tests for a new learned game. Validate real UI interaction in a browser as well as native and actual-Wasm
protocol tests; a compile check alone does not establish graphical behaviour.

## Private-game presentation

Love Letter supplies `HIDDEN_INFORMATION`, `display_board` and `display_settings`
through `BrowserGame`. Shared runtime view selection keeps human observers during
AI turns, suppresses AI action candidates/proposals and limits historical move
lists to their selected prefix. A separate presentation command changes observer,
public-spectator mode or privileged reveal without changing search identity/RNG.
The [Love Letter reference](../games/loveletter.md#browser-views-controllers-and-persistence)
owns its exact data/seed/history boundaries and save-proof limits. New private
games must supply their own permitted board conversion; registration alone does
not grant access to full state in ordinary player views.
