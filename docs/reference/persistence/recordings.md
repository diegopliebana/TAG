# Adding save/load support to a game

For operating instructions, use the [short manual](../../guides/games/new-game.md). This page contains
the detailed reference.

This page explains the implementation steps
for new games. [File formats and limits](formats.md) defines the portable schemas;
[framework contracts](../framework/contracts.md) explains sufficient state,
independent snapshots and portable actions.
For the overall order of game implementation, start with the
[new-game workflow](../framework/implementing-games.md). This guide supplies its save/load step.

If Rust is new to you, start with [Reading this framework from Java or C++](../framework/rust-for-java-cpp.md)
for ownership, references, traits, associated types, enums and error handling.

## 1. Establish what must survive saving

Saving a state means preserving enough information to resume the rules, not
just drawing the same board. Decide which points in play can be saved, and list
every field needed to resume from those points.
For Terraforming Mars, that may include player resources/production, cards and decks,
board tiles and ownership, global parameters, generation, active player/phase,
passed-player flags, pending effects and any history consulted by the rules.
This is an illustrative checklist; establish the actual fields from its implementation.

Preserve identifiers and their references: a card referenced by a pending effect
must resolve to the same card after loading. Declare how IDs/indices are encoded
and checked. State includes sufficient rule history even when navigation history
is omitted. Preserve environment RNG progress or declared future chance inputs
for stochastic games. A seed alone restarts a sequence.

Full state and player observations remain separate. Native state exports may
contain hidden information; player-facing adapters need an explicit access policy.

## 2. Define a game-specific file representation

Add src/games/<game>/persistence.rs and register it in the game module:

```rust
#[cfg(feature = "replay")]
mod persistence;
```

Define SavedState as the data written to the file. It should contain values with
their own storage, so the file data remains usable after the live game changes. Serialize/Deserialize are serde
traits that turn these values into file text and read them back. Choose field names
and a version so a later loader knows how to interpret the file. This layout is
called a schema. The SavedState associated type can be any complete representation. BoardStateData is useful for the current placement games;
Terraforming Mars should define its own appropriate data structure.
Keep internal State fields private. Read a file into separate data values first. Check them before constructing a
live State. For example, reject a Connect 4 board with a piece floating above an
empty cell rather than accepting or silently repairing it. Use stable IDs for
references and decimal strings for integers too large for JavaScript to represent exactly as Number when crossing JavaScript.

Parameters and actions also need feature-gated serialization for replay files.
Actions encode stable move identity rather than pointers into an old state.
Game rules stay free of filesystem/browser/Python dependencies.

## 3. Implement the optional persistence capability

**Persistence** means keeping data after the current program run, usually in a
file. The [PersistableGame trait](../../../src/replay/mod.rs) is implemented on the forward
model alongside its other traits. It adds capabilities to the same model type.
Implement:

| Item | Responsibility |
|---|---|
| SavedState | Complete owned file representation for this game's simulation state. |
| GAME_ID | Stable unique game identifier, independent of display labels. |
| RULES_VERSION | Compatibility identifier; change it when old files imply different rules. |
| storage_units | Checked estimate of required state storage under the parameters. |
| validation_work | Checked estimate for restoring the supplied state data. |
| state_data | Read State and return complete file data with separate storage. |
| restore_state | Borrow parameters/data, validate them and return owned State or PersistenceError. |

To load a recording and continue using its original agents, also implement
ReplayGame on the same model. It adds
SavedEvaluator and verification_work(parameters, osla), the checked transition/AI
work estimate. Saving a state does not require agent support and can be implemented separately,
even when the rules type cannot be cloned. The current replay implementation only supports evaluators that keep no changing
internal memory between calls; controller checkpoints themselves remain an optional capability.

Study [Tic-Tac-Toe persistence](../../../src/games/tictactoe/persistence.rs) and
[Connect 4 persistence](../../../src/games/connect4/persistence.rs) for registration,
ownership and the validation boundary. Their alternating-count, grid, gravity and
last-move checks apply to their own rules. A new game's validation must reflect
its phases, turn order, components and outcomes.

Document named inputs, ownership, returns and errors in Rustdoc. Use checked
arithmetic and explicit limits before allocations, reference resolution and
expensive verification. Work units are estimates; measure and refine them.

## 4. Validate before constructing live state

Files record versions for the file layout, game rules, supported agents/evaluators
and random-number generator. These values let a loader reject a file whose meaning
it does not understand. They are assembled by versions() in
[shared file checks](../../../src/replay/common.rs). Update the relevant format/agent/evaluator/
RNG identity when its interpretation or future choices change. File compatibility
needs explicit identifiers; a Cargo package version alone does not establish it.
State import checks format/game/rules. Replay import also checks agent/evaluator/RNG
identities. Version-1 state files retain their existing metadata fields for schema
compatibility; agent metadata has no effect on their game-state restoration.

Typical checks include supported parameters/schema, collection limits, valid
player/component IDs, unique identities, resolved references, legal resource
ranges, phase/actor consistency and agreement of outcome/counters with state.
The new game determines which apparently unusual combinations are legal.

Avoid silently repairing invalid files. Produce a descriptive error. Build an
independent candidate state and replace live execution after successful validation.
Test that modifying the loaded state cannot change its source.

Declare the extent of reachability validation. Invariants may establish that a
state is internally consistent without proving its complete history from setup.
A replay establishes its recorded continuation by executing actions from its
validated root. Complex games need an explicit account of any stronger guarantees.

## 5. Integrate state files and replay files

For saved game state without a move recording saving, export_state(model, parameters, state) returns JSON.
import_state(text, model, limits) returns validated parameters/state. Choose fresh
controllers and pass those values to Session::from_state; BranchSession can then
make that loaded position its root. Keep file I/O with the adapter/caller.

For replay saving, retain initial parameters, sufficient state, controllers and
accepted moves through BranchSession or run_game_recorded/GameHistory. Convert
trusted headless history with BranchSession::from_history. export_replay/import_replay
handle shared ancestry and supported controller checkpoints generically.

Built-in replay support covers Human, Random, FirstAvailable and deterministic
OSLA with a stateless evaluator. Adapter `ReplayCodec` implementations provide
versioned `SeatData::Extension` policies for native/neural MCTS, learned placement
players and stochastic Can't Stop controllers. The `*_with` replay functions
accept these codecs without requiring deterministic state reconstruction.
`validate_fresh` checks extra controller streams at replacement boundaries;
the common importer checks the ordinary action RNG separately.
Capture every cross-move field affecting future choices. Keep per-decision MCTS
trees transient as agreed; preserve RNG progress and relevant persistent memory.

Stochastic games retain environment RNG progress in their saved state. Set
`ReplayGame::HAS_CHANCE` and implement `chance_event(action, after)` to describe
public realized chance after a checked transition. `MoveData.chance` is optional
and omitted from deterministic files, preserving existing placement formats.
Stochastic export reconstructs chance on an independent root copy; import checks
each event against normal seeded execution. Can't Stop records four dice per
Roll. Its [game reference](../games/cantstop.md#controlled-events-and-portable-persistence)
owns state invariants, independent simulation streams and continuation limits.
Browser game selection additionally needs adapter registration; follow the
[implemented browser guide](../browser/integration.md). The browser supports state, branch
and session downloads/uploads for all three implemented games.

## 6. Verify with meaningful cases and retain examples

Test initial, ongoing, terminal and interrupted/phase-specific states; configured
variants; independent copies; state-file continuation with new controllers; and
replay continuation with exact original controllers/RNG. Include malformed fields,
invalid references/versions, resource limits and failed loads preserving live play.
Use exhaustive small states or independent invariants where practical.

Generate actual example files through the engine exporter, load them back, and
keep loader regression tests. [Current sample files](../../../examples/saves/README.md)
and their [generator](../../../examples/save_files.rs) show this pattern. Register
new game tests in its tests/games/<game>/ folder; shared codec tests remain under
core/sessions/ while persistence is exercised at that execution boundary.

Run formatting, Clippy, native tests with replay enabled, Rustdoc and wasm checks.
Document exact supported versions, validation coverage and remaining limitations.

## Single-line recordings and controller compatibility

export_branch writes only the selected line of play, including its moves inherited
from parent branches and any recorded changes of controller. The file no longer
stores the relationships to other branches. Replay format 2 supports these events; format 1 without them stays supported.
import_replay checks who acted, what action was taken, who acts next, and the
saved agent progress after each accepted move. A newly installed agent must begin
at its entered seed; a preserved agent must retain its progress. Full configuration and persistent agent
state must survive a checkpoint: a seed alone cannot provide exact continuation.
A stateless evaluator may have an empty snapshot. Parameterized/stateful
evaluators need complete versioned configuration/state; local learned controllers
also bind immutable model identities. Declare and validate those representations.

State-only import checks state/game/rules compatibility independently of agents;
its caller selects new controllers. Unsupported replay agent/evaluator/RNG versions
fail; a recorded-moves-only fallback is deferred. Browser file workflows and limits
are described in the [browser protocol](../browser/protocol.md).

## Adapter controller formats

`ReplayCodec` defines how an adapter encodes a complete controller checkpoint and
restores an independent controller from file data. `export_replay_with`,
`export_branch_with` and `import_replay_with` accept that codec. Existing built-in
entry points retain their supported evaluator contract and normal file formats.
Agent scheme 3 permits explicitly interpreted extension entries; the default
built-in codec rejects them. Unknown policies are never silently replaced.

The browser learned codec records policy/version/model identity and exact Rust
RNG, with required immutable models embedded once in a separate browser bundle.
It also charges inference work during replay verification. See
[browser learned players](../browser/learned-controllers.md) for the selected formats,
ownership, byte/work limits, supported tasks and validation requirements.
