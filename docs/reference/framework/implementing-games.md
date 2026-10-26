# Build a new game: step-by-step workflow

For operating instructions, use the [short manual](../../guides/games/new-game.md). This page contains
the detailed reference.

This workflow connects native rules to optional agents, files, Python and browser
execution. Start with a concrete ruleset and list which capabilities the game
will expose. [Framework contracts](contracts.md) explain state, observation,
action, randomness and outcome guarantees. [Rust from Java/C++](rust-for-java-cpp.md)
explains the language constructs used below.

## 1. Settle the game rules and scope

First explain the game in ordinary game terms: how many people play, how it starts,
what a player can do on a turn, how turns change, and when someone wins or the game
draws. List adjustable settings and their valid ranges. For example, Connect 4 has
board width/height and a target number of consecutive pieces.

Also describe random events, information hidden from other players, and any phases
such as setup or scoring. State whether actions can trigger further decisions.
Resolve ambiguous rules before writing the implementation.

For a public TAG port, inspect its public Java implementation and relevant game
rules. Record the exact reference revision and controlled comparison scenarios.
Preserve behaviour through idiomatic Rust choices. Document deliberate differences, their rules source and comparison cases. Preserve adapted-source
notices in [source notices](../../source-notices.md).

Completion: a concrete design and acceptance cases, including edge configurations,
with agreed scope. Rules needing missing engine capabilities have an explicit plan.

## 2. Check framework support and shared components

| Requirement | Current support / action |
|---|---|
| Sequential player decision and whole-game completion | Decision::Player / Finished and ForwardModel. |
| Generic player counts, per-player result and score | Shared interfaces; placement games have two players; Can't Stop and Love Letter support 2-4. |
| Board storage, checks on player IDs, move/turn counts, line detection and a single-winner result | Reuse existing helpers when their behavior matches the game. |
| Simulating moves from player-visible facts | Deterministic games implement PublicInformationModel. Stochastic MCTS/sampled OSLA use SimulationModel with independent simulation RNGs. |
| Human/AI input, recording moves, saving agent progress and trying alternate continuations | Available through Session, RecordedSession, Checkpoint and BranchSession; implement the traits needed for the selected features. |
| State/replay JSON | Optional replay feature; supported controller/evaluator restrictions are in the persistence guide. |
| Hidden-state sampling | Love Letter implements observation-derived `SimulationModel` forks; define and test the new game's constraints/distribution. |
| Explicit chance decisions or simultaneous decisions | Not represented by the current `Decision` variants; these require a coordinated execution/search/adapter design. |
| Browser adapter/worker and basic text play | Implemented for current games; use [the browser guide](../browser/integration.md). |
| Browser graphics, animation and sound | Available for the current games; a new game needs its own display/input mapping. |
| Numeric observations for learning | LearningGame is available in Rust; see the learning-input guide. |
| Python game operations | Implemented for all four games; register new games in the Rust adapter and extend native/Python comparisons. |
| External evaluation and neural search | Supported by external OSLA and native/resumable neural MCTS; each game needs compatible encoding and simulation capabilities. |

List reusable mechanics and evaluate existing core implementations before adding
new ones. Extract substantial shared logic when concrete overlap justifies it,
and migrate earlier games where their semantics match. Small read-only accessors
can remain game-specific. SingleWinnerOutcome is optional: a game with teams,
multiple winners or richer results needs its own appropriate representation.

Completion: a component map stating what is shared, game-specific or still missing.

## 3. Create a focused module layout

Register the game in [src/games/mod.rs](../../../src/games/mod.rs). Start from the
layout in [Tic-Tac-Toe](../../../src/games/tictactoe/mod.rs) or
[Connect 4](../../../src/games/connect4/mod.rs):

```text
src/games/<game>/
  mod.rs              public exports and private module registration
  types.rs            parameters, actions, errors and small value types
  state.rs            owned simulation data and read-only queries
  observation.rs      player-visible facts and observer identity
  forward_model.rs    setup, choices, transitions and outcomes
  evaluation.rs       optional position estimates
  persistence.rs      optional validated state/file conversion
```

Create files as their functionality is implemented. Split large games further by
responsibility: boards, cards, resources, phases or rules for content families.
Avoid placing all card effects or phases in one huge forward-model function.
Use shared types by their shared names and keep public exports intentional.
Platform APIs, graphics, file I/O and GPU/Python dependencies belong to adapters.

Completion: a compiling module with clear responsibility boundaries.

## 4. Implement parameters, actions and errors

Parameters express rule choices and explicit resource limits. Permit reasonable
experimental values and reject invalid configurations with checked arithmetic
before allocation/indexing. Distinguish true semantic limits from example presets.

An action describes a choice, such as PlaceMark { x, y } or DropPiece { column }.
It must still identify the same move when applied to an independent copy of the state.
Use values, coordinates or IDs rather than references to objects inside the old state. State whose current player
has a forced pass offers an explicit pass action. Empty choices at an ongoing
acting decision are an execution error in the current runner/agents.

Errors distinguish invalid setup/player/move and execution failure from normal
game results. Document rejection order where behaviour depends on it.

Completion: setup domains, action identity and error cases have meaningful checks.

## 5. Implement sufficient state and player observations

Store everything needed to continue play in State: board/cards/resources, active
player or phase, move and turn counts, effects still to resolve, results, and any
past events the rules need. For example, a rule that depends on an earlier action
requires the state to retain the relevant information about that action.
Fixed rule tables may be shared when their identity is clear.

Keep fields private and expose queries that read them. This stops outside code
from creating impossible states, such as a floating Connect 4 piece. Copies used
for simulation must have independent mutable data so a simulated move leaves the
source unchanged. Implement `Clone` with that requirement in mind and test it by
changing a copy and inspecting the source.

Observation contains only facts the observing player is allowed to know. The current public games
own a state copy plus observer ID; use this pattern when every copied fact is public.
A hidden-information game needs its own visibility and sampling design before
agents receive observations. Its sampled possible state must be a complete hypothesis
based on what that player knows. Do not let the evaluator inspect the true hidden data.

Completion: setup and independent-copy tests pass; observation tests establish
per-player visibility. Missing hidden/chance capabilities are explicitly deferred
or implemented in the selected scope.

## 6. Implement the forward model

Implement [ForwardModel](../../../src/core/mod.rs) directly on the game model.
The struct defines data; its impl blocks add methods/traits to that same type.

| Operation | Required game behaviour |
|---|---|
| init_state | Validate parameters/player count and return an owned initial state. |
| player_count / next_player | Report how many players exist and who can act next, or that play has ended. |
| observe | Return that player's owned view containing only information the player is allowed to see. |
| offered_actions | Build the list of choices from the observation alone, in the documented order. |
| available_actions | List legal moves in a supplied simulation state for a particular player. Planning code must be allowed to inspect that state. |
| next | Check the player and move first; then update state and resolve automatic effects. Invalid moves must leave state unchanged. |
| player_result / player_score | Return game-defined results and scores for an explicit player. |

Define phases, reactions, automatic effects and terminal boundaries carefully.
Counters follow the game's rules; terminal moves and completed turns need not mean
the same thing. Preserve action ordering, scoring perspectives and tie/winner rules.
Invalid moves leave state unchanged. Use multiplayer/non-zero-sum scoring where
appropriate rather than assuming complementary two-player values.

After validation, mutate the supplied state directly. Report rare internal
execution errors with a descriptive cause and abort that simulation or live game;
the failed state may contain partial changes and must not be retried. Do not clone
the entire state before every transition merely to roll back such failures.
Independent search branches still need their own simulation states.

Completion: representative full games, outcomes, counters, invalid moves and
configured variants pass. Add exhaustive small cases or independent checks where
practical. Keep available/offered action semantics explicit; check that both queries agree on permitted choices without exposing hidden facts.

## 7. Integrate native execution and relevant agents

First run the game through run_game with FirstAvailableAgent and controlled moves.
Add a native example. Exercise humans and supported AI policies through Session
when that execution mode is selected; compare states/actions with runs without a graphical interface.

To use OSLA (one-step lookahead), implement PublicInformationModel when the
observation contains enough public facts to recreate the full simulation state.
Implement Evaluator<State> to estimate a position for a specified player. Values
must not be NaN or infinite, and state copies must be independent. OSLA evaluates
one successor per move, including terminal outcomes, then chooses randomly among
values within 0.001 of the best. Each agent owns a separate seeded RNG.

For stochastic public games, implement `SimulationModel` to preserve observed
facts while assigning independent future chance. Native MCTS obtains a new fork
per iteration. `SampledOslaAgent` averages R independently simulated successors
per action for the deciding player; exact equal highest means use seeded random
selection. Checkpoints retain R and separate action-selection/simulation RNGs.
See [Can't Stop's integration](../games/cantstop.md#native-search-and-experiment-integration).

Native matches can register Random support without supplying any search,
heuristic, learning or replay capability. Add each supported optional capability
explicitly through `MatchGame`; declare final ranks only if game rules or a named
research convention provide them. Unsupported choices must fail before play.

Test fixed-seed repeatability and error handling. Java/Rust seed equality is not
proof of matching stochastic games; use controlled chance outcomes or recorded
inputs. Measure release-mode workloads before claiming speed improvements.

Completion: a runnable native game and meaningful controller/evaluator tests.
Unsupported agent/sampling capabilities remain clearly identified.

## 8. Add save/load and replay when selected

Follow [game persistence](../persistence/recordings.md) for the exact state schema,
PersistableGame state conversion, ReplayGame agent support, validation, version identities,
controller support, chance continuation and sample files. Use shared sessions,
recording and branches. Keep state saving sufficient to continue the rules even
when navigation history is omitted.

Completion: state and replay round trips, continued execution, independence,
invalid-file rejection and actual engine-generated examples. Declare validation
coverage and exact continuation assumptions.

## 9. Integrate platform presentation and learning when available

The browser supports all four implemented games. Follow
[browser integration](../browser/integration.md) for typed registration, worker
messages, permitted displays, presentation, history and files. Compare every
recorded graphical position with native rules. Preserve asset attribution and
verify redistribution rights; asset manifests live under `web/assets`.

Implement [LearningGame](../learning/learning-game.md) in Rust to declare and
produce numeric observations, fixed action IDs and legal-action masks. Register
basic game operations in the [Python adapter](../python/game-api.md). Basic
registration does not automatically enable attached controllers, learner-step
Gym episodes or training: each has additional observation, opponent, reward and
value-perspective requirements. Tic-Tac-Toe and Connect 4 support learner-step
Gym/teacher training; all four games support complete-game MCTS self-play and
frozen-model evaluation within their documented model contracts.

Use [adding learning support](../learning/adding-support.md),
[external evaluation](../python/external-osla.md) and
[neural MCTS](../learning/neural-mcts.md) for those integration points. Keep
simulation/tree traversal in Rust, batch model evaluation through Python and
preserve owned buffer, cancellation and pending-request semantics. Hidden-game
encoders must read only the appropriate observer's information.

Completion: native/Python/Wasm checks exercise every enabled capability, including
seeded continuation, invalid inputs and information boundaries. Unsupported
combinations fail clearly before play.

## 10. Validate and hand over the game

Run checks appropriate to the selected scope:

```powershell
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets --all-features -- -D warnings
cargo test --offline -p tag-rust --no-default-features
cargo test --offline --workspace
cargo test --offline --workspace --all-features
cargo check --offline --workspace --all-targets --all-features --target wasm32-unknown-unknown
cargo doc --offline --workspace --all-features --no-deps
```

These are verification instructions; run them for the new implementation and
report unavailable dependencies/tools with exact manual setup commands. Add real
browser/Python checks when those adapters exist. Use the
[test organisation](../../../tests/README.md): tests/games/<game>/ for game behaviour,
core tests for shared implementations and nearby unit tests for private invariants.

Record the rules source, deliberate differences, parameter domains, information
boundaries and supported execution modes in the game's technical reference.
Keep its manual short: one working command, configuration and links to shared
operations. Preserve adaptation provenance in [source notices](../../source-notices.md).
Report unavailable checks and limits explicitly; performance claims require
reproducible measurements.
