# Add learning support to a game

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md) · [General game workflow](../framework/implementing-games.md)

Implement `LearningGame` on the game's existing forward-model type. This lets the
game describe its numeric inputs and action mapping once in Rust. The generic
Python adapter calls those methods for each registered game.

Game-name discovery comes from the Rust factory registry in
`adapters/python/src/engine.rs`: it supplies both `create()` and
`tag_rust.registered_games()`. Register the new constructor there and implement
the required typed adapter operations. Generic Python examples and training
configuration must use that registry, rather than add game-name allowlists.
The native match/tournament application similarly registers each concrete model
once in `examples/support/matches/execution.rs` for validation and execution.
Browser views, episode support and replay handling remain explicit integrations
when the game requires them; a discovered name does not supply those capabilities.

Use [Tic-Tac-Toe](../../../src/games/tictactoe/learning.rs) and
[Connect 4](../../../src/games/connect4/learning.rs) as examples, alongside the
[explained trait walkthrough](learning-game.md).

## Decide what the observer's numbers mean

Start with an observation the player is allowed to receive. Choose which facts
an algorithm needs and how to represent them: a feature vector, grid, graph or
image. Define the ordering, dimensions, number types and viewpoint of every
representation. For hidden information, specify how unknown facts are represented.

Tic-Tac-Toe and Connect 4 use +1/-1/0 cells and own/opposing/empty channels. Their shared
encoder assumes two players and a rectangular placement board. Another game can
reuse that helper when these assumptions fit. A card game or a multiplayer game
may need another shared encoder or its own representation. Keep the observation's
information restrictions in the implementation and tests.

## Describe the format and implement encoding

`learning_spec` should validate the parameters and return the available formats
with `EncodingIdentity`. Advertise payloads that the game can actually produce.
Read [data descriptions](data-descriptions.md) for the fields and current limits.

`encode` should check requested names and sizes, read the supplied observation,
and return independently stored arrays plus the required player/progress/action
information. Explain how each shape follows from the configuration. For example,
Connect 4's `[3, 6, 7]` follows from three occupancy channels and six rows of seven
columns. Each output value must have a documented meaning.

The current helpers `placement_spec`, `PlacementView` and `encode_placement` live
inside the engine's learning module. Game implementations supply a cell-reading
closure, dimensions, counters, outcomes and offered IDs. These helpers are
crate-internal and apply to this family of games. General games use
`EncodingConfiguration::Named` for their settings and may set `heuristic` to None.
See [neural inference](neural-inference.md#extension-boundaries) for the general
model boundary and explicit current observation-encoding and adapter limits.

## Give actions fixed meanings

Define the complete action ID set for each configuration. Decide how concrete
actions map to IDs and back, and check that both conversions agree. The game's
offered-action list determines the true entries in the current mask.

Tic-Tac-Toe assigns an ID to every cell; Connect 4 assigns one to every column.
The [action guide](action-ids.md) explains why occupied cells and full columns
retain their IDs. Games with richer actions need an explicit mapping design;
`ActionMapping::Indexed` supports game-defined permanent IDs. `Candidates` describes
variable action lists with decision-local identities. The coordinate helpers
support only cells and columns; the game's own mapping must handle richer actions.
Candidate execution requires an action-feature encoder, retained decision/action mapping
and adapter checks before a game can use it end to end.

## Test the meanings, as well as the lengths

Use the [existing tests](../../../tests/core/learning.rs) as a starting point:

1. Check default and changed dimensions against the advertised shapes and ID count.
2. Apply moves at asymmetric coordinates and check their exact vector and channel
   positions. A center-only example can hide a row/column ordering mistake.
3. Observe the same state as different players and check the viewpoint-dependent
   values, absolute result order and turn-dependent masks.
4. Convert all representable IDs to actions and back. Check occupied/full targets,
   out-of-range inputs and agreement with the offered-action list.
5. Exercise completion, draws and progress counters around terminal moves.
6. Modify returned arrays and advance the live game; verify that the observation
   and earlier encoded data retain their values.
7. Check unknown/repeated names, cell/byte limits and overflowing sizes. Verify
   that failures leave input data unchanged.
8. If supporting the hand-written Python heuristic comparison, export its line
   groups and compare estimates calculated from them with the native evaluator on
   representative positions, including terminal states. Document these groups as
   evaluator support, separately from the observation inputs intended for models.

Run the relevant commands from [the entry point](overview.md#follow-the-code).
Add a dispatch entry and parameter parsing to
[the Rust adapter registry](../../../adapters/python/src/engine.rs), then extend
[its comparison cases](../../../adapters/python/tests/cases.json) and Python
tests. The common Python classes discover formats from the Rust descriptions.
Use the same worked positions to confirm agreed types, shapes and values reach
Python. [The connection guide](../python/game-api.md) explains the current API.

To support OSLAPython, implement PublicInformationModel only when the game's
observation really supplies the information needed to reconstruct a permitted
simulation. The current adapter registers both public games with this capability.
For hidden information, select a sampling design that respects the observer's
knowledge before enabling lookahead. Test the new game's candidate arrays and
acting-player viewpoint using the [OSLAPython checks](../../../adapters/python/tests/test_evaluation.py).

For Rust-owned execution, add the game's enum dispatch in
[execution.rs](../../../adapters/python/src/execution.rs). For grouped execution,
also extend the session dispatch in
[group_limits.rs](../../../adapters/python/src/group_limits.rs). The common
coordinator then combines the game's requests using its Rust encoding description;
all current grouped numeric inputs are float32 tensors. Add the game to
[test_grouping.py](../../../adapters/python/tests/test_grouping.py), comparing
individual and grouped moves, exact controller checkpoints and row viewpoints.
Select an encoding extension before introducing another number type or a new
image or graph observation encoder. [Grouped evaluation](../python/grouped-evaluation.md) explains the
current compatibility and size limits.

## Keep the explanations beside the implementation

For direct neural play, construct a model input/output specification
(`ModelContract`) with explicit observation-array names from the game's
description. For neural-value OSLA, use the existing external
request path with a frozen snapshot and declared reward/player conventions.
The [neural play guide](neural-play.md) owns this integration. New episode games
still require native runtime/registry dispatch and transition/teacher tests;
general model inputs alone do not supply game rules or hidden-state sampling.

Update this guide folder with a worked position for the new game. Show its board
or other visible information, the resulting numbers and the action mapping.
Explain each conversion beside the code that performs it, using matching names.
Link the new implementation and its tests from the game workflow. Describe unsupported Python or neural combinations explicitly. Add them to the
capability list only when their implementation and relevant checks exist.

For training, the current [neural training workflow](neural-training.md) uses complete episodes,
fixed action IDs and one observer value. New game registration must preserve
reward/turn semantics and pass target/bootstrap tests before enabling training.
Model descriptions retain their general interfaces; additional player-value or
candidate-policy training requires a selected algorithm.
