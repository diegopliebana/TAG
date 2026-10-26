# Portable state and replay formats

The optional `replay` feature provides JSON conversion without adding file I/O
to game rules. [Format types](../../../src/replay/format.rs) define the documents;
[persistence integration](recordings.md) explains the game/controller traits.

## State files

A `.tag-state.json` document contains `kind="tag-state"`, `versions`, full
`parameters` and game-specific sufficient `state`. It omits navigation and
controller history. `import_state` returns independently owned parameters/state;
the caller chooses controllers and creates a session rooted at that position.
Reset then returns to that root, not necessarily the game's empty setup.

State validation checks the game's invariants and applicable resource limits.
Placement games validate dimensions, occupancy, player counts, counters, outcomes,
winning cells and gravity where applicable. A terminal position needs a legal
last-move predecessor without an earlier win. Arbitrary Connect 4 invariant checks
do not establish a complete chronological history from an empty board; a replay
establishes its recorded history from a validated root.

Can't Stop includes environment RNG/phase/dice/temporary progress. Love Letter
includes private state, exact chance progress and validated retained-round
knowledge in a complete snapshot; older proof-based files remain readable. These files
are privileged research/local-save data, not ordinary player observations.
Their [game references](../games/README.md) define additional validation and
proof limits.

## Replay files

A `.tag-replay.json` document contains `kind="tag-replay"`, `versions`, full
`parameters`, `initial_state`, `branches`, `selected_branch`, `cursor` and
`delay_ms`. A branch stores an ID, parent/fork position, starting controllers,
local accepted moves and any controller-change boundaries. A move contains
actor, portable action, optional realized chance data, resulting next player
(null at completion) and the acting controller's resulting checkpoint.

The root may be an imported position rather than an empty initial game. Its rule
counters survive, while recording positions start at zero. Branch IDs and parent
links are validated; a branch refers backward to an existing parent's valid
position. Inherited records plus local records form its chronological line.

`export_branch` emits one line of play, retaining inherited moves and controller
boundaries without unrelated branches. Format version 2 represents ordered
controller changes with absolute positions and complete seat checkpoints. Version 1
remains supported and is exported when those events are absent. Game-specific
chance metadata and rule versions are validated separately.

Replay validation executes recorded moves. AI records also reproduce the choice
and compare the action, resulting decision and checkpoint. Human records pass
ordinary offered-action/rule checks. Unknown controller/evaluator/model versions
are rejected; they are not silently replaced with Random or human controllers.
Replay validation can therefore perform search/inference work, unlike display-only
navigation of an already validated recording.

## Versions and RNG data

`Versions` records `format`, `game`, `rules`, `agents`, `evaluators` and `rng`.
The built-in codec supports its declared controller schemes; extension entries
require an explicit codec. Random-containing built-in records use agent scheme 2;
extension entries use scheme 3. Model-backed browser bundles add explicit model
identities and immutable model documents. Do not edit version fields to make an
unknown file appear compatible.

RNG data records version, initial seed, stream and word position. Large integer
fields are decimal strings to preserve u64/u128 values across JavaScript. An
exact checkpoint restores progress; entering the initial seed again restarts it.
Search trees are rebuilt per decision unless a separate continuation contract
explicitly preserves pending in-memory work.

## Default native import limits

| `ReplayLimits` field | Default |
|---|---:|
| `max_bytes` | 10 MiB |
| `max_depth` | 64 |
| `max_branches` | 256 |
| `max_moves` | 100,000 stored local moves |
| `max_cells` | 4,096 game-declared storage units |
| `max_verification_work` | 100,000,000 estimated work units |

Estimates charge root validation, reconstruction, transitions and controller
verification. Overflow fails validation. Callers may choose other limits, but
work units are conservative estimates rather than timing guarantees. The browser
adds its own envelope, nesting and rendering limits. Imports validate before
installing the new session. See [browser messages](../browser/protocol.md#limits).

## Compare restored views

Game-state continuation and presentation settings are separate. Imports can reset
observer/public/reveal preferences. Compare the same observer and permitted
presentation mode before treating different displayed hands/history as a failed
round trip. Privileged complete state in a save does not grant private information
to the normal post-import view or acting agents.
