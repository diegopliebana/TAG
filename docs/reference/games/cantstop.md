# Can't Stop rules, state and encoding

Use the [game manual](../../guides/games/cantstop.md) for commands and rules sources.
This page records the precise game-specific behaviour that callers and models
depend on: legal parameter ranges, action/phase effects, state-copy and dice
semantics, observation-array shapes, action IDs and player-value ordering.
For example, it defines why Roll can leave the same player acting, what copying
preserves about future dice, and which network output identifies Stop.
Shared API usage and MCTS algorithms belong in the linked topic references.


## Rules and actions

`games::cantstop` exports `CantStopForwardModel`, `CantStopParameters`,
`CantStopState`, `CantStopObservation`, `CantStopPosition`, `CantStopAction`,
`Allocation`, `CantStopBaseline` and `CantStopProgress`.

| Setting | Default | Supported domain |
|---|---|---|
| Player count supplied to `init_state` | Example defaults to 2 | 2, 3 or 4 |
| `rules.column_heights` | [3,5,7,9,11,13,11,9,7,5,3] | Eleven positive u32 heights, indexed by column minus two |
| `rules.markers` | 3 | 1-11 |
| `rules.columns_to_win` | 3 | 1 through ceil(11/player_count): maxima 6, 4, 3 |
| `seed` | 0 in `Default`; example uses OS randomness when omitted | Any u64; explicitly resolved by the native caller |

Reject targets above that maximum: all eleven columns could otherwise close
without a winner. For example, four players targeting four columns could close
all columns with scores 3,3,3,2. Rejecting that configuration avoids inventing an
exhausted-board draw. This is an explicit supported-parameter restriction, not a
claim that the published variant mandates those tuning ranges. Heights do not
control allocation size: positions use eleven counters per player regardless of
height. Automatic OS seed resolution belongs to the application adapters.

| Phase | Offered actions | Effect |
|---|---|---|
| `StartTurn` | Roll | Roll four dice internally; enter Allocation. |
| `Allocation` | Allocate one/two sums, or forced Bust | Move temporary markers; successful allocation enters Decision. Bust loses temporary progress and advances the player. |
| `Decision` | Stop, then Roll | Stop banks all temporary markers, claims top spaces and checks victory. Roll risks all temporary progress. |

A Bust is represented by a single offered acknowledgement action so the generic
runner can process it; there is no choice to avoid a failed roll. Each Roll,
Allocate, Stop or Bust increments accepted moves once. Only nonterminal Stop/Bust
increments completed turns. The terminal Stop retains the winning actor and does
not complete an ordinary turn. Cycling players does not create a game round;
the current round counter remains zero, following Java's turn-counter semantics.
Starting player is seat 0. No random initial-player selection is added.

`Allocation::single(n)` and `Allocation::pair(a,b)` validate sums 2-12.
Pair equality/hashing ignores order; repeated sums represent two advances in the
same column. Offered pairs follow the three dice pairings in index order
(01|23, 02|13, 03|12); duplicates are removed, then legal single-sum choices from
unusable pairs follow in encounter order. A legal full pair cannot be reduced
voluntarily to one sum. Near the top a double may supply only one usable advance.

`CantStopState::position()` and `CantStopObservation::position()` borrow immutable
public facts. Queries expose rules, dice, phase, actor, counters, column owners,
permanent/temporary positions, score and outcome. Nonacting observers receive no
offered actions. Unknown seats and invalid columns return contextual errors.
A rejected action or controlled roll leaves position and RNG progress unchanged.
Score is claimed-column count; outcome/reward is a separate quantity.

## Exact snapshots and independent simulations

- `CantStopState::clone()` or `Checkpoint::snapshot/from_snapshot` preserves every
  fact and exact game RNG progress. Mutable arrays are independently owned.
- `CantStopObservation` contains a copied `CantStopPosition` and observing seat.
  It contains no seed, generator state or future events. Its type cannot be used
  to reconstruct the actual real-game RNG.
- `core::simulation::SimulationModel::simulation_fork(observation, &mut random)`
  builds an independent state using a caller-owned `SimulationRng`. Its public
  position matches the observation but its future chance is deliberately new.
  Use a new fork per trajectory/MCTS iteration, not identical clones of one root.
- `SimulationRng` implements `Checkpoint`; checkpoint it alongside agent decisions
  as implemented in native MCTS and sampled OSLA. Forks consume this source only; observing,
  enumerating choices and discarding simulations cannot affect real dice.
- Existing `PublicInformationModel` implementations get a deterministic fork
  implementation that delegates exact reconstruction and consumes no random draws.
  Can't Stop deliberately does not implement that older exact-reconstruction trait.

Real games own `ChanceRng`. Agents still own their unchanged `AgentRng`.
Both environment types reuse the existing pinned ChaCha8 generator/sampler.
Version-one stream names are:

| Purpose | ChaCha stream |
|---|---|
| Existing agent RNG default | 0 (unchanged) |
| Real-game chance | 0x5441475f47414d45 |
| Simulation seed source | 0x5441475f53494d53 |
| Simulated chance | 0x5441475f53494d43 |

`SimulationRng::fork_chance` draws one u64 seed from its source, then initializes
that seed on the simulated-chance stream. Die draws use the existing unbiased
rejection sampler with six outcomes, plus one. This preserves the previous agent
sequences and separates stream purposes even when callers reuse an integer seed.
State equality includes chance checkpoints; position equality compares public
facts only. Do not treat equal visible positions as equal exact snapshots.

## Controlled events and portable persistence

`next_recorded` applies one action and returns `Option<RollEvent>` containing the
realized four dice when rolling. `next_with_roll` applies a supplied roll after
validating four values in 1-6 and checking actor/phase. It still consumes four
normal die samples from the state's RNG, replacing only their realized values.
Thus controlled event replay retains the normal generator's advancement. Inputs
must use the recorded initial RNG/version and action order for exact continuation.

With the `replay` feature, `CantStopSavedState` stores rules-independent mutable
facts: all players' banked positions, temporary markers, owners, dice, actor,
phase, counters, outcome and the exact real-game chance checkpoint. Parameters
retain the rules and resolved seed. `PersistableGame::restore_state` checks player
counts, column ownership/progress, marker limits, actor/turn/phase/outcome
consistency and RNG version/seed/stream/progress. These checks establish internal
consistency, not complete historical reachability from a new game.

The shared [persistence API](../persistence/recordings.md) supports standalone positions,
branch recordings and full sessions. Each recorded Roll includes
`chance: {dice: [d0,d1,d2,d3]}`; import executes the seeded transition and checks
the realized dice. No other action carries chance data. Export reconstructs the
recorded line on an independent root copy, leaving live game/controller RNGs
unchanged. Replay validates the recorded continuation from its checked root;
cross-version deterministic guarantees remain outside the supported contract.

### Browser presentation and continuation

The browser uses SVG tracks for columns 2–12 and distinct colored circle,
diamond, square and triangle markers for absolute players 0–3. White circles
indicate temporary progress. HTML action buttons preserve Rust's offered order
and allocation identity, including repeated sums. Phase, visible dice, owners,
positions, counters, scores and outcomes come from player observations. RNG
checkpoints are private save data, excluded from `BoardView`.

Browser configuration accepts heights 1–128, 1–11 markers and the native winning
target domain above. Native counters still support u32 heights. Browser sampled
OSLA uses the native `SampledOslaAgent<CantStopProgress>`, with 1–128 repetitions
per action, mean-value selection and seeded random ties. Its checkpoints retain
both the action RNG and simulation RNG. Native/neural MCTS retains a full
`MctsSnapshot`, including evaluator checkpoints and both RNG streams; each
decision creates a fresh tree. Discarding prepared work drops an independent
controller copy, retaining all live streams.

Neural MCTS runs locally in Rust/Wasm through `LocalMctsModel`. Models must match
rules/player count and provide terminal-reward values in absolute seat order,
gamma=1, the environment-action clock and all acting observers. Browser saves
bundle referenced models. Import reruns Random/First Available/sampled OSLA
choices; MCTS imports validate recorded legal actions, unchanged policy/evaluator
identity and non-reversing RNG progress without rerunning time-dependent search.
They do not prove that rerunning MCTS would select the recorded action.

Game creation accepts a decimal-string u64 seed or null/omission/`auto` for Web
Crypto. The frontend resolves automatic game/agent seeds once and displays the
actual integers. Save setup returned to JavaScript uses decimal strings; raw
portable JSON is parsed by Rust to preserve native integer parameters. Imported
state files ask for fresh controllers; imported replay files restore controller
progress. Native CLI action/chance research logs retain their existing separate
format and do not become browser replay files automatically.

## Native state evaluation

`CantStopBaseline` returns claimed columns divided by the winning target.
`CantStopProgress` estimates nonterminal progress as
`(C/K + alpha*B + beta*T) / (1 + alpha + beta)`. Here C is the player's claimed
column count, K is the winning target, B is the banked-progress term and T is the
temporary-progress term. Sum over columns not claimed by any player:
`B = sum(permanent_position / column_height) / 11`. For the active player,
`T = sum((temporary_position - permanent_position) / column_height) / 11`, using
the permanent position when a column has no temporary marker. T is zero for
other players. Claimed columns are excluded to avoid counting them twice. Alpha and beta are independently
configurable finite nonnegative weights (defaults 0.5 and 0.25).
Both evaluate any valid player perspective and support the `Evaluator` and
`Checkpoint` traits. They also expose `evaluate_position` for public facts without
requiring access to a state RNG. Every terminal state overrides estimates with
+1 win, 0 draw, -1 loss. Evaluate each player to obtain a complete vector; do not
negate one player's estimate to invent the others.

## Native search and experiment integration

`SampledOslaAgent<E>::new(evaluator, seed, repetitions_per_action)` takes a positive
`NonZeroU32` repetition count. For each action it reconstructs R independent
observation-derived successors and averages their values for the original player.
It selects a highest mean; exact equal means are resolved with the separately
seeded action-selection RNG. `last_transitions()` counts successfully applied
candidate transitions. Typed checkpoints retain the evaluator, R and both random
streams. Evaluator changes before failure are retained; the real game is untouched.
The CLI defaults R to one and uses `CantStopProgress`; direct Rust callers can
supply either evaluator. Original deterministic OSLA remains unchanged.

`MctsAgent<CantStopForwardModel>` uses `SimulationModel`. Each iteration receives
a fresh simulation chance stream. A temporary fork validates root actor/actions
without advancing the persistent stream. Typed checkpoints also retain
`simulation_rng`. Fixed-work deterministic decisions retain their existing streams.
Cancellation may advance a directly used search agent's private streams; restore
an independent checkpoint before searching when discarded work must preserve them.
Search cannot advance the real game's chance generator.

`CantStopForwardModel::native_evaluators()` and
`EvaluatorRegistry::<CantStopForwardModel>::for_game()` provide:

| State evaluator | Parameters |
|---|---|
| `game_heuristic` | None; default progress heuristic |
| `progress` | Optional finite nonnegative `alpha`/`beta`, defaults 0.5/0.25 |
| `baseline` | None; claimed-column fraction |

Each returns a complete vector in player-seat order. The shared `successor_state`
action evaluator independently selects a nested state evaluator. All its simulated
transitions count against the same strict MCTS budget as traversal and rollout.
Within an iteration it copies the hypothetical trajectory's chance state;
repeated OSLA instead obtains independent forks. Successor-state candidates within
one MCTS iteration are not claimed to have independent chance samples.

[Matches](../experiments/matches.md) rotate agents through all seats by default.
[Tournaments](../experiments/tournaments.md) offer separate outcome and finishing-place points.
Can't Stop supplies a full finishing order only with opt-in
`ranking: "score"`; equal counts share occupied places' points.
`max_actions` belongs to runners, defaults to null, and leaves game rules unchanged.
Native action/chance logs retain realized dice and actual seeds. Browser/state
replay imports are not supported by the native research-log format. Python arrays and search are described below.

## Rules sources and implementation differences

Published [Yucata rules](https://www.yucata.de/en/Rules/CantStop) take precedence
over discrepant Java behaviour. Board lengths are also specified in Glenn and
Aloi's [2009 paper, page 1](https://cdn.aaai.org/ocs/123/123-2430-1-PB.pdf).
Compared with the inspected public Java implementation, Rust requires the initial
roll, uses the published 3-to-13-step board, removes opponents' permanent markers
from claimed columns, and banks all columns atomically before testing victory.
The last choice ensures the final state's score includes all banked columns.
The [source notices](../../source-notices.md) identify the read-only reference sources
and retained MIT notice. Source inspection and Rust rule tests do not prove
executed Java/Rust equivalence.

## Python game operations and learning arrays

The common Python `Game` API registers `cantstop` for 2-4 players, supporting
creation, reset, independent copies, observations, action IDs, move application
and outcomes. `MctsSearch` and `MctsBatch` calculate native/neural moves.
[Self-play](../learning/mcts-selfplay.md) provides training and completed-cycle
checkpoints; [frozen-model evaluation](../learning/model-evaluation.md) records
native lineups, candidate seat rotation and separate truncation/rank accounting.
Attached controllers and Gym episodes are separate capabilities. The
[game-operations guide](../../guides/games/cantstop.md#use-from-python)
provides commands.

`LearningGame::learning_spec_for_players(parameters, players, limits)` describes
the configured player count. Its default preserves existing fixed-count game
implementations. Can't Stop's original `learning_spec` method describes two
players; explicit 3/4-player callers use the new method. `encode` reads the actual
observer's public position and returns independently owned arrays.

Let N be the configured player count. Encoding version 1 uses absolute seat order:

| Payload | float32 shape | Values |
|---|---|---|
| `permanent` | [N,11,2] | Each permanent column position as [high16,low16]. Columns are 2-12. |
| `temporary` | [11,3] | [present,high16,low16] for active temporary markers. An absent marker has [0,0,0]. |
| `owners` | [11,N+1] | One-hot column owner: unclaimed, seat 0, ..., seat N-1. |
| `dice` | [4,7] | One-hot face for each die: absent,1,...,6. Dice are absent before the first roll. |
| `context` | [3+2*N] | Three phase bits [StartTurn,Allocation,Decision], N observer bits, N next-player bits. Terminal views have no next-player bit. |

The exact integer position is `high16*65536 + low16`; both components are exactly
representable as float32 throughout the native u32 range. For example, 65537 is
[1,1]. This avoids rounding adjacent large track positions into the same value.
These raw components have bounds [0,65535]. Indicator arrays use zeros and ones;
`temporary` mixes an indicator and the two position components. Rule heights,
marker count and winning target are recorded exactly in the named configuration
identity. This encoding is specific to its player count/rules, with no seed/RNG
state in the identity or observations. Seeded runs of the same rules share an
identity. No grid or open-line heuristic is advertised.

Action IDs are fixed across supported rule configurations:

- 0=Roll, 1=Stop, 2=Bust.
- 3-13 are single allocations for columns 2-12, in ascending order.
- 14-79 are unordered pairs with repetition in lexicographic order: (2,2),
  (2,3), ..., (2,12), (3,3), ..., (12,12).

For example, roll [1,2,3,4] initially offers (3,7), (4,6) and (5,5): IDs 29,37,44.
The legal mask always has 80 entries. `offered_action_ids` follows native rule
order, which is distinct from ascending ID order. Nonacting and terminal views
have no offered actions. Decoding an ID does not make it legal; application uses
the ordinary Rust rule checks and position token.

Progress/outcome metadata uses the common int64 counters and absolute-seat int8
outcomes. `max_cells` bounds N*11 permanent track entries; `max_bytes` bounds
returned numeric buffers before allocation. Describing arrays does not allocate
their values: metadata-only requests can succeed when requesting all arrays would
exceed the configured byte limit. Returned arrays remain valid after the game
changes or closes.

Python accepts nested `rules` with `column_heights`, `markers`, `columns_to_win`,
and optional `seed`, using the same native parameter domains. Python/NumPy integer
indices are accepted; Boolean, negative, floating and out-of-range inputs fail.
Omitted/null seed resolves once from OS randomness; `game.seed` returns the actual
u64 integer. Keep this seed with research setup metadata, outside acting-agent
inputs. `game.reset()` repeats the original resolved setup; it does not draw a new
seed. `game.clone()` retains current state/RNG progress independently. Create a
new game to select a new automatic seed.

## Python stochastic MCTS

`MctsSearch` captures public facts from the acting player's observation. Rust's
`SimulationModel` constructs independent hypothetical chance streams for MCTS
iterations; it never receives the real game's RNG. The Python worker now uses
this same interface for deterministic and stochastic games. For deterministic
games, the existing reconstruction implementation supplies the simulation.
Game rules, transition accounting, tree traversal and back-propagation remain in
Rust. See [neural search](../learning/neural-mcts.md) for request/reply and batching APIs.

For Can't Stop, omitted `value_observers` selects `deciding_player`. External
state evaluation requires this mode and `value_completion="require_full_vector"`.
Each value request contains one acting-player observation and expects N finite
values in absolute seat order. A value estimates expected terminal outcome reward:
win +1, loss -1, draw 0. The neural model must declare
`ValueSpec(players=tuple(range(N)), gamma=1)` and support every acting observer.
`turn_context="own_turn"` is sufficient; `any_turn` also admits these requests.
Its encoding identity binds the player count and rules. Scalar models, reordered
value declarations, discounted values and mismatched rules/counts are rejected
before search. Existing placement-game scalar/vector conventions remain supported.

Policy requests use the next decision maker and the simulated position's offered
actions. Repeated Roll/Allocate/Stop decisions can belong to the same player;
search follows `next_player` at every node. Terminal game outcomes bypass neural
evaluation and supply the entire player vector directly. Scores and finishing
positions do not replace these rewards.

Batching combines independent trees, with at most one pending request per tree.
While suspended, the Rust future retains tree, simulated state and RNG progress.
Polling or rejecting a malformed reply leaves that pending continuation intact;
a valid reply resumes it. Cancellation discards it without applying a real move.
This is in-memory continuation. Each new `MctsSearch` creates a fresh agent/tree;
Python search checkpoint export or continuation between separate decisions is
not implemented. Completed-cycle training checkpoints are supported through the
[self-play workflow](../learning/mcts-selfplay.md#reproducibility-and-checkpoints).

Game and search seeds resolve independently from OS randomness when omitted;
retain `game.seed` and `search.settings["seed"]` to reproduce fixed-work decisions.
The [demonstration](../../guides/agents/rust-mcts-from-python.md#run-independent-searches)
uses `cantstop-mcts-demo-v1`: hash the ASCII string
`cantstop-mcts-demo-v1:{master}:{role}:{index}` with SHA-256, then interpret its
first eight digest bytes as a little-endian u64. Roles are `game`, `search` and
`model` (model index 0). The output records master/game seeds, model identity,
root seats and reports containing resolved search seeds. This demonstration recipe
does not change existing workflows' seed derivations.

Python receives owned arrays. Constructing inference batches copies those arrays;
long Rust work releases Python's interpreter lock. No GPU, zero-copy, throughput
or playing-strength claim follows from this integration.
