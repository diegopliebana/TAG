# Love Letter rules, state and observations

Use the [short manual](../../guides/games/loveletter.md) for the native command.
The selected authority is the classic option (page 14) in the user's
[Z-Man rulebook](https://cdn.svc.asmodee.net/production-zman/uploads/2026/04/LL_Rulebook_with_Bag.pdf).
The game supports native rules, observation-derived simulations, existing agents
and shared experiments; Python operations/encoding, native/neural MCTS, complete-game
self-play/checkpoints and frozen-model evaluation; and browser play with separate
private/reveal views and checked actual-game persistence.

## Native types and setup

`games::loveletter` exports `LoveLetterForwardModel`, `LoveLetterParameters`,
`LoveLetterRules`, `LoveLetterState`, `LoveLetterObservation`,
`LoveLetterPublicState`, `LoveLetterAction`, `LoveLetterOutcome`, `Card`, and
typed public/private events. `ForwardModel` supplies creation, observation,
offered/available actions, transitions, player results and scores.

| Setting | Default | Supported values |
|---|---|---|
| Player count passed to `init_state` | Example: 2 | 2, 3, 4 |
| `rules.winning_tokens` | `[6,5,4]` | Three positive u32 targets for 2, 3, 4 players |
| `seed` | Native `Default`: 0 | Resolved u64; example offers independent OS-random setup |
| `first_player` | Seat 0 | Any valid seat; digital setup replaces the social first-player criterion |

The fixed sixteen-card multiset is `Card::DECK`. Ranks are
Guard 1 (five copies), Priest 2 / Baron 3 / Handmaid 4 / Prince 5 (two each), and
King 7 / Countess 8 / Princess 9 (one each). Spy and Chancellor are absent.
Card counts, hand size and visible-reserve size are not configurable parameters.
Invalid counts, zero targets and invalid starter seats fail before a state is returned.

Round setup shuffles, sets aside one hidden card, removes three further cards
faceup for two players, deals each player one card and draws for the starter.
At a decision the current player therefore holds two cards. Effects and automatic
draws are part of checked transitions, rather than separate player actions.

## Actions and effects

| Action value | Purpose |
|---|---|
| `Play(card)` | Handmaid/Countess/Princess play, or explicit no-effect play when a card requiring another target has none |
| `Target {card,target}` | Priest/Baron/King target another eligible player; Prince may also target self |
| `Guess {target,card}` | Play Guard and name a non-Guard character |

Actions own copied card types and seat IDs. They contain no inspection values,
actual opponent card, hidden physical-card identifier or references into a state.
Equivalent plays of duplicate cards are deduplicated. Stable action order is
ascending card rank, then ascending target seat, then ascending Guard guess rank.
There is no neural action-ID mapping yet.

Only the actor's observation offers actions. Off-turn and terminal observations
offer an empty list. Every non-Guard character remains a legal Guard guess even
if public information makes it implausible; the game does not prune strategic
choices or reveal whether a guess would hit. Countess alongside Prince or King
forces `Play(Countess)`, with no public indication whether the play was forced.

Handmaid protects against other players until the start of the holder's next
turn. Guard/Priest/Baron/King have no effect when all other active players are
protected. Prince must target self in that case. Eliminated players cannot be
targeted. Priest reveals privately to the actor; Baron compares privately between
its two participants, eliminating the lower card while equal cards survive.
King exchanges retained hands. Prince publicly discards its target's retained
card and replaces it; a discarded Princess instead eliminates its holder without
a replacement. With an empty draw pile, the hidden set-aside card is consumed as
the replacement, removing it from that zone. The public reserve is never drawn.

Invalid plays and invalid actors return `LoveLetterError` before changing state,
private memory or real-game randomness. Legal actions execute directly in the
supplied real or simulated state, without a defensive full-state copy. An internal
failure such as counter overflow returns a descriptive error and stops that
execution; its state may contain partial effects and must not be retried. Search
discards the failed simulation, while an interactive live session blocks further
moves until reset or restoration of a valid position.

## Rounds, shared outcomes and counters

Round elimination is temporary; whole-game results stay `Ongoing` until the match
finishes. Eliminated hands become public discards without triggering their effects.
The last survivor wins immediately, without automatically exposing their retained
card. If the pile is empty after an action resolves, surviving hands are revealed;
every highest-rank survivor wins the round. Discard totals do not break ties.

Each round winner receives one favor token. Every player reaching the configured
target wins the whole game simultaneously. `LoveLetterOutcome::Won(Vec<PlayerId>)`
keeps all winners in seat order: each has `PlayerResult::Win`, with `Loss` for
others. A shared win is not a draw or a fixed two-player zero-sum result. Score is
the player's token count. Search evaluators and learned values are described below.

If play continues, a sole winner starts next round; tied winners are sampled
uniformly using the real-game chance stream and stable seat order. All players
return, protection clears, cards reshuffle and current-round information restarts.
After game termination there are no new draws, starter samples or next decisions.
The preserved current/last actor fields are metadata; use `next_player` for control.

Each accepted card play increments move count and completed-turn count, including
a terminal play. Round resolution increments completed rounds and resets turns
within the round to zero. The new round's setup/draws add no moves or turns.
Targets, tokens and counters use checked arithmetic; overflow is an explicit
execution error. This is this game's counter convention, not a universal turn model.

## Information boundary and retained experiences

`LoveLetterState` owns all hands, actual draw order, removed card, chance RNG and
each player's private experiences. Its fields are private. Explicitly named
`privileged_hand`, `privileged_draw_pile`, `privileged_removed_card` and
`chance_snapshot` are research/audit access. Acting agents receive observations,
not these full-state values.

An observation owns:

- `public()`: rules, public actor/round metadata, tokens, participation/protection,
  public reserve/discards, draw/removed counts, outcomes and public events.
- `hand()`: only the observer's current hand, including their own new draw.
- `private_events()`: only that observer's current-round receipts, Priest
  inspections, Baron comparisons and King exchange values.
- `last_round_private_events()`: only that observer's most recently completed
  round experiences; `public().last_round` holds the corresponding public record.

Unknown values and seeds/RNG checkpoints are absent, rather than stored behind a
visibility flag. A private event's `event` field indexes the public transcript of
its round. Public events never include private effect values. Their counts and
indices describe public chronology, not hidden card identities.

The transcript describes experiences at their time: seeing a Prince does not
assert that the target still holds it after replacing or exchanging cards. A
failed Guard guess records its public constraint. Public exchanges plus prior
private inspections permit deductions without revealing the actual unknown cards.
Both Baron participants retain their comparison; outsiders receive only the public
comparison/elimination outcome and any public discard. Private data from other
players is never copied into an observer's view.

Live state retains the current and most recent completed round, keeping storage
bounded across long matches. Earlier review requires retaining earlier snapshots
in the caller's history. After a reshuffle, previous-round private events remain
separate from current-round card inference. This is not a full-match event archive.

Native paired-world tests vary unknown allocation, deck order and game RNG while
keeping the observer's information fixed. Observations (including Debug text),
action values/order and existing Random-agent choices agree. Further pairs cover
forced/voluntary Countess, Priest, Baron, King and failed Guard effects. Tests
cover native inputs; future Python/browser/model serialization needs its own
leakage checks when implemented.

## Exact snapshots, seeds and controlled deals

`Clone` and `Checkpoint` duplicate all mutable state and exact RNG progress.
Snapshots are typed in-memory values with private fields. Independently continuing
a restored state and agent checkpoints reproduces all decisions, effects, future
round deals and starters under the same versions. Queries draw no randomness.

Setup uses `ChanceRng` and a Fisher–Yates shuffle. Draws consume the fixed deck
order; only shuffling and tied-starter selection sample chance. The example's
game and agent seed sources are independent, preventing agent seeds from directly
revealing the shuffle seed. Both sources offer explicit integers/OS randomness,
are resolved once and are printed for reproduction. There is no clock fallback.

`init_with_deck(parameters, players, top_first)` is a privileged controlled-outcome
tool. The sixteen-card array must match `Card::DECK` exactly. A normal shuffle
still advances the real RNG, then the supplied top-first order replaces its
outcome before removal/dealing. Reproducing such a fixture needs both its seed and
supplied deck, not the seed alone. Controlled deals do not assume Java/Rust seed
equivalence. Browser file workflows use the separate actual-origin validation
[described below](#browser-views-controllers-and-persistence).

## Observation-derived simulations

`SimulationModel::simulation_fork(observation, random)` constructs a hypothetical
complete state. Its only inputs are the observation and a caller-owned
`SimulationRng`; it cannot read the real hidden state or the game's RNG.
The sampler assigns ranks to positions in a hypothetical sixteen-card shuffle.
These positions are local variables, not identities from the real game.

It walks the current round's events, tracking which hypothetical positions are in
each hand. Public discards, Guard results, Baron comparisons and revealed final
hands constrain their ranks. The observer's draws, Priest inspections, comparisons
and exchanges add private constraints at the time they occurred. Later replacement
draws and exchanges move the relevant variables: earlier knowledge does not become
an assertion about an unrelated replacement card. Prince/King plays exclude a
retained Countess. The observer's ordered hand and public reserve are fixed.

Constraint propagation enforces comparisons and card counts. Randomized
backtracking chooses compatible assignments. Each complete candidate is verified
by replaying the round through the normal rules; the resulting observation must
equal the input exactly. This checks legality, protection, turn order, effects,
outcomes and private chronology. Replay also reconstructs other players'
hypothetical current-round private experiences, supporting subsequent forks from
their observations.

The first feasible assignment is returned. This is **not uniform sampling** over
decks or information sets, and does not estimate a posterior conditioned on
opponents' strategic preferences. Earlier actions must be legally possible, but
no opponent model assigns strategic likelihoods. The removed card and remaining
draw order are sampled along with opponent hands. Actual hidden values are never
a fallback.

`DEFAULT_SAMPLING_BUDGET` is 100,000 search nodes per fork, counting event and
card-assignment branches. Native callers may use
`simulation_fork_with_budget(observation, random, nodes)` for another limit.
Exhaustion returns `SamplingBudgetExceeded`; an impossible observation returns
`InvalidState`. Failure leaves the caller's random stream unchanged. The bound
measures construction work, not elapsed time or individual instructions. A final
candidate needs at most one bounded round of rule replay.

Each successful fork obtains two child chance streams from `SimulationRng`: one
for constructing the deck, one for future simulated shuffles and tied starters.
Backtracking effort therefore does not advance future chance. Checkpoints retain
the source position; real chance, agent tie-breaking, construction and simulated
future chance remain separate. Construction/replay work is not included in
MCTS's forward-model-call counter, which measures simulated action transitions
through its existing accounting context.

Public records/counters and the observer's last completed-round private memory
are preserved. Other players' unknown previous-round private memories are not
reconstructed for an ongoing round; their current-round experiences are.
Independent shuffles make previous rounds irrelevant to current card counts.
The supplied policies/evaluator do not learn opponent strategies from previous
private experiences; such strategic memory remains future work.

## Native evaluation and existing agents

`LoveLetterHeuristic` evaluates a player's permitted observation. During play:

```text
0.7 * (own favor tokens / winning target) + 0.3 * (highest own card rank / 9)
```

An eliminated player has no hand contribution but retains match progress. The
highest rank covers both one-card hands and two-card decision hands. The estimate
lies in [0,1) during reachable ongoing play and is a baseline, not a calibrated
win probability. Whole-game winners receive +1 and losers -1, independently for
every seat: two shared winners both receive +1. Round elimination alone is not
a terminal loss.

The native registry exposes `game_heuristic` without parameters. Sampled OSLA
averages successor evaluations over `repetitions_per_action` independent forks,
always evaluating the original deciding player. Existing MCTS uses a fresh root
sample per iteration, the current action-history tree and full player-value
vectors. Neither agent receives the actual hidden state. Each player's heuristic
component uses that player's observation in the sampled successor.

These are approximate planning agents. Simulated choices may exploit a
hypothetical hand the real player would not know, and one action-history tree can
combine incompatible hidden situations. This can bias decisions even without
actual-information leakage. The sampler makes no claim of information-set search, optimal
strategy or improved playing strength. IS-MCTS is not implemented.

## Native experiment capabilities and limits

`loveletter` is registered once in the shared native experiment registry. It
supplies parameter/action serialization, sampled OSLA, its evaluator registry and
game-seed replacement. Shared code owns scheduling, seat rotation, seed derivation,
action limits and reporting. Configure `rules.winning_tokens` (default 6/5/4) and
zero-based `first_player` as needed. See the shared [seed convention](../experiments/reproducibility.md).

All game winners remain `win` in reports and count towards their win totals.
Tournament outcome points split one point equally among winners; this statistic
is separate from game scores and search rewards. Truncations award no points.
Action records omit hidden draw events, but their rules, resolved game seeds and
actions still reproduce hidden deals. Complete experiment reports are privileged
research files, not safe player observations. They are never acting-agent inputs.

Love Letter also supports basic Python operations and `LearningGame`, described
below. Python MCTS calculates moves without attaching game controllers. There is
no `PublicInformationModel`: an ordinary observation cannot reconstruct the real
hidden state. Browser registration, private/reveal presentation and checked
state/replay files use the separate integration described below. Card artwork
origins and rights are recorded in [source notices](../../source-notices.md).
Unsupported capability combinations fail explicitly.

The rule, observation and simulation tests exercise controlled deals, retained
knowledge, inaccessible-world equivalence and complete matches.

## Python operations and learning arrays

`tag_rust.create("loveletter", parameters, players=2)` uses the shared Python
`Game` API for discovery, observations, action application, copies and reset.
`parameters` accepts `rules.winning_tokens`, zero-based `first_player` and `seed`.
Integer seeds reproduce setup; `None`/omission resolves OS randomness once.
`game.seed` exposes that resolved integer to the experiment owner. It is privileged
setup data that reproduces hidden deals, and is absent from observation arrays,
metadata and encoding identity. Do not pass game handles or seeds to acting
policies; pass their observations. There is no clock fallback.

Copies preserve full private state and chance progress in independently owned Rust
storage. Reset repeats the original resolved setup and invalidates old position
tokens. Arrays are independent NumPy buffers, remaining valid after the game
changes or closes. Rust rule/encoding work uses the existing detached execution
path; no zero-copy or new throughput claim is made. Shared
[Python operations](../../guides/running/game-operations-from-python.md) explain calls.

The encoding identity is game=`loveletter`, rules version=1, encoding version=1,
player count, and all three canonical `winning_tokens` values. Seed and initial
starter are not encoding configuration: starter is an observed state fact.
All feature arrays are `float32`; seat axes use absolute zero-based seat order.
Let N be the configured player count:

| Array | Shape | Meaning |
|---|---|---|
| `hands` | `[N,2]` | Ordered own cards by rank; another player's occupied slots are -1 (unknown), empty slots 0. Public terminal reveals fill their known ranks. |
| `players` | `[N,6]` | Active in round, protected, token high16, token low16, observer indicator, next-player indicator. |
| `reserve` | `[3]` | Public reserve ranks; zero padding. All zero for 3/4 players. |
| `discards` | `[N,16]` | Public ranks in discard order, then zero padding. |
| `context` | `[10]` | Draw count, removed count, four round-number limbs, target high16/low16, last-round-present, last-actor-plus-one. |
| `public_history` | `[256,22]` | Current-round public events, in order. |
| `private_history` | `[256,7]` | Only the observer's current-round private experiences. |
| `last_public_history` | `[256,22]` | Most recent completed round's public events. |
| `last_private_history` | `[256,7]` | Only the observer's corresponding private experiences. |
| `last_tokens` | `[N,2]` | Tokens at that round's end, high16/low16; zero when absent. |
| `terminal_values` | `[N]` | Whole-game Win=+1, Loss=-1; all zero while ongoing. Shared winners each get +1. |

Ranks are 1,2,3,4,5,7,8,9. In history, rank zero means the field is unused. A
private revelation describes its event time, not necessarily the current hand.
`hands` does not attempt to perform deductions from private history; the full
permitted evidence remains available in the history arrays. For example, a Priest
row retains Prince after the target later replaces it, while the replacement
remains -1 in `hands`. No sampled or actual unknown card is encoded as a fact.

Each history row starts with a validity value: 1 for a real event, 0 for padding.
Padding is entirely zero; private rows are packed in that observer's own order.
256 rows is a conservative bound: a round has only sixteen cards, at most sixteen
plays, and fewer than twelve public events per play plus bounded setup/end events.
Private event counts are smaller. Overflow returns an error rather than truncating
retained information. Current/last round are separate, including at game end when
the terminal round is also the last completed round.

Public history columns, in order:

```text
valid, event_type, actor_plus_one, target_plus_one, card_rank, guess_rank,
correct_or_enabled, eliminated_plus_one, draw_source, round_end_reason,
round_limb_0, round_limb_1, round_limb_2, round_limb_3,
winner_seat_0, winner_seat_1, winner_seat_2, winner_seat_3,
revealed_rank_seat_0, revealed_rank_seat_1, revealed_rank_seat_2, revealed_rank_seat_3
```

Event types 1–11 are RoundStarted, Received, Played, Discarded, GuardResult,
Compared, Exchanged, Protection, Eliminated, RoundEnded, GameEnded. Actor refers to
the starter/player/actor named by that event; unused fields are zero. A Played
row's rank, target and guess distinguish untargeted plays, targeted effects and
Guard guesses. Winner columns are independent indicators, not one exclusive class.

Private history columns:

```text
valid, public_event_index, event_type, other_plus_one,
card_rank, other_card_rank, draw_source
```

Private types 1–4 are Received, Priest, Comparison, Exchange. Received uses card
and source; Priest uses target in `other` and the inspected card; Comparison uses
opponent, own card, other card; Exchange uses other player, given card, received
card. The zero-based index points into the corresponding public history. That
index may be zero for a real event; validity is given by the separate first column.
Plus-one seats reserve zero for an unused field. Draw sources are Deal=1, Pile=2,
Removed=3. Round-end reasons are LastPlayer=1, DeckEmpty=2.

To avoid float32 rounding, reconstruct a token/target as `high16*65536+low16`.
The round number's four limbs are least significant first:
`sum(limb[i] * 65536**i for i in range(4))`. Every limb is an exactly representable
integer in [0,65535]. Generic progress uses checked int64 counts. Result metadata
retains the shared labels 0=ongoing, 1=win, 2=draw, 3=loss; these codes are distinct
from `terminal_values`. Round elimination leaves all whole-game values ongoing/0.

## Fixed policy action mapping

There are 52 permanent action IDs across 2/3/4-player configurations:

| IDs | Mapping |
|---|---|
| 0–7 | Untargeted Play in card order Guard, Priest, Baron, Handmaid, Prince, King, Countess, Princess. |
| 8–23 | `8 + 4*card_index + target_seat`, where cards are Priest, Baron, Prince, King. |
| 24–51 | `24 + 7*target_seat + guess_index`, with guesses Priest, Baron, Handmaid, Prince, King, Countess, Princess. |

Decoding represents an intent; it does not make it legal. Missing seats, self
targets, protected/eliminated targets, cards not held, mandatory Countess and
otherwise unsupported untargeted plays are excluded by the mask. IDs 52 and above
are rejected. Offered IDs preserve the forward model's action ordering. Off-turn
and terminal observations have no offered choices and an all-false mask. A Guard
guess remains offered regardless of the actual unknown target card.

`max_cells` must allow sixteen card positions. `max_bytes` bounds requested numeric
arrays/indices; metadata-only requests avoid allocating histories. Unknown or
duplicate array names fail explicitly. `spec().heuristic` is absent: it describes
the placement-game open-lines facility, not Love Letter's native heuristic.
Attached evaluation controllers and learner episodes remain unsupported. The search contract follows below.

## Python native and neural MCTS

Use the shared `MctsSearch`/`MctsBatch` API from
[Run Rust MCTS from Python](../../guides/agents/rust-mcts-from-python.md). Native
`game_heuristic` and frozen PyTorch policy/value models both use the same Rust
search, observation-derived sampling, legal actions and independent RNG streams.
Search captures the root observation; it does not pass the actual hidden state
or environment RNG to the worker. Calculating a move does not apply it.

Love Letter requires `value_observers="deciding_player"` (the default) and
`value_completion="require_full_vector"`. This is checked in both Python and the
compiled adapter before search begins. Root-player/all-player value input modes
and scalar completion/negation are rejected. Each external request carries one
observation, belonging to the simulated deciding player. Policy requests always
use that player too. No request combines different players' private views.

Search evaluators must omit `last_private_history`. The sampler reconstructs
all hypothetical current-round private experiences, but preserves previous-round
private memory only for its root observer. Passing other simulated actors' missing
previous-round records as if complete would misrepresent their knowledge. Custom
bindings and neural models requesting that array are rejected in both Python and
Rust. Current `private_history`, previous-round public history and public tokens
remain available. The basic game API still returns the real observer's complete
retained previous-round private record. This is an explicit search-input
restriction, not a change to the observation encoder.

For a model created from `ModelContract.from_spec`, use
`ValueSpec(players=tuple(range(N)), gamma=1)`. Values follow absolute seat order;
every output predicts one player's outcome using the same single permitted
observation. It is a vector prediction, not multiple private inputs. All acting
observers must be supported, with a compatible encoding, rules, tensor layouts
and 52-action domain. A model may declare `turn_context="own_turn"` because
evaluation requests use the deciding player's own turn. Policy logits cover all
52 IDs; only currently offered actions receive preferences. Scalar, missing-seat,
reordered-seat and discounted value contracts fail explicitly. Native evaluation
also returns complete independent player vectors.

Terminal simulations bypass estimated values and return actual +1 for each game
winner and -1 for each loser. Round elimination alone remains nonterminal.
Suspended searches retain their Rust tree and agent/simulation RNG progress;
rejected responses preserve the pending request. Request IDs, evaluator identity,
action mapping, vector length and finite values are validated before resumption.
Cancellation, closing or an outdated root discards work without applying a move
or advancing the real game's hidden deck/RNG. Python search checkpoint export
remains unsupported; suspension is not an exported checkpoint.

`MctsBatch` groups compatible requests across independent trees, one pending
request per tree. It uses existing row/byte bounds and frozen CPU PyTorch
prediction; no within-tree batching or new planning algorithm was introduced.
Native simulations/traversal/accounting remain in Rust, with the Python interpreter
lock released during suitable worker operations. No GPU or throughput claim is
made. Iteration, time and forward-model-call budgets keep their established
meanings: hypothesis construction/replay is bounded separately by the sampler,
and is not included in transition-call counts. Interruption is checked around
simulation forks; constraint construction itself runs to completion or its node
bound before returning control.

The [shared demonstration](../../../adapters/python/examples/sampled_mcts.py) supports
`--game loveletter`, `--players 2|3|4`, native evaluation or neural evaluation,
independent searches and explicit/automatic seeds. Root starter seats rotate
through `first_player`. `--model` loads compatible evaluation JSON; omission in
neural mode creates an explicitly untrained network and starts no training.
The [game-specific entry point](../../../adapters/python/examples/loveletter_mcts.py)
forwards to that same implementation. Existing Can't Stop commands remain wrappers
with their original seed recipe and seeded decisions preserved.

Love Letter's demonstration recipe is `loveletter-mcts-demo-v1`: hash the ASCII
text `loveletter-mcts-demo-v1:{master}:{role}:{index}` with SHA-256 and interpret
the first eight bytes as a little-endian u64. Roles `game`, `search` and `model`
derive separate seeds; model uses index 0. The master comes from an explicit
integer or OS randomness and is printed along with resolved game/search seeds.
These are privileged research outputs: they reproduce actual deals, so do not
include them or game handles in model inputs. Neural requests receive only the
specified observation arrays and permitted decision metadata.

All deeper observations describe sampled hypothetical worlds. Actual-hidden-state
independence does not remove ordinary determinization's strategic limitations:
simulated players can exploit hypothetical knowledge, and one action-history tree
can combine incompatible hidden situations. Search retains the first-feasible,
nonuniform sampling distribution and previous-round-memory limitations. IS-MCTS
and new search algorithms remain excluded. The following sections describe
complete-game self-play, frozen-model evaluation and browser integration. 

## Complete-game self-play and training

The existing `SelfPlaySession`, `collect_games`, training buffer, policy/value
losses, CPU Adam updates, model export and completed-cycle checkpoint workflow
support Love Letter with 2–4 players. No separate game-specific learner, Gym
environment or search algorithm was added. The
[shared manual](../../guides/learning/mcts-selfplay.md#train-love-letter)
owns commands; [self-play reference](../learning/mcts-selfplay.md) owns common options.

Each collection group uses one frozen model. At every actual decision it copies
only the acting player's real permitted observation, then obtains policy targets
from that root's completed MCTS visits. Sampled leaf observations stay inside
search; they are not collected as real-game training examples. After the whole
game ends, every retained decision receives the actual terminal vector in
absolute seat order: all winners +1, all losers -1. Gamma is 1 and the declared
clock is `environment-action`. Round elimination does not finish the game or
assign final outcomes; eliminated players return in later rounds.

Set `model.value_output="players"`; `value_players` is omitted or exactly the
absolute seat sequence. Search uses one deciding-player observation and strict
full-vector completion. `last_private_history` remains excluded under the
sampler limitation. The supplied model uses current/public state, current private
history, previous public history and public previous tokens. It omits the
`terminal_values` input because real decisions precede the outcome and that array
would be all zero. Targets are stored separately and never inserted into decision
inputs. Other supported feature subsets remain configurable.

The example [configuration](../../../examples/selfplay/loveletter.json) uses default
classic rules/targets, 3 players, two cycles, three games per cycle, eight MCTS
iterations per decision, `[32,32]` hidden layers and eight updates per cycle.
`games_per_cycle` need not be a multiple of the player count, but full multiples
balance starters within a cycle. The short example disables evaluation; the
shared evaluation workflow can run after each cycle. Model, byte,
position, action-count and checkpoint limits retain their shared meanings.

### Starters, randomness and recorded outcomes

The default starter is `logical_game_id % players`, including across cycles and
regardless of active-game scheduling. An explicit `parameters.first_player`
selects a fixed starter. The experiment seed is top-level `seed`; an integer
reproduces the workflow, while null/omission resolves OS randomness once and
records/prints the resulting integer. `parameters.seed` is rejected because the
workflow supplies a separately derived actual game seed for every logical game.

Love Letter uses the existing
[stochastic self-play seed recipe, version 2](../learning/mcts-selfplay.md#stochastic-self-play-seed-recipe),
unchanged from Can't Stop. That shared reference owns derivation, purposes and
known seed fixtures. Rust's search-owned named streams further separate agent
choices, hypothesis construction and future simulated chance, as described above.

Reports preserve actual game seeds, starters, full action traces and outcomes.
Per-seat and combined counts use the shared outcome helper. Combined totals count
player appearances; every shared winner contributes one win. Each cycle records
its losses, sampled indices, model identities and bounded buffer digest. Actor,
game ID, search/noise seeds and search reports are research metadata, not model
features. Exact numerical continuation assumes recorded software/CPU settings and
fixed work/batching configuration; changing batch shapes is not a bitwise numerical
equivalence guarantee.

### Completed-cycle checkpoints and information boundaries

Checkpoints are accepted only after a successful completed cycle, or before any
cycle begins. They preserve model/Adam state, replay-buffer examples, Torch and
minibatch RNG progress, logical game counters and recorded history. New games use
the saved next logical ID; resumes do not obtain a new master seed. There are no
active game/search states to resume at these boundaries. Mid-game/cycle resume
remains unsupported.

Love Letter uses checkpoint format version 2 with its recorded game identity,
without changing existing placement/Can't Stop formats or seed recipes. Loading
checks settings/provenance, counters, outcomes, RNG/optimizer state, numeric
limits and buffer digests. It then replays each recorded complete action trace
through the Rust rules using its derived seed/starter, including traces whose
early examples were evicted from the buffer. Every retained input must exactly
match the real actor's permitted observation at that position, and the final
outcomes must match. This rejects forged private input even if its digest is
recomputed. It does not rerun historical MCTS or optimization. Replay work grows
with completed history, bounded per cycle by collection-position limits and per
game by the configured move limit when enabled.

If any game hits a move/collection limit or fails, the entire collection group
aborts before buffer insertion or updates. A later optimization failure marks
the session failed. Neither can overwrite the last successful checkpoint; recovery
uses that completed checkpoint. No truncated game receives final value targets.

Checkpoint datasets combine different players' private experiences, and recorded
seeds/action traces reproduce hidden deals. Checkpoints, reports and exported-model
provenance carry `information_access="privileged-research"`. They belong to the
experiment owner; policies receive only their selected permitted arrays. Exported
numeric weights can be loaded into Python MCTS with matching rules/player count
and full-vector contract. Export is not a browser registration or proof of strength.


## Frozen-model opponent evaluation

The shared `evaluate_model` API, standalone `selfplay_evaluate.py` command and
optional post-cycle evaluation support 2-4-player Love Letter. Use the
[operating guide](../../guides/games/loveletter.md#train-and-evaluate)
for the supplied configuration and commands. The
[evaluation reference](../learning/model-evaluation.md) owns lineup configuration,
independent candidate/starter rotation, seed streams, results, limits and native
controller continuation.

Each candidate decision uses the neural contract and each native opponent
uses its own permitted observation. Random, sampled OSLA and existing MCTS can be
mixed without changing game rules or adding a planning algorithm. Shared winners
remain wins; public scores are favor tokens. Frozen weights and training RNGs
remain unchanged by evaluation. Reports are labelled privileged research data;
actual seeds and reproducible action traces stay separate from model inputs.

## Browser views, controllers and persistence

Use the [shared browser guide](../../guides/games/loveletter.md#play-in-the-browser)
for operating instructions. The existing worker hosts Human, First Available,
Random, sampled OSLA, native MCTS and compatible local neural MCTS controllers.
The stochastic controller/checkpoint codec is shared with Can't Stop. Rules and
all accepted transitions remain in Rust.

`BoardView.loveletter` contains public round/tokens/protection/discards/events and
only the selected observer's hand/current and previous private experiences. An
explicit public-spectator view removes those private fields' contents. Opponent
cards are represented by card backs in JavaScript; identities do not enter normal
payloads. Legal choices are supplied only for an observed human who acts next;
prepared AI choices remain internal until committed. Controller seed display is
redacted. Historical lists stop at the selected cursor, and private facts come
from that historical observation. Shared outcomes use browser `shared_win` with
the complete winner list and independent Win/Loss results.

`presentation` accepts `observer` (null for current human), `public_only` and
`reveal_hidden_information`. It has no rule/search position effect. Automatic
view selection switches directly to a deciding human, otherwise retains the last
human observer; AI-only sessions default to public spectator. Explicit observer
selection is a research display feature, without user authentication.

Reveal adds a separately labelled `privileged` object containing all hands, the
removed card and draw-pile order. It never changes observations, evaluator inputs,
search ownership or any RNG. View-request epochs are separate from game/search
identity: stale responses refresh display while preserving proposal IDs. The DOM
is rebuilt after removing privileged data so image sources/alt text cannot retain
hidden card identities. New, reset and imported sessions start with reveal off;
presentation settings are not serialized in game saves.

The optional `replay` feature implements `PersistableGame`/`ReplayGame` using
`LoveLetterSavedState`. New files contain a `snapshot` with complete public and
private state, card zones, retained current/previous-round experiences and exact
environment RNG progress. Snapshot data is constructed when saving; the simulation
state stores no match-long action recipe and appends no persistence actions.
Full move/controller history belongs to the optional recording layer. A bare
state save resumes one position and provides no earlier navigable moves.

Loading checks player/zone dimensions, card conservation, rules, scores, counters
and RNG seed/stream/progress. It reconstructs at most the two retained rounds from
their private draws and public actions, checking all retained public/private facts
against rule execution. Unused unknown cards from the completed round can be
filled consistently because they affected none of its retained experiences.
This validates local knowledge and continuation invariants; proving the entire
earlier match requires a recording. Retained histories and validation work have
explicit limits independent of the number of earlier rounds. Unknown fields and
inconsistent cards/memory are rejected before replacing live play.

Older files with `players`, optional `initial_deck` and `actions` remain readable:
their bounded reproduction recipe is replayed once during import and then
discarded. New exports use snapshots, including when an older file was imported.
Older binaries that only understand recipes cannot read the new snapshot shape.
Rule behavior and game rules version 1 are unchanged; the saved-state payload
distinguishes the two representations structurally. Sampler-created hypotheses
use a separate chance stream and are not accepted as ordinary real-game saves.
Native Clone snapshots continue to support those roots. No serialized in-flight
search or new planning algorithm is introduced.

The full supplied model uses 13,154 inputs at three players. Local reference
inference now permits 16,384 features while preserving its 10 MiB model and
one-million-parameter limits. Full absolute-seat values, gamma=1, deciding-player
requests and the `last_private_history` exclusion remain mandatory. Card asset
origin/credits are recorded under `web/assets/games/loveletter/`; rank captions
follow this rulebook independently of the older Java edition.

### Action results and permitted reveals

Browser `human_move` and `commit_ai` commands accept an optional `observer` for
that response only. The UI requests its preceding viewer, then shows a result
step before returning to the normal automatic/fixed observer. Invalid response
observers are rejected before committing a move. Responses contain one permitted
observation, never pooled private experiences from multiple participants.

Rules still apply one complete rule step, including any next-round deal. The UI
holds the preceding display while showing effect-time facts. Results use the
new public-event suffix and that observer's matching private-event suffix. At a
round boundary they use the completed round's transcripts, not the newly dealt
hand. MCTS, authoritative RNG and controller progress are unchanged by reading
or acknowledging a result. When a result screen is shown, automatic live and recording playback await Continue;
Pause acknowledges the result and leaves playback paused. This is an action
review, without a separate human handover screen.

A correct Guard guess reveals the eliminated hand; an incorrect guess does not
reveal its identity. Baron participants see their exact private comparison;
other observers see only public discarded cards. Priest and exchange details
remain audience-specific. Round notifications preserve every shared winner and
its token award; a public last-round notice remains after acknowledgement.
Dialog close/cancellation removes its private image/text nodes. Global privileged
reveal is separate: removed/draw-pile images are smaller and ordered top first.

Result screens default to showing only when at least one Human controller is
still active after the effect in its round. At a round rollover, eligibility is
computed from the completed round's prior active flags and new elimination
events, not the automatically reactivated next deal. Thus AI-only remainder moves
stay uninterrupted; normal screens resume in the next round with active humans.
The page setting `always-action-results` defaults false and overrides this policy
when enabled. Public round notices and game outcomes remain visible even when
result screens are skipped. This preference affects no observations, RNG or save
content. Draw-pile headings and token separators use plain punctuation.

The user-facing term is action information screens. Their played-card thumbnail
comes from the public Played event. Guard guess thumbnails represent the public
guess, marked correct/incorrect; comparison thumbnails retain the event-time
private/public disclosure rules above. Marker labels accompany the colored tick,
X or equals sign so the meaning does not depend on color alone.
