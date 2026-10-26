# Frozen-model multiplayer evaluation

Use the [operating guide](../../guides/learning/evaluate-mcts-models.md) for commands.
`tag_rust.learning.selfplay.evaluate_model(settings, snapshot, cycle=0)` evaluates
a `FrozenModel` using `SelfPlaySettings` (or its mapping). It resolves an omitted
master seed once and returns a report. It does not construct a training session,
update model weights, add examples or consume training minibatch/PyTorch streams.

The existing placement-game evaluation path keeps its two-player native-MCTS
opponent behaviour and version-1 seeds. Can't Stop and Love Letter use the lineup implementation
for 2-4 players, including the two-player case.

## Configuration and compatibility

The standalone CLI reads the model's tensor/value descriptions and fills the
corresponding `model` settings if omitted. It also defaults game/player count from
the artifact. Supplied game parameters must yield the same encoding identity.
Model compatibility, all native agent constructors and evaluator names are
validated before scheduled games. Invalid setup preserves previous output files.

| `evaluation` field | Meaning |
|---|---|
| `enabled` | Enables post-cycle evaluation; must be true for standalone CLI use. |
| `games` | Positive requested game count. |
| `positions` | `rotate` (default) or `fixed`. |
| `candidate_seat` | Required valid seat for fixed; omitted/null for rotate. |
| `opponents` | N-1 unique named `{name,agent}` entries in declared order. |
| `opponent` | Homogeneous shorthand; copies receive suffixes `-1`, `-2`, etc. |
| `opponent_search` | Legacy native MCTS profile. Lineup games repeat it over N-1 seats when no explicit lineup is supplied. |
| `ranking` | Null by default; `score` enables positions based on public game scores. `claimed_columns` remains an alias. |
| `max_actions` | Positive limit or null (default disabled) for lineup games. Legacy placement evaluation retains `limits.max_moves`. |
| `include_actions` | False by default; retain accepted public action events when true. |

Lineup entries support Random, sampled OSLA and native MCTS. OSLA uses
`CantStopProgress` with its default weights or `LoveLetterHeuristic`, depending
on the game, and positive u32 `repetitions_per_action` (default 1). MCTS accepts exactly one inline `config` or
`config_file`; relative files resolve from the experiment JSON directory. The
native registry validates all state/action evaluator settings. Per-game seeds
override MCTS template seeds; final selection is overridden to most visits.
Neural candidate evaluation also uses most visits and no root noise.

`opponents` and `opponent` are mutually exclusive and cannot be combined with
`opponent_search`/`opponent_config_file` in JSON. Names must be nonempty and unique;
`candidate` is reserved. Explicit lineups, score ranking, action traces and the new
action limit support Can't Stop and Love Letter. Model/game registration alone
does not advertise those features for every game.

## Seats and randomness

Logical game ID is `cycle*evaluation.games + index`. A rotating candidate uses
`logical_game_id % player_count`, so rotation continues across training cycles.
For Love Letter, the initial starter independently rotates as
`(logical_game_id // player_count) % player_count`. Thus N-squared games cover
all candidate/starter pairings. An explicit `parameters.first_player` fixes the
starter; this is independent of `evaluation.positions`, which controls candidate
seats only. Self-play keeps its existing starter schedule.

Native opponents fill increasing remaining seat IDs in declared order. Fixed
seats retain that same opponent placement convention. Reports preserve each
actual lineup instead of inferring it from a combined result.

Both lineup games use the [seed recipe version 2](mcts-selfplay.md#stochastic-self-play-seed-recipe),
with phase `evaluation` separating it from training collection:

- `game-chance`, logical game ID and ply 0: real-game chance seed.
- `opponent:{name}`, logical game ID and ply 0: each native controller's initial
  seed, tied to its identity rather than its seat. Its native RNG progress then
  continues throughout the game.
- `search`, logical game ID and actual accepted-action index: each candidate
  MCTS decision's seed. Sampling is disabled, while search's seeded tie handling
  and independent simulation streams retain normal semantics.

Same master/rules/model/settings and fixed work reproduce the supported traces.
Elapsed-time budgets and general floating-point batching retain the existing
neural-search limitations. Neither neural evaluation nor native simulations
consume authoritative game chance.

## Rust execution and ownership

`_engine.NativeOpponents` binds an independently seeded native lineup to one game
position. `advance` executes Rust opponent decisions until candidate input,
completion, the remaining action limit or failure. Python's interpreter lock is
released. Candidate search still uses the existing resumable MCTS worker and
frozen Python evaluator; Python performs no native opponent transition loop.

Only owned native checkpoints persist between detached calls. Local agent objects
are restored, run and snapshotted inside Rust. This preserves evaluator/RNG progress
without making evaluator traits thread-safe or transferring Python objects into
native search. It makes no throughput claim. Random, sampled OSLA and MCTS use
the same core agents as native matches.

Candidate recommendations are applied through `apply_candidate`, which checks the
captured game position and actual next player. Reset, copies, foreign games and
external moves invalidate the retained position. Invalid candidate actions preserve
state. Accepted moves before a later native failure remain accepted and advance
the game token. Further lineup execution is rejected after failure, while public
state remains available for diagnosis. This helper is separate from attached OSLA
controllers and has no reset, clone or portable checkpoint API.

Optional traces contain actor and permanent action ID. Can't Stop adds actual
dice for Roll; Love Letter adds no hidden draw, hand or private effect values.
Metadata comes from the Rust state; Python does not reconstruct rule effects.
Trace storage is bounded to 100,000 entries per game. Ordinary model observations
remain independently owned numeric arrays containing no private RNG progress.

## Results, limits and failures

Each raw game has `completed`, `truncated` or `failed` status, its actual move
count, candidate seat, seeds, lineup and candidate decision seeds/counters.
Completed games include game scores and absolute-seat outcome labels (1 win,
2 draw, 3 loss). Optional score ranking gives each player
`1 + count(greater scores) + (count(equal scores)-1)/2`.

Per-seat candidate summaries track attempted/completed games, wins/draws/losses,
truncations and failures. Combined summaries sum actual candidate appearances;
win rate divides wins by completed games, and average finishing position divides
the sum of candidate places by those same completed games. Zero denominators
produce null. Unlike self-play collection's all-seat summaries, evaluation counts
one candidate appearance per completed game.

Terminal completion is checked before the action limit. Truncated rows have
`outcomes: null` and no finishing positions; their current scores remain public
diagnostics. They never become wins/losses or terminal training targets. The
schedule continues after truncation. Runtime failure records its row, preserves
completed peers, stops further games and reports unplayed entries.

The CLI writes `tag-mcts-model-evaluation/1` JSON with resolved settings and the
evaluation report, using atomic replacement. Runtime failures produce that report
and exit status 1; invalid setup exits before replacement. Post-cycle evaluation
failures remain diagnostics after valid model updates. Training checkpoints retain
the resolved evaluation settings and histories under their existing version.

## Love Letter information and outcomes

Love Letter candidates retain the [game's neural-search contract](../games/loveletter.md):
one deciding-player view, full absolute-seat values, gamma=1 and no
`last_private_history` model input. Native opponents observe their own permitted
facts and construct hypothetical worlds through the existing sampler. No search
receives the actual hidden world. Evaluation uses ordinary existing MCTS, with
the sampler's documented strategic and previous-round-memory limits.

Scores are public favor-token totals. Multiple winners each receive outcome 1;
shared wins contribute to candidate win rate just like sole wins. Optional ranks
use token scores and average tied places, independently of the outcome labels.
Love Letter rows record the initial `first_player`; the report records rule
parameters and `information_access: "privileged-research"`. Seeds reproduce deals
and candidate action traces can reveal private decisions, so these research
reports must not become model inputs or ordinary player-facing data.

The existing native lineup retains checkpoints internally between calls; it has
no public clone/save interface. Training checkpoints retain completed-cycle
continuation. Standalone evaluation cannot resume a partially played report.
