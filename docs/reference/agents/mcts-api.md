# Run and customize native MCTS

For operating instructions, use the [short manual](../../guides/agents/mcts.md). This page contains
the detailed reference.

Monte Carlo tree search (MCTS) estimates action values by repeatedly simulating
play from the current position. Each iteration has four main steps:

1. **Selection:** follow the tree policy through actions already explored.
2. **Expansion:** choose an unvisited available action and simulate it. This
   produces the expanded leaf position.
3. **Rollout and evaluation:** simulate further actions with the rollout policy,
   then evaluate the resulting position. A zero rollout length evaluates the
   expanded leaf immediately. A tree-depth limit can stop selection earlier and
   evaluate that position directly.
4. **Back-propagation:** update visit counts and value sums along the selected
   path. Only completed iterations contribute these statistics.

For player i, Q_i(s,a) is the mean of the evaluated values propagated through
action a at a tree node. Selection uses the component for the player deciding
there. UCB1 adds an exploration term to that mean; PUCT weights exploration by
prior action probabilities. Progressive bias optionally adds h(s,a), an action
heuristic whose contribution decreases with visits. These policies estimate
values for every player without assuming a two-player zero-sum game.

The [parameter reference](mcts-configuration.md) owns every setting,
allowed value and default.

## Simultaneous search

When several players choose from one state, each open-loop tree edge stores
their complete ordered joint action. The focal player chooses through UCB1;
simulated opponents use `joint_opponent_policy` (`uniform_random` by default or
`first_available`) on their own observations of a state sampled from the focal
player's view. No real pending choice or hidden hand enters the search. Rollouts
sample complete decisions. One joint transition consumes one
`forward_model_calls` unit, and tree depth and rollout length each count complete
decisions. A completed iteration back-propagates one value per absolute seat.
Root visits for one focal action aggregate all explored joint edges containing
that action.

Opponent observations are constructed inside a hypothetical state sampled from
the focal observation. The focal policy never receives those observations or
the real opponents' hidden information. Since tree paths store complete joint
actions, this search mode requires every joint choice to become public before
the next decision. `joint_actions_public_after` declares that rule, and search
rejects simultaneous decisions that do not satisfy it. The game must verify
the declaration with observation tests; hidden-action games need an
information-set tree instead.

Joint search currently supports UCB1 with random expansion and random rollouts;
it rejects PUCT, progressive bias and heuristic expansion for a simultaneous
decision. A native or externally supplied state evaluator may still estimate
the complete successor. The explicit joint-action tree can branch by the
product of participants' offered action counts; resource limits bound explored
nodes and edges. Sequential games retain their existing search settings.

## Run a JSON-configured agent

For configurable opponents, rule parameters and agent-versus-agent experiments,
use the [match runner](../experiments/matches.md). The commands below demonstrate the MCTS API
against a fixed Random opponent.

From the project root:

```powershell
cargo run --offline --release --features mcts-json --example mcts -- examples/mcts/basic.json tictactoe 2
cargo run --offline --release --features mcts-json --example mcts -- examples/mcts/puct.json connect4 2
```

The example alternates MCTS between both seats against Random and reports wins,
draws and losses separately and together. It prints resolved settings and both
seeds. Null or omitted seed selects a system-generated seed once when creating
the agent. Copy the printed integer into your JSON to repeat fixed-work runs.
Time-limited runs can explore different trees even with the same seed.

The application reads the file; `Settings::from_json` reads only JSON text. No
Python, model file or inference runtime is needed. The optional mcts-json feature
uses the existing Serde dependencies. Typed native settings need no optional feature.

## Follow a decision through the Rust API

```rust
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::tictactoe::{TicTacToeForwardModel, TicTacToeParameters};
use tag_rust::search::mcts::{EvaluatorRegistry, MctsAgent, Settings};

let model = TicTacToeForwardModel;
let state = model.init_state(&TicTacToeParameters::default(), 2)?;
let actor = PlayerId::new(0);
let observation = model.observe(&state, actor)?;
let actions = model.available_actions(&state, actor)?;
let registry = EvaluatorRegistry::<TicTacToeForwardModel>::for_game();
let settings = Settings { seed: Some(42), ..Settings::default() };
let mut agent = MctsAgent::new(settings, &registry)?;
let report = agent.search(&model, actor, &observation, &actions)?;
// The selected index refers to this offered list, not a neural output number.
// Applying the move is separate from calculating it.
let mut continuation = state.clone();
model.next(&mut continuation, actor, &actions[report.selected_index])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The example uses shared references (`&`) to read the original observation and
actions. The agent owns its evaluators and random generator. Its simulation copies
are independent. Only the last line mutates the separate continuation through an
exclusive reference (`&mut`), comparable to passing a mutable C++ reference while
Rust checks that no competing reference can mutate that state simultaneously.

`MctsAgent` also implements `Agent<F>::choose`, so ordinary native runners can use
it. `choose` retains `last_report()`; `search` directly returns an owned report and
clears any previous report. [Browser native/neural MCTS and model-backed
experiments](../learning/local-mcts.md) use separate
adapter controllers. The existing BuiltinAgent enum remains unchanged.

## Understand open-loop simulations and limits

Set `structured_actions: true` in MCTS settings to visit component nodes.
A partial node retains its parent's simulation state and deciding player. Its
edge receives visits and player-value back-propagation, while game depth and
the forward-model counter advance only when a complete action is applied.
Root reports still list complete actions in offered order. Existing
`structured_actions: false` checkpoints use algorithm version 1; structured
checkpoints use version 2, and restoration preserves the selected mode. The
existing complete-action neural policy mapping is reused: descendant action
probabilities are summed and conditioned at each prefix. A factorized neural
output would require a separate mapping version.

An open-loop node represents an action history. Each iteration obtains a simulation fork from the root observation
and applies the actions selected along that history. Nodes retain visit
counts and value sums. Those simulated actions
consume forward-model budget, alongside expansion, rollout and heuristic-generated
successors. Legal choices and heuristic scores are recomputed from the current
simulation. Existing tests cover changed legal choices and decision makers using
controlled fixtures and real Can't Stop games. `SimulationModel` supplies a fresh
chance stream per iteration, separate from real dice and action selection.

Depth counts actions from the root. At depth 1, one tree action is applied and its
result evaluated directly. Rollout length zero instead evaluates the expanded leaf
wherever expansion happens. Positive rollout length counts further simulated
actions after expansion. Null disables either limit independently.

An iteration budget counts completed iterations. It cannot interrupt an unlimited
rollout in a game that can cycle forever. For such a game, enable a rollout-length,
time or forward-model limit. Placement games always end after finitely many moves. Can't Stop needs a finite
rollout, depth or transition budget because repeated busts can continue indefinitely.

Budgets stop at the first enabled limit. At forward-model exhaustion an in-progress
iteration evaluates its current position and completes back-propagation without another
action. At a deadline unfinished statistics are discarded. A native evaluator
cannot be interrupted while its function is executing. If no root action acquired
a completed visit, the agent returns a reported seeded random legal fallback.
In structured mode, an iteration interrupted during a partial path is discarded;
the partial selection has no state transition to evaluate.
When the validated root has exactly one action, MCTS returns it immediately.
The report sets `forced=true`, `fallback=false`, one root action with zero visits,
one root node, no edges or stop reasons, and zero iterations, transitions and
evaluator calls. Its mean-value entries are zero because no state was evaluated.
Root validation still makes one permitted simulation copy;
the agent/search RNGs do not advance. This applies to native, Python and browser
consumers of the shared search engine.

`search_controlled` accepts `BudgetOverrides`, a `Clock` and a cancellation probe.
For JSON overrides use `BudgetOverrides::from_json`: an omitted field retains the
default, null disables it, and a number replaces it. All-disabled budgets fail.
Cancellation returns `Error::Cancelled` and never returns a move. RNG/evaluator
work already performed remains advanced. To calculate a decision while preserving
the live agent, restore a separate agent from its snapshot and search with that copy.

Timed Wasm calls need a browser-backed Clock; the native SystemClock reports an
unsupported-clock error there. Automatic Wasm seeds use `with_seed_source` with
an adapter-supplied source, or supply an explicit integer. The browser
adapter supplies a monotonic browser clock and independent seat seed controls.

## Provide your own evaluator

An evaluator specification identifies a registry name plus parameters. State and
action registries are separate. Use `register_state` or `register_action` to supply
a factory closure returning a fresh boxed implementation. A trait is Rust's
interface; `Box<dyn StateEvaluator<F>>` owns an implementation selected at runtime,
similar to owning an object behind a Java interface or C++ unique_ptr to a base type.

StateEvaluator returns `StateValues::Full` in absolute seat order, or
`StateValues::Scalar { player, value }`. The selected completion policy requires a
complete vector, fills other players with zero, or fills them with the negation.
For example scalar .7 for Player 1 becomes [-.7,.7,-.7] under negation in a
three-player game. Filled entries are explicit assumptions and are recorded in
the effective settings. Selection uses the actual deciding player's component.

The default terminal provider maps win/draw/loss to +1/0/-1. Games with other
reward conventions override `StateEvaluator::terminal` to return actual per-player
rewards. Terminal outputs must be complete and finite; scalar completion never
changes actual terminal outcomes. PlayerEvaluator adapts existing `Evaluator<State>`
implementations by querying each player separately.

ActionEvaluator scores the current offered moves for their deciding player. It
receives `SimulationContext`; use its `copy_state` and `apply` methods for internal
simulations, and `require_transitions` before starting a whole batch. The built-in
successor_state evaluator scores each one-step successor. Before starting, it
checks that the remaining forward-model budget can score every action. If the
budget is insufficient, the iteration
evaluates its current position and stops with a forward-model reason, possibly
before consuming the entire numerical limit.

The registry supplies game_heuristic for all registered native games, uniform_actions for
zero action scores, and successor_state with a nested state evaluator. Bias,
PUCT preferences, expansion and rollout settings build independent evaluator
instances even if they name the same factory. User factories validate their own
parameters and output meaning. Custom code must not bypass simulation accounting
through direct rule calls. Arbitrary private evaluator allocations cannot be bounded
by the framework; output and tree limits are not a process-memory ceiling.

For restoration, implement checkpoint/restore on custom evaluator traits. They
default to an explicit unsupported error. `agent.snapshot()` saves effective
settings, evaluator bytes, action-selection RNG progress and the separate
simulation RNG checkpoint. `MctsAgent::from_snapshot` builds
fresh evaluator instances through the supplied compatible registry and restores
those bytes. Factory authors must version names/parameters and byte formats when
their meaning changes. Native snapshots are Rust values, not a file format.

## Read or extend the implementation

- [Setup](../../../src/search/mcts/engine/setup.rs) creates evaluators and restores agents between decisions.
- [Decision loop](../../../src/search/mcts/engine.rs) runs iterations and back-propagates completed evaluations.
- [Traversal](../../../src/search/mcts/engine/traversal.rs) separates selection, expansion and rollout.
- [Scoring](../../../src/search/mcts/selection.rs) contains UCB1, PUCT, bias and stable probability calculations.
- [Accounting](../../../src/search/mcts/budgets.rs) is shared by search and native evaluators.
- [Native evaluators](../../../src/search/mcts/evaluation.rs) own registry factories and player-value completion.
- [Tests](../../../tests/core/agents/mcts.rs) demonstrate limits, continuation and multiplayer fixtures.

[External neural requests and Python batching](../learning/neural-mcts.md)
use the same search loop. Native ordinary-agent use remains available.
No playing-strength or speed improvement is established by these integration tests.

The standalone example accepts `mcts <config.json> [game ID] [game count]
[opponent seed|auto]`. Its Random opponent uses OS randomness when the fourth
argument is omitted; both resolved seeds are printed. See the
[seed manual](../experiments/reproducibility.md) for framework-wide conventions.
