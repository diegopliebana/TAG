# MCTS parameters and JSON configuration

For operating instructions, use the [short manual](../../guides/agents/mcts.md). This page contains
the detailed reference.

For training experiments that use these search settings, see the separate
[self-play JSON parameter reference](../learning/mcts-selfplay.md).
It covers collection, exploration, replay data, optimization and output files.

This reference describes native MCTS settings and external evaluator bindings.
The [API guide](mcts-api.md) explains Rust operations;
[neural MCTS](../learning/neural-mcts.md) explains Python request/resume.

## Loading configuration

Applications read JSON files and pass text to Rust; core rules do not open files.
Named evaluators are resolved by an explicit Rust registry, similar to a Java
factory mapping names to constructors. Each has its own parameter object:
`{"name": "example_evaluator", "parameters": {}}`. This name illustrates custom registration. Built-in names and parameters are listed below.
Unknown names, invalid settings and incompatible contracts produce errors.

## Parameters

These are the accepted JSON names. Effective settings can be exported with Settings::to_json.

| Field | Values and meaning |
|---|---|
| `schema_version` | Required `1`; identifies the configuration format. |
| `seed` | Explicit unsigned integer, or `null`/omitted for automatic seeding at agent creation. Expose and record the resolved integer. Checkpoints also retain current RNG progress. |
| `search_mode` | `open_loop` is the selected default. Each iteration reconstructs a simulation from the root and applies tree actions. Version 1 rejects other modes. |
| `tree_policy` | `ucb1` or `puct`. |
| `exploration_constant` | Finite nonnegative number; controls exploration relative to the declared value scale. Default sqrt(2). |
| `tree_depth` | Positive integer or `null`. Root depth is zero. At the limit evaluate the selected node directly, skipping expansion and rollout. |
| `rollout_length` | Nonnegative integer or `null`. Zero evaluates the expanded leaf. Disabled means continue until terminal or decision-budget exhaustion. |
| `state_evaluator` | Named evaluator and parameters; provides values for every player. Native Rust always available; external neural evaluation optional. |
| `value_completion` | `require_full_vector`, `fill_zeros`, `negation`; Converts explicitly identified scalar values or accepts a complete vector. Default: require_full_vector. |
| `progressive_bias.mode` | `none`, `additive`, `tag_blend`. |
| `progressive_bias.evaluator` | Independent action evaluator providing h(s,a) for the deciding player; required with bias. |
| `progressive_bias.bias_visits` | Positive finite number; used by TAG's blend. Default 1 when omitted. |
| `puct_preferences.source` | `uniform`, `native_softmax`, `probabilities`. All three selected. |
| `puct_preferences.evaluator` | Required for native scores or declared probabilities. |
| `puct_preferences.temperature` | Finite positive number for native softmax. |
| `expansion.order` | `random` default or `heuristic`. |
| `expansion.evaluator` | Required for heuristic expansion; higher scores expand first. |
| `rollout.policy` | `random`, `greedy`, `softmax`. Default random. |
| `rollout.evaluator` | Independent action evaluator, required for greedy/softmax. |
| `rollout.temperature` | Finite positive number for softmax rollouts. |
| `final_selection.policy` | `most_visits` default, `highest_mean`, `visit_sampling`. Uses completed root statistics. |
| `final_selection.temperature` | Finite positive temperature; weights are visits^(1/T), default 1. |
| `budgets.time_ms` | Finite nonnegative milliseconds or `null`; default disabled. External evaluation waits count toward elapsed time. |
| `budgets.iterations` | u64 or `null`; counts completed iterations. Zero allowed. Default 1000 if budgets object omitted. |
| `budgets.forward_model_calls` | u64 or `null`; strict simulated action-application limit including traversal, rollout and evaluator simulations. Default disabled; zero allowed. Copies reported separately. |

`null` disables a limit. At least one decision budget must be enabled. Stop at the
first enabled budget reached. Override fields omitted before a decision retain
defaults; null disables them; numbers replace them. Rust optional values handle
disabled limits easily; the override parser must also distinguish omitted fields.

Tree bias, PUCT preferences, expansion and rollout evaluators are configured
independently. Temperatures are independent. Trees rebuild for each real move.
Initial neural batching uses independent searches, one pending leaf per tree.

## Formulas and boundaries

For visited actions UCB1 uses `Q + K * sqrt(log(N) / n)`, where Q is the deciding
player's mean value, N is parent visits and n is action visits. Canonical bias
adds `h(s,a)/(1+n)`. TAG's blend uses
`(1-beta)*Q + beta*h(s,a) + exploration`, with
`beta = sqrt(bias_visits/(bias_visits+3*n))`.
See [TAG's progressive-bias definition](https://tabletopgames.ai/wiki/agents/MCTS.html#progressivebias).
Unvisited legal actions expand first. Either tree policy can use either bias formula.
Values are not adaptively rescaled. PUCT uses Q + K*P(a)*sqrt(max(1,N))/(1+n).

Softmax uses `P(a)=exp(h(s,a)/T)/sum_b exp(h(s,b)/T)`, calculated stably.
Low T favors higher scores strongly; high T approaches uniform sampling.
For scores 1 and 0, T=1 gives approximately 73% probability to the first action.
Score scale affects temperature. Declared probabilities must be finite,
nonnegative and normalized over legal actions; all-zero mass is rejected.

At forward-model exhaustion during rollout, evaluate the current position and
complete back-propagation without more actions. Time expiry discards unfinished results while
preserving completed statistics. Synchronous native calls cannot be interrupted
mid-call. No completed iteration means a reported seeded random legal fallback.
The same current-position completion applies during traversal and after expansion.
Cancellation takes precedence over time; time takes precedence over finishing work.
Reports include all observed stop reasons. Insufficient whole-list heuristic budget
can stop search before the numerical transition cap is fully consumed.

## Example JSON

Valid native example using zero-valued action scores for additive bias. For useful
successor-state action scores use the nested evaluator shown below or the runnable
[PUCT configuration](../../../examples/mcts/puct.json).

```json
{
  "schema_version": 1,
  "seed": 42,
  "search_mode": "open_loop",
  "tree_policy": "ucb1",
  "exploration_constant": 1.4142135623730951,
  "tree_depth": 30,
  "rollout_length": 10,
  "state_evaluator": {"name": "game_heuristic", "parameters": {}},
  "progressive_bias": {
    "mode": "additive",
    "evaluator": {"name": "uniform_actions", "parameters": {}}
  },
  "expansion": {"order": "random"},
  "rollout": {"policy": "random"},
  "final_selection": {"policy": "most_visits"},
  "budgets": {"time_ms": null, "iterations": 1000, "forward_model_calls": 20000}
}
```

For TAG bias choose `tag_blend` and add `bias_visits`; disabled bias is
`{"mode":"none"}`. Greedy rollouts require an evaluator; softmax also requires
temperature. For PUCT choose `puct` and configure `puct_preferences` with source
`uniform`, `native_softmax` (evaluator and temperature), or `probabilities`
(evaluator). External neural configuration needs a selected model/service contract.

## Compatibility and platform support

Automatic seeds are resolved once, exposed and recorded as actual integers.
Windows setup uses BCryptGenRandom; Unix setup reads /dev/urandom. No dependency
was added. Other platforms can supply with_seed_source or an explicit seed.
Timed Wasm calls require a supplied Clock; the browser adapter supplies it.
Windows runtime is tested; Unix seeding is implemented but not runtime-tested here.
Restoration retains seed and RNG progress. Fixed work and evaluator ordering are
required for reproduction; equal time limits alone do not guarantee it.

State evaluators declare either a full vector or a scalar with its player ID.
Selected completion policies are require_full_vector (reject missing values),
fill_zeros (other players receive zero), and negation (other players receive -v).
The last two accept a complete vector unchanged or complete one scalar; reject
other partial vectors. Native and neural evaluators follow the same rule.
For scalar .7 for Player 1 of three, outputs are rejection, [0,.7,0] and
[-.7,.7,-.7], respectively. Completion never bypasses model-input compatibility.
Filled values are explicit search assumptions; record the policy in reports and
checkpoints. Actual terminal outcomes bypass completion. See the design for details.

Open-loop simulations can encounter different legal actions after stochastic
outcomes. Define action identity, legality handling and recomputation of heuristic
scores/preferences. Open-loop storage alone does not establish stochastic support;
chance streams, games and tests need separate design.

The settings below also govern external request/resume and neural model
compatibility through the Python search API. For browser/file registration, see the [local MCTS guide](../learning/local-mcts.md).

## Defaults and validation

| Setting | Default/validation |
|---|---|
| schema_version | Required; exactly 1. Reject unknown fields. |
| seed | Omitted/null means automatic platform seeding; explicit u64 retained. |
| search_mode | open_loop; reject unsupported alternatives in version 1. |
| tree_policy / exploration_constant | ucb1 / sqrt(2); constant finite, nonnegative. |
| tree_depth / rollout_length | 100 / 10; null disables either independently. |
| state_evaluator | game_heuristic, empty parameters, validated for the game. |
| value_completion | require_full_vector; zero filling and negation require explicit selection. |
| progressive_bias | none; enabled modes require evaluator. tag_blend bias_visits defaults to 1. |
| puct_preferences | uniform; native softmax temperature defaults to 1. |
| expansion / rollout | random / random; heuristic modes require separate evaluators. |
| rollout.temperature | 1 for softmax; finite positive. |
| final_selection | most_visits; visit_sampling temperature 1, finite positive. |
| budgets | Omitted object means 1000 iterations, other limits disabled. Supplied object's omitted limits are disabled; at least one enabled required. |
| Budget numbers | Zero valid as immediate stop/fallback; finite nonnegative f64 milliseconds, u64 counts. |
| limits.max_nodes | 100000, positive, includes root. |
| limits.max_edges | 1000000, positive. |
| limits.max_actions_per_decision | 4096, positive. |
| limits.max_request_bytes | 67108864, positive; checked numeric output buffers, including root mean values. Python external request encodings also respect this bound. |

Only omitted fields receive defaults. Null is accepted only for seed and the
explicitly disableable limits. Reject irrelevant mode-specific fields. Minimal
configuration is `{"schema_version":1}`. Per-decision budget overrides
retain the separate omitted/disable/replace semantics defined earlier.

Can't Stop additionally supplies `baseline` and configurable `progress` state
evaluators; see its [search reference](../games/cantstop.md#native-search-and-experiment-integration).

Native registry names are game_heuristic (existing per-player evaluator),
uniform_actions (zero action scores, no parameters), and successor_state (one-step successors
evaluated through a nested parameters.state_evaluator specification). The latter
must charge every simulated transition/copy to search accounting. These implementations are available for native search. Example rollout fragment:

```json
{
  "policy": "greedy",
  "evaluator": {
    "name": "successor_state",
    "parameters": {
      "state_evaluator": {"name": "game_heuristic", "parameters": {}}
    }
  }
}
```

PUCT is `Q + K*P(a)*sqrt(max(1,N))/(1+n)`. Try unvisited legal actions
first. Bias can modify either tree policy. Visit sampling uses weights
`visits^(1/T)`; unvisited actions have zero weight. Exact ties use seeded uniform
sampling. Reject all-zero legal probabilities; do not silently rescale values.
The design specifies budget precedence, checkpoints and evaluator accounting.

External neural bindings use separate state-value and policy registry names:

```json
{
  "state_evaluator": {"name": "network_value"},
  "puct_preferences": {
    "source": "probabilities",
    "evaluator": {"name": "network_policy"}
  }
}
```

This is a fragment of a PUCT configuration. Python's `MctsSearch(bindings=...)`
resolves these names to declared frozen evaluators; external core parameters are
empty. JSON loading never imports arbitrary classes. The Python binding selects
`value_observers=all_players`, `root_player` or `deciding_player` and validates the
model's supported contexts. The selected completion rule still applies to an
explicitly identified scalar. Bias/expansion/rollout heuristics remain native.
Use [neural.json](../../../examples/mcts/neural.json) for a complete configuration
and the [neural MCTS guide](../learning/neural-mcts.md) for binding,
requests, batching, ownership and errors. Native CLIs do not execute Python
callbacks. [Local browser and match/tournament integration](../learning/local-mcts.md)
uses the same search settings with explicit portable-model bindings.
