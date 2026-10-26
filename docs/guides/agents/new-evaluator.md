# Add a position evaluator

An evaluator estimates a position's value for a specified player. Use it to
change how an existing OSLA or MCTS policy rates simulations without replacing
the game's rules or writing a new action-selection algorithm. Define its player
perspective, numeric units, terminal rewards and computational cost first.

## Evaluate candidate positions in Python

Start with the runnable [external evaluation example](../running/external-evaluation.md#supply-your-own-estimates).
It receives batches of permitted successor observations, returns one estimate per
row and restores exact terminal outcome values. Change its ongoing-position
formula, retaining row order and player perspective. Give the evaluator a stable
name/version and record parameters that affect its result.

Use the observation arrays and legal IDs advertised by the game. A hidden game's
actual full state, environment seed or another player's private history must not
enter an ordinary evaluator. Hypothetical sampled facts remain hypotheses.

## Implement a native evaluator

Implement `Evaluator<State>` for native one-step evaluation and `Checkpoint` if
its settings or mutable state must survive controller restoration. The
[placement evaluator](../../../src/games/tictactoe/evaluation.rs) is a small
stateless example. Check the perspective before reading player-specific data,
return finite estimates and let terminal outcomes override ongoing heuristics.

Native MCTS uses `StateEvaluator`/`ActionEvaluator` factories in an
`EvaluatorRegistry`. That interface carries explicit player vectors, search
context and budget accounting. Register the requested role/name and validate its
parameters; an OSLA `Evaluator<State>` implementation alone does not register a
new MCTS configuration name. See the [MCTS extension reference](../../reference/agents/mcts-api.md#read-or-extend-the-implementation).

## Check and document the contract

Compare both player perspectives and terminal outcomes on controlled positions.
For hidden information, vary inaccessible facts while retaining the same
permitted observation: the evaluator input must not change. Check batching row
order, incompatible identities, invalid replies, cancellation and checkpoint
continuation where applicable. Stateful evaluators must restore independent
memory/RNG progress; reconstructing a cache must preserve decisions.

Attach your evaluator to the existing policy and run a fixed-work comparison with
rotated seats. Report the formula and measurement conditions; faster evaluation
or a successful run does not establish stronger play. Generic listener/metric
registries and search-wide action-filter decorators are not implemented extension
mechanisms; select a concrete capability before relying on them.

## Reference

[Python evaluator contract and full example](../../reference/learning/adding-evaluator.md),
[native MCTS API](../../reference/agents/mcts-api.md),
[game learning registration](../../reference/learning/adding-support.md),
[agent implementation](new-agent.md).
