# Use First Available, Random and OSLA

Start with these agents to check that a game runs and establish comparison
baselines before introducing MCTS or a learned model. An agent receives its
player's permitted observation and offered actions, then chooses one. All three
use the same Rust rule interface and need no trained weights or Python.

## Choose a decision method

| Agent | How it chooses | What it helps check |
|---|---|---|
| First Available (`first_available`) | Always chooses entry 0 from the offered list | Action ordering, rule execution and complete-game plumbing |
| Random (`random`) | Gives every offered entry equal probability | Repeated play without a heuristic; a simple opponent baseline |
| OSLA (`osla`) | Simulates one action ahead and compares the resulting position estimates | The effect of a game heuristic before deeper tree search |

First Available is not a strategy for choosing a tactically good action: its
behaviour follows the game's action ordering. Random is uniform over offered
actions, which need not give equal probabilities to broader strategic outcomes.

## Run First Available against Random

After [setup](../setup.md), run the supplied two-game match from the repository root:

```powershell
cargo run --release --features mcts-json,replay --example match_play -- examples/matches/basic-agents.json
```

The [configuration](../../../examples/matches/basic-agents.json) starts with:

```json
"players": [
  {"name": "first", "agent": {"type": "first_available"}},
  {"name": "random", "agent": {"type": "random"}}
]
```

`positions: "rotate"` gives each agent one game in each seat. The runner prints
resolved seeds and outcomes by seat and combined. The seed is fixed for this
example; set `seed` to null for automatic setup. See [matches](../running/matches.md)
for selecting another registered game and the required number of competitors.

For Random versus Random on both placement games, the direct native example is:

```powershell
cargo run --release --example random -- 7 8
```

Omit the arguments, or use `auto auto`, for independent OS-random agent seeds.
First Available retains the common RNG/checkpoint interface but consumes no random
draw when selecting entry 0. Random saves its own selection RNG progress.

## Add one-step lookahead

OSLA means **one-step lookahead**. It simulates each offered action and asks an
**evaluator** to estimate the resulting position for the deciding player. For
example, an open line with friendly pieces can receive a higher estimate in a
placement game. A heuristic estimate is separate from the game score or outcome.

Change an entry in the match to `{"type": "osla"}`, or run:

```powershell
cargo run --release --example osla -- 7 8
```

The direct example plays both placement games with native Rust heuristics. It
compares successor estimates and uses its agent RNG to choose among values within
0.001 of the best. Terminal candidates use exact game outcomes.

Games with chance/private information use **sampled OSLA** instead: for each
choice it averages estimates across independent simulated successors. Those
simulations come from the player's information and a separate simulation RNG.
The native match's `osla` registration selects the appropriate implementation;
`repetitions_per_action` controls sampling where supported (default 1). Sampled
OSLA selects a greatest mean and resolves exact ties with its choice RNG. Game
manuals explain their available heuristic and information limits.

OSLA considers the immediate successor, rather than searching a tree of future
choices. More samples reduce sampling variation but do not turn it into deeper
search. [MCTS](mcts.md) explores longer action sequences and shares statistics
between repeated simulations.

## Use the same agents in other workflows

The [browser](../running/browser.md) offers registered First Available, Random and
OSLA controllers. [Tournaments](../running/tournaments.md) use the same native
agent descriptions as matches. Check a game's manual for the supported player
count and sampled/deterministic variant.

In Rust, the concrete types are `FirstAvailableAgent`, `RandomAgent`, `OslaAgent`
and `SampledOslaAgent`. Their constructors accept explicit seeds. The runner asks
`Agent::choose` for an index into its current action list; the game applies that
action. To implement another policy, continue with [Create an agent](new-agent.md).
[OSLAPython](../../reference/python/external-osla.md) is the separate path for
batching externally evaluated successor values.

## Reference

[Agent interfaces and integration](../../reference/agents/implementing-agents.md),
[seed/RNG semantics](../running/README.md#reproducibility),
[First Available](../../../src/agents/first_available.rs), [Random](../../../src/agents/random.rs),
[OSLA](../../../src/agents/osla.rs), [sampled OSLA](../../../src/agents/sampled_osla.rs).
