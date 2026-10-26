# Can't Stop

## Run

Run a complete game with Random agents from the repository root:

```powershell
cargo run --release --example cantstop -- 2
```

The arguments are player count (2, 3 or 4; default 2) and an optional game seed.
Omitting the seed selects one from operating-system randomness and prints it.
Append an integer, for example `2 42`, to reproduce that seeded run.
The program prints game/agent seeds, accepted action count, completed turns,
claimed-column scores and each player's result. Try `3` or `4` for multiplayer with automatic seeds.
The example has no action limit. Equal seeds and unchanged versions reproduce
its choices and dice; this is not a Java/Rust seed-equivalence guarantee.

## Rules

The game starts with a compulsory Roll. Allocate one of the offered pairings,
then Stop to bank progress or Roll again. A forced Bust discards temporary
progress. Only banking a top space claims a column. The first player to claim
three columns wins under the default rules.

- [Hasbro's official rulebook (PDF)](https://www.hasbro.com/common/instruct/Can%27t_Stop.pdf).
- [Yucata's readable rules](https://www.yucata.de/en/Rules/CantStop), including
  separately described variants. The current implementation uses the base rules
  and configurable columns/markers/winning target, not Jumping or Forced Move.

## Configure the game

Use this as the `game` object in a [match](../running/matches.md), with 2-4 agent
entries in its `players` list:

```json
{"name": "cantstop", "parameters": {"rules": {"markers": 3, "columns_to_win": 3}}}
```

`rules.column_heights` accepts eleven positive heights for columns 2-12; defaults
are 3,5,7,9,11,13,11,9,7,5,3. `markers` accepts 1-11. The winning target must be
between 1 and ceil(11/player_count): maxima 6, 4 and 3 for 2, 3 and 4 players.
Seat 0 starts; experiment defaults rotate agents through seats.

For mixed opponents, start with
[the supplied match](../../../examples/matches/cantstop.json), which uses Random,
sampled OSLA and MCTS. Follow the [match manual](../running/matches.md) to change
agents, seeds and action limits, or the [tournament manual](../running/tournaments.md)
with [the supplied tournament](../../../examples/tournaments/cantstop.json).
For finishing-place statistics, `ranking: "score"` ranks claimed-column counts.
The supplied tournament has one game entry, all six seat permutations repeated
twice, and 12 played games. Its native metadata retains actions/dice/seeds; the
native tournament exporter does not produce browser action recordings for this
game, separate from the browser's own session save/load support.


## Play in the browser

Follow [browser setup and shared controls](../running/browser.md), then select
**Can't Stop**. Expand **Change game parameters** to choose 2–4 players, track
heights, markers, winning target and the game dice seed. An integer repeats its
dice stream; `auto` uses Web Crypto. AI seeds are independent and also accept
`auto`. The resolved game seed appears above the board. Choose **Roll**, an
offered allocation, **Stop** or forced **Bust** below the tracks. White circles
show temporary progress; colored player shapes show banked positions. Sampled
OSLA averages 1–128 independently sampled successors per action. Neural MCTS
requires a matching player count/rules export from [self-play](../learning/mcts-selfplay.md).

## Use from Python

After [Python setup](../setup.md), create a game with
`tag_rust.create("cantstop", {"seed": 42}, players=3)`. Observations can request
`("dice", "context")`. The [shared API guide](../running/game-operations-from-python.md)
explains action application, copies and reset.

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/cantstop.py --players 3 --seed 42 --max-actions 1000
```

All seats use First Available, which banks when Stop is offered. `--players`
changes the seat count, not the policies. The script prints its resolved seed,
action count, truncation flag, scores and outcomes. This fixed run completes in
260 actions with scores [1,1,3] and outcomes [loss,loss,win]. Omit `--seed` or use
`auto` for OS randomness. The action limit defaults to disabled; truncation does
not assign final outcomes. Use a native match for mixed agents.

To call Rust MCTS from Python, run:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/cantstop_mcts.py --players 3 --searches 3 --seed 42 --native
.\.venv\Scripts\python.exe adapters/python/examples/cantstop_mcts.py --players 3 --searches 3 --seed 42
```

The second command uses optional PyTorch and an untrained network unless
`--model` supplies weights. Both calculate independent recommendations without
applying them. `--iterations` defaults to 32. The three roots use seats 0,1,2;
`--players 4 --searches 4` covers four seats. Omit `--seed` or use `auto` for an
OS-random master seed. Printed output includes derived game/search seeds, action
IDs, visits, full player values and counters. See [Rust MCTS from Python](../agents/rust-mcts-from-python.md)
to apply a recommendation; the [reference](../../reference/games/cantstop.md#python-stochastic-mcts)
defines this demonstration's versioned seed recipe.

## Train and evaluate

Use [MCTS self-play](../learning/mcts-selfplay.md#train-cant-stop-with-2-4-players)
with `examples/selfplay/cantstop.json` for training/checkpoints, and
[frozen-model evaluation](../learning/evaluate-mcts-models.md) for opponent lineups.
The supplied three-player evaluation command is:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_evaluate.py --model examples/selfplay/results/cantstop-model.json --config examples/evaluation/cantstop.json
```

It runs six games against Random and sampled OSLA with four successor samples
per action, giving the candidate two games in each seat. Root noise and candidate
visit sampling are disabled. Results go to
`examples/evaluation/results/cantstop-evaluation.json`. Copy the configuration
and change `output.report_file` for another experiment. `ranking: "score"` uses claimed-column counts for finishing places.
Learner-step Gym episodes and local neural agents in the native match CLI are not
registered. Browser Neural MCTS and Python neural search/evaluation are supported.

## Reference

Native callers use `CantStopForwardModel`, `CantStopParameters`, `CantStopAction`,
`CantStopBaseline` and `CantStopProgress` from
[games::cantstop](../../../src/games/cantstop/mod.rs). The
[game reference](../../reference/games/cantstop.md) defines transitions, snapshots,
dice simulations, action IDs and observation arrays.
