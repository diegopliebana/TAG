# Collect transitions and learn a Tic-Tac-Toe table

For operating instructions, use the [short manual](../../guides/learning/tabular-learning.md). This page contains
the detailed reference.

A learning program uses an observation to choose an action, then uses the returned
reward and next observation to improve future choices. This guide follows that
cycle with `collect_episodes` and the runnable [Q-learning example](../../../adapters/python/examples/tabular_q.py).
Use the [Python setup](../python/setup.md) and optional dependency setup in the
[episode guide](../python/gym-episodes.md) first.

## Collect complete episodes

`collect_episodes(batch, choose_action, seeds=...)` resets every member of an
existing `GameEnvBatch`. It asks the supplied function for each active learner's
legal action, steps the batch, and yields one `Transition` per successful member.
Each member stops when its game terminates or reaches its collection horizon.
The following example collects both learner seats for each current game:

```python
from contextlib import ExitStack
import numpy as np
from tag_rust.episodes.collection import collect_episodes
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch

def choose_action(member, observation, info):
    return int(np.flatnonzero(observation["action_mask"])[0])

for game in ("tictactoe", "connect4"):
    with ExitStack() as stack:
        environments = []
        for learner in (0, 1):
            env = GameEnv(game, learner=learner, opponent="random", payloads=("cells",))
            stack.callback(env.close)
            environments.append(env)
        batch = stack.enter_context(GameEnvBatch(environments))
        records = list(collect_episodes(batch, choose_action, seeds=[40, 41]))
        endings = [record for record in records if record.terminated or record.truncated]
        assert len(endings) == 2
        assert {record.member for record in endings} == {0, 1}
        print(game, len(records), [record.reward for record in endings])
```

The caller owns the batch and environments. Consume the iterator before using
them elsewhere. Python starts the reset when iteration begins. A subsequent call
with `seeds=None` continues each member's ordinary reset seed generator.

| Transition field | Meaning |
|---|---|
| member | Original batch row, retained throughout that episode. |
| observation / info | Learner decision before the selected action. info includes seat, encoding identity, episode index and effective seeds. |
| action | Permanent action ID chosen from that decision's legal mask. |
| reward | Reward for that learner action and intervening opponent play. |
| next_observation / next_info | Returned learner decision or final board, including counters, results and opponent RNG checkpoint. |
| terminated / truncated | Separate flags for rule completion and an external collection horizon. |

The collector copies arrays and metadata into each record and gives the policy
separate input copies. Records remain usable after later steps, reset and close.
The final observation stays in its ending transition. A new collection begins
with fresh initial observations. Storing the iterator in a list retains all its
records; streaming it through a learning loop allows earlier records to be released.
These Python copies add storage and processing work to the underlying batch API.

On a per-member step failure, the collector yields successful peer transitions
from that round, then raises `CollectionError`. Its `errors` and `statuses` retain
the batch diagnostics. The failed member requires reset. A reset failure raises
before action collection. A policy exception propagates before that round's step.
Abandoning an iterator leaves environments at their current positions; the next
collection explicitly resets them. The [bulk guide](../python/batched-episodes.md) defines
the underlying failure and ownership contract.

## Learn values for legal actions

The example stores a number for each observed default Tic-Tac-Toe board and action.
That number, written `Q(s,a)`, estimates the learner's future discounted reward
after action `a` at board `s`. The table starts with zero estimates. Its board key
is a tuple of nine learner-relative cells: own piece +1, opposing piece -1 and
empty cell 0. A row contains nine values indexed by permanent action ID.

Training alternates first and second player by default while keeping the Rust
opponent fixed. All members share one table. Its cells always use the acting
learner's viewpoint. At a learner decision in Tic-Tac-Toe, the first player has
equal own/opposing piece counts and the second player has one fewer own piece.
These board keys distinguish the two decision contexts. Each collected transition
keeps the same learner viewpoint before and after its action and opponent reply.
The example supports the four registered Rust opponents.
The collector also works with Connect 4; the teaching table uses the small default
Tic-Tac-Toe board so its entries are easy to inspect.

The learner explores by choosing a random legal action with probability epsilon.
Otherwise it chooses a legal action with the highest estimate, sampling uniformly
among equal best values. This is called an epsilon-greedy policy. Evaluation uses
epsilon=0. Legal masks apply to exploration, greedy choices and continuation values.

After a transition, `QTable.update` calculates a target `y` and changes one entry:

```text
Rule termination:       y = reward
Ongoing or truncated:  y = reward + gamma * max(Q(next_board, legal_next_action))
Update:                 Q(board, action) += alpha * (y - Q(board, action))
```

Alpha is the learning rate: 0.2 moves the estimate one fifth of the way toward the
target. Gamma is the discount factor, applied once per learner step, including the
opponent reply. At gamma=0.99, a reward one learner step farther away receives 0.99
times the weight. Each value and reward uses that episode's learner viewpoint.

For example, an old estimate of 0.2, reward 0.1, gamma=0.9 and best legal next
value 0.8 give target 0.82. With alpha=0.5, the new estimate is 0.51.
At a rule ending with reward -1, the same old estimate and alpha give -0.4.

Using an existing estimate to form the next target is called bootstrapping. A
truncated board still represents an ongoing game, so the update uses its returned
legal mask and estimated continuation. Both ending flags stop collection for that
member. Very short horizons may collect only zero rewards; their updates then
depend on existing continuation estimates. Gymnasium's [time-limit explanation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
documents this distinction.

Within one batch round, all actions are chosen before stepping. The learning loop
then processes returned records in member order. The next round uses the updated
table. Keep the member count and iteration order fixed when reproducing training.

## Seat schedule

Training and evaluation assign logical game i to learner seat i modulo 2, starting
with seat 0. In a batch, i is round_index*members+member_index. An even game count
gives equal games in both seats; an odd count gives seat 0 one extra game.
For example, five games use seats 0,1,0,1,0. The report includes combined totals
and a by_seat entry for each position with its own outcome counts and rates.

Each GameEnv retains a fixed learner seat throughout its lifetime. The runner
caches batches with the required seat assignments and selects the appropriate
batch for each round. Each batch is seeded once with seed+member_index when first
used, then continues its member reset generators. With an odd batch size, two
alternating seat layouts have independent generators initialized from those same
member seeds. The learner's action generator advances across the full run.

Pass `--learner 0` or `--learner 1` for an explicitly fixed-seat experiment.
Training stores the selected schedule in the task: learner=null means alternating,
while 0 or 1 means fixed. Standalone evaluation alternates by default for every
loaded model; its own `--learner` option requests a fixed evaluation seat.

## Train, save and evaluate the reloaded table

Run from the repository root after building the extension:

```powershell
New-Item -ItemType Directory -Force .scratch | Out-Null
.\.venv\Scripts\python.exe adapters/python/examples/tabular_q.py --output .scratch/tictactoe-q.json
.\.venv\Scripts\python.exe adapters/python/examples/tabular_q.py --learner 1 --opponent osla_python --rounds 8 --eval-rounds 2 --members 4 --output .scratch/tictactoe-q-seat1.json
```

The default run evaluates the zero-initialized table, trains for 128 rounds with
eight members (1,024 episodes), saves JSON, reloads it and evaluates that reloaded
table. Training has 512 games per seat; each evaluation has 64 per seat.
Each evaluation uses 16 rounds of eight episodes. The output records the
task, seed, member/round counts, learning settings, outcome counts, number of
stored boards and number of entries whose estimates changed. Preserve that output
alongside the command and software versions for an experiment record.

`--learner` selects a fixed seat 0 or 1; omitting it alternates seats.
`--opponent` accepts `first_available`, `random`,
`osla` or `osla_python`. `--alpha`, `--gamma`, `--epsilon`, `--seed` and
`--max-episode-steps` expose the teaching choices. Learner exploration and tie-breaking
use NumPy's explicit PCG64 generator. The seat schedule above defines member reset
seeding. Evaluation uses
seed+10000 and epsilon=0 for both the initial and reloaded policies.

The JSON file records format/version, task configuration, encoding identity,
alpha, gamma and sorted board/value rows. `QTable.load(path)` reads the saved task
and validates the row format. Supply `task=expected_task` to check an expected
configuration as well. Using the loaded table checks encoding identity and a
supported learner viewpoint. Training updates also check the recorded opponent,
any fixed training seat and matching before/after viewpoints.
Evaluation reads estimates and leaves the stored rows unchanged. Reload restores
the learned values; each `run` starts its
action-sampling generator from the supplied seed.

New tables use file format version 2 to record alternating or fixed training.
Version 1 tables retain their original fixed-seat training metadata when loaded.
Retrain with the default command to give a table experience in both positions.

## Evaluate a saved table against a selected Rust opponent

[evaluate_tabular_q.py](../../../adapters/python/examples/evaluate_tabular_q.py)
loads an existing model and plays exactly the requested number of episodes:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/evaluate_tabular_q.py --model .scratch/tictactoe-q.json --games 1000 --opponent osla --seed 90001
```

`--opponent` selects `first_available`, `random`, `osla` or `osla_python`.
Omitting it uses the saved training opponent. Evaluation may select a different
opponent while preserving the table's training metadata. The collection horizon
comes from the saved task. The command above plays 500 games in each seat, using
the cached seat-specific environments described above. Greedy action selection
uses seeded random tie-breaking. Append `--learner 0` or `--learner 1` to evaluate
all 1,000 games in the requested seat.

The JSON report includes separate training/evaluation tasks, seed, requested game
count, seat_schedule, wins, losses, draws, truncations, rates and mean reward.
by_seat["0"] describes first-player results and by_seat["1"] second-player results.
Combined rates divide by the requested game count; per-seat rates divide by that
seat's episode count. An unused seat has zero counts and null rates/mean reward.
Mean reward is (wins-losses) divided by the corresponding episode count.
A saved collection horizon can produce truncated episodes,
which have their own reported count. Evaluation keeps the table values and model
file unchanged. Repeat with several seeds to assess variation across runs.

The outcome counts describe this opponent, seed set and training budget. Further
opponents and repeated independent seeds are useful when studying playing strength.
[Evaluation examples](../../guides/learning/tabular-learning.md) report outcomes by
seat and combined totals.

The selected [browser learned-player extension](../browser/learned-controllers.md) can load this
Q-table JSON for a Q-learning seat. It preserves greedy legal action-value choice;
Rust owns seeded ties. Its guide explains model compatibility and browser
validation requirements.
