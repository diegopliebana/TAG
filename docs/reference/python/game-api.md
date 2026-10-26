# Use the Python connection

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](../learning/overview.md) · [Build and check](setup.md)

Python can create Tic-Tac-Toe, Connect 4, Can't Stop and Love Letter, inspect their formats, obtain NumPy observations,
apply individual actions and copy game states. Rust owns the game states and
performs rule execution, observation encoding and state copying. The small Python
classes give the returned information readable names.

## Create a game and discover its arrays

After [building the package](setup.md), import it and create a game:

```python
import tag_rust

game = tag_rust.create("connect4")
spec = game.spec()
print(spec.identity.width, spec.identity.height)  # 7 6
print(spec.actions.size)                         # 7
for payload in spec.payloads:
    print(payload.name, payload.tensor.shape, payload.tensor.dtype)
```

The observation-array specifications are `cells` with shape `(42,)` and `board` with shape
`(3, 6, 7)`, both float32. The descriptions come from Rust's `LearningGame`.
Python uses the same discovery and basic game operations for registered games.
Can't Stop supports 2-4 players with different named arrays; see its
[API reference](../games/cantstop.md#python-game-operations-and-learning-arrays).
Love Letter also supports 2–4 players, with observer-only hand/history arrays,
explicit unknown cards and 52 fixed action IDs; see its
[encoding reference](../games/loveletter.md#python-operations-and-learning-arrays).
Its [Python native/neural MCTS](../games/loveletter.md#python-native-and-neural-mcts)
uses one deciding-player observation and full player values. Its
[complete-game self-play](../games/loveletter.md#complete-game-self-play-and-training)
uses those same information restrictions. [Frozen-model evaluation](../learning/model-evaluation.md)
compares the model with native opponents.

For changed settings, supply a parameter dictionary:

```python
small = tag_rust.create("connect4", {"width": 4, "height": 3, "winning_length": 3})
ttt = tag_rust.create("tictactoe", {"grid_size": 5})
```

`small` has 12 cells and four column IDs. `ttt` has 25 cells and 25 cell IDs.
Unknown parameter names fail explicitly. The placement games require two players. Can't Stop supports 2-4 and accepts
nested rule parameters. Optional search/episode/controller capabilities are
implemented per game; basic registration does not imply those capabilities.
The optional `max_cells` and `max_bytes` arguments use the
[encoding limits](../learning/data-descriptions.md#limits-and-failures); defaults are 4,096
cells and 64 MiB of numeric/index data per operation.

## Observe, select and apply one move

On the initial Connect 4 board, Player 0 acts next:

```python
view = game.observe(observer=0, payloads=("cells", "board"))
board = view.payloads["board"]
assert board.shape == (3, 6, 7)
assert view.offered_action_ids.tolist() == [0, 1, 2, 3, 4, 5, 6]

game.apply(action_id=2, position=view.position)
reply = game.observe(observer=1, payloads=("cells",))
assert reply.payloads["cells"][17] == -1
```

`apply` asks Rust to decode ID 2 and apply the column-2 action. Gravity places
Player 0's piece at `(2, 5)`. Player 1's observation has -1 at vector index
`2 * 6 + 5 = 17`. The [array guide](../learning/observation-arrays.md) explains both layouts.

`view.position` identifies the game instance and the revision that was observed.
Every accepted move and reset changes the revision. Passing an older token or a
token from another game/copy raises `StalePositionError`. Obtain a fresh
observation before making another choice.

By default, `apply` uses the current acting seat. An optional `actor` argument
allows the caller to check a seat explicitly; a wrong seat is rejected by Rust.
For [OSLAPython controllers](external-osla.md), direct apply is accepted for human seats,
while AI seats use prepared evaluation and commit.

## Read the returned fields

`view.payloads` maps the requested names to NumPy arrays in request order.
Omitting `payloads` requests all arrays advertised by the game.
`payloads=()` requests metadata and choices only. Every observation also contains:

| Field | Python representation and meaning |
|---|---|
| `identity` | Game/configuration/version description, matching GameSpec.identity. |
| `position` | Token containing the game instance ID and current revision. |
| `observer` | Integer seat whose viewpoint defines own/opposing values. |
| `next_player` | Next acting seat, or None at game completion. |
| `progress` | int64 array of four counters: moves, completed turns, turns this round, completed rounds. |
| `results` | int8 array in absolute seat order: ongoing=0, win=1, draw=2, loss=3. |
| `action_mask` | Boolean array of length spec.actions.size, marking actions offered to this observer. |
| `offered_action_ids` | int64 array of offered IDs, in the game's action-list order. |
| `done` | True when the whole game is finished. |

An inactive player can observe the board but has an all-false mask and an empty
offered-ID array. The action IDs retain the meanings described in
[the action guide](../learning/action-ids.md).

`spec.heuristic` is separate support for the hand-written Python heuristic
comparison. Its fixed line indices are not included in `view.payloads`. The
observation arrays are the inputs intended for learning models; see
[the distinction](../learning/overview.md#model-inputs-and-support-for-the-python-heuristic).

## Copy, reset and release a game

```python
branch = game.clone()
branch_view = branch.observe(branch.next_player, payloads=())
branch.apply(3, position=branch_view.position)
assert game.observe(0, payloads=()).progress[0] == 1

game.reset()
assert game.observe(0, payloads=()).progress[0] == 0
game.close()
assert reply.payloads["cells"][17] == -1
```

`clone()` copies the current state and original settings into independent Rust
storage. The copy receives a new instance ID, so it needs its own position token.
Python's `copy.copy` and `copy.deepcopy` use the same independent-copy behavior.
Attached OSLAPython controllers are copied with their exact settings/RNG progress.
Finish/commit or cancel pending evaluations before cloning; a pending request or
proposal causes clone to raise EvaluationError.

`reset()` restores the original setup and invalidates old tokens. `close()`
releases the state; later game operations raise `ClosedGameError`. Closing twice
is allowed. A `with tag_rust.create(...) as game:` block closes automatically
when execution leaves that block.

Arrays already returned remain usable after moves, reset, close and deletion of
the game. They have their own data, and editing them leaves Rust state unchanged.
Rust creates independent output vectors and transfers their buffers to Python
array owners. The arrays are writable and contiguous. Some NumPy operations that
resize storage require a NumPy-owned copy, obtainable with `array.copy()`.
Heuristic line indices returned by `spec()` are read-only description arrays.
The adapter retains each fixed configuration's validated format description and
returns separate copies, so returned data cannot change later format queries.
[CPU measurements](../../benchmarks/osla-python/README.md) cover complete matches
and evaluation routes. See [neural inference](../learning/neural-inference.md) for CPU model/buffer
semantics. GPU inference/transfers are not implemented by this executor.

## Errors and execution

Bad names/settings, invalid observers, repeated payloads, illegal actions and
actions after completion raise `ValueError`. Integer range failures raise
`OverflowError`; exceeding configured encoding limits raises `MemoryError`.
The public interface rejects Boolean and noninteger indices/settings. Rejected
moves leave both the state and its position revision unchanged.

Rust creates/copies states, copies retained descriptions, encodes observations and applies
moves while detached from Python's interpreter lock. NumPy object creation happens
after reattaching. This allows other Python threads to run during that Rust work;
it does not create parallel game execution automatically. Concurrent conflicting
access to the same game can raise a borrowing error. Use separate copies for
independent simulations.

For a full match loop, read and run
[play.py](../../../adapters/python/examples/play.py). The
[OSLAPython guide](external-osla.md) explains external batches and controller
ownership. See [Gym episodes](gym-episodes.md), [neural inference](../learning/neural-inference.md)
and [neural MCTS](../learning/neural-mcts.md) for their separate execution contracts.
