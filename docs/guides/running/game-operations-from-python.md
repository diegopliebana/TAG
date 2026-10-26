# Create and control games from Python

Use this API when a Python script should own the game loop, inspect permitted
observations, apply choices or create independent branches. Rules and state stay
in Rust. Complete [Python setup](../setup.md), then run scripts with
`.\.venv\Scripts\python.exe your-script.py` from the repository root.

## Create a game and apply moves

```python
import tag_rust

print(tag_rust.registered_games())
with tag_rust.create("connect4") as game:
    while game.next_player is not None:
        observation = game.observe(game.next_player)
        action_id = int(observation.offered_action_ids[0])
        game.apply(action_id, position=observation.position)
    print(game.observe(0, payloads=()).results.tolist())
```

This example uses Connect 4 and always takes the first available action. For
another registered game, change its name, parameters and `players` as described
in its [game manual](../README.md#game-manuals); the loop is unchanged.
Read `next_player` after each action: consecutive decisions may belong to the
same seat. A position token rejects choices based on an outdated state.

## Configure, copy and reset

```python
import tag_rust as tag

with tag.create("connect4", {"width": 4, "height": 4, "winning_length": 3}) as game:
    with game.clone() as branch:
        view = branch.observe(branch.next_player)
        branch.apply(int(view.offered_action_ids[0]), position=view.position)
    game.reset()
```

The parameters are game-specific; basic copy/reset operations are shared.
Copies have independent state and retain any current environment RNG position.
Basic reset repeats the original setup and invalidates old position tokens.
Returned observation arrays own their storage and survive game changes/closure.
See [reproducibility](README.md#reproducibility) for automatic/fixed seeds and
how basic-game reset differs from Gym reset.

## Read observations

`game.spec()` describes available arrays and fixed action IDs. `game.observe(seat)`
returns that game's advertised arrays. Supply a sequence of names to request a
subset; an empty tuple requests only metadata/actions. Names, shapes and values
belong to the [game references](../../reference/games/README.md).

Results use absolute seat order and label codes: 0=ongoing, 1=win, 2=draw, 3=loss.
These labels differ from learning rewards. For hidden-information games, expose
only the permitted observation to the policy. A full game handle, state export
or resolved environment seed is privileged experiment data.

## Let Rust run native decisions

```python
import tag_rust

with tag_rust.create("connect4") as game:
    for seat in range(2):
        game.set_controller(seat, tag_rust.OSLAPython.rust_heuristic(seed=7 + seat))
    result = game.advance(max_moves=4096)
    print(result.reason, game.observe(0, payloads=()).results.tolist())
```

This uses an attached Rust-scored OSLAPython controller. Attached-controller
registration is narrower than basic game access: check the [capability matrix](../../reference/framework/overview.md#supported-workflows)
and game reference. Other scoring functions can return requests for Python to
evaluate, then resume Rust with the values. Native JSON experiments use the
[match](matches.md) and [tournament](tournaments.md) runners.

## Search and learning

[Run Rust MCTS from Python](../agents/rust-mcts-from-python.md) to obtain and apply
search recommendations or batch independent trees. Use [learning guides](../learning/README.md)
for training and frozen-model evaluation. `GameEnv`/`GameEnvBatch` provide
learner-step environments where registered; basic game access does not imply
support for every controller or learning workflow.

## Reference

[Game API](../../reference/python/game-api.md), [Rust-owned execution](../../reference/python/rust-owned-execution.md),
[external OSLA](../../reference/python/external-osla.md), [Gym](../../reference/python/gym-episodes.md),
[bulk episodes](../../reference/python/batched-episodes.md), [neural search](../../reference/learning/neural-mcts.md).
