# Create a game

Use [Tic-Tac-Toe](../../../src/games/tictactoe/mod.rs) as the smallest complete
example, or [Connect 4](../../../src/games/connect4/mod.rs) for configurable rules.
Rust code belongs in `src/games/<game>/`; adapters handle files, Python and display.

## Implement the rules

1. Define parameters, actions, errors and a state containing everything needed
   to continue the game. Separate player-visible observations from full state.
2. Implement `ForwardModel` from [core/mod.rs](../../../src/core/mod.rs). It is the
   rules interface, comparable to a Java interface with game-specific type parameters.
3. Export the new module from [games/mod.rs](../../../src/games/mod.rs).

| Operation | What your implementation supplies |
|---|---|
| `init_state` | Validate parameters/player count and create the initial state |
| `player_count`, `next_player` | Number of seats and who acts next, or completion |
| `observe` | Information the requested player may see |
| `available_actions`, `offered_actions` | Legal simulation choices and player-visible choices |
| `next` | Validate/apply an action and update turns/outcomes |
| `player_result`, `player_score` | Game outcome and numeric score |

Keep chance events, repeated turns and victory rules explicit. Do not assume
players always alternate. Deterministic public games implement
`PublicInformationModel` for exact reconstruction from an observation. Stochastic
MCTS and sampled OSLA use `SimulationModel`: preserve visible facts and supply
independent simulation randomness, as in [Can't Stop](cantstop.md).
Hidden-information sampling needs a separate concrete design. Adapter support
must still be implemented and tested for each game.

## Run it before adding integrations

Follow the small [native Random example](../../../examples/random.rs): replace its
forward model, parameters and player count with your game's types, then call the
same runner. Test initial state, action ordering, legal/illegal transitions,
independent state copies, observations and all terminal outcomes.

```powershell
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets --all-features -- -D warnings
cargo test --offline --workspace --all-features
```

## Make it available to users

| Where it should run | Registration/integration |
|---|---|
| Match and tournament JSON | Add `MatchGame` support in [agents.rs](../../../examples/support/matches/agents.rs) and a registration in [execution.rs](../../../examples/support/matches/execution.rs). Add an example JSON using the new name. |
| Native MCTS | Implement `SimulationModel`, native evaluator factories and terminal player values. Follow the existing game registrations in [evaluation.rs](../../../src/search/mcts/evaluation.rs). |
| Browser | Add game creation, permitted views/actions and presentation in [browser adapter](../../../adapters/browser/src/games.rs), its protocol/dispatch and frontend. |
| Python and learning | Implement `LearningGame` to encode observations and map actions; register creation in [Python engine](../../../adapters/python/src/engine.rs), then add the supported typed operations. |
| Save/load and browser replay | Implement the game's replay/state validation and register compatible codecs. |

These are separate capabilities: a new name in the registry does not implement
them automatically. Generic scripts should discover registered games rather than
gain another hard-coded list. Test the exact rule configuration on each supported
platform, then add the game to the [running manuals](../running/matches.md).

## Add the game's short manual

Every game needs `docs/guides/games/<game>.md`, linked from the
[manual index](../README.md). Use [Can't Stop](cantstop.md),
[Tic-Tac-Toe](tictactoe.md) or [Connect 4](connect4.md) as a model. Include a working
command, what its arguments/defaults mean, a short rules summary, configurable
parameters and the integrations currently supported. Link shared workflows rather
than repeating the browser, tournament or training manuals.

Link the publisher's official rules where available. Otherwise identify a reliable
rules reference as such. State framework variants and deliberate differences from
the published rules. Keep this manual current as integrations are added.

Details: [game implementation checklist](../../reference/framework/implementing-games.md),
[browser integration](../../reference/browser/integration.md),
[persistence](../../reference/persistence/recordings.md),
[learning integration](../../reference/learning/adding-support.md).

Game/experiment setup must support explicit and OS-random automatic seeds, expose
the resolved integer, and preserve checkpoint continuation. Follow the
[seed convention](../running/README.md#reproducibility); keep entropy resolution outside game rules.
