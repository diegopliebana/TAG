# Find and run tests

A test checks a specific behavior against an expected result. Unit tests can inspect
private implementation details beside the code. Integration tests use the library's
public API. A fixture is saved input/expected output, such as a move sequence recorded
from Java. An independent calculation is useful because merely calling the same
helper twice would repeat the same possible bug.

## Where tests live

- core/: shared helpers, evaluation arithmetic and independent simulation copies.
- core/learning.rs: encoded cell/grid values, action IDs/masks, array independence and comparisons with existing evaluators.
- core/agents/: separate random generators, restoration of the next random draws, and OSLA move selection.
- core/agents/mcts.rs: open-loop accounting, per-player decisions, all native policies, JSON overrides, deadlines, completion and restoration. Private formula tests live in src/search/mcts/selection.rs.
- core/agents/osla_python.rs: candidate batches, native OSLA comparison, checkpoint continuation and the prepared-evaluation session boundaries.
- core/runners/: execution, failures and shared runs across games.
- core/runners/osla.rs: seeded OSLA matches, mixed policies and failure recovery.
- core/sessions/: one-move interaction, human/mixed AI seats, position numbers used to reject outdated requests, recreating earlier moves, alternate continuations and failures; optional persistence tests validate JSON/state loading.
- games/tictactoe/: Tic-Tac-Toe behaviour, parameters, exhaustive checks and comparison.
- games/connect4/: Connect 4 initialization, moves, observations, replays and comparison.
- games/loveletter.rs: public native runner, OSLA/MCTS and exact game/agent continuation;
  rules/conservation tests live in src/games/loveletter/tests.rs, and sampling,
  history, independent-card-oracle and search privacy tests in simulation_tests.rs there.
  learning/tests.rs there checks numeric encoding, action IDs and history privacy;
  adapters/python/tests/test_loveletter.py checks the compiled shared Python API
  against complete native trajectories and paired actual hidden worlds.
  test_loveletter_mcts.py covers native/Python search parity, suspended batches,
  neural predictions, full-player contracts and paired-world request independence.
  test_loveletter_selfplay.py checks real decision-input collection, shared-win
  targets, starter rotation, training/export, both buffer modes, exact cycle resumes
  and replay validation of checkpoint inputs/outcomes.
- fixtures/: saved comparison data and provenance.
- support/: shared test helpers and independent evaluation arithmetic.

integration.rs registers the folders as one Cargo integration-test target.
Each folder's mod.rs registers its test modules; add new modules there.
Private-state unit tests stay beside their game implementations in src/games/.

Python adapter comparisons and request/lifecycle tests live in
[adapters/python/tests](../adapters/python/tests/test_evaluation.py). Run them with
the commands in [the Python setup guide](../docs/reference/python/setup.md).
The [owned-execution tests](../adapters/python/tests/test_execution.py) check
advance/resume, native and external controller parity, per-call move budgets,
accepted-move accounting after later failures and checkpoint execution choices.

Run all tests with cargo test --offline --workspace. Run a group with:

```powershell
cargo test --offline --test integration core::
cargo test --offline --test integration core::agents::
cargo test --offline --test integration core::runners::
cargo test --offline --test integration core::sessions::
cargo test --offline --test integration games::tictactoe::
cargo test --offline --test integration games::connect4::
```

Run the optional replay/state checks with:

```powershell
cargo test --offline --workspace --all-features
cargo test --offline --features replay --test integration core::sessions::persistence::
```

The [grouping tests](../adapters/python/tests/test_grouping.py) compare each move
and exact RNG against individual execution across both games/configurations and
three chunk sizes. They exercise mixed players/formats, whole-reply rejection,
stale cancellation, limits, ownership and per-game errors after accepted moves.
