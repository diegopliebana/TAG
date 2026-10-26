# Seeds, RNG streams and reproducibility

Every user-facing run must offer both choices: an explicit integer seed, or an
automatic seed from operating-system randomness. Automatic runs print or save
the actual integer so you can repeat them. Never use the clock as a fallback.

The [running overview](../../guides/running/README.md#reproducibility) summarises
seed options for CLI, JSON, Python and browser workflows. Each game manual owns
its native example's arguments; experiment references own derived-stream recipes.
Environment, agent-choice and simulation streams have independent identities.

## Use the same policy from code

Native code resolves an automatic integer at setup, then passes it to the existing
typed constructors:

```rust
let seed = tag_rust::core::random::automatic_seed()?;
println!("Resolved seed: {seed}");
let agent = tag_rust::agents::RandomAgent::new(seed);
```

Python's `tag_rust.randomness.resolve_seed(None)` uses `secrets.randbits(64)`.
Training sessions retain the chosen master seed in their settings/provenance and
checkpoints; evaluation reports contain it. Direct episode collection retains it
in `EpisodeGroup.master_seed`. Keep that value when collecting subsequent groups
of the same experiment. Low-level seed-derivation functions take the already
resolved master integer rather than drawing new entropy.

Browser generation uses `crypto.getRandomValues`, retaining seeds as decimal
strings so all 64 bits survive JavaScript/JSON boundaries. Rust uses the existing
Windows BCrypt or Unix /dev/urandom source. An unavailable entropy source produces
an error; it does not silently switch to time-based seeding.

## Continue an experiment

Loading a checkpoint preserves the saved seed and RNG progress; it does not draw
a new automatic seed. Gym `reset(seed=None)` retains its existing meaning:
initialize from OS entropy on first use, then continue that environment's stream.
An explicit reset seed restarts its sequence.

A basic `Game.reset()` repeats its original resolved setup. `Game.clone()` retains
the current environment RNG position independently. Creating a new game with an
automatic seed starts a new stream. Game manuals/references define rule-specific
seed parameters and which data can be revealed to players.

For exact reproduction, retain configuration, engine/model versions and RNG
checkpoints where continuation is required. Use fixed work budgets for search;
elapsed-time budgets can complete different amounts of work on another run.
Tests, fixtures and published measurements may deliberately provide fixed seeds.
Those recorded values and versioned derived-seed recipes remain unchanged.
