# Create an agent

An agent receives the rules, acting player, permitted observation and available
action list. It returns an **index into that list**. Use
[RandomAgent](../../../src/agents/random.rs) as a complete example with RNG/checkpoints.

## Implement action selection

Create `src/agents/<agent>.rs` and export it from
[agents/mod.rs](../../../src/agents/mod.rs). A minimal deterministic policy looks like:

```rust
use tag_rust::agents::{Agent, EmptyActionList};
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::core::random::AgentRng;

pub struct MyAgent { rng: AgentRng }
impl MyAgent {
    pub fn new(seed: u64) -> Self { Self { rng: AgentRng::new(seed) } }
}
impl<F: ForwardModel> Agent<F> for MyAgent {
    type Error = EmptyActionList;
    fn rng(&self) -> &AgentRng { &self.rng }
    fn rng_mut(&mut self) -> &mut AgentRng { &mut self.rng }
    fn choose(&mut self, _game: &F, _actor: PlayerId,
        _observation: &F::Observation, actions: &[F::Action])
        -> Result<usize, Self::Error> {
        if actions.is_empty() { Err(EmptyActionList) } else { Ok(0) }
    }
}
```

This example is written as external code importing `tag_rust`. Inside the engine
crate, use `crate::` in place of `tag_rust::`. Replace the final selection with
your policy. For example, `self.rng.index(actions.len()).ok_or(EmptyActionList)`
selects uniformly. For `[move_a, move_b]`, return `1` to choose `move_b`.

Every agent owns its RNG, even a policy that currently uses no random draws.
Store only permitted information. State evaluation and search must use the
correct player perspective; multiplayer values need an explicit value per player.

## Expose configuration and run it

1. Define and validate the settings that affect choices. Native MCTS already
   supports [JSON tree policies, budgets and evaluator roles](mcts.md); changing
   those settings does not require creating another agent class.
2. For native matches/tournaments, extend `AgentConfig` in
   [config.rs](../../../examples/support/matches/config.rs), construction/dispatch in
   [agents.rs](../../../examples/support/matches/agents.rs), and report description.
   Add a named agent entry to a [match](../running/matches.md) or [tournament](../running/tournaments.md).
3. To support interactive recording, implement `Checkpoint` with the complete
   settings, RNG progress and private memory. Follow Random's snapshot/restore.
4. For browser use, add protocol configuration, controller construction/settings,
   UI selection/labels and the compatible replay codec. Long search should support
   cooperative `choose_async` preparation, as browser MCTS does.

Test legal selection, both seats, independent RNGs, discarded proposals, restored
choices, reset and historical forks. A failed evaluation is an execution error,
not a game loss. Run the Rust checks in [Create a game](../games/new-game.md).

For a learned policy, you may only need to [train and attach a model](models.md)
to an existing controller. For Python evaluators or a new native neural backend,
use the reference below.

Details: [agent integration checklist](../../reference/agents/implementing-agents.md),
[external evaluators](../../reference/learning/adding-evaluator.md),
[Rust syntax from Java/C++](../../reference/framework/rust-for-java-cpp.md).

Every user-facing agent setup must offer an explicit seed and an automatic seed
from OS randomness, and expose the integer actually used. Resolve it once at
construction; snapshots retain RNG progress. Follow the [seed convention](../running/README.md#reproducibility).
