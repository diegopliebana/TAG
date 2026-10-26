# Technical reference

Use the [guides](../guides/README.md) for short commands and worked tasks.
This reference groups API contracts, algorithms, configuration and limits by
subject. Each topic has one main home.

| Subject | Start here | Detailed pages |
|---|---|---|
| Framework | [Overview and capabilities](framework/overview.md) | [State and information contracts](framework/contracts.md), [glossary](framework/glossary.md), [Rust from Java/C++](framework/rust-for-java-cpp.md), [implementing games](framework/implementing-games.md) |
| Games | [Game contracts](games/README.md) | [Tic-Tac-Toe](games/tictactoe.md), [Connect 4](games/connect4.md), [Can't Stop](games/cantstop.md), [Love Letter](games/loveletter.md) |
| Agents and search | [Agents](agents/README.md) | [Implementation](agents/implementing-agents.md), [MCTS API](agents/mcts-api.md), [MCTS configuration and formulas](agents/mcts-configuration.md) |
| Experiments | [Matches and tournaments](experiments/README.md) | [Match configuration](experiments/matches.md), [scheduling, rankings and reports](experiments/tournaments.md), [seeds and reproducibility](experiments/reproducibility.md) |
| Persistence | [Saved states and recordings](persistence/README.md) | [Formats, validation, branches and controller checkpoints](persistence/recordings.md) |
| Python | [Adapter APIs](python/README.md) | [Build](python/setup.md), [game operations](python/game-api.md), [Rust-owned execution](python/rust-owned-execution.md), [external OSLA](python/external-osla.md), [grouped evaluation](python/grouped-evaluation.md), [Gym episodes](python/gym-episodes.md), [bulk episodes](python/batched-episodes.md) |
| Learning | [Learning reference](learning/README.md) | Observation/action encoding, model contracts, training, self-play and frozen-model evaluation |
| Browser | [Browser integration](browser/README.md) | [Game integration](browser/integration.md), [protocol and limits](browser/protocol.md), [learned controllers](browser/learned-controllers.md) |

[Architecture diagrams and standalone exports](../images/architecture/README.md) give
short visual introductions to these sections.

[Source notices](../source-notices.md) record adaptation and asset provenance.
[Benchmarks](../benchmarks/README.md) keep measurements separate from API contracts.

[Python CLI options](python/command-line.md).
