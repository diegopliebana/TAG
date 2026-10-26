# Tabletop Games in Rust

A framework for game simulation and research. Implement rules in Rust, develop
agents, run experiments, train models in Python/PyTorch, and play in the browser.
The implemented games are Tic-Tac-Toe, Connect 4, Can't Stop and Love Letter.

**[Start here: user manuals](docs/guides/README.md)**

| Task | Manual |
|---|---|
| Install/build | [Setup](docs/guides/setup.md) |
| Play, save/load or review | [Browser](docs/guides/running/browser.md) |
| Select games and agents | [Matches](docs/guides/running/matches.md), [tournaments](docs/guides/running/tournaments.md) |
| Configure search | [MCTS](docs/guides/agents/mcts.md) |
| Train models | [Training](docs/guides/learning/README.md) |
| Find/load model files | [Saved models](docs/guides/agents/models.md) |
| Create and control games from Python | [Game operations](docs/guides/running/game-operations-from-python.md) |
| Extend the framework | [Create a game](docs/guides/games/new-game.md), [create an agent](docs/guides/agents/new-agent.md) |

[Technical reference](docs/reference/README.md) covers APIs, algorithms,
implementation details and limits. The [framework overview](docs/reference/framework/overview.md)
shows supported capabilities and execution boundaries.

The project ports the public Java TabletopGames framework while preserving game
behaviour and using Rust's design features. See [source notices](docs/source-notices.md).

Code and documentation are [MIT licensed](LICENSE); third-party assets retain
their recorded licences. See [contributing](CONTRIBUTING.md) for checks and changes.

[Public source inventory](docs/benchmarks/public-candidate-size.md) records this export.
