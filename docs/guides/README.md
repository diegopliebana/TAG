# User manuals

Choose the task you want to perform. You do not need to read these in order.
Commands use PowerShell from the repository root.
[Understanding the commands](setup.md#understanding-the-commands) explains Cargo
flags, Python invocation and argument notation.

Manuals are grouped under `agents/`, `games/`, `learning/` and `running/`.
Only this index and [setup](setup.md) stay at the guide root.

| I want to… | Manual |
|---|---|
| Install/build the framework | [Setup](setup.md) |
| Play, review, save or load a game | [Browser](running/browser.md) |
| Run one matchup and choose its agents | [Matches](running/matches.md) |
| Compare an agent pool across games | [Tournaments](running/tournaments.md) |
| Choose an execution mode and reproduce runs | [Running overview](running/README.md) |
| Choose First Available, Random or OSLA | [Simple agents](agents/basic-agents.md) |
| Configure search budgets and heuristics | [MCTS settings](agents/mcts.md) |
| Supply external values or group independent games | [External evaluation](running/external-evaluation.md) |
| Collect learner episodes and trajectories | [Episodes and collection](learning/episodes-and-collection.md) |
| Train a table or neural model | [Training](learning/README.md) |
| Create, observe, copy/reset games and apply actions from Python | [Game operations from Python](running/game-operations-from-python.md) |
| Run Rust MCTS from a Python script | [Rust MCTS from Python](agents/rust-mcts-from-python.md) |
| Evaluate a frozen MCTS model against opponents | [Model evaluation](learning/evaluate-mcts-models.md) |
| Find saved models and load them into agents | [Saved models](agents/models.md) |
| Implement a game | [Create a game](games/new-game.md) |
| Implement an agent | [Create an agent](agents/new-agent.md) |
| Change a position evaluator | [Create an evaluator](agents/new-evaluator.md) |

## Game manuals

| Game | Manual and rules links | Current support |
|---|---|---|
| Tic-Tac-Toe (`tictactoe`) | [Run Tic-Tac-Toe](games/tictactoe.md) | Native, matches/tournaments, Python/learning and browser |
| Connect 4 (`connect4`) | [Run Connect 4](games/connect4.md) | Native, matches/tournaments, Python/learning and browser |
| Can't Stop (`cantstop`) | [Run Can't Stop](games/cantstop.md) | Native experiments, Python/learning, portable saves and browser |
| Love Letter (`loveletter`) | [Run Love Letter](games/loveletter.md) | Native experiments, private observations, Python self-play/evaluation, portable saves and browser |

For API contracts, algorithm explanations, implementation details and measurements,
use the separate [technical reference](../reference/README.md). Each manual links
to the relevant details at the end.

For a visual introduction, see the [architecture diagrams](../images/architecture/README.md).

The [complete workflow map](../reference/framework/workflows.md) connects entry
points to guides and exhaustive option references.
