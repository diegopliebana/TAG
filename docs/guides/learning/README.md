# Train a model

Complete [Python setup](../setup.md) once, then choose a workflow. Each manual
shows the command, output files and how to continue or evaluate training.

| What to train | Manual | Where the result runs |
|---|---|---|
| A small Q-table | [Q-learning](tabular-learning.md) | Python evaluation and browser Q-learning; default Tic-Tac-Toe |
| A network using OSLA examples and actor-critic learning | [Neural training](neural-training.md) | Python policy/Neural OSLA evaluation and browser Neural OSLA |
| A network using MCTS self-play | [MCTS self-play](mcts-selfplay.md) | Browser Neural MCTS; native neural matches/tournaments for placement games |

For arrays, learner-step environments and teacher/policy trajectories, start
with [episodes and collection](episodes-and-collection.md).

For the current **Neural MCTS** browser controller, choose **MCTS self-play**.
That workflow also supports 2-4-player Can't Stop and Love Letter. Compatible
exports run through Python MCTS, [opponent evaluation](evaluate-mcts-models.md)
and browser Neural MCTS.
These workflows use different value conventions; their models are not freely
interchangeable between controllers.

After training, [evaluate a frozen MCTS model](evaluate-mcts-models.md) against
independent opponents without changing its weights.

[Saved models](../agents/models.md) is the common guide to output locations, `.json`
versus `.pt`, and loading the result into a match, tournament or browser seat.

Training success means that the workflow executed. Use separate evaluation
against named opponents to assess strength. The supplied examples alternate
player seats by default.

Details: [learning APIs, encodings, algorithms and measurements](../../reference/learning/README.md).

[Complete Python command options](../../reference/python/command-line.md) lists
all flags/defaults; each algorithm reference owns its value and checkpoint semantics.
