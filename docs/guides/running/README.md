# Choose how to run games

Complete [setup](../setup.md), then choose the entry point that fits your task.
The same Rust rules support native experiments, Python scripts and the browser.

| Task | Manual |
|---|---|
| Interactive play, review, branches and saves | [Browser](browser.md) |
| Compare one configured lineup | [Matches](matches.md) |
| Compare an agent pool across configurations | [Tournaments](tournaments.md) |
| Supply external values while Rust runs, or group independent games | [External evaluation](external-evaluation.md) |
| Create/control games and inspect observations in a script | [Game operations from Python](game-operations-from-python.md) |

[Game manuals](../README.md#game-manuals) own rules, parameters, game-specific
controls and example configurations. [Agent guides](../agents/basic-agents.md)
explain decision methods. [Learning](../learning/README.md) covers training and
frozen-model evaluation.

## Reproducibility

Randomness affects all execution modes. Use an explicit integer seed to repeat a
run, or automatic seeding to generate a new experiment. Automatic setup obtains
entropy from the operating system and prints or records the resolved integer.
Keep it together with the configuration, versions, seat assignment and model.

| Entry point | New automatic seed | Repeat a run |
|---|---|---|
| Match/tournament/search/training JSON | `"seed": null`, or omission where optional | Set `"seed": 42` |
| Python API | `seed=None`, or omit an optional seed | Supply the recorded integer |
| Python example CLI | Omit `--seed`, or use `--seed auto` | `--seed 42` |
| Native example CLI | Omit seeds or use `auto`; argument positions are in its game/agent guide | Supply all recorded game/agent seeds |
| Browser | Keep the generated value or use **Regenerate** | Enter the recorded decimal seed |

A game can have separate streams for rule chance, agent choices and simulated
search. Reproducing a complete game may therefore require more than one seed.
Training/evaluation derive and record their per-game streams from a master seed.
Seeds and complete game handles are privileged research data for hidden-information
games; acting policies receive permitted observations instead.

Resume a checkpoint to continue saved RNG progress. Entering its original seed
starts the stream again. Basic game reset repeats the initial setup; Gym reset
has its own stream convention. Use fixed computational budgets when repeating
search: time limits can complete different amounts of work on each run.

The [reproducibility reference](../../reference/experiments/reproducibility.md)
defines integer representations, seed sources, reset/clone/resume behaviour and
versioned experiment requirements.

## Command options and errors

Start with [command syntax and build options](../setup.md#understanding-the-commands)
for the meaning of `--release`, optional `--offline`, features and the `--` separator.

[Native positional arguments](../../reference/framework/command-line.md) and
[Python CLI flags](../../reference/python/command-line.md) list defaults, choices
and required files. A CLI seed, a game's chance seed and an agent seed can have
different roles; use the relevant configuration reference.

| Symptom | Check |
|---|---|
| Unknown option/JSON field | Use the current entry point's option/schema reference; Python commands accept `--help` |
| Invalid or BOM-prefixed JSON | Use quoted keys and UTF-8 without BOM for portability; self-play readers also accept UTF-8 BOM |
| Model/game mismatch | Match player count, rules, observation/action descriptions and value contract; a training `.pt` is not an evaluation model |
| Output file already exists | Native experiment writers reject collisions; choose new result/log paths |
| Unsupported agent/environment | Registration is capability-specific; consult the workflow map rather than assuming every game supports every runner |
| Pending/stale request | Finish or cancel the current work and use the correct generation/position; do not apply an old reply to a new game |

See the [workflow map](../../reference/framework/workflows.md) for supported
combinations and separate game/agent/evaluator extension paths.
