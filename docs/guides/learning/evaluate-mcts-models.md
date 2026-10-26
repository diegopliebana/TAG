# Evaluate a frozen MCTS model

Use this after [training](mcts-selfplay.md) to measure a policy against fixed
opponents, separately from its training games. The model is frozen: evaluation
changes neither its weights nor its training buffer. Complete [setup](../setup.md),
including optional PyTorch.

## Run evaluation

From the repository root, this worked command uses a supplied model/configuration
pair (produce its model with the matching self-play configuration first):

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_evaluate.py --model examples/selfplay/results/cantstop-model.json --config examples/evaluation/cantstop.json
```

The command prints the resolved master seed, model identity and JSON report.
The configuration's `output.report_file` selects where full results are saved;
change it for a separate experiment. Input model/config files cannot also be the
report destination. [Game manuals](../README.md#game-manuals) describe their
supplied evaluation settings, starter rules, scores and information restrictions.

## Configure opponents and seats

For the multiplayer lineup format, declare one named opponent for each of the
other N-1 seats. A three-player example is:

```json
"opponents": [
  {"name": "random", "agent": {"type": "random"}},
  {"name": "osla-r4", "agent": {"type": "osla", "repetitions_per_action": 4}}
]
```

For MCTS, use `{"name":"mcts","agent":{"type":"mcts","config_file":"../mcts/basic.json"}}`.
An inline `config` works too. OSLA's positive `repetitions_per_action` defaults to
one. To fill all opponent seats with the same configuration, use `opponent`
instead of `opponents`, for example
`{"name":"random","agent":{"type":"random"}}`. Each copy has a distinct name
and RNG. Avoid combining these forms with the legacy `opponent_search` field.

`positions: "rotate"` moves the candidate through every seat. Opponents occupy
the remaining seats in their declared order; every actual lineup is recorded.
For a fixed seat, set `positions: "fixed"` and `candidate_seat` to a zero-based
seat number. Use multiples of player count for a balanced single run.

Top-level `seed` accepts an integer or null/omission for OS randomness. The model
declares its observation arrays/value outputs; the standalone configuration does
not duplicate them. Game parameters must match the saved model. Set rules under
`parameters.rules` and omit `parameters.seed`; recorded game seeds are derived
from the resolved master independently of all agent/search streams.

## Read outcomes and finishing positions

The report retains raw games, candidate seats, actual lineups, game/agent/search
seeds, scores and outcomes. `per_seat` and `combined` describe the **candidate**.
Completed games contribute to wins/draws/losses and win rate; combined results
weight every completed candidate appearance equally.

`ranking: "score"` opts into finishing positions based on public game scores,
as defined in the game manual. For scores
[3,1,1], positions are [1,2.5,2.5]. Average finishing position is lower-is-better
and uses those same completed games. Omit ranking to report outcomes alone.

`max_actions` limits accepted actions; null/omission disables it. A terminal move
at the limit completes normally. Other limited games are marked `truncated`,
with no final outcomes or finishing positions, and the schedule continues.
Truncations are counted separately and excluded from outcome/rank denominators.
Runtime failures stop the schedule, retain successful peers, and report the number
of unplayed games; the CLI saves the report and exits unsuccessfully.

Set `include_actions: true` for accepted actor/action traces and supported public
chance events. The optional
trace is limited to 100,000 entries per game. Traces and recorded seeds support
research reproduction; browser import is a separate game capability.

## Evaluate after training cycles

Put the same `evaluation` object into a self-play training configuration and set
`enabled: true`. The updated frozen model is evaluated after each completed
training cycle. Results are diagnostics; they do not promote models or alter
training weights, examples or RNG progress.

Placement-game evaluation uses the two-player `opponent_search` configuration;
multiplayer lineup fields are a separate format. For API use, call
`evaluate_model(settings, FrozenModel(model))` from `tag_rust.learning.selfplay`.
For placement-game evaluation, omit named `opponents`/`opponent`, `ranking`,
`max_actions` and `include_actions`; they are rejected there. Use the legacy
`opponent_search` profile and `limits.max_moves`. This is a configuration-format
boundary, even when the game uses two players in both cases.

## Reference

See the [evaluation reference](../../reference/learning/model-evaluation.md) for
the exact configuration, seed recipe, accounting and Rust adapter boundaries.

[Complete CLI flags](../../reference/python/command-line.md#selfplay_evaluatepy).
