# Check a neural learning integration

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

Use this checklist after adding a game or changing how a model receives inputs.
[Training](neural-training.md) explains model updates and file use;
[framework capabilities](../framework/overview.md) lists supported workflows.

## Follow one observation through the model

Start with a legal player observation, then its named numeric arrays, public
turn context, legal-action mask and action IDs. Keep the observing player fixed
when scoring a simulated successor, including a successor where someone else
moves next. For example, after player 0 places a token, an OSLA candidate remains
player 0's value even though next_player is 1. The mask is empty for that observer
at an opponent-turn position; value evaluation still applies.

For the current two-player placement encodings, context has five entries: two
observer indicators, two next-player indicators and one indicator for whether
the observer moves next. Cells plus context give 14 input numbers for default
Tic-Tac-Toe and 47 for Connect 4. These dimensions describe the current games;
general contracts derive context from the declared player count and can describe
named configurations, multiple typed inputs and candidate actions.

Check actual values at asymmetric positions, both observers, intermediate and
terminal positions. Check that only legal IDs receive policy probability and
that terminal results map to declared rewards. Win, draw and loss map to +1,0,-1
in the current reference trainer; other games need an explicit reward design.
MCTS requires an explicit player-value completion policy: require a full vector,
fill other players with zero, or negate an identified scalar for the other players.
The latter two are configured search assumptions. Their suitability for a game
must be assessed; they do not establish the actual rewards of those players.
See the [MCTS configuration reference](../agents/mcts-configuration.md).
The observation encoder and evaluator inputs must preserve the game's
hidden-information restrictions.

## Add a game in the right layers

Follow [adding learning support](adding-support.md) for LearningGame, game
registration, native execution and encoding tests. General model descriptions
alone do not register a playable game or supply its rules. For teacher/training
support, review legal action mapping, rewards, ending versus truncation,
learner-step discount timing and optional intermediate observations. Add tests
for those meanings before enabling a training workflow for the game.

The reference training workflow currently supports default Tic-Tac-Toe and
Connect 4, fixed action IDs and one observer value. Python inference supports a
wider model boundary, including synthetic multiplayer/card/token fixtures.
Graph and image observation encoders, simultaneous decision semantics, hidden-state sampling,
candidate-action training and additional playable games require selected work.
The [inference guide](neural-inference.md#extension-boundaries) owns those limits.

## Validate models and checkpoints before playing

An evaluation JSON identifies the game/configuration, rules/encoding/feature
versions, named tensor layout, action mapping, value meaning, architecture and
weights digest. Matching array lengths alone are insufficient. Pass
expected_contract when loading a model for a specific Python task; the browser
validates a model against the selected game's full description before play.

A training .pt checkpoint additionally retains optimizer state, completed groups,
phase/counters, history, RNG and recorded CPU/software conditions. It resumes
only under those checked conditions. Use JSON in the browser's Neural OSLA model
picker. Q-learning needs its separate saved Q-table; a neural JSON does not supply
Q(s,a) estimates. Files → Load file loads a game state/session rather than an
individual model. See [browser learned players](../browser/learned-controllers.md).

Model JSON must remain opaque when passed through JavaScript. Parsing and
serializing it again can change floating metadata or signed zero and invalidate
an otherwise compatible artifact. The browser receives Rust-serialized model
strings after bundle import and retains them unchanged for future candidate
workers. Invalid model/file loads preserve the current session.

## Measure ownership and complete work

Rust encoders export independently owned CPU arrays. Neural packing stacks/copies
them. Inference copies named inputs into local NumPy allocations and wraps those
allocations in CPU tensors; the network concatenates its flattened inputs.
Predictions are copied to independent NumPy arrays. Neural OSLA converts observer
values into an owned float64 reply which Rust validates before selecting a move.
This workflow includes several copies and does not establish zero-copy operation.

Measure encoding, packing, copies and prepared-tensor forward execution separately
from checked inference and complete collection/training. Include snapshot creation
when measuring a fresh training group. These timings overlap; do not add them as
independent pipeline stages. Report batch sizes, repetitions, warmed timings,
spread, CPU/thread/software identity and outcome/decision counts. CPU-only runs
have no CPU/GPU transfer; they provide no GPU cost estimate. The [benchmark collection](../../benchmarks/README.md) records existing workloads
and measurement conditions.
