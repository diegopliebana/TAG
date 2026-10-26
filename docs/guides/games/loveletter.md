# Love Letter

## Run

From the repository root, run a complete classic game with existing Random agents:

```powershell
cargo run --release --example loveletter -- 2 42 100
```

Arguments are player count (2–4), game seed and agent seed base. Omit either seed
or use `auto` for operating-system randomness. The example resolves game and agent
seeds independently, prints them, then gives seat i the agent seed base plus i
(wrapping u64 arithmetic). Supply both printed seed values to repeat the full run.

For this fixed example the output includes:

```text
Actions: 39; completed turns: 39; completed rounds: 7
Player 0: Win, 6 favor tokens
Player 1: Loss, 1 favor tokens
Winners: [0]
```

## Rules

Each turn draws a card, then plays one of the two cards in hand. Card effects can
inspect or exchange cards, protect a player or eliminate them from the round.
The last survivor wins; if the deck empties, the highest surviving hands share
the round. Collect enough favor tokens to win the game; shared game wins are allowed.

Use the **classic option on page 14** of the
[selected rulebook](https://cdn.svc.asmodee.net/production-zman/uploads/2026/04/LL_Rulebook_with_Bag.pdf).
It has 16 cards and ranks 1,2,3,4,5,7,8,9. Default token targets for 2/3/4 players
are 6/5/4. These rules take precedence over the older public Java implementation.

## Configure the game

Use this as the `game` object in a [match](../running/matches.md), with 2-4 entries
in its `players` list:

```json
{"name": "loveletter", "parameters": {"rules": {"winning_tokens": [6, 5, 4]}, "first_player": 0}}
```

`winning_tokens` supplies positive targets for 2, 3 and 4 players respectively.
`first_player` is a zero-based initial seat, default 0. Deck and hand size stay
fixed. Experiment defaults rotate agent seats; a fixed initial starter and
rotated agent seats are separate choices.

To compare Random, sampled OSLA and MCTS:

```powershell
cargo run --release --features mcts-json,replay --example match_play -- examples/matches/loveletter.json
```

The example uses automatic seeds and prints them. Set `seed` to an integer in the
JSON to reproduce the experiment. A tournament example is
`examples/tournaments/loveletter.json`; follow the shared [tournament guide](../running/tournaments.md)
for scheduling. Each shared winner counts as a win and receives an equal share of
one tournament outcome point. Game scores remain favor-token counts. Native
tournament finishing places are unavailable; `ranking: "score"` is rejected
there. Python frozen-model evaluation separately supports optional token-score
ranking. Research
reports include seeds that reproduce hidden deals; keep them separate from agents
and ordinary player displays.

## Play in the browser

Follow [browser setup and shared controls](../running/browser.md), then select
**Love Letter**. Select 2-4 players and optional token targets under **Change game parameters**.
Choose a card, then a target and Guard guess when offered. Ranks follow the
selected classic rules: King 7, Countess 8, Princess 9. TAG's original card icons
are used, with credits linked below the table.

Action information screens explain actions and show permitted card reveals while at least
one human remains active in that round. They are skipped in AI-only games or
once all humans are eliminated. **Settings > Always show action information screens** overrides
this rule; it defaults off. It is a page preference, not a saved-game field. **Continue** resumes play and switches to the next human's view;
**Pause** closes the result and pauses automatic play. Guard shows hit/miss;
a missed guess leaves the opponent's card hidden. Baron participants see both
compared cards, and Priest information is shown only to its viewer. Round wins
show the winner(s), favor tokens and any public final hands; a round notice
remains above the table afterward.

Each action information screen shows the played card. Guard guesses show the
guessed card with a red X for a miss or a green tick for a hit. Baron comparisons
mark the higher hand with a green tick and the eliminated hand with a red X;
equal ranks use an equals sign. Short labels remain alongside the images.

In privileged reveal mode, the removed card and draw pile use smaller card
images. The pile is numbered in draw order, with the top card first.

**Board display > Fit** reflows the card table to the available width, with
vertical scrolling for long histories. **Scrollable** enables zoom for the whole
table; use **Reset zoom to 100%** to restore its size.

**Settings > View** selects the current human, a fixed player or public spectator.
Current-human mode switches directly to the next human's own view. During AI
turns it retains the previous human's view; AI-only games default to public view.
**Reveal hidden information (privileged)** shows all hands, the removed card and
pile order. It changes display only and can be toggled during MCTS. New/reset/
loaded games start with reveal off. Turning it off clears private display data.

Choose **Neural MCTS**, load `results/loveletter-model.json` from
[self-play](../learning/mcts-selfplay.md#train-love-letter), and select it for a seat.
The model must match player count and token targets. The supplied `[32,32]` model
fits the browser's limits; larger training networks may exceed the 10 MiB model
limit. Inference stays local. Saves contain enough information to reproduce all
hidden cards and are labelled **privileged**; keep them separate from player views.

## Use from Python

After [Python setup](../setup.md), create a game with
`tag_rust.create("loveletter", {"seed": 42}, players=3)`. Use the
[shared game API](../running/game-operations-from-python.md) to apply actions,
copy or reset it. Request `("hands", "private_history")` for your own cards and
remembered information. Unknown opponents' cards have value -1; historical
revelations describe their event time. Use `seed=None` or omit it for OS
randomness. The game handle and `game.seed` are privileged experiment data;
acting policies receive only their permitted observations.

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/sampled_mcts.py --game loveletter --players 3 --searches 3 --seed 42 --native
.\.venv\Scripts\python.exe adapters/python/examples/sampled_mcts.py --game loveletter --players 3 --searches 3 --seed 42
```

These scripts call Rust MCTS from Python and print recommendations without
applying them. The second uses an untrained network unless `--model` supplies
compatible weights. Search uses one deciding-player observation, a full-seat value
vector and approximate sampled worlds; IS-MCTS is not implemented. It omits
`last_private_history` because hypothetical opponents' previous-round private
memories are not reconstructed; the basic game API retains the actual observer's
memories. The example selects supported arrays. See
[Rust MCTS from Python](../agents/rust-mcts-from-python.md) for applying results
and batching independent searches.

## Train and evaluate

Use [MCTS self-play](../learning/mcts-selfplay.md#train-love-letter) with
`examples/selfplay/loveletter.json`. Frozen-model evaluation is:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_evaluate.py --model results/loveletter-model.json --config examples/evaluation/loveletter.json
```

The supplied configuration uses three players and nine games against Random and
sampled OSLA. Candidate seats and initial starters rotate independently, covering
every pair. Results go to `results/loveletter-evaluation.json`. Use N-squared games
for balanced pairs, or set `parameters.first_player` for a fixed starter. The
model's rules and player count must match. Every shared winner counts as a win.
Recorded seeds can reproduce private deals, so reports are privileged. Small
runs demonstrate execution; use larger independent experiments to assess strength.
The [evaluation manual](../learning/evaluate-mcts-models.md) explains opponent
configuration and report interpretation. Attached controllers and learner-step
Gym episodes are not registered; native local-neural match CLI registration is
also absent, separate from supported Python/browser neural MCTS.

## Reference

Native callers use `LoveLetterForwardModel` and `LoveLetterParameters` from
[games::loveletter](../../../src/games/loveletter/mod.rs). The
[game reference](../../reference/games/loveletter.md) defines actions, retained
private information, controlled deals, sampling, seeds, shared outcomes and save
validation.
