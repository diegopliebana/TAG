# The Game

## Run

From the repository root, run a two-player native match with Random agents:

```powershell
cargo run --example thegame -- 2 42 100
```

The arguments are player count (1–5), game seed, agent seed base and optional
first seat (zero-based). Omit both seeds for automatic values, or write `auto`
in either seed position to draw it from operating-system randomness. The
example prints the resolved seeds so the match can be repeated. This command
prints:

```text
Team: loss; score: 10/98 cards placed; cards remaining: 88; completed turns: 3
Seat 0: Loss
Seat 1: Loss
```

To compare native MCTS with Random over two complete games while rotating seats:

```powershell
cargo run --offline --release --features mcts-json,replay --example match_play -- examples/matches/thegame-mcts.json
```

To run a local Neural MCTS model in the same native match runner:

```powershell
cargo run --offline --release --features mcts-json,replay,inference --example match_play -- examples/matches/thegame-neural-mcts.json
```

The supplied model is a small compatibility fixture, not a trained strategy.
Replace its `model_file` with a compatible two-seat export for an experiment.
The model must match the communication mode and seat count, use The Game's
406-action map, and return a full absolute-seat team value vector. Search samples
hidden cards from the acting seat's permitted observation.

## Rules

The players cooperate to place cards 2–99 on four piles. Two start at 1 and
rise; two start at 100 and fall. A card exactly ten in the opposite direction
is also legal. Play at least two cards in order on each turn, then refill your
hand while the draw pile has cards. When it empties, the minimum becomes one
card. Everyone wins if all 98 cards are placed; otherwise everyone loses when
a player cannot place the required minimum. The number of unplaced cards is a
separate measure of the team's progress.

The [publisher's two-page rulebook](https://www.nsv.de/wp-content/uploads/2024/04/TheGame_GB.pdf)
governs the original game. This first native version uses its standard card
rules and a selected limited-choice communication mode. Before each card turn,
each seat chooses silence or one of three requests about a named pile: leave it,
make only a small jump, or indicate that they can use it. The messages become
public together before the active player places a card. Open talk and the
rulebook's expert variants are deferred.

## Configure the game

`Parameters` supplies a resolved integer `seed`, `first_player` and
`communication`. `LimitedChoices` is the default. `Silent` supports a controlled
run without communication. Solo play skips the message phase. The native
example accepts an optional first seat after the seeds:

```powershell
cargo run --example thegame -- 3 42 100 2
```

## Play in the browser

Build and open the [browser](../running/browser.md), then choose **The Game**.
Set 1–5 players, an optional shuffle seed and the first seat. Each seat can be
Human, Random, MCTS or Neural MCTS. The four piles show their current tops. Select an available
message before each card turn, then play at least two cards while the deck has
cards, or one after it empties. Select a card in your hand to see its legal
piles; cards without a legal pile are disabled. **End turn and draw** appears when the minimum
has been met. The page shows the active seat's hand, other hand sizes, the
deck count and the team's shared result.
MCTS defaults to structured card-then-pile search with 256 iterations; its JSON
settings can be edited per seat. It samples hidden cards from the acting
player's observation. Neural MCTS uses a compatible exported model with full
absolute-seat values, `gamma=1` terminal outcomes and declared support for
every deciding player's own-turn observation. Load the model JSON in seat setup
before selecting it; browser inference runs locally in Wasm.
The result also shows **cards placed out of 98**, so two losses can be compared
by how far the team progressed.

With two or more players, each seat sends one limited-choice message in seat
order. Messages appear together once all seats have spoken. The page advances
to the next human's view automatically. This research interface does not
provide private device handoff: a person at the screen can see the current
human's cards. **Reveal hidden information** explicitly lists every player's
hand, including the active player's. The **View** selector is for Love Letter
and does not appear while playing The Game.

**Save state**, **Save branch** and **Save whole session** include hidden cards
and deck order. State files ask for fresh controllers on load; branch and
session files preserve controller progress.

## Use from Python

After [Python setup](../setup.md), create a game and read the acting seat's
permitted observation:

```python
import tag_rust as tag
game = tag.create("thegame", {"seed": 42, "first_player": 0,
                              "communication": "limited_choices"}, players=2)
view = game.observe(game.next_player, ("hand", "piles", "context"))
game.apply(int(view.offered_action_ids[0]), position=view.position)
```

Omit `seed` or use `None` for an operating-system seed; `game.seed` reports the
resolved integer. `silent` is the other communication setting. The
[shared game API](../running/game-operations-from-python.md) describes reset,
copy and action validation. For a learner that acts whenever its seat comes up,
use `GameEnv("thegame", players=2, learner=0)` as described in the
[Gym episode guide](../learning/episodes-and-collection.md). Its default
observation includes the player's hand, public piles and history. A Gym step
advances other seats with native Random actions until the learner acts again or
the team finishes; terminal reward is +1 for a win and -1 for a loss.

## Train and evaluate

Run observation-based Rust MCTS from Python with a fixed work limit:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/sampled_mcts.py --game thegame --players 2 --searches 2 --iterations 32 --seed 42 --native
```

The example recommends moves without applying them. `--native` uses the
cards-placed progress heuristic; omit it to use an untrained neural model or
pass `--model` with compatible weights. Search samples unseen cards into
hypothetical hands and draw orders. It needs a full team-value vector in
absolute seat order, with the same value for every seat. This is a baseline
determinisation, not an information-set MCTS algorithm. See
[Rust MCTS from Python](../agents/rust-mcts-from-python.md) for search settings.
For complete-game training from MCTS visits and shared terminal outcomes, run:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_train.py --config examples/selfplay/thegame.json --max-cycles 1
```

The [training guide](../learning/mcts-selfplay.md) explains resume and output
files. The example trains two-seat limited-choice play with structured
card-then-pile search, then evaluates the frozen model with a Random teammate.
Set `seed` to `null` for an OS-random master seed; the actual integer is printed
and saved. The model has one value per absolute seat, but all seats receive the
same +1 win or -1 loss target. Cards placed out of 98 is reported separately.
These short runs establish operation, not playing strength.

To evaluate the exported model separately:

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/selfplay_evaluate.py --model results/thegame-model.json --config examples/evaluation/thegame.json
```

The evaluation rotates the candidate seat by default and runs independently
seeded native teammates. It reports team win rate and average cards placed per
candidate seat and combined. Match the model's player count and communication
setting. The [evaluation guide](../learning/evaluate-mcts-models.md) covers
fixed seats, native teammate types and report status.

## Reference

The [technical game contract](../../reference/games/thegame.md) defines the
native state, actions, observations, team results and current limits.
