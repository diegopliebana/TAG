# Run Sushi Go! natively

From the repository root, play three Random-agent games with three players:

```powershell
cargo run --example sushigo -- --players 3 --games 3 --seed 42
```

The command prints the resolved seed, each game's scores and Pudding counts,
and each agent's outcomes by seat and in total. Outcome arrays are
`[wins, draws, losses]`. Agent identities rotate through seats across
games. Omit `--seed` to obtain and print a seed from operating-system randomness;
use the printed number to repeat the run.

For the command above, the output includes:

```text
seed=42 players=3 games=3
game=1 seed=42 scores=[42, 44, 37] pudding=[1, 3, 4]
agent=1 totals=[2, 0, 1] seat_outcomes=[[1, 0, 0], [1, 0, 0], [0, 0, 1]]
```

Sushi Go! is a **simultaneous-choice** game. Each player chooses from their own
current hand. Once every choice is collected, the rules reveal the cards,
apply Chopsticks and Wasabi effects, and pass hands left. The final card of
each hand passes left and is revealed automatically. A player may use
Chopsticks to choose two cards in order; Wasabi followed by Nigiri in that
pair scores as a combination at round end. Players tied on both final score
and Pudding share victory.

## Use the Python adapter

After [Python setup](../setup.md), calculate native MCTS moves for three
independent searches. Each search completes every simulated joint decision
before evaluating its next state:

```powershell
py -3.14 adapters/python/examples/sushigo_mcts.py --native --players 3 --searches 3 --iterations 8 --seed 42
```

The JSON output gives one recommended action ID per seat, the resolved game
and search seeds, completed iterations and forward-model calls. An iteration
may traverse several joint states, so calls can exceed iterations.

With optional Gymnasium installed, run one learner episode or a batch with
learner seats 0, 1 and 2:

```powershell
py -3.14 adapters/python/examples/sushigo_episode.py --seed 42
py -3.14 adapters/python/examples/sushigo_episode.py --batch --seed 42
```

With optional PyTorch installed, run the small three-player self-play example.
It uses UCB1 joint search with a neural value model, rotates candidate seats
during frozen-model evaluation and saves a model, checkpoint and report:

```powershell
py -3.14 adapters/python/examples/selfplay_train.py --config examples/selfplay/sushigo.json
```

## Play in the browser

Build and open the [browser](../running/browser.md#build-and-open). Select
**Sushi Go!**, choose 2–5 seats and a deal seed or **auto**. Human, Random,
First Available, sampled OSLA, native MCTS and compatible local neural MCTS
controllers are available. For neural MCTS, load a Sushi Go! model trained for
the selected player count; use UCB1 and full absolute-seat values.

Each human selects one card, or an ordered two-card Chopsticks choice. A choice
stays hidden until every player has chosen. The next waiting human's hand and
choices appear automatically after submission. The interface offers the next
choice until every seat has submitted, then reveals the played cards.
Round points appear after the round ends. Previous round scores, Pudding,
played cards appear in the card view. New Sushi Go! games start in Scrollable
display at 80% zoom; change either setting whenever needed.
Played cards are visible to every player. Opponent hands show card backs by
default. **Reveal hidden information (privileged)** shows their card faces for
local inspection. The browser does not display remembered passing hands; Rust
still supplies that justified knowledge to agents through their observations.
This research interface does not provide hot-seat privacy.
The card images come from the public TAG checkout, with
[origin and licence details](../../../web/assets/sushigo/README.md).

Save a state, branch or session at a complete choice boundary. A state asks for
new controllers when loaded; a branch or session restores recorded controller
progress. These files contain hidden hands and seeds, so share them as
privileged data. [File details](../../reference/persistence/formats.md#sushi-go-joint-files)
explain the exact continuation proof.

The [technical game contract](../../reference/games/sushigo.md) defines every
numeric input, legal-action mask, value perspective and snapshot boundary.
