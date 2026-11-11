# Sushi Go! native contract

`SushiGoForwardModel` implements the original 108-card, 2–5-player game in
Rust. It follows the [Gamewright rules](https://gamewright.com/pdfs/Rules/SushiGoTM-RULES.pdf)
and the [designer's Chopsticks order clarification](https://boardgamegeek.com/thread/1014756/chopsticks-with-wasabi-and-nigiri).
Public TAG's `games/sushigo` was used as a behavioural comparison. The final
card differs intentionally: Gamewright passes it to the next player without
asking for another choice. The implementation is new Rust code and copies no
Java source or artwork.

The **full state** owns every hand, draw pile, public played cards and scores,
round progress, knowledge records and chance RNG. An **observation** copies
only the requested player's hand, public information and exact card multisets
of passing hands that player has seen. It excludes hidden hands, draw order
and real RNG progress. A simulation fork constructs a possible full state
from that observation using a separate simulation RNG. It preserves known
cards, card counts and hand sizes.

`Action::Pick(card)` and `Action::Chopsticks { first, second }` identify card
values, not positions in a private hand. Duplicate copies of one value create
one logical action; two identical copies can form a Chopsticks pair. The
stable `Action::id()` maps picks to 0–11 and ordered pairs to 12–155; legal
actions are a subset of these 156 IDs. The
ordered pair is public after reveal, including when the order has no scoring
effect. This is necessary for the current MCTS joint tree, which identifies
each edge by the complete action. Each tree edge applies all players' choices
before reaching a successor. A hidden hand index could not safely identify
that edge because turning over a card does not reveal its former index.

Use `core::joint::apply` or `runner::run_joint_game` for a complete decision.
Direct `next` returns `SimultaneousOnly`; no partial game transition occurs.
The [game manual](../../guides/games/sushigo.md) has runnable native and Python
commands.

## Python and learning contract

`tag_rust.create("sushigo", {"seed": 42}, players=3)` creates a game with a
resolved 64-bit integer seed. Omit `seed` or use `null` to obtain one from
operating-system randomness; `game.seed` reports the resolved number. The
Python game handle is privileged experiment state. Give an actor only
`game.observe(seat)` and its legal action IDs.

Each player may choose from one unchanged state. `game.next_player` identifies
the next seat awaiting input, while `game.observe(seat).next_player` identifies
that observation's own active participant. `game.apply(action_id,
position=view.position, actor=seat)` stores a choice privately until all
seats have chosen. The last submission resolves the complete joint action,
reveals the cards and passes hands. Submissions may arrive in any order.
Previously submitted seats have empty legal masks until reveal. A Python copy
requires a complete joint decision boundary. Observations taken before any
submission retain the same position token through the pending phase, so all
players can submit choices made from that original state. The token changes
after the complete joint reveal.

`LearningGame` uses `rules_version=1`, `encoding_version=1`, edition
`original-108`, and the configured player count as its model identity. All
payloads are independently owned `float32` arrays in absolute seat order;
`card_id` follows the `Card` order in the Rust enum. The seed is absent from
every array.

| Payload | Shape | Meaning |
|---|---|---|
| `hand` | `[12]` | Counts of the observer's card values. |
| `known_hands` | `[players,12]` | Exact remaining counts for previously held hand IDs; `-1` means that hand is unknown. |
| `played` | `[players,12]` | Public played-card counts by seat and card value. |
| `paired_nigiri` | `[players,3]` | Egg, Salmon and Squid Nigiri placed on Wasabi. |
| `players` | `[players,4]` | Score, Pudding count, current hand ID and observer marker. |
| `context` | `[5]` | Round, cards per hand, draw count, finished flag, completed rounds. |
| `discard` | `[12]` | Public discarded-card counts. |
| `last_reveal` | `[players,3]` | First card ID plus one, second card ID plus one or zero, and Chopsticks flag. |
| `round_scores` | `[3,players]` | Completed round points; unfinished rounds are zero. |
| `terminal_values` | `[players]` | Ongoing zero; final winners +1 and losers -1. |

The 156 fixed action outputs use picks 0–11 and ordered Chopsticks pairs
`12 + 12 * first_card_id + second_card_id`. Each observation returns a Boolean
legal mask and compact offered-ID list. This supports direct policy heads and
legal-action masking without exposing private hand positions. The progress
array counts submitted card choices, completed joint decisions, joint
decisions this round and completed rounds. The `results` array labels each
seat's ongoing, win or loss status; shared final winners each have `Win`.

Single and batched Gym episodes let a configured learner seat act once per
joint decision and run the other seats with native Random, first-available or
sampled OSLA. The native OSLA teacher is available. Native MCTS and Python
neural MCTS use the same joint traversal; neural values require a full
absolute-seat vector from the deciding player's observation and gamma 1.
The joint search tree uses UCB1 and can batch external PyTorch evaluation across
independent searches. Self-play records one policy target per participant
choice, then attaches the completed game's full outcome vector for training.
Frozen-model evaluation rotates candidate seats by default and supports
Random, sampled OSLA and native MCTS opponents. Complete-cycle checkpoints
retain optimizer, model, sample RNG and logical game progress; recorded Sushi
Go! action sequences can be replayed when a checkpoint is loaded.

One complete joint transition counts as one forward-model call. Deeper tree
traversal may make several such calls per iteration. A first-round hand has at
most ten cards, so the largest possible legal list has 100 actions when a
played Chopsticks card permits two picks. The policy retains 156 IDs because
it describes every card-value pair across all positions. Browser execution
and portable game persistence are not yet available.
