# The Game contract

`TheGameForwardModel` implements the original standard 1–5-player card rules
from [Nürnberger-Spielkarten-Verlag's English rulebook](https://www.nsv.de/wp-content/uploads/2024/04/TheGame_GB.pdf).
This game has no public TAG Java implementation. The Rust implementation
adapts the rulebook's behaviour and does not copy Java code or publisher art.

## State and setup

`Parameters { seed, first_player, communication }` supplies a resolved `u64`
shuffle seed, an agreed starting seat and the communication mode. The native
example resolves an omitted or `auto` game seed and an independent agent seed
from operating-system randomness. Rule code receives the resolved game seed.
`init_with_deck` accepts all 98 distinct cards in draw order for controlled
fixtures. It rejects missing or duplicate cards and advances the normal
shuffle stream before installing the supplied order.

`State` owns every hand, the remaining draw order, pile tops, public placements,
turn progress, pending messages and chance RNG. An exact `Checkpoint` snapshot
copies all of these independently, including RNG progress. The public state
contains four pile tops in the order ascending one, ascending two, descending
one, descending two. Earlier placements were public when made, so the native
observation retains their ordered transcript as player memory even though
only pile tops are currently displayed. The public message transcript retains
each completed communication round; pending messages stay in the full state.

`Observation` contains one seat's hand plus public facts. Other hands, draw
order, pending messages and the real chance stream are omitted. `observe` and
available-action queries do not advance chance. `privileged_hand` and
`privileged_deck` inspect full state and must stay out of player inputs.

`SimulationModel::simulation_fork` takes only a permitted observation and a
caller-owned `SimulationRng`. It removes the observer's hand and every public
placement from the one-copy 2–99 inventory, then uniformly shuffles the unseen
cards into other hands and the draw pile. It rejects samples where the active
seat could not finish its known turn minimum. At most 4,096 attempts are made;
exhaustion returns `SamplingBudgetExceeded` without advancing the caller's RNG.
The future chance stream is independent of the real game. Unrevealed messages
from earlier speakers are represented as silence, a point-mass prior that does
not affect card legality. This is an approximate hidden-world sampler, not an
information-set search algorithm or a model of truthful communication.

## Actions and turn sequence

`Action::Speak(message)` is offered during the selected communication phase:
one message or silence from each seat in fixed seat order before every card
turn. All 13 options are offered to each speaker. Solo play and `Silent` mode
skip that phase. The completed round is revealed together to all seats before
card play. A message records an intention; the engine does not claim that its
content is true. It does not change card legality.

`Action::Play { card, pile }` places one owned card. The ascending piles accept
a higher value or exactly ten less than the top. The descending piles accept
a lower value or exactly ten more. The next action sees the updated top.
`Action::EndTurn` becomes available after two placements while the draw pile
has cards, or after one when it is empty. It draws up to the starting hand size
and advances clockwise, skipping players with empty hands once drawing stops.
An already chosen card placement that makes the required minimum impossible
ends the game as a loss; a different placement may have succeeded. Invalid
external actions are rejected before mutation.

Component selection offers each message and `EndTurn` as one choice. A card
choice leads to its currently legal piles. The browser receives these
rule-owned paths and applies only a completed `Play { card, pile }`; selecting a
card alone leaves the position and chance stream unchanged. `RandomAgent`
remains uniform over complete placements. `StructuredRandomAgent` chooses a
card uniformly, then one legal pile uniformly.

The standard hand sizes are eight for one player, seven for two, and six for
three to five. The draw pile contains the remainder of cards 2–99. The
rulebook's expert minimum of three and optional smaller hand are not part of
the current native implementation. Open talk is also deferred.

## Team result and scope

`TeamOutcome` is `Ongoing`, `Won` or `Lost`. Every seat receives the same
`PlayerResult`: `Win` when all 98 cards were placed, `Loss` when a required
turn minimum cannot be reached, and `Ongoing` during play. There is no
individual rank. `cards_remaining()` counts cards in hands and draw pile;
`player_score` returns the number placed for every seat. The score reports
progress and does not turn a loss with fewer than ten remaining into a win.
`game_result` includes one team `OutcomeScore` in `cards_placed` units, from
zero to 98. It is the number of cards on the four piles, including losing
games. The browser result shows this number alongside the shared outcome.
Native match and tournament records store it separately from win/loss;
The Game tournament rankings include mean cards placed and support
`ranking_metric: "average_score"` for a single game entry. Native match and
tournament setup supports Random, First Available and native MCTS agents for
The Game. Python can calculate native or neural MCTS moves; local neural MCTS
is not registered in the native match CLI.

## Numeric observations and search values

`LearningGame` encoding version 1 uses a fixed 406-output policy. IDs 0–12
identify the messages. ID `13 + 4 * (card - 2) + pile` identifies a card–pile
placement for card 2–99 and pile index 0–3. ID 405 ends a turn. The legal mask
marks only currently offered complete actions. Structured MCTS can select a
card and then a pile through component nodes; only the complete action changes
the game state.

Requested `float32` arrays are `hand` (98 card-presence entries), `piles` (four
tops), `players` (hand counts and observer marker by absolute seat), `context`
(seats, communication phase, draw and turn counts, outcome), `placements` (98
padded rows of actor, pile and card), `messages` (98 padded rounds of active
seat and message ID by absolute seat), and `terminal_values` (one identical
team value per seat). Actor, pile and message entries in history are one-based,
leaving zero for an unused slot; cards retain their printed values. Zero rows
mark unused history capacity. Opponent hands,
draw order, pending messages and real RNG progress have no array representation.

Gym learner steps return zero reward until termination, then +1 for a team win
or -1 for a loss. Truncation at a learner-step horizon is separate from game
termination. Native `TheGameHeuristic` returns +1/-1 at terminal states and
`0.9 * cards_placed / 98` while ongoing, for every seat. This is a progress
estimate, not a calibrated win probability or the game's cards-placed score.
Neural search expects a full value vector in absolute seat order with
`gamma=1`; every seat shares the same target.

The browser adapter uses the same native forward model. It exposes only the
observed seat's hand and public facts during ordinary play. A public spectator
view omits the hand and legal actions; privileged reveal adds every seat's hand
for inspection. The board reports `shared_win` for all seats or `team_loss`.
Human, seeded Random, native MCTS and local Neural MCTS controllers are supported.
Both MCTS modes use structured card-then-pile nodes by default in the browser.
Neural models must match seat count, communication mode and the 406-action
encoding, and provide full absolute-seat outcomes from deciding-player own-turn
observations. The native match CLI registers native MCTS; its neural MCTS agent
still requires a separate hidden-game integration. Existing Random files retain their controller format and
continuation behavior. Each accepted speech or
card action is an ordered recorded move, so history and forks retain the
communication timing.

Version 1 state files store public progress, all hands, remaining deck order,
pending messages and chance progress. The loader checks the card inventory,
placement sequence, phase, team result and setup seed before installing a
state. Branch and session files also retain controller RNG progress and verify
recorded moves. These formats contain hidden information.

Complete-game self-play supports one to five seats. The trainer uses one
deciding-player observation per action, the fixed 406-action policy and a full
absolute-seat value vector. Every terminal target is the same +1 team win or
-1 team loss, with `gamma=1`; the cards-placed score is an independent metric.
The stochastic seed recipe derives each game shuffle from the resolved master
seed, and self-play rotates the first card player unless explicitly fixed.
Communication still begins with the first speaker in fixed seat order, so that
speaker can differ from the starting card player. Checkpoint replay checks the
actual starting player, full action trace, permitted retained observations,
team result and score. Training files are privileged research records because
the seed and actions reproduce the deal; network inputs remain player views.

Frozen evaluation places a neural candidate in rotated or fixed seats with
native Random, sampled OSLA or MCTS teammates. It reports team win rate and
mean cards placed for each candidate seat and combined. All seats share the
score, so individual score ranking is unsupported. Incomplete games have no
terminal outcome or training target. See the
[self-play reference](../learning/mcts-selfplay.md) and
[evaluation reference](../learning/model-evaluation.md) for shared formats.

The [game manual](../../guides/games/thegame.md) gives runnable native,
browser, Python, training and evaluation instructions.
