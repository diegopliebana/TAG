# TAG concepts

Terms used in the framework, with examples. [How the framework works](../../start-here.md)
connects these concepts. [Framework contracts](contracts.md) defines their guarantees
and [capabilities](overview.md) distinguishes supported and unsupported workflows.
Java names identify related reference APIs; Rust types use the interfaces described here.

## State and information

| Term | Meaning |
|---|---|
| Simulation state | The complete data the rules use to continue a game: board, active player, progress and any hidden cards. A player may not be allowed to see all of it. |
| Observation | Information one player is allowed to know. For example, a card-game observation can contain their hand and public discards, but not the opponent's hidden hand. |
| Retained knowledge | Information a player may remember from earlier play even when it is no longer displayed; observations must include such legitimate memory when needed. |
| Information set | All complete game states that could be true given what a player knows. Several possible opponent hands can belong to the same information set. |
| Determinisation | A complete possible state obtained by filling unknown facts with guesses consistent with the player's knowledge. For example, it assigns one possible hidden hand to an opponent so an agent can simulate play under that assumption. |
| Belief sampler | A function that samples possible complete states from a stated probability distribution. Its random-number source must be explicit. |
| Exact snapshot / snapshot | Independent copy preserving simulation data and the position of any RNG owned by the simulation; supplied future chance inputs remain external. |
| Search fork | Independent simulation branch that may deliberately derive new randomness; distinct from an exact snapshot. |
| Replay | Recreate positions by starting from the saved initial situation and applying recorded moves/random events. Loading one saved state instead skips those earlier moves. |

Java `AbstractGameState.copy(playerId)` combines copying, visibility and sampling.
Rust keeps their meanings separate; signatures and sampler algorithms remain open.

## Rules and decisions

An **action** describes a choice or operation. A **rule** determines effects and
automatic consequences. A **forward model** supplies initialization, available
choices and transitions. None requires a Java-style object hierarchy.

A **decision point** identifies the next player(s), chance, or terminal condition.
**Turn ownership** and the **decision maker** can differ during reactions.
At game termination there is no next decision maker; retained current-player
or last-actor metadata does not grant permission to act.
A **phase** changes applicable rules; a **sequence** tracks unfinished decisions.
Turn and round boundaries are game-specific; one action need not end a turn.
**Chance** is environment randomness, distinct from agent and belief randomness.

**Action space** is the declared set of representable choices, possibly under a
specified abstraction. **Legal actions** are choices permitted at a particular
decision point. An **agent search filter** selects which offered choices to explore;
it does not redefine game legality.

**Terminal for a player** means that player's participation has ended under the
rules for an explicitly stated scope: decision, phase, round or whole game.
**Terminal for the game** means the whole game has ended. Round elimination may
allow return next round; temporary inactivity is distinct from elimination.

## Game data

**Components** are entities or collections: cards, tokens, dice, counters, ordered
decks, grouped areas, spatial grids and graphs. This vocabulary does not require
a component base type or universal ID registry.

**Parameters** configure rules, including categorical choices.
**Static game data** defines immutable content such as card effects or topology;
loading and file formats belong to adapters.
**Identity** distinguishes entities; **equality** compares values for a declared
purpose. A **hash** is a calculated value used to find candidates quickly in a
lookup table. Several different objects can have the same hash, so equality checks
must compare the relevant data. Saved recordings use explicitly defined identifiers.

## Agents, values and execution

| Term | Meaning |
|---|---|
| Player | A participant identified by a seat number, such as Player 0. The program supplying moves for that seat is a controller. |
| Agent | A program that chooses moves for a player. A policy is its rule for choosing. A human can supply moves instead of an agent. |
| Runner / game loop | Code that repeatedly obtains the next player's observation, asks for a move and applies it through the rules. |
| Budget | A limit on search work or time. For example, 1,000 simulated moves and 10 milliseconds are different limits; specify which one is counted. |
| Result | A player's outcome (win/draw/loss) or whether the whole game is over. A player can finish participating before a multiplayer game ends. |
| Game score | A number awarded by the game rules, such as victory points. It is separate from an agent's estimate of how promising a position is. |
| Rank | Ordinal placement including ties and tiebreaks. |
| Heuristic | A function written by a programmer to estimate how good a position is for a specified player. Its formula defines what the numbers mean; for example, our line-game heuristics give higher estimates to boards with more promising open lines. |
| Privileged evaluator | Labelled research/training evaluator permitted to access full hidden state; separate from ordinary acting-time evaluation. |
| Reward | Feedback r from a transition, expressed for a specified player. The current learning environments give +1 for a win, 0 for a draw and -1 for a loss at termination. |
| Return | Accumulated future rewards, with a declared discount factor and time step. For example, G_t = r_t + gamma G_(t+1). |
| Value function | An estimate of expected return from a state or observation under a specified policy. State the player, reward units and discount convention. |
| Action value | An estimate Q_i(s,a) of the return for player i after action a in state s. MCTS estimates it from the values back-propagated through that action. |
| Policy | A rule for choosing actions, often a probability distribution pi(a given o) over the available actions in observation o. |
| Trajectory / episode | A trajectory is a sequence of observations, actions and rewards. An episode is one interaction with a defined start and ending condition; a training horizon can truncate it before the game ends. |
| Encoding / policy mapping | An encoding turns an observation into numbers. A policy mapping connects each action to a network output, such as one output per Connect 4 column. |
| Event / metric | Occurrence / measurement with declared timing and information access. |
| GUI / adapter | A GUI displays the game and accepts input. An adapter converts between engine types and another platform's calls/messages; it must not reimplement the rules. |

Observations include all rule-defined retained knowledge. Agents separately
maintain strategic memory and inferred beliefs. Beliefs describe what an agent
considers possible or likely and should be labelled with that meaning.
Privileged training targets do not authorise privileged inputs or oracle outputs
during ordinary play.

Exact replay is scoped to recorded engine, RNG and schema versions, initial
conditions, actions and chance events. Deterministic search also requires fixed
work budgets, evaluator behaviour and execution ordering. Cross-version and
cross-platform guarantees require separate selection and validation.

## Terms used in files, arrays and execution

| Term | Meaning and example |
|---|---|
| Descriptor / specification | Data describing how other data is organized and interpreted. For example, TensorSpec gives an array's shape and number type, while Float32Tensor stores the numbers for one array. |
| Tensor / shape / dtype | A tensor is an array with one or more dimensions. Shape lists dimension sizes; dtype is its number type. A three-channel 6 by 7 board has shape [3,6,7]. |
| Payload | The existing API name for a named observation array, such as `cells`. An unsupported name means the game has no encoder for that requested input. Use observation features or observation arrays when explaining learning algorithms. |
| Metadata | Information explaining other data or its context: observing player, dimensions, version or request ID. |
| Schema | The agreed fields, types and meanings in a file or message. Version numbers identify changes that affect how data is interpreted. |
| Stable action ID | An action's identifier keeps the same meaning within a game configuration, even when the action is temporarily illegal. A full column retains its ID. |
| Legal-action mask | A true/false entry per action ID showing what can be chosen now. It is separate from the compact list of available actions. |
| Window / line group | Consecutive board cells long enough to win, such as four positions in a diagonal. A heuristic can count pieces in each group. |
| Checkpoint | Saved settings, memory and RNG progress sufficient to restore an agent with the same future behavior. A seed alone restarts the sequence. |
| RNG / stream / offset | A random-number generator produces a sequence from a seed. A stream identifies a sequence within the generator; an offset says how far it has advanced. |
| Batch | Several inputs processed together. Rust may send all candidate boards from one decision to one Python evaluation call. |
| Prepare / commit / discard | Calculate a possible move separately; then either validate/apply it (commit) or throw it away (discard) without changing live progress. |
| Stale request | A request based on an old game or position. For example, a delayed click must not be applied after Reset creates another continuation. |
| Correlated reply | A reply includes the ID of the request it answers, so the caller can match it correctly. |
| Atomic failure | If an operation fails, it leaves the original live data unchanged instead of partially applying a result. Each API must state what it guarantees. |
| Invariant | A condition valid states must always satisfy, such as no floating pieces in Connect 4. |
| Serialization / deserialization | Convert values to a file/message representation, or read that representation back. Reading text does not itself prove the values form a legal state. |
| Native / headless / Wasm | Native means ordinary compiled Rust execution. Headless means no graphical interface. WebAssembly (Wasm) is the build used by the browser; these distinctions are different axes. |
| Regression check | Repeat an existing test to catch a new change breaking previously working behavior. |
| Parity / equivalence | Two implementations produce the same relevant results on specified inputs. State which boards, actions, counters, values or outcomes were compared. |
| Value perspective | The player whose future reward a number estimates. A favorable position for Player 0 need not be favorable for Player 1. |

Rust-specific words such as ownership, borrowing, trait, associated type and enum
are explained with code in [Reading Rust](rust-for-java-cpp.md). Algorithm
abbreviations OSLA, RL, NN and MCTS are introduced in [Start here](../../start-here.md).

## Monte Carlo tree search

One **MCTS iteration** consists of selection, expansion when possible, an optional
rollout, evaluation and back-propagation. A **rollout** is the simulated sequence
after expansion. One **forward-model call** applies one action in a simulation;
an iteration usually makes several such calls. Always state which of these units
a computational budget counts.

| Term | Meaning |
|---|---|
| Tree policy | The rule used to select among actions already explored at a tree node. UCB1 balances estimated value and exploration; PUCT also uses prior action probabilities. |
| Expansion policy | The rule for choosing an available action not yet represented by an outgoing edge. |
| Rollout policy | The rule for choosing simulated actions after expansion, such as uniform random selection or a heuristic-based softmax distribution. |
| State evaluation | Estimate the value of the position where simulation stops. For terminal states, use the game's outcomes or rewards. |
| Action heuristic | A function h(s,a) scoring an action a in state s for the deciding player. Its uses in expansion, progressive bias, PUCT and rollouts are configured separately. |
| Progressive bias | Add an action heuristic to the tree-selection score with a contribution that decreases as visits accumulate. |
| Back-propagation | Update visit counts and value sums along the selected tree path. In MCTS this propagates evaluated returns; neural-network training separately back-propagates gradients. |
| Open-loop tree | A tree whose nodes represent action histories. Each iteration simulates a state from the root by applying the selected actions. A history can produce different states in a stochastic game. |
| Closed-loop tree | A tree whose nodes retain game states. The current native MCTS uses open-loop search. |
| Player value vector | One estimated value per player, indexed by absolute player ID. Selection uses the component for the player deciding at that node. The evaluator's scalar-completion policy must be explicit. |
| Tree depth / rollout length | Tree depth counts actions from the root; rollout length counts actions after expansion. Reaching the configured tree-depth limit evaluates immediately. Rollout length zero evaluates the expanded leaf. |

The [MCTS guide](../agents/mcts-api.md) explains these steps in the implementation.
The search references define the algorithm terms in context.

## Related terminology sources

These definitions use established game-AI vocabulary, including
[TAG definitions](https://tabletopgames.ai/wiki/definitions),
[core responsibilities](https://tabletopgames.ai/wiki/games/creating/core_classes),
[information hiding](https://tabletopgames.ai/wiki/conventions/info_hiding), and
[scores and heuristics](https://tabletopgames.ai/wiki/games/creating/heuristics).
The per-game references define precise rules and supported variants.

Related scientific sources include [Design and Implementation of TAG](https://diego-perez.net/assets/pdf/papers/DesignImplementationTAG.pdf),
[MultiTree MCTS in Tabletop Games](https://diego-perez.net/assets/pdf/papers/MultitreeMCTSTabletop-CoG22.pdf),
[PyTAG: Tabletop Games for Multi-Agent Reinforcement Learning](https://diego-perez.net/assets/pdf/papers/PyTAG-ToG24.pdf),
and [Seeding for Success: Skill and Stochasticity in Tabletop Games](https://diego-perez.net/assets/pdf/papers/SeedingForSuccess-TOG25.pdf).
