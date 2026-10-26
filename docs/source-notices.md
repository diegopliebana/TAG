# Source and asset notices

The framework uses public Java TabletopGames as a behavioural reference and
implements rules with Rust types and shared components. Preserve the applicable
public-framework MIT notice below for reference-derived work. This notice records
origins. The project's original code and documentation use the [MIT licence](../LICENSE).
Third-party assets retain their separately recorded terms.

## Placement games and shared mechanics

Tic-Tac-Toe and Connect 4 semantics were adapted from public TAG sources under
`src/main/java/`:

- `games/tictactoe/TicTacToeGameState.java`, `TicTacToeGameParameters.java`,
  `TicTacToeConstants.java` and `TicTacToeForwardModel.java`.
- `games/connect4/Connect4GameParameters.java`, `Connect4GameState.java`,
  `Connect4ForwardModel.java` and `Connect4Constants.java`.
- `games/GameType.java`, `core/AbstractGameState.java`, `core/CoreConstants.java`,
  `core/AbstractForwardModel.java`, `core/StandardForwardModel.java` and
  `core/forwardModels/SequentialActionForwardModel.java` for relevant shared
  initialization, move/counter, turn/round and copy semantics.

The recorded baseline for placement comparison is public commit
`d65ad79f04992e238663c37c812dc2c5f6ce7ff2`, supplied by the repository owner.
The comparison fixtures retain that provenance in
[tests/fixtures](../tests/fixtures). Export/benchmark tools call public APIs in a
disposable build; they do not copy rule implementations or modify the reference.
[Java/Rust measurements](benchmarks/java-rust/README.md) describe the validated
workloads and conditions. Controlled configurable boards and the Connect 4
target-one behaviour are documented in the game references.

Shared grid/validation/counter/outcome helpers are Rust implementations, rather
than a translation of Java class structure. Owned observations and exact controller
snapshots have explicit independent-copy semantics. Current game modules reside
under `src/games/tictactoe/` and `src/games/connect4/`.

## Can't Stop

`src/games/cantstop/` was independently implemented after reading public
`games/cantstop/CantStopParameters.java`, `CantStopGameState.java`,
`CantStopForwardModel.java`, `actions/AllocateDice.java`, `actions/RollDice.java`,
`actions/Pass.java` and the shared forward models. No Java source was copied
verbatim. Published rules govern the first roll, default column heights and
removal of opponents' claimed-column markers. Final banking changes are applied
before victory evaluation. The [game reference](reference/games/cantstop.md)
records rules sources and differences. No board artwork or rulebook text was
copied for the rules implementation.

## Love Letter

`src/games/loveletter/` was independently implemented after reading public
`games/loveletter/` parameters, forward model, game state, CardType, card actions
and heuristic as comparison sources. No Java source was copied verbatim. The
Z-Man rulebook's classic option governs ranks, token targets, shared round/game
wins and random tied starters. [The game reference](reference/games/loveletter.md)
records exact sources, rules and information contracts.

Observation-derived constraint sampling, the native heuristic, numeric encoding,
experiment registration and Python integration are original implementations using
shared framework APIs. Public `LoveLetterHeuristic.java` was inspected for
comparison; its tunable feature implementation was not copied. Self-play,
checkpoint validation and neural search reuse the existing Rust/Python framework.
The sampled-search examples adapt the project's Can't Stop examples and preserve
their seed recipe. No rulebook text was copied into game assets.

## Agents and search

Random action selection follows public `players/simple/RandomPlayer.java`.
Its Java copy operation draws a new seed and advances the source RNG. Rust exact
checkpoints instead retain independent saved progress; equal seeds do not imply
Java/Rust sequence equality.

Native MCTS follows the algorithm structure of public
`players/basicMCTS/BasicMCTSPlayer`, `BasicMCTSParams` and `BasicTreeNode`.
Public `json/players/mcts.json` illustrates configuration, but its larger-agent
class loading and parameter set were not copied. Open-loop traversal, per-player
values, strict accounting and independent evaluator roles are original Rust
implementations. [MCTS formulas](reference/agents/mcts-configuration.md) explain
UCB1, PUCT and progressive bias. `tag_blend` follows the documented weighted-blend
formula, which differs from the public `SingleTreeNode.getBiasValue` additive
implementation. No website prose or implementation code was copied.

## Portable learned-model identities and randomness

The optional `src/inference/hash.rs` implements SHA-256 from the operations and
constants defined by NIST [FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf),
sections 4.1.2, 4.2.2, 5 and 6.2. It is original project code. Standard
empty/abc/million-a vectors and independently generated Python model digests test
content identities; this is not a NIST validation claim.

Windows automatic seed setup uses the platform signature/flags documented by
[Microsoft BCryptGenRandom](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom).
The small FFI boundary is original code; its safety comment states buffer, length
and handle requirements. Unix setup uses `/dev/urandom`; cross-platform runtime
validation must be reported separately.

## Browser assets

Asset origins, licence texts and hashes are recorded under
[web/assets](../web/assets/README.md). The public distribution uses original MIT
Love Letter symbols. The development checkout may instead contain separately
credited TAG icons; those third-party images are excluded from the public export.
Rules implementations and numeric observation code do not contain
copied rulebook prose or card artwork.

## MIT notice from the public checkout

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
