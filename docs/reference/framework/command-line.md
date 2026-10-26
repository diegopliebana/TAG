# Native commands and example programs

Run Cargo commands from the repository root. The
[command explanation in setup](../../guides/setup.md#understanding-the-commands)
explains build profiles, features, packages, manifests, targets and argument syntax.
`cargo run --release --example NAME -- ARGS...` builds and runs an optimized
example; arguments after `--` belong to that program. Omit `--release` for a
development build. Add `--offline` only when dependencies are already cached and
you want to prevent Cargo network access. These programs use positional arguments,
rather than a shared `--help` option parser.

## Game and agent runners

| Example | Required features | Arguments in order | Defaults and output |
|---|---|---|---|
| `random` | None | `[player-0 seed] [player-1 seed]` | Independent automatic agent seeds; plays both default placement games and prints outcomes/move counts. |
| `osla` | None | `[player-0 seed] [player-1 seed]` | Independent automatic agent seeds; native one-step heuristics, both default placement games. |
| `osla_python` | None | `[player-0 seed] [player-1 seed]` | Independent automatic seeds; native demonstration of prepared external value batches and commit. No Python process is required by this Rust example. |
| `cantstop` | None | `[players] [game seed]` | Two players, automatic game seed; derives each Random agent seed as game seed+seat with wrapping arithmetic. Prints all seeds, progress, scores and outcomes. |
| `loveletter` | None | `[players] [game seed] [agent seed base]` | Two players, independent automatic game/base seeds; each seat uses base+seat. Prints progress, token scores and all winners. |
| `mcts` | `mcts-json` | `config.json [game ID] [game count] [opponent seed]` | Config required; default tictactoe, two games, automatic Random opponent seed. Only tictactoe/connect4 in this standalone example; generic matches use registration instead. |
| `match_play` | `mcts-json,replay` | `match.json` | Exactly one configuration path. Defaults, agents, seats, outputs and errors are in the [match reference](../experiments/matches.md). |
| `tournament` | `mcts-json,replay` | `tournament.json` | Exactly one configuration path. See [scheduling/ranking](../experiments/tournaments.md). |

Seeds accept unsigned 64-bit decimal integers or `auto`/`null`; omitted seeds
resolve OS entropy. Player counts for game examples are 2-4. MCTS game count is
positive; its configuration owns the search seed. Agent and game seed streams
remain separate even when an example derives their initial values from one
recorded seed. [Game manuals](../../guides/README.md#game-manuals) explain actual
rules, defaults and game-specific parameters. [Simple agents](../../guides/agents/basic-agents.md)
introduces the decision methods.

```powershell
cargo run --release --example random -- 7 8
cargo run --release --features mcts-json,replay --example match_play -- examples/matches/basic-agents.json
```

Add `inference` for a native `neural_mcts` match/tournament. Only games registered
for local neural inference accept it; browser/Python support does not imply
native CLI registration. File paths inside experiment JSON resolve relative to
that JSON, while the command's input path resolves from the current directory.
Output collision policies differ: native experiment result/log writers reject
existing destinations; training runners explicitly replace their output files.

## Inspect models, encodings and saved games

| Example | Features | Arguments / effects |
|---|---|---|
| `learning` | None | No arguments. Prints portable placement observation shapes, action counts and heuristic-window counts. |
| `model_inference` | `inference` | First argument is a portable model JSON path. Validates the model and prints kind/content identity; it does not train or run a game. |
| `replay` | `replay` | No arguments validates an in-memory branch/state/replay demonstration. Exactly two arguments additionally write replay and state JSON to those paths, replacing existing files. Fixed fixture seeds, not an experiment seed CLI. |
| `save_files` | `replay` | No configurable arguments. Regenerates the tracked examples under `examples/saves/`; run only when intentionally updating fixtures. Fixed recorded seeds are deliberate. |

For example:

```powershell
cargo run --release --example learning
cargo run --release --features inference --example model_inference -- web/tests/fixtures/mcts/tictactoe-vector.json
cargo run --release --features replay --example replay
```

These are format/inspection demonstrations, not a general native replay file
viewer. The browser supplies interactive import/navigation, and the Rust API
supplies [validated import/export operations](../persistence/recordings.md).

## Browser-adapter and measurement helpers

`cargo run --release -p tag-browser --example native_workload` exports
fixed native cases used by actual-browser comparisons. `mcts_workload` in the same
package takes one or more compatible self-play model JSON paths and measures
native adapter prediction/search preparation; it omits browser rendering and
worker transport. See the [recorded method](../../benchmarks/mcts-adapter/README.md).

The root `history_profile` example has no options. `osla_measure` and
`java_comparison_measure` each accept a first repetition count, default 500,
range 1-10,000. They use fixed fixture seeds and controlled workloads, not general
match options. The latter times Rust against the recorded corpus; running it
alone does not generate a Java comparison. [Benchmarks](../../benchmarks/README.md)
owns reproduction, raw data and limits. Use release builds and compare matching
work before interpreting measurements.

Python-adapter Rust examples listed in its separate Cargo manifest are comparison
or fixture helpers. Build them with `--manifest-path adapters/python/Cargo.toml`
and their declared feature set; they are not additional user game runners.

## Configuration, JSON and failures

Use strict JSON with quoted keys/strings and no comments/trailing commas. JSON
null is distinct from the string "null" except where a seed parser explicitly
accepts the spelling. Check [MCTS settings](../agents/mcts-configuration.md),
[match fields](../experiments/matches.md) and [tournament fields](../experiments/tournaments.md)
for nested defaults and validation.

PowerShell 5 can add a UTF-8 BOM when writing files. A portable way to write JSON
without one is:

```powershell
$text = $config | ConvertTo-Json -Depth 32
[System.IO.File]::WriteAllText("my-config.json", $text, [System.Text.UTF8Encoding]::new($false))
```

Python self-play configuration loading also accepts a UTF-8 BOM. Do not infer that
all native/model readers do. An invalid configuration, unavailable capability,
unknown evaluator/model or illegal external request produces a nonzero process
exit and an error, not a game-defined loss. Runtime experiment failures can retain
completed-game reports; read their failure/truncation fields before computing rates.

The [research measurement tools](../../../tools/benchmarks/research/README.md)
explain `research_measure` (eight positional settings), `research_features`
(game/player/seed process probe), and the browser package's `research_adapter`
(portable model paths). These are measurement helpers with separate timing and
memory protocols, not ordinary game runners.
