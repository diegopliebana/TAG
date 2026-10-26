# Browser messages and limits

The [protocol types](../../../adapters/browser/src/protocol.rs) define requests,
replies, game views, seat configuration and file operations. The
[engine](../../../adapters/browser/src/engine.rs) validates envelopes and dispatches
them to the typed game runtime. [BrowserClient](../../../web/client.js) handles
worker readiness and request/reply identity.

## Ordinary commands

```json
{"request_id":"1","generation":"0","position_version":null,"command":{"type":"create","game":"tictactoe"}}
```

Request IDs, generations and position versions use decimal strings. Each is
checked as an unsigned 64-bit integer, avoiding JavaScript Number precision loss.
Unknown fields/commands fail validation. `create` accepts game parameters and
optional seat controllers; omitted controllers default to two human seats.
Omit `parameters` to select game defaults. A supplied object is decoded by that
game and can require complete fields, including resource limits; an empty or
partial object is not a universal default/merge operation. Native match JSON
parameter merging is a separate contract.

| Command | Purpose |
|---|---|
| `create` | Create a game from identifier, parameters and controllers; advance the generation and invalidate old work. |
| `view` | Project the current position for the requested observer. |
| `history` | Project a branch/cursor without changing the live position. |
| `select_branch` | Restore a recorded branch tip, including controller progress. |
| `fork` | Continue from a branch/cursor with optional replacement controllers; null replacements retain saved progress. |
| `human_move` | Validate actor/action against the expected live position and apply it. |
| `presentation` | Change observer, public-only mode or privileged reveal without advancing rules/RNG. |
| `reset` | Restore the session root and starting controller state in a new continuation. |
| `prepare_ai` | Calculate on an independent controller; return a proposal or pending work. |
| `poll_ai` | Continue/poll the identified pending calculation. |
| `commit_ai` | Validate and apply a prepared proposal, optionally choosing the response observer. |
| `discard_ai` | Drop prepared work without applying a move. |

Commands that depend on a position carry its `position_version`. Replies echo
request identity and distinguish the requested generation from the current one.
Reset, replacement, branch changes and changed controllers invalidate obsolete
work. The client must discard late responses belonging to an old user intention.
Cancellation is cooperative; a synchronous worker call can finish before the
worker processes its cancellation message.

Views include game/branch/observer identity, position/cursor/tip, permitted board
facts and actions, results/scores and controller labels. Full checkpoints remain
inside Rust. Historical views are read-only. For hidden-information games,
display settings redact private information, seeds and AI candidates according
to that game's contract. [Love Letter](../games/loveletter.md#browser-views-controllers-and-persistence)
defines its event-time views and privileged reveal mode.

## File channel

Large text uses `BrowserEngine::handle_file`, separate from ordinary commands.
`FileRequest` has request/generation identity and one tagged operation:

| Operation | Purpose |
|---|---|
| `load_model` | Validate and register portable model JSON. |
| `models` | List registered model summaries. |
| `predict_model` | Evaluate a registered model for a specified observer; an adapter diagnostic operation. |
| `export` | Save `session`, `branch` or `state`, with cursor and playback delay. |
| `import` | Validate a state/replay/model bundle; state imports choose fresh controllers. |

The UI loads files in a temporary worker and replaces the live worker only after
validation succeeds. Invalid input preserves the current game. Replay bundles
can carry immutable models required by their controller checkpoints; registry
replacement is staged before installation. See [formats](../persistence/formats.md)
and [learned controllers](learned-controllers.md).

## Limits

| Quantity | Browser bound |
|---|---|
| Ordinary command or file envelope | 32 KiB UTF-8 text |
| Separate file text | 10 MiB UTF-8 text |
| Board dimensions/storage | At most 128 per dimension and 4,096 cells |
| Stored accepted moves | 100,000 across stored history |
| Model registry | At most 16 models, also bounded by aggregate model bytes |

Replay parsing applies byte, depth, branch, storage and estimated-work checks.
The browser reserves one nesting level for its bundle and uses replay depth 63;
the native default is 64. Search and model inference apply their own operation,
shape and resource limits. A work estimate is not a wall-clock guarantee.
Consult the source-linked types and game/model references for those independent
limits; changing one does not waive the others.
