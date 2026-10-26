# Play and review games in the browser

The browser displays a Rust game running in a Web Worker. Use it for human/AI
play, agent comparisons and reviewing saved continuations. Game-specific inputs,
parameters and private-information controls are in the [game manuals](../README.md#game-manuals).

[Browser execution diagram](../../reference/browser/README.md#a-browser-move) shows how the page and worker handle a move.

## Build and open

After [setup](../setup.md), run from the repository root:

```powershell
cargo build --release -p tag-browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tag_browser.wasm
.\serve-browser.ps1
```

Open `http://127.0.0.1:8080`. Rebuild/regenerate bindings after Rust changes;
refresh with **Ctrl+F5** after JavaScript/CSS changes.
The launcher accepts `-Port 8081` (valid range 1-65,535; default 8080). On POSIX,
use `.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory web`.

## Start a game

1. Open **New game**, select a game and optionally expand **Change game parameters**.
2. Choose each seat's controller. Available choices depend on the game; see
   [simple agents](../agents/basic-agents.md) and [MCTS settings](../agents/mcts.md).
3. For MCTS, edit its JSON or load a configuration file. The seat seed overrides
   the search JSON seed. For a learned controller, load its evaluation `.json`
   and select the matching model in the picker.
4. Press **Start game**. Human/AI games respond automatically; AI-only games start
   paused. Use **AI step**, **Play live** or **Pause**, and adjust the move delay.

[Model compatibility](../agents/models.md) explains which training exports each
controller accepts. Inference runs locally in Rust/Wasm. Seeds come from Web
Crypto unless entered explicitly; follow the shared [reproducibility instructions](README.md#reproducibility).

## Review and continue

Previous/Next and recording playback visit recorded positions without advancing
the live game or its agents. Historical positions are read-only. **Fork** creates
a continuation; retain controllers or replace seats. **Reset** returns to that
session's initial position. Reviewing history never grants access to information
that the selected observer could not know at that position.

## Save and load

**Save session** retains branches and resumable controllers. **Save branch** keeps
one continuation. **Save state** keeps a position and asks for fresh controllers
when loaded. Learned-agent saves include their required models. Pending searches
are not saved. **Load file** accepts supported state/session saves or tournament
action recordings; imported games start paused. The latter replay recorded moves
with human controllers rather than restoring the original tournament agents.

Hidden-information saves can contain privileged state and seeds. Follow the
selected game's manual before sharing them. Public/player views and privileged
inspection are separate display modes where supported.

## Display controls

Settings offers theme, sound, reduced motion, **Fit** and **Scrollable** board
display. Scrollable display enables zoom. On narrow screens, setup fields stack
vertically. Each game's manual explains any additional views, card/board controls
and action information screens.

## Check interactive behaviour

After rebuilding Wasm, `node --test web/tests/*.test.js` checks frontend logic
and actual Wasm/protocol behaviour. `node tools/browser/test-browser.mjs` also
launches installed Chrome/Edge for real UI checks. Set `TAG_BROWSER_EXE` if the
browser is outside its usual Windows installation path. See [tool setup](../setup.md).

For manual checking:

1. Start a game with supported human/AI controllers; try keyboard and pointer input.
2. Use AI step, automatic play and pause. Confirm that reset/replacement discards
   obsolete prepared choices.
3. Change Fit/Scrollable, zoom, sound and reduced motion; check narrow layouts.
4. Review an earlier position, return to the tip and create a fork. Review must
   not advance the live agents.
5. Save/load a state, branch and session. Check model-backed continuation where
   supported; a state-only import asks for new controllers.
6. Try malformed input and verify that the current game survives rejection.
7. Follow the game's manual for private/public/reveal views and action information
   screens. Confirm that changing display access does not alter agent decisions.

A Node/mock pass does not establish graphical interaction. If browser launch is
unavailable, record that limitation and run the graphical/manual checks in an
environment that can launch the installed browser normally.

## Reference

[Build/launch troubleshooting](../../../web/README.md), [protocol and limits](../../reference/browser/protocol.md),
[file formats](../../reference/persistence/formats.md), [local model bindings](../../reference/learning/local-mcts.md).
