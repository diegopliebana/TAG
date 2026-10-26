# Browser build and validation

The browser supports Tic-Tac-Toe, Connect 4 and 2-4-player Can't Stop and Love
Letter. For setup, play, models, zoom, action information screens and saves, use
the [browser manual](../docs/guides/running/browser.md). This page owns build,
launch and developer validation commands.

## Prerequisites

Follow [setup](../docs/guides/setup.md) for Rust, the Wasm target, the matching
wasm-bindgen CLI, Python for the static server and optional Node/Chrome checks.
The binding generator must match the pinned wasm-bindgen package version.
No frontend npm dependencies or browser driver are required.

## Build

Run from the repository root:

```powershell
.\tools\browser\build.ps1
```

Or run its commands directly:

```powershell
cargo build --offline --release -p tag-browser --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tag_browser.wasm
```

Generated `web/pkg` files are ignored by Git. Build once for a fresh checkout and
again after Rust/dependency changes. Native example builds do not update Wasm.
For HTML/CSS/JavaScript changes, refresh with Ctrl+F5. Documentation changes need
no browser rebuild. Save a session before refreshing if you want to retain it.

## Serve

```powershell
.\serve-browser.ps1
```

Leave the server running and open the printed address, normally
http://127.0.0.1:8080. Use `-Port 8081` if that port is busy. The launcher resolves
`web` relative to its own path. A direct equivalent from the repository root is:

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory web
```

If the engine cannot load, check that `web/pkg` exists and the server terminal is
still running. A local server serves files only; game rules and local model
inference execute in the browser worker, with no Python inference service.

## Validate

Build Wasm first, then run:

```powershell
node --test web/tests/*.test.js
node tools/browser/test-browser.mjs
```

The Node suite includes actual generated Wasm/client tests as well as frontend
logic checks. The second command launches installed Chrome/Edge headlessly with
an isolated workspace profile. It generates native comparison workloads, serves
test files over loopback and exercises actual UI clicks and module-worker play.
Set `TAG_BROWSER_EXE` if the standard Windows browser locations do not match your
installation. Profiles/workloads and screenshots stay under ignored `.scratch`.
A browser launch failure must be reported separately from a failed rule or
protocol check; do not change browser sandbox settings to bypass one.

Open `tests/loveletter.html` for Love Letter's real-app graphical checks. Also
check sound, reduced motion, fit/zoom, keyboard controls, hidden-information views,
action information screens, file pickers and exact continuation manually when
those behaviours change. The standalone presentation preview is
http://127.0.0.1:8080/presentation.html; it uses the same Rust engine and asset
manifest. [Asset documentation](assets/README.md) records provenance and licences.

## Source map

| Files | Responsibility |
|---|---|
| `app.js`, `setup.js`, `seat-fields.js` | Setup, validated parameters, OS-random/entered seeds, controller choices and feedback |
| `board.js`, `cantstop-board.js`, `loveletter-board.js` | Display/input mappings from Rust-produced permitted views |
| `ai-playback.js`, `recording-playback.js` | Live AI intent versus playback of existing records |
| `review-panel.js`, `files.js` | Branch/cursor controls, file pickers and staged worker replacement |
| `layout.js`, `styles.css`, `presentation/` | Fit/zoom, layout, artwork, animation and sound |
| `client.js`, `worker.js` | Request identity and worker-owned Wasm engine |
| `adapters/browser` | Shared sessions, permitted views, controller construction and file validation |

See [integration](../docs/reference/browser/integration.md),
[protocol](../docs/reference/browser/protocol.md),
[learned controllers](../docs/reference/browser/learned-controllers.md) and
[local MCTS](../docs/reference/learning/local-mcts.md) for technical contracts.
