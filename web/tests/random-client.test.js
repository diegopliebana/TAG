import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
import { BrowserClient } from "../client.js";
initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });
class EngineWorker {
  listeners = {}; engine = new BrowserEngine();
  addEventListener(type, listener) { this.listeners[type] = listener; }
  postMessage(data) { this.listeners.message({ data: JSON.parse(data?.type === "file" ? this.engine.handle_file(data.envelope, data.text) : this.engine.handle(data)) }); }
  terminate() { this.engine.free(); }
}
function makeClient() { const worker = new EngineWorker(), client = new BrowserClient(worker); worker.listeners.message({ data: { type: "ready" } }); return client; }
async function step(client, view) {
  const prepared = await client.request({ type: "prepare_ai" }, { positionVersion: view.position_version });
  return (await client.request({ type: "commit_ai", proposal: prepared.proposal.id }, { positionVersion: view.position_version })).view;
}
test("actual Wasm Random checkpoints survive discard, forks and all file scopes", async () => {
  for (const game of ["tictactoe", "connect4"]) {
    const client = makeClient(), loaded = makeClient();
    try {
      let view = (await client.request({ type: "create", game, controllers: [{ kind: "random", seed: "7" }, { kind: "random", seed: "18446744073709551615" }] })).view;
      const first = await client.request({ type: "prepare_ai" }, { positionVersion: view.position_version });
      await client.request({ type: "discard_ai", proposal: first.proposal.id }, { positionVersion: view.position_version });
      const cancelled = JSON.parse((await client.requestFile({ type: "export", scope: "session", cursor: 0, delay_ms: 0 })).text);
      assert.equal(cancelled.versions.agents, 2); assert.equal(cancelled.branches[0].controllers[0].rng.word_position, "0");
      const positions = [view]; for (let i = 0; i < 4; i++) { view = await step(client, view); positions.push(view); }
      view = (await client.request({ type: "fork", branch: 0, cursor: 2, replacements: [null, null] }, { positionVersion: view.position_version })).view;
      assert.deepEqual((await step(client, view)).board, positions[3].board);
      view = (await client.request({ type: "view" })).view;
      const session = await client.requestFile({ type: "export", scope: "session", cursor: 1, delay_ms: 23 });
      const branch = await client.requestFile({ type: "export", scope: "branch", cursor: 1, delay_ms: 23 });
      const state = await client.requestFile({ type: "export", scope: "state", cursor: 1, delay_ms: 0 });
      const expected = await step(client, view);
      for (const text of [session.text, branch.text]) {
        const imported = await loaded.requestFile({ type: "import", controllers: null }, text);
        assert.deepEqual(imported.view.board, positions[1].board);
        const tip = await loaded.request({ type: "history", branch: imported.live_view.branch, cursor: imported.live_view.tip });
        assert.deepEqual((await step(loaded, tip.view)).board, expected.board);
      }
      const imported = await loaded.requestFile({ type: "import", controllers: [{ kind: "random", seed: "8" }, { kind: "human" }] }, state.text);
      assert.equal(imported.view.cursor, 0); assert.equal(imported.view.controllers[0].kind, "random");
      assert.deepEqual(imported.view.board, positions[1].board);
    } finally { client.dispose(); loaded.dispose(); }
  }
});
