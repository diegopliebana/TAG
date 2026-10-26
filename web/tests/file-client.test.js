import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
import { BrowserClient, MAX_FILE_BYTES } from "../client.js";
initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });
class EngineWorker {
  listeners = {}; engine = new BrowserEngine();
  addEventListener(type, listener) { this.listeners[type] = listener; }
  postMessage(data) { this.listeners.message({ data: JSON.parse(data?.type === "file" ? this.engine.handle_file(data.envelope, data.text) : this.engine.handle(data)) }); }
  terminate() { this.terminated = true; }
}
function makeClient() { const worker = new EngineWorker(); const client = new BrowserClient(worker); worker.listeners.message({ data: { type: "ready" } }); return client; }
test("actual Wasm file channel roundtrips native files and leaves failed replacements intact", async () => {
  for (const game of ["tictactoe", "connect4"]) {
    const client = makeClient();
    const native = readFileSync(new URL(`../../examples/saves/${game}.tag-replay.json`, import.meta.url), "utf8");
    const imported = await client.requestFile({ type: "import", controllers: null }, native + " ".repeat(40_000));
    assert.equal(imported.view.cursor, 2); assert.equal(imported.delay_ms, 250);
    assert.equal(imported.view.branches.length, 2);
    const saved = await client.requestFile({ type: "export", scope: "branch", cursor: 2, delay_ms: 250 });
    const branch = JSON.parse(saved.text); assert.equal(branch.branches.length, 1); assert.equal(branch.branches[0].parent, null);
    assert.equal(branch.versions.format, 2);
    await assert.rejects(client.requestFile({ type: "import", controllers: null }, "{invalid"), error => error.code === "invalid_file");
    const unchanged = await client.request({ type: "view" }); assert.deepEqual(unchanged.view.board, imported.view.board);
    const fresh = makeClient(); const restored = await fresh.requestFile({ type: "import", controllers: null }, saved.text);
    assert.deepEqual(restored.view.board, imported.view.board);
    const state = await client.requestFile({ type: "export", scope: "state", cursor: 2, delay_ms: 0 });
    const loaded = await fresh.requestFile({ type: "import", controllers: [{ kind: "human" }, { kind: "first_available", seed: "18446744073709551615" }] }, state.text);
    assert.equal(loaded.view.cursor, 0); assert.deepEqual(loaded.view.board, imported.view.board);
    assert.equal(loaded.view.controllers[1].seed, "18446744073709551615");
    client.dispose(); fresh.dispose();
  }
});
test("file byte limits preserve ordinary-command limits and settle without posting", async () => {
  const client = makeClient();
  await assert.rejects(client.requestFile({ type: "import" }, "é".repeat(MAX_FILE_BYTES / 2 + 1)), error => error.code === "resource_limit");
  await assert.rejects(client.request({ type: "view", padding: "x".repeat(40_000) }), error => error.code === "resource_limit");
  assert.equal(client.pending.size, 0); client.dispose();
});
test("terminating candidate worker rejects pending validation and late replies cannot revive it", async () => {
  const worker = new EngineWorker(); worker.postMessage = () => {};
  const client = new BrowserClient(worker); worker.listeners.message({ data: { type: "ready" } });
  const pending = client.requestFile({ type: "import" }, "{}");
  await Promise.resolve(); client.dispose();
  await assert.rejects(pending, error => error.code === "cancelled");
  assert.equal(worker.terminated, true); assert.equal(client.pending.size, 0);
  worker.listeners.message({ data: { request_id: "1", request_generation: "0", generation: "1", view: {} } });
  assert.equal(client.generation, "0");
});
