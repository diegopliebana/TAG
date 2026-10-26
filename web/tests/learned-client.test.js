import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
import { BrowserClient } from "../client.js";
initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });
const text = name => readFileSync(new URL(`./fixtures/learned/${name}`, import.meta.url), "utf8");
const fixtures = JSON.parse(text("fixtures.json"));
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
function close(a, b) { assert.ok(Math.abs(a - b) <= fixtures.absolute_tolerance + fixtures.relative_tolerance * Math.abs(b), `${a} differs from PyTorch ${b}`); }

test("bundle import preserves neural source text for later standalone-state candidate loading", async () => {
  const source = makeClient(), restored = makeClient(), candidate = makeClient();
  try {
    const loaded = await source.requestFile({ type: "load_model" }, text("tictactoe-trained.json"));
    const controllers = [{ kind: "neural_osla", seed: "7", model_id: loaded.text }, { kind: "human" }];
    const view = (await source.request({ type: "create", game: "tictactoe", controllers })).view;
    const bundle = await source.requestFile({ type: "export", scope: "session", cursor: view.cursor, delay_ms: 0 });
    await restored.requestFile({ type: "import", controllers: null }, bundle.text);
    assert.equal(restored.modelTexts.length, 1);
    const state = await restored.requestFile({ type: "export", scope: "state", cursor: 0, delay_ms: 0 });
    // FileControls copies this cache into a new worker before loading a state.
    for (const model of restored.modelTexts) {
      const reply = await candidate.requestFile({ type: "load_model" }, model);
      assert.equal(reply.text, loaded.text);
    }
    const imported = await candidate.requestFile({ type: "import", controllers }, state.text);
    assert.equal(imported.needs_controllers, false);
    assert.deepEqual(imported.live_view.board, view.board);
    const moved = await step(candidate, imported.live_view);
    assert.equal(moved.cursor, 1);
  } finally { source.dispose(); restored.dispose(); candidate.dispose(); }
});

test("actual Wasm inputs, logits and observer values match trained, zero and combined PyTorch fixtures", async () => {
  for (const model of fixtures.models) {
    const client = makeClient();
    try {
      const document = JSON.parse(text(model.model_file));
      await client.requestFile({ type: "load_model" }, text(model.model_file));
      for (const row of model.rows) {
        let view = (await client.request({ type: "create", game: model.game })).view;
        for (const [index, id] of row.prefix.entries()) {
          const action = model.game === "tictactoe" ? { x: Math.floor(id / 3), y: id % 3 } : { column: id };
          view = (await client.request({ type: "human_move", actor: index % 2, action }, { positionVersion: view.position_version })).view;
        }
        const raw = await client.requestFile({ type: "predict_model", model_id: document.model_identity, observer: row.observer });
        const result = JSON.parse(raw.text);
        assert.deepEqual(result.inputs, row.inputs);
        assert.equal(result.logits.length, row.logits.length);
        result.logits.forEach((value, i) => close(value, row.logits[i])); close(result.value, row.value);
      }
    } finally { client.dispose(); }
  }
});

test("actual Wasm neural OSLA completes both seats with the Python fixture actions", async () => {
  for (const model of fixtures.models) for (const trace of model.traces) {
    const client = makeClient();
    try {
      const document = JSON.parse(text(model.model_file));
      await client.requestFile({ type: "load_model" }, text(model.model_file));
      const controllers = [{ kind: "osla", seed: "13" }, { kind: "osla", seed: "13" }];
      controllers[trace.seat] = { kind: "neural_osla", seed: "7", model_id: document.model_identity };
      let view = (await client.request({ type: "create", game: model.game, controllers })).view;
      for (const expected of trace.actions) {
        view = await step(client, view);
        const action = view.accepted_move.action;
        assert.equal(model.game === "tictactoe" ? action.x * 3 + action.y : action.column, expected);
      }
      assert.equal(view.next_player, null);
    } finally { client.dispose(); }
  }
});

test("actual Wasm Q-learning masks scores and learned bundles restore independent continuation", async () => {
  const source = makeClient(), loaded = makeClient();
  try {
    const reply = await source.requestFile({ type: "load_model" }, text("q-table.json"));
    const id = reply.models[0].id;
    const controllers = [0, 1].map(() => ({ kind: "q_learning", seed: "7", model_id: id }));
    let view = (await source.request({ type: "create", game: "tictactoe", controllers })).view;
    const prepared = await source.request({ type: "prepare_ai" }, { positionVersion: view.position_version });
    await source.request({ type: "discard_ai", proposal: prepared.proposal.id }, { positionVersion: view.position_version });
    for (let n = 0; n < 2; n++) {
      const prediction = JSON.parse((await source.requestFile({ type: "predict_model", model_id: id, observer: view.next_player })).text);
      const legal = view.actions.map(a => a.x * 3 + a.y);
      const best = Math.max(...legal.map(a => prediction.q_values[a]));
      view = await step(source, view);
      const action = view.accepted_move.action.x * 3 + view.accepted_move.action.y;
      assert.equal(prediction.q_values[action], best);
      if (n === 0) assert.deepEqual(view.accepted_move.action, prepared.proposal.action);
    }
    const exported = await source.requestFile({ type: "export", scope: "session", cursor: view.cursor, delay_ms: 0 });
    const bundle = JSON.parse(exported.text);
    assert.equal(bundle.kind, "tag-learned-browser"); assert.equal(bundle.models.length, 1);
    const restored = await loaded.requestFile({ type: "import", controllers: null }, exported.text);
    let other = restored.live_view;
    while (view.next_player !== null) {
      view = await step(source, view); other = await step(loaded, other);
      assert.deepEqual(other.board, view.board);
      assert.deepEqual(other.accepted_move.action, view.accepted_move.action);
    }
    bundle.models = [];
    await assert.rejects(loaded.requestFile({ type: "import", controllers: null }, JSON.stringify(bundle)));
    assert.deepEqual((await loaded.request({ type: "view" })).view.board, view.board);
  } finally { source.dispose(); loaded.dispose(); }
});
