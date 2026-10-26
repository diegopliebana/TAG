import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
import { BrowserClient } from "../client.js";
import { scenes, buildSample } from "../presentation/scenes.js";
import { AiPlayback } from "../ai-playback.js";

initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });

class EngineWorker {
  listeners = {};
  engine = new BrowserEngine();
  addEventListener(type, callback) { this.listeners[type] = callback; }
  postMessage(text) { this.listeners.message({ data: JSON.parse(text?.type === "file" ? this.engine.handle_file(text.envelope, text.text) : this.engine.handle(text)) }); }
  terminate() {}
}

test("actual Wasm decoding errors reject their client request and permit subsequent play", async () => {
  const worker = new EngineWorker(); const client = new BrowserClient(worker);
  worker.listeners.message({ data: { type: "ready" } });
  try {
    for (const command of [{ type: "unknown_command" }, { type: "create", game: "tictactoe", controllers: "wrong type" }]) {
      await assert.rejects(client.request(command), error => error.code === "invalid_request");
      assert.equal(client.pending.size, 0);
    }
    const game = await client.request({ type: "create", game: "tictactoe" });
    const moved = await client.request({ type: "human_move", actor: 0, action: game.view.actions[0] }, { positionVersion: game.view.position_version });
    assert.equal(moved.view.board.counters[0], 1);
    await assert.rejects(client.request({ type: "human_move", actor: 0, action: moved.view.actions[0] }, { positionVersion: moved.view.position_version }), error => error.code === "wrong_actor");
    assert.equal(client.pending.size, 0);
  } finally { worker.engine.free(); }
});

test("actual Wasm AI scheduling preserves cancelled RNG and every seeded step/automatic position", async () => {
  const worker = new EngineWorker(); const client = new BrowserClient(worker);
  worker.listeners.message({ data: { type: "ready" } });
  let view, trace = [], errors = [];
  const playback = new AiPlayback({ client, view: () => view, ready: () => true, apply: async after => { view = after; trace.push(JSON.stringify(after.board)); }, changed: () => {}, failed: error => errors.push(error) });
  playback.setDelay(0);
  try {
    for (const game of ["tictactoe", "connect4"]) {
      view = (await client.request({ type: "create", game, controllers: [{ kind: "osla", seed: "18446744073709551615" }, { kind: "osla", seed: "9007199254740993" }] }, { replace: true })).view;
      playback.step(); playback.pause(); await playback.settled();
      assert.equal(view.position_version, "0"); assert.equal(trace.length, 0);
      while (view.next_player !== null) { playback.step(); await playback.settled(); }
      const expected = trace; trace = [];
      view = (await client.request({ type: "reset" }, { positionVersion: view.position_version })).view;
      playback.play();
      for (let i = 0; i < 500 && view.next_player !== null; i++) await new Promise(resolve => setTimeout(resolve, 1));
      await playback.settled();
      assert.equal(view.next_player, null); assert.deepEqual(trace, expected);
      assert.equal(playback.playing, false); assert.deepEqual(errors, []); trace = [];
    }
  } finally { playback.pause(); worker.engine.free(); }
});

test("presentation samples contain engine-produced initial, ongoing and terminal positions", async () => {
  const worker = new EngineWorker(); const client = new BrowserClient(worker);
  worker.listeners.message({ data: { type: "ready" } });
  try {
    for (const scene of scenes) {
      const sample = await buildSample(client, scene);
      assert.equal(sample.views.length, scene.moves.length + 1);
      assert.equal(sample.views[0].board.outcome.kind, "ongoing");
      sample.views.forEach((view, index) => assert.equal(view.board.counters[0], index));
      assert.equal(sample.views.at(-1).next_player, null);
      assert.equal(sample.views.at(-1).board.outcome.kind, scene.outcome);
      if (scene.emphasis) assert.deepEqual(sample.views.at(-1).board.winning_cells, scene.emphasis);
      if (scene.outcome === "draw") assert.deepEqual(sample.views.at(-1).board.winning_cells, []);
    }
  } finally { worker.engine.free(); }
});

test("actual Wasm history and forks preserve source continuations and reject historical moves", async () => {
  const worker = new EngineWorker(), client = new BrowserClient(worker);
  worker.listeners.message({ data: { type: "ready" } });
  try {
    const sample = await buildSample(client, scenes[0]);
    const end = sample.views.at(-1);
    const history = await client.request({ type: "history", branch: 0, cursor: 2 });
    assert.deepEqual(history.view.board, sample.views[2].board);
    assert.equal(history.view.position_version, end.position_version);
    assert.equal(history.view.read_only, true);
    await assert.rejects(client.request({ type: "human_move", actor: 0, action: { x: 2, y: 2 } }, { positionVersion: end.position_version }), error => error.code === "historical_position");
    const fork = await client.request({ type: "fork", branch: 0, cursor: 2, replacements: [null, null] }, { positionVersion: end.position_version });
    assert.equal(fork.view.branch, 1); assert.equal(fork.view.moves.length, 2); assert.equal(fork.view.read_only, false);
    const accepted = await client.request({ type: "human_move", actor: 0, action: { x: 2, y: 2 } }, { positionVersion: fork.view.position_version });
    const restored = await client.request({ type: "select_branch", branch: 0 }, { positionVersion: accepted.view.position_version });
    assert.deepEqual(restored.view.board, end.board);
    assert.equal(restored.view.branches[1].move_count, 3);
    const list = restored.view.moves;
    assert.deepEqual(list.map(move => move.action), scenes[0].moves);
  } finally { worker.engine.free(); }
});
