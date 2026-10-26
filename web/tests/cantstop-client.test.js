import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });
const fixtures = JSON.parse(readFileSync(new URL("./fixtures/mcts/cantstop-predictions.json", import.meta.url)));
const native = { schema_version: 1, rollout_length: 0, budgets: { iterations: 16 } };
const neural = { ...native, tree_policy: "puct", state_evaluator: { name: "network_value" },
  puct_preferences: { source: "probabilities", evaluator: { name: "network_policy" } } };
function command(e, generation, view, command) {
  const r = JSON.parse(e.handle(JSON.stringify({ request_id: "1", generation, position_version: view?.position_version ?? null, command })));
  assert.equal(r.error, null, JSON.stringify(r.error)); return r;
}
function file(e, generation, command, text = "") {
  const r = JSON.parse(e.handle_file(JSON.stringify({ request_id: "1", generation, command }), text));
  assert.equal(r.error, null, JSON.stringify(r.error)); return r;
}
function prepare(e, v) {
  let r = command(e, "1", v, { type: "prepare_ai" }); let polls = 0;
  while (r.proposal.pending) { assert.ok(polls++ < 10000); r = command(e, "1", v, { type: "poll_ai", proposal: r.proposal.id }); }
  return r.proposal;
}
function step(e, v) { return command(e, "1", v, { type: "commit_ai", proposal: prepare(e, v).id }).view; }
function id(a) {
  if (typeof a === "string") return { Roll: 0, Stop: 1, Bust: 2 }[a];
  const { columns: [a0, b], length } = a.Allocate;
  if (length === 1) return 3 + a0 - 2;
  let k = 14; for (let x = 2; x <= 12; x++) for (let y = x; y <= 12; y++, k++) if (x === a0 && y === b) return k;
  throw new Error("unknown allocation");
}
function params(f) { return { ...f.parameters, seed: "18446744073709551615" }; }
function load(e, f) { return file(e, "0", { type: "load_model" }, readFileSync(new URL(`./fixtures/mcts/${f.model_file}`, import.meta.url), "utf8")).text; }
const close = (a, b) => assert.ok(Math.abs(a - b) <= 2e-6 + 2e-5 * Math.abs(b), `${a} != ${b}`);

test("actual Wasm Can't Stop public arrays, logits and full value vectors match PyTorch", () => {
  for (const f of fixtures) {
    const e = new BrowserEngine();
    try {
      const model_id = load(e, f); let generation = 0;
      for (const row of f.rows) {
        let v = command(e, String(generation++), null, { type: "create", game: "cantstop", parameters: params(f), controllers: Array.from({ length: f.players }, () => ({ kind: "human" })) }).view;
        for (const actionId of row.prefix) {
          const action = v.actions.find(a => id(a) === actionId); assert.ok(action);
          v = command(e, String(generation), v, { type: "human_move", actor: v.next_player, action }).view;
        }
        const prediction = JSON.parse(file(e, String(generation), { type: "predict_model", model_id, observer: row.observer }).text);
        assert.deepEqual(prediction.inputs, row.inputs);
        assert.deepEqual(prediction.value_players, Array.from({ length: f.players }, (_, i) => i));
        prediction.logits.forEach((v, i) => close(v, row.logits[i]));
        prediction.values.forEach((v, i) => close(v, row.values[i]));
        assert.equal(JSON.stringify(v.board).includes("rng"), false);
      }
    } finally { e.free(); }
  }
});

test("actual Wasm multiplayer neural/native matches finish and retain both controller streams in bundled continuations", () => {
  for (const f of fixtures) {
    const e = new BrowserEngine(), restored = new BrowserEngine();
    try {
      const model_id = load(e, f);
      const controllers = Array.from({ length: f.players }, (_, seat) => seat === 0
        ? { kind: "neural_mcts", seed: "18446744073709551615", model_id, configuration: neural }
        : seat % 2 ? { kind: "sampled_osla", seed: "7", repetitions_per_action: 2 } : { kind: "mcts", seed: "8", configuration: native });
      let v = command(e, "0", null, { type: "create", game: "cantstop", parameters: params(f), controllers }).view;
      let resumed = null;
      for (let moves = 0; v.next_player !== null; moves++) {
        assert.ok(moves < 1000);
        if (moves === 5) {
          const before = file(e, "1", { type: "export", scope: "session", cursor: v.cursor, delay_ms: 0 }).text;
          const p = prepare(e, v);
          assert.deepEqual(command(e, "1", v, { type: "view" }).view.board, v.board);
          command(e, "1", v, { type: "discard_ai", proposal: p.id });
          assert.equal(file(e, "1", { type: "export", scope: "session", cursor: v.cursor, delay_ms: 0 }).text, before);
          assert.equal(JSON.parse(before).kind, "tag-learned-browser");
          resumed = file(restored, "0", { type: "import", controllers: null }, before).view;
          const repeated = prepare(e, v); assert.deepEqual(repeated.action, p.action);
          const pending = command(e, "1", v, { type: "discard_ai", proposal: repeated.id });
          assert.deepEqual(pending.view.board, v.board);
        }
        v = step(e, v);
        if (resumed) { resumed = step(restored, resumed); assert.deepEqual(resumed.board, v.board); }
      }
      assert.equal(v.board.outcome.kind, "won");
      assert.equal(v.results.filter(r => r === "win").length, 1);
      const saved = file(e, "1", { type: "export", scope: "session", cursor: v.cursor, delay_ms: 0 }).text;
      const d = JSON.parse(saved); assert.ok(d.replay.branches[0].moves.some(m => m.chance?.dice));
    } finally { e.free(); restored.free(); }
  }
});

test("actual Wasm game seeds use exact strings or Web Crypto and invalid setup preserves the live game", () => {
  const e = new BrowserEngine();
  try {
    const v = command(e, "0", null, { type: "create", game: "cantstop", parameters: { seed: null } }).view;
    const saved = file(e, "1", { type: "export", scope: "state", cursor: 0, delay_ms: 0 });
    assert.match(JSON.parse(saved.text).state.chance.initial_seed, /^\d+$/);
    for (const parameters of [[], 7, "invalid", { seed: 18446744073709551615 }, { seed: "18446744073709551616" }, { seed: "1", rules: { column_heights: Array(11).fill(129) } }]) {
      const r = JSON.parse(e.handle(JSON.stringify({ request_id: "1", generation: "1", position_version: v.position_version, command: { type: "create", game: "cantstop", parameters } })));
      assert.ok(r.error); assert.deepEqual(command(e, "1", v, { type: "view" }).view.board, v.board);
    }
  } finally { e.free(); }
});

test("actual Wasm suspends stochastic MCTS and rejects reversed simulation progress and mismatched model seats", () => {
  const e = new BrowserEngine();
  try {
    const long = { kind: "mcts", seed: "42", configuration: { ...native, budgets: { iterations: 100000 } } };
    let v = command(e, "0", null, { type: "create", game: "cantstop", parameters: { seed: "42" }, controllers: [long, long] }).view;
    const before = file(e, "1", { type: "export", scope: "session", cursor: 0, delay_ms: 0 }).text;
    const p = command(e, "1", v, { type: "prepare_ai" }).proposal;
    assert.equal(p.pending, true);
    command(e, "1", v, { type: "discard_ai", proposal: p.id });
    assert.equal(file(e, "1", { type: "export", scope: "session", cursor: 0, delay_ms: 0 }).text, before);
    const f = fixtures[0], model_id = loadModelAfterCreation(e, f);
    const wrong = { kind: "neural_mcts", seed: "7", model_id, configuration: neural };
    const bad = JSON.parse(e.handle(JSON.stringify({ request_id: "2", generation: "1", position_version: null,
      command: { type: "create", game: "cantstop", parameters: params(f), controllers: [wrong, { kind: "human" }, { kind: "human" }] } })));
    assert.ok(bad.error); assert.deepEqual(command(e, "1", v, { type: "view" }).view.board, v.board);
    const short = { kind: "mcts", seed: "7", configuration: native };
    v = command(e, "1", null, { type: "create", game: "cantstop", parameters: { seed: "42" }, controllers: [short, short] }).view;
    // The new lifecycle uses generation 2. Accepted records retain simulated RNG progress.
    for (let i = 0; i < 4; i++) {
      let r = command(e, "2", v, { type: "prepare_ai" });
      while (r.proposal.pending) r = command(e, "2", v, { type: "poll_ai", proposal: r.proposal.id });
      v = command(e, "2", v, { type: "commit_ai", proposal: r.proposal.id }).view;
    }
    const saved = JSON.parse(file(e, "2", { type: "export", scope: "session", cursor: v.cursor, delay_ms: 0 }).text);
    saved.branches[0].moves[2].controller_after.rng.policy.simulation.word_position = "0";
    const r = JSON.parse(e.handle_file(JSON.stringify({ request_id: "3", generation: "2", command: { type: "import", controllers: null } }), JSON.stringify(saved)));
    assert.ok(r.error); assert.deepEqual(command(e, "2", v, { type: "view" }).view.board, v.board);
  } finally { e.free(); }
});
function loadModelAfterCreation(e, f) {
  return file(e, "1", { type: "load_model" }, readFileSync(new URL(`./fixtures/mcts/${f.model_file}`, import.meta.url), "utf8")).text;
}

test("actual S4c three-player default-rule export loads into local browser neural MCTS", () => {
  const e = new BrowserEngine();
  try {
    const model_id = file(e, "0", { type: "load_model" }, readFileSync(new URL("../../examples/selfplay/results/cantstop-model.json", import.meta.url), "utf8")).text;
    const controllers = [{ kind: "neural_mcts", seed: "7", model_id, configuration: neural }, { kind: "random", seed: "8" }, { kind: "random", seed: "9" }];
    let v = command(e, "0", null, { type: "create", game: "cantstop", parameters: { seed: "42" }, controllers }).view;
    for (let i = 0; i < 6; i++) v = step(e, v);
    assert.equal(v.cursor, 6);
  } finally { e.free(); }
});
