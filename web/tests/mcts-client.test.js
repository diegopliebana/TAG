import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
initSync({ module: readFileSync(new URL("../pkg/tag_browser_bg.wasm", import.meta.url)) });
const native = { schema_version: 1, rollout_length: 0, budgets: { iterations: 64 } };
const neural = { ...native, tree_policy: "puct", state_evaluator: { name: "network_value" },
  puct_preferences: { source: "probabilities", evaluator: { name: "network_policy" } } };
function command(engine, generation, version, command) {
  const reply = JSON.parse(engine.handle(JSON.stringify({ request_id: "1", generation,
    position_version: version, command })));
  assert.equal(reply.error, null, JSON.stringify(reply.error)); return reply;
}
function file(engine, generation, command, text = "") {
  const reply = JSON.parse(engine.handle_file(JSON.stringify({ request_id: "1", generation, command }), text));
  assert.equal(reply.error, null, JSON.stringify(reply.error)); return reply;
}
function prepare(engine, view) {
  let reply = command(engine, "1", view.position_version, { type: "prepare_ai" });
  let count = 0;
  while (reply.proposal.pending) {
    assert.ok(count++ < 1000);
    reply = command(engine, "1", view.position_version, { type: "poll_ai", proposal: reply.proposal.id });
  }
  return reply.proposal;
}
const close = (a, b) => assert.ok(Math.abs(a - b) <= 2e-6 + 2e-5 * Math.abs(b), `${a} != ${b}`);

test("actual Wasm self-play scalar/vector predictions agree with PyTorch", () => {
  const cases = JSON.parse(readFileSync(new URL("./fixtures/mcts/predictions.json", import.meta.url)));
  for (const fixture of cases) {
    const engine = new BrowserEngine();
    try {
      const model = file(engine, "0", { type: "load_model" }, readFileSync(new URL(`./fixtures/mcts/${fixture.model_file}`, import.meta.url), "utf8")).text;
      let generation = 0;
      for (const row of fixture.rows) {
        let view = command(engine, String(generation++), null, { type: "create", game: fixture.game }).view;
        for (const id of row.prefix) {
          const selected = view.actions.find(a => fixture.game === "connect4" ? a.column === id : a.x * 3 + a.y === id);
          view = command(engine, String(generation), view.position_version,
            { type: "human_move", actor: view.next_player, action: selected }).view;
        }
        const result = JSON.parse(file(engine, String(generation), { type: "predict_model", model_id: model, observer: row.observer }).text);
        assert.deepEqual(result.inputs, row.inputs);
        assert.equal(result.logits.length, row.logits.length);
        result.logits.forEach((v, i) => close(v, row.logits[i]));
        result.values.forEach((v, i) => close(v, row.values[i]));
      }
    } finally { engine.free(); }
  }
});

test("actual Wasm MCTS plays complete mixed games and resumes self-contained files", () => {
  for (const game of ["tictactoe", "connect4"]) for (const head of ["scalar", "vector"]) for (const seat of [0, 1]) {
    const engine = new BrowserEngine(); const restored = new BrowserEngine();
    try {
      const loaded = file(engine, "0", { type: "load_model" }, readFileSync(new URL(`./fixtures/mcts/${game}-${head}.json`, import.meta.url), "utf8"));
      const controllers = [0, 1].map(() => ({ kind: "mcts", seed: "13", configuration: native }));
      controllers[seat] = { kind: "neural_mcts", seed: "7", configuration: neural, model_id: loaded.text };
      let view = command(engine, "0", null, { type: "create", game, controllers }).view;
      for (let move = 0; view.next_player !== null; move++) {
        assert.ok(move < 43);
        if (move === 4) {
          const saved = file(engine, "1", { type: "export", scope: "session", cursor: 4, delay_ms: 0 }).text;
          const recovered = file(restored, "0", { type: "import", controllers: null }, saved).view;
          assert.deepEqual(recovered.board, view.board);
          const original = prepare(engine, view), repeat = prepare(restored, recovered);
          assert.deepEqual(original.action, repeat.action);
          command(engine, "1", view.position_version, { type: "discard_ai", proposal: original.id });
          command(restored, "1", recovered.position_version, { type: "discard_ai", proposal: repeat.id });
        }
        const proposal = prepare(engine, view);
        view = command(engine, "1", view.position_version, { type: "commit_ai", proposal: proposal.id }).view;
      }
      assert.ok(view.results.every(r => r !== "ongoing"));
    } finally { engine.free(); restored.free(); }
  }
});

test("actual Wasm timed MCTS yields, cancels, and replays recorded choices", () => {
  const engine = new BrowserEngine(), restored = new BrowserEngine();
  try {
    const setup = { kind: "mcts", seed: "42", configuration: { ...native, budgets: { time_ms: 20, iterations: null, forward_model_calls: null } } };
    let view = command(engine, "0", null, { type: "create", game: "connect4", controllers: [setup, setup] }).view;
    const pending = command(engine, "1", "0", { type: "prepare_ai" });
    assert.equal(pending.proposal.pending, true);
    command(engine, "1", "0", { type: "discard_ai", proposal: pending.proposal.id });
    const choice = prepare(engine, view);
    view = command(engine, "1", "0", { type: "commit_ai", proposal: choice.id }).view;
    const saved = file(engine, "1", { type: "export", scope: "session", cursor: 1, delay_ms: 0 }).text;
    const loaded = file(restored, "0", { type: "import", controllers: null }, saved);
    assert.deepEqual(loaded.view.board, view.board);
    const illegal = JSON.parse(saved); illegal.branches[0].moves[0].action.column = 99;
    const badAction = JSON.parse(restored.handle_file(JSON.stringify({ request_id: "2", generation: "1",
      command: { type: "import", controllers: null } }), JSON.stringify(illegal)));
    assert.ok(badAction.error);
    const changedPolicy = JSON.parse(saved);
    changedPolicy.branches[0].moves[0].controller_after.rng.policy.configuration.exploration_constant = 9;
    const badPolicy = JSON.parse(restored.handle_file(JSON.stringify({ request_id: "3", generation: "1",
      command: { type: "import", controllers: null } }), JSON.stringify(changedPolicy)));
    assert.ok(badPolicy.error);
    assert.deepEqual(command(restored, "1", null, { type: "view" }).view.board, view.board);
  } finally { engine.free(); restored.free(); }
});

test("actual Wasm reset drops suspended MCTS and invalid configurations preserve the game", () => {
  const engine = new BrowserEngine();
  try {
    const seat = { kind: "mcts", seed: "18446744073709551615", configuration: { ...native, budgets: { iterations: 100000 } } };
    const initial = command(engine, "0", null, { type: "create", game: "connect4", controllers: [seat, { kind: "human" }] });
    assert.equal(initial.view.controllers[0].seed, seat.seed);
    assert.equal(initial.view.controllers[0].configuration.seed, null);
    const pending = command(engine, "1", "0", { type: "prepare_ai" }).proposal;
    assert.equal(pending.pending, true);
    const reset = command(engine, "1", "0", { type: "reset" }).view;
    const stale = JSON.parse(engine.handle(JSON.stringify({ request_id: "2", generation: "1", position_version: reset.position_version,
      command: { type: "poll_ai", proposal: pending.id } })));
    assert.equal(stale.error.code, "invalid_proposal");
    const bad = { ...seat, configuration: { ...native, state_evaluator: { name: "unknown" } } };
    const rejected = JSON.parse(engine.handle(JSON.stringify({ request_id: "3", generation: "1", position_version: null,
      command: { type: "create", game: "connect4", controllers: [bad, { kind: "human" }] } })));
    assert.ok(rejected.error);
    assert.deepEqual(command(engine, "1", null, { type: "view" }).view.board, reset.board);
  } finally { engine.free(); }
});
