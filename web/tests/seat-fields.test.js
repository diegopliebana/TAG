import test from "node:test";
import assert from "node:assert/strict";
import { createSeatFields, fillSeats, readControllers, setModels, setModelLoader } from "../seat-fields.js";
class Element {
  constructor(document, tag) { this.ownerDocument = document; this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {}; this.value = ""; }
  append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } }
  replaceChildren(...children) { this.children = []; this.append(...children); this.value = children[0]?.value ?? ""; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(key, value) { (this.listeners[key] ??= []).push(value); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] ?? []) fn(event); }
  async dispatchAsync(type) { for (const fn of this.listeners[type] ?? []) await fn({ type }); }
  get options() { return this.children; }
  set selectedIndex(index) { this.value = this.children[index]?.value ?? ""; }
  querySelector(selector) { return this.children.find(child => selector === '[role="status"]' && child.attributes.role === "status") ?? null; }
}
function fixture(count = 2, game = null) {
  const elements = [];
  const document = { createElement(tag) { const node = new Element(document, tag); elements.push(node); return node; }, getElementById(id) { return elements.find(node => node.id === id); } };
  const container = document.createElement("div");
  createSeatFields(container, { next: () => "7" }, "", count, game);
  return { document, container };
}
test("four-player setup preserves sampled OSLA work and exact seeds and filters neural models by game and seat count", () => {
  const { document } = fixture(4, "cantstop");
  setModels([{ id: "cs4", kind: "neural_mcts", game: "cantstop", player_count: 4 },
    { id: "cs3", kind: "neural_mcts", game: "cantstop", player_count: 3 },
    { id: "c4", kind: "neural_mcts", game: "connect4", player_count: 2 }]);
  const seats = [{ kind: "human" }, { kind: "sampled_osla", seed: "18446744073709551615", repetitions_per_action: 17 },
    { kind: "random", seed: "7" }, { kind: "first_available", seed: "0" }];
  fillSeats(document, seats);
  assert.deepEqual(readControllers(document, { rules: {} }), seats);
  const select = document.getElementById("controller-3"); select.value = "neural_mcts"; select.dispatchEvent(new Event("change"));
  assert.deepEqual(document.getElementById("model-3").options.map(o => o.value), ["", "cs4"]);
  assert.equal(select.options.some(o => o.value === "neural_osla"), false);
  document.getElementById("repetitions-1").value = "0";
  assert.throws(() => readControllers(document, { rules: {} }), /successor samples/);
});
test("learned seat controls preserve loaded identities, filtering, seeds and historical settings", () => {
  const { document } = fixture();
  setModels([{ id: "q-id", kind: "q_learning", game: "tictactoe" }, { id: "nn-id", kind: "neural_osla", game: "connect4" }]);
  const seats = [{ kind: "q_learning", seed: "7", model_id: "q-id" }, { kind: "neural_osla", seed: "8", model_id: "nn-id" }];
  fillSeats(document, seats);
  assert.deepEqual(readControllers(document, { grid_size: 3 }), seats);
  assert.deepEqual(document.getElementById("model-0").options.map(option => option.value), ["", "q-id"]);
  setModels([{ id: "q-id", kind: "q_learning", game: "tictactoe" }]);
  assert.equal(document.getElementById("model-0").value, "q-id");
  assert.throws(() => readControllers(document, { grid_size: 3 }), /Load and select/);
  setModels([{ id: "nn-only", kind: "neural_osla", game: "tictactoe" }]);
  fillSeats(document, [{ kind: "q_learning", seed: "7" }, { kind: "human" }]);
  assert.throws(() => readControllers(document, { grid_size: 3 }), /Q-table JSON.*Neural OSLA/);
});
test("model upload success and failure are visible without clearing prior valid choices", async () => {
  const { document } = fixture();
  fillSeats(document, [{ kind: "q_learning", seed: "7" }, { kind: "human" }]);
  const input = document.getElementById("model-file-0");
  setModelLoader(async () => setModels([{ id: "q-valid", kind: "q_learning", game: "tictactoe" }]));
  input.files = [{ name: "trained.json" }];
  await input.dispatchAsync("change");
  const status = input.parentElement.querySelector('[role="status"]');
  assert.equal(status.textContent, "Loaded trained.json");
  assert.equal(document.getElementById("model-0").value, "q-valid");
  setModelLoader(async () => { throw new Error("Model digest mismatch"); });
  await input.dispatchAsync("change");
  assert.equal(status.textContent, "Model digest mismatch");
  assert.equal(document.getElementById("model-0").value, "q-valid");
});

test("MCTS seat controls retain separate search and model-binding JSON", () => {
  const { document } = fixture();
  setModels([{ id: "mcts-model", kind: "neural_mcts", game: "connect4" }]);
  const configuration = { schema_version: 1, tree_policy: "puct", budgets: { iterations: 47 },
    state_evaluator: { name: "v" }, puct_preferences: { source: "probabilities", evaluator: { name: "p" } } };
  const seats = [{ kind: "neural_mcts", seed: "42", model_id: "mcts-model", configuration,
    bindings: { value: "v", policy: "p", value_observers: "deciding_player" } },
    { kind: "mcts", seed: "9", configuration: { schema_version: 1, budgets: { time_ms: 50, iterations: null, forward_model_calls: null } } }];
  fillSeats(document, seats);
  assert.deepEqual(readControllers(document, { width: 7, height: 6 }), seats);
  document.getElementById("mcts-0").value = "bad json";
  assert.throws(() => readControllers(document, { width: 7, height: 6 }), /Player 0/);
});
