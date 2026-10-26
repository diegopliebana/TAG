import { readSeed, CryptoSeeds } from "./setup.js";
export const controllerName = kind => ({ human: "Human", random: "Random", first_available: "First Available", osla: "OSLA", sampled_osla: "Sampled OSLA", mcts: "MCTS", neural_mcts: "Neural MCTS", q_learning: "Q-learning", neural_osla: "Neural OSLA" })[kind];
let models = [], loadModel = null;
const modelFields = new Set();
const fieldGroups = new WeakMap(), seatCounts = new WeakMap();
export function setModelLoader(loader) { loadModel = loader; }
export function setModels(loaded) {
  models = loaded;
  for (const refresh of modelFields) refresh();
}
export function createSeatFields(container, seeds, prefix = "", count = 2, game = null) {
  const document = container.ownerDocument;
  for (const refresh of fieldGroups.get(container) ?? []) modelFields.delete(refresh);
  const refreshes = []; fieldGroups.set(container, refreshes);
  container.replaceChildren();
  const counts = seatCounts.get(document) ?? new Map(); counts.set(prefix, count); seatCounts.set(document, counts);
  for (let seat = 0; seat < count; seat++) {
    const section = document.createElement("div"); section.className = "seat-settings";
    const label = document.createElement("label"); label.textContent = `Player ${seat}`;
    const select = document.createElement("select"); select.id = `${prefix}controller-${seat}`;
    for (const kind of ["cantstop","loveletter"].includes(game) ? ["human", "random", "first_available", "sampled_osla", "mcts", "neural_mcts"] : ["human", "random", "first_available", "osla", "mcts", "neural_mcts", "q_learning", "neural_osla"]) {
      const option = document.createElement("option"); option.value = kind; option.textContent = controllerName(kind); select.append(option);
    }
    select.value = seat === 0 ? "human" : ["cantstop","loveletter"].includes(game) ? "sampled_osla" : "osla"; label.append(select);
    const seedLabel = document.createElement("label"); seedLabel.textContent = `Player ${seat} seed`; seedLabel.hidden = select.value === "human";
    const controls = document.createElement("div"); controls.className = "seed-controls";
    const input = document.createElement("input"); input.type = "text"; input.inputMode = "numeric"; input.id = `${prefix}seed-${seat}`;
    input.setAttribute("aria-label", `Player ${seat} seed`);
    input.value = seeds.next(); input.placeholder = "auto";
    const regenerate = document.createElement("button"); regenerate.type = "button"; regenerate.textContent = "Regenerate"; regenerate.id = `${prefix}regenerate-${seat}`;
    regenerate.addEventListener("click", () => { input.value = seeds.next(); });
    select.addEventListener("change", () => { seedLabel.hidden = select.value === "human"; });
    controls.append(input, regenerate); seedLabel.append(controls); section.append(label, seedLabel); container.append(section);
    const modelLabel = document.createElement("label"); modelLabel.textContent = `Player ${seat} model`;
    const modelSelect = document.createElement("select"); modelSelect.id = `${prefix}model-${seat}`;
    modelSelect.setAttribute("aria-label", `Player ${seat} model`);
    const upload = document.createElement("input"); upload.type = "file"; upload.accept = ".json,application/json";
    upload.id = `${prefix}model-file-${seat}`; upload.setAttribute("aria-label", `Load model for player ${seat}`);
    const status = document.createElement("span"); status.setAttribute("role", "status");
    function refresh() {
      const selected = modelSelect.value;
      modelSelect.replaceChildren();
      const placeholder = document.createElement("option"); placeholder.value = ""; placeholder.textContent = "Choose a loaded model"; modelSelect.append(placeholder);
      for (const model of models.filter(m => m.kind === select.value && (!game || m.game === game) && (!["cantstop","loveletter"].includes(game) || m.player_count === count))) {
        const option = document.createElement("option"); option.value = model.id;
        option.textContent = `${model.game === "loveletter" ? "Love Letter" : model.game === "cantstop" ? "Can't Stop" : model.game === "tictactoe" ? "Tic-Tac-Toe" : "Connect 4"} / ${model.id.slice(0, 12)}`;
        option.title = model.id; modelSelect.append(option);
      }
      modelSelect.value = [...modelSelect.options].some(o => o.value === selected) ? selected : "";
      if (!modelSelect.value && modelSelect.options.length === 2) modelSelect.selectedIndex = 1;
      modelLabel.hidden = !["q_learning", "neural_osla", "neural_mcts"].includes(select.value);
    }
    modelFields.add(refresh); refreshes.push(refresh); select.addEventListener("change", refresh);
    upload.addEventListener("change", async () => {
      const file = upload.files[0]; if (!file) return;
      status.textContent = "Loading model...";
      try {
        if (!loadModel) throw new Error("Model loading is unavailable.");
        await loadModel(file); refresh(); status.textContent = `Loaded ${file.name}`;
      } catch (error) { status.textContent = error.message; }
      finally { upload.value = ""; }
    });
    modelLabel.append(modelSelect, upload, status); section.append(modelLabel); refresh();
    const searchLabel = document.createElement("label"); searchLabel.textContent = `Player ${seat} MCTS JSON`;
    const search = document.createElement("textarea"); search.id = `${prefix}mcts-${seat}`;
    search.rows = 8; search.value = JSON.stringify({ schema_version: 1, rollout_length: 0,
      budgets: { iterations: 256 }, state_evaluator: { name: "game_heuristic" } }, null, 2);
    search.setAttribute("aria-label", `Player ${seat} MCTS JSON`);
    const searchFile = document.createElement("input"); searchFile.type = "file";
    const searchStatus = document.createElement("span"); searchStatus.setAttribute("role", "status");
    searchFile.accept = ".json,application/json";
    searchFile.addEventListener("change", async () => {
      const file = searchFile.files[0]; if (!file) return;
      try {
        if (file.size > 16384) throw new Error("MCTS configuration exceeds 16 KiB.");
        const text = await file.text(); JSON.parse(text); search.value = text;
        searchStatus.textContent = `Loaded ${file.name}`;
      } catch (error) { searchStatus.textContent = error.message; }
      finally { searchFile.value = ""; }
    });
    const nativeDefault = search.value;
    const neuralDefault = JSON.stringify({ schema_version: 1, tree_policy: "puct", rollout_length: 0,
      budgets: { iterations: 256 }, state_evaluator: { name: "network_value" },
      puct_preferences: { source: "probabilities", evaluator: { name: "network_policy" } } }, null, 2);
    const bindingsLabel = document.createElement("label"); bindingsLabel.textContent = `Player ${seat} neural MCTS bindings JSON`;
    const bindings = document.createElement("textarea"); bindings.id = `${prefix}bindings-${seat}`;
    bindings.rows = 4; bindings.value = JSON.stringify({ value: "network_value", policy: "network_policy", value_observers: null }, null, 2);
    bindingsLabel.append(bindings); section.append(bindingsLabel);
    const showSearch = () => {
      searchLabel.hidden = !["mcts", "neural_mcts"].includes(select.value);
      bindingsLabel.hidden = select.value !== "neural_mcts";
      if (select.value === "neural_mcts" && search.value === nativeDefault) search.value = neuralDefault;
      if (select.value === "mcts" && search.value === neuralDefault) search.value = nativeDefault;
    };
    select.addEventListener("change", showSearch); showSearch();
    searchLabel.append(search, searchFile, searchStatus); section.append(searchLabel);
    const repetitionsLabel = document.createElement("label"); repetitionsLabel.textContent = "Successor samples per action (1–128)";
    const repetitions = document.createElement("input"); repetitions.id = `${prefix}repetitions-${seat}`;
    repetitions.type = "number"; repetitions.min = "1"; repetitions.max = "128"; repetitions.step = "1"; repetitions.value = "1";
    repetitionsLabel.append(repetitions); section.append(repetitionsLabel);
    const showSamples = () => { repetitionsLabel.hidden = select.value !== "sampled_osla"; };
    select.addEventListener("change", showSamples); showSamples();
  }
}
export function fillSeats(document, controllers, prefix = "") {
  controllers.forEach((controller, seat) => {
    const select = document.getElementById(`${prefix}controller-${seat}`);
    select.value = controller.kind; select.dispatchEvent(new Event("change"));
    document.getElementById(`${prefix}seed-${seat}`).value = controller.seed ?? "";
    document.getElementById(`${prefix}model-${seat}`).value = controller.model_id ?? "";
    if (controller.configuration) document.getElementById(`${prefix}mcts-${seat}`).value = JSON.stringify(controller.configuration, null, 2);
    if (controller.bindings) document.getElementById(`${prefix}bindings-${seat}`).value = JSON.stringify(controller.bindings, null, 2);
    document.getElementById(`${prefix}repetitions-${seat}`).value = controller.repetitions_per_action ?? 1;
  });
}
export function readControllers(document, parameters, prefix = "") {
  return Array.from({ length: seatCounts.get(document)?.get(prefix) ?? 2 }, (_, seat) => {
    const kind = document.getElementById(`${prefix}controller-${seat}`).value;
    if (kind === "human") return { kind };
    if (kind === "osla" && (parameters.grid_size ? parameters.grid_size ** 2 : parameters.width * parameters.height) > 256) throw new Error(`Player ${seat}: OSLA supports boards with at most 256 cells.`);
    try {
      const raw = document.getElementById(`${prefix}seed-${seat}`).value.trim();
      const seed = !raw || raw === "auto" ? new CryptoSeeds().next() : readSeed(raw);
      const controller = { kind, seed };
      if (kind === "sampled_osla") {
        const repetitions = Number(document.getElementById(`${prefix}repetitions-${seat}`).value);
        if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 128) throw new Error("Use 1–128 successor samples per action.");
        controller.repetitions_per_action = repetitions;
      }
      if (["mcts", "neural_mcts"].includes(kind)) controller.configuration = JSON.parse(document.getElementById(`${prefix}mcts-${seat}`).value);
      if (kind === "neural_mcts") controller.bindings = JSON.parse(document.getElementById(`${prefix}bindings-${seat}`).value);
      if (["q_learning", "neural_osla", "neural_mcts"].includes(kind)) {
        controller.model_id = document.getElementById(`${prefix}model-${seat}`).value;
        if (!controller.model_id) throw new Error(kind === "q_learning"
          ? "Load and select a Q-table JSON for Q-learning. Neural model JSON uses Neural OSLA."
          : "Load and select a neural evaluation JSON for Neural OSLA. Q-tables use Q-learning.");
      }
      return controller;
    }
    catch (error) { throw new Error(`Player ${seat}: ${error.message}`); }
  });
}
