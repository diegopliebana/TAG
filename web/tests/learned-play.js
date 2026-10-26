// Real-browser proof: file pickers, model errors and full games use the app's worker.
import { controllerName } from "../seat-fields.js";
async function until(check, label) {
  for (let n = 0; n < 1000; n++) {
    if (check()) return;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error(`Timed out: ${label}`);
}
function select(id, value) {
  const element = document.getElementById(id); element.value = value;
  element.dispatchEvent(new Event("change"));
}
async function upload(seat, name, text) {
  const input = document.getElementById(`model-file-${seat}`);
  const transfer = new DataTransfer(); transfer.items.add(new File([text], name, { type: "application/json" }));
  input.files = transfer.files; input.dispatchEvent(new Event("change"));
  const status = input.parentElement.querySelector('[role="status"]');
  await until(() => status.textContent !== "Loading model..." && status.textContent !== "", "model load");
  return status.textContent;
}
export async function runLearnedPlayChecks() {
  await window.gameReady;
  document.getElementById("reduced").checked = true;
  document.getElementById("mute").checked = true;
  document.getElementById("mute").dispatchEvent(new Event("change"));
  const delay = document.getElementById("ai-delay"); delay.value = "0"; delay.dispatchEvent(new Event("change"));
  const cases = [
    ["tictactoe", "q_learning", "q-table.json", 0], ["tictactoe", "q_learning", "q-table.json", 1],
    ["tictactoe", "neural_osla", "tictactoe-trained.json", 0], ["tictactoe", "neural_osla", "tictactoe-trained.json", 1],
    ["connect4", "neural_osla", "connect4-trained.json", 0], ["connect4", "neural_osla", "connect4-trained.json", 1],
    ["tictactoe", "neural_osla", "tictactoe-trained.json", 0],
    ["tictactoe", "neural_mcts", "../mcts/tictactoe-scalar.json", 0],
    ["tictactoe", "neural_mcts", "../mcts/tictactoe-vector.json", 1],
    ["connect4", "neural_mcts", "../mcts/connect4-scalar.json", 1],
    ["connect4", "neural_mcts", "../mcts/connect4-vector.json", 0],
  ];
  let games = 0, rejectedModels = 0;
  for (const [game, kind, filename, seat] of cases) {
    if (!document.getElementById("setup").open) {
      document.getElementById("new-game").click();
      await until(() => document.getElementById("setup").open, "new setup");
    }
    select("game", game);
    select(`controller-${seat}`, kind);
    select(`controller-${1-seat}`, kind === "neural_mcts" ? "mcts" : games === 6 ? "q_learning" : "osla");
    const source = await (await fetch(`./tests/fixtures/learned/${filename}`)).text();
    const status = await upload(seat, filename, source);
    if (!status.startsWith("Loaded ")) throw new Error(status);
    // Upload adds a model without replacing a valid existing selection. Several
    // cases use the same controller type for different games, so choose this
    // case's model explicitly rather than retaining the previous game's model.
    const model = JSON.parse(source);
    if (model.model_identity) select(`model-${seat}`, model.model_identity);
    if (games === 0) {
      const id = document.getElementById(`model-${seat}`).value;
      const failure = await upload(seat, "bad-model.json", "{}");
      if (failure.startsWith("Loaded ") || document.getElementById(`model-${seat}`).value !== id) throw new Error("Invalid model replaced loaded choices.");
      rejectedModels++;
    }
    document.getElementById(`seed-${seat}`).value = "7";
    document.getElementById(`seed-${1-seat}`).value = "13";
    document.getElementById("create").click();
    // Later games preserve the previous session until the existing discard dialog is accepted.
    await until(() => !document.getElementById("setup").open || document.getElementById("discard-dialog").open || document.getElementById("setup-error").textContent, "create/discard");
    if (document.getElementById("discard-dialog").open) document.querySelector('#discard-dialog button[value="discard"]').click();
    await until(() => !document.getElementById("setup").open || document.getElementById("setup-error").textContent, "created game");
    if (document.getElementById("setup-error").textContent) throw new Error(`${game}, ${kind}, ${filename}, player ${seat}: ${document.getElementById("setup-error").textContent}`);
    if (!document.getElementById("match-title").textContent.includes(controllerName(kind))) throw new Error(`${game}, ${kind}, player ${seat}: learned controller label missing.`);
    document.getElementById("play-pause").click();
    await until(() => /wins|Draw/.test(document.getElementById("result").textContent), "complete learned game");
    if (document.getElementById("status").textContent.includes("undefined")) throw new Error("Missing learned move label.");
    games++;
  }
  return { games, rejectedModels, checks: ["model picker", "both seats", "Q-learning", "neural OSLA", "native/neural MCTS", "scalar/vector models", "complete worker games"] };
}
