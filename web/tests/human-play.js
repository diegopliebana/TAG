// Run against the real main page, using its controls and module worker.
import { scenes } from "../presentation/scenes.js";
const ui = id => document.getElementById(id);
function check(value, message) { if (!value) throw new Error(message); }
async function until(predicate, message) {
  for (let i = 0; i < 200; i++) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(message);
}
function chooseGame(game) { ui("game").value = game; ui("game").dispatchEvent(new Event("change")); }
function setParameters(parameters) {
  for (const input of ui("setup-parameters").querySelectorAll("input")) input.value = parameters[input.name];
}
async function respondDiscard(value) {
  const dialog = ui("discard-dialog");
  await new Promise(resolve => {
    dialog.addEventListener("close", resolve, { once: true });
    dialog.querySelector(`[value="${value}"]`).click();
  });
}
export async function createGame(game, parameters, controllers = [{ kind: "human" }, { kind: "human" }], pauseAi = true) {
  if (!ui("setup").open) ui("new-game").click();
  await until(() => ui("setup").open && !ui("create").disabled, "Setup did not open");
  chooseGame(game);
  if (["cantstop","loveletter"].includes(game)) {
    const fields = ui("setup-parameters");
    const players = fields.querySelector('[name="players"]'); players.value = controllers.length; players.dispatchEvent(new Event("change"));
    for (const input of fields.querySelectorAll("input")) {
      if (input.name === "seed") input.value = parameters?.seed ?? "42";
      else if (input.name === "first_player") input.value = parameters?.first_player ?? 0;
      else if (input.name.startsWith("target_") && parameters?.rules?.winning_tokens) input.value = parameters.rules.winning_tokens[Number(input.name.slice(7))-2];
      else if (input.name.startsWith("column_") && parameters?.rules?.column_heights) input.value = parameters.rules.column_heights[Number(input.name.slice(7)) - 2];
      else if (parameters?.rules?.[input.name] !== undefined) input.value = parameters.rules[input.name];
    }
  } else if (parameters) setParameters(parameters);
  controllers.forEach((controller, seat) => {
    ui(`controller-${seat}`).value = controller.kind; ui(`controller-${seat}`).dispatchEvent(new Event("change"));
    if (controller.seed !== undefined) ui(`seed-${seat}`).value = controller.seed;
    if (controller.repetitions_per_action !== undefined) ui(`repetitions-${seat}`).value = controller.repetitions_per_action;
  });
  const hasAi = controllers.some(controller => controller.kind !== "human");
  const previousDelay = ui("ai-delay").value;
  // Manual-control tests explicitly pause after creation, before the first commitment.
  if (pauseAi && hasAi) { ui("ai-delay").value = "5000"; ui("ai-delay").dispatchEvent(new Event("change")); }
  ui("create").click();
  await until(() => ui("discard-dialog").open || !ui("setup").open, "Setup did not submit");
  if (ui("discard-dialog").open) await respondDiscard("discard");
  await until(() => !ui("setup").open && !ui("reset").disabled, "Game did not start");
  if (pauseAi && hasAi) {
    if (ui("play-pause").textContent === "Pause live") ui("play-pause").click();
    await until(() => ui("play-pause").textContent === "Play live", "Initial test pause did not settle");
    ui("ai-delay").value = previousDelay; ui("ai-delay").dispatchEvent(new Event("change"));
  }
}
async function move(action, count) {
  const button = "column" in action ? ui("board").querySelectorAll("thead button")[action.column] : ui("board").querySelector(`[data-x="${action.x}"][data-y="${action.y}"]`);
  check(button && !button.disabled, "Expected legal input was disabled"); button.click();
  await until(() => ui("details").textContent.includes(`| ${count} moves |`) && !ui("reset").disabled, "Move was not displayed");
  await until(() => ui("board").getAnimations({ subtree: true }).length === 0 && (ui("result").textContent.includes("wins") || ui("result").textContent === "Draw" || ui("board").querySelector("button:enabled")), "Move feedback did not finish");
}
export async function runHumanChecks() {
  await window.gameReady;
  check(ui("controller-0").value === "human" && ui("controller-1").value === "osla", "Default controllers failed");
  check(/^\d+$/.test(ui("seed-0").value) && ui("seed-0").value !== ui("seed-1").value, "Time seed defaults failed");
  for (let seat = 0; seat < 2; seat++) { ui(`controller-${seat}`).value = "human"; ui(`controller-${seat}`).dispatchEvent(new Event("change")); }
  check(ui("setup").open && document.body.dataset.theme === "dark", "Initial setup/theme failed");
  check(ui("setup-parameters").querySelector("input").value === "3", "Tic-Tac-Toe defaults failed");
  // Closing the parameter section must retain edits.
  ui("parameter-section").open = true; setParameters({ grid_size: 5 }); ui("parameter-section").open = false;
  ui("create").click(); await until(() => !ui("setup").open, "Configured setup failed");
  check(ui("board").querySelectorAll("tbody button").length === 25, "Collapsed parameters lost edits");
  const table = ui("board").querySelector("table");
  const button = ui("board").querySelector("button:enabled"); button.focus(); button.click(); button.click();
  await until(() => ui("details").textContent.includes("| 1 moves |") && ui("board").getAnimations({ subtree: true }).length > 0, "Placement did not animate");
  check(ui("board").querySelector("table") === table && !ui("board").querySelector("button:enabled"), "Duplicate-input lock failed");
  ui("reset").click(); await until(() => ui("details").textContent.includes("| 0 moves |") && !ui("reset").disabled, "Reset during animation failed");
  await new Promise(resolve => setTimeout(resolve, 550));
  check(ui("details").textContent.includes("| 0 moves |") && ui("board").getAnimations({ subtree: true }).length === 0, "Old feedback changed reset game");
  // Reset retains a played continuation, so replacement still requires consent.
  ui("new-game").click(); await until(() => ui("setup").open && !ui("create").disabled, "Setup did not open after reset");
  ui("create").click(); await until(() => ui("discard-dialog").open, "Missing discard prompt after reset");
  await respondDiscard("cancel"); await until(() => !ui("create").disabled, "Cancel did not settle setup");
  check(ui("setup").open && ui("board").querySelector("table") === table, "Cancel replaced the session");
  // Invalid setup preserves the current table and stays in the dialog.
  setParameters({ grid_size: 65 }); ui("setup-form").dispatchEvent(new Event("submit", { cancelable: true }));
  check(ui("setup-error").textContent.includes("4,096") && ui("board").querySelector("table") === table, "Invalid setup was not atomic");
  ui("defaults").click(); check(ui("setup-parameters").querySelector("input").value === "3", "Use defaults failed");
  ui("setup-cancel").click();
  ui("reduced").checked = true; ui("reduced").dispatchEvent(new Event("change"));
  ui("mute").checked = true; ui("mute").dispatchEvent(new Event("change"));
  ui("theme").value = "light"; ui("theme").dispatchEvent(new Event("change"));
  check(document.body.dataset.theme === "light", "Light theme failed");
  for (const scene of scenes) {
    await createGame(scene.game, scene.parameters);
    const savedTable = ui("board").querySelector("table");
    for (let i = 0; i < scene.moves.length; i++) await move(scene.moves[i], i + 1);
    check(ui("board").querySelector("table") === savedTable, "Board controls were rebuilt during play");
    check(scene.outcome === "won" ? ui("result").textContent.includes("wins") : ui("result").textContent === "Draw", `Incorrect outcome: ${scene.id}; result=${ui("result").textContent}; details=${ui("details").textContent}; status=${ui("status").textContent}; cells=${JSON.stringify([...ui("board").querySelectorAll("tbody [aria-label]")].map(cell => cell.getAttribute("aria-label")))}`);
    check(!ui("board").querySelector("button:enabled"), "Terminal board accepts input");
    check(scene.outcome === "won" ? ui("board").querySelectorAll('[data-winning="true"]').length > 0 : !ui("board").querySelector('[data-winning="true"]'), "Winning emphasis failed");
    check(ui("board").getAnimations({ subtree: true }).length === 0, "Reduced motion animated");
    await Promise.all([...ui("board").querySelectorAll("img")].filter(image => !image.hidden).map(image => image.decode()));
  }
  // A full column is unavailable before the game ends.
  await createGame("connect4", { width: 4, height: 3, winning_length: 4 });
  for (let i = 0; i < 3; i++) await move({ column: 0 }, i + 1);
  check(ui("board").querySelector("thead button").disabled, "Full column remained available");
  await createGame("connect4", { width: 128, height: 32, winning_length: 4 });
  check(ui("board").querySelectorAll("tbody td").length === 4096, "Maximum configured board failed");
  await createGame("tictactoe");
  ui("theme").value = "dark"; ui("theme").dispatchEvent(new Event("change"));
  return { samples: scenes.length, checks: ["setup", "retained edits", "defaults", "invalid setup", "discard/cancel", "reset during animation", "duplicate input", "terminal input", "winning cells", "images", "themes", "reduced motion", "full column", "4096-cell board"] };
}
