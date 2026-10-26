// Exercise actual file inputs, downloads, candidate workers and replacement dialogs.
import { createGame } from "./human-play.js";
const ui = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(value, message) { if (!value) throw new Error(message); }
async function until(predicate, message) {
  for (let i = 0; i < 1200; i++) { if (predicate()) return; await sleep(5); }
  throw new Error(`${message}; status=${ui("status").textContent}`);
}
const cursor = () => Number(ui("review-position").textContent.match(/Move (\d+) of/)?.[1]);
const position = () => JSON.stringify({ cells: [...ui("board").querySelectorAll("tbody [aria-label]")].map(cell => cell.getAttribute("aria-label")), result: ui("result").textContent, details: ui("details").textContent });
async function jump(value) { ui("history-moves").querySelector(`[data-position="${value}"]`).click(); await until(() => cursor() === value && !ui("reset").disabled, "File-test navigation failed"); }
async function step() { const before = cursor(); ui("ai-step").click(); await until(() => cursor() === before + 1 && (!ui("ai-step").disabled || ui("board").querySelector("button:enabled") || ui("result").textContent.includes("wins") || ui("result").textContent === "Draw"), "File-test AI step failed"); }
function upload(text, filename = "test.tag-replay.json") {
  const transfer = new DataTransfer(); transfer.items.add(new File([text], filename, { type: "application/json" }));
  ui("file-input").files = transfer.files; ui("file-input").dispatchEvent(new Event("change"));
}
async function dismiss(value) {
  const dialog = ui("discard-dialog");
  await new Promise(resolve => { dialog.addEventListener("close", resolve, { once: true }); dialog.querySelector(`[value="${value}"]`).click(); });
}
async function load(text, { controllers, choice = "discard" } = {}) {
  upload(text);
  if (controllers) {
    await until(() => ui("state-setup").open, "State controller selection did not open");
    controllers.forEach((controller, seat) => { ui(`state-controller-${seat}`).value = controller.kind; ui(`state-controller-${seat}`).dispatchEvent(new Event("change")); if (controller.seed) ui(`state-seed-${seat}`).value = controller.seed; });
    ui("state-form").requestSubmit();
  }
  await until(() => ui("discard-dialog").open || (!ui("file-progress").open && !ui("reset").disabled), "File did not validate or reach replacement");
  if (ui("discard-dialog").open) await dismiss(choice);
  await until(() => !ui("file-progress").open && !ui("reset").disabled, "File load did not settle");
}
export async function runFileChecks() {
  const downloads = []; const originalUrl = URL.createObjectURL;
  URL.createObjectURL = function(blob) { downloads.push(blob); return originalUrl.call(URL, blob); };
  async function save(scope) {
    const count = downloads.length; ui(`save-${scope}`).click();
    await until(() => downloads.length === count + 1 && !ui("reset").disabled, "File download did not settle");
    return downloads.at(-1).text();
  }
  let historicalPositions = 0;
  try {
    for (const [game, policy] of [["tictactoe", "osla"], ["connect4", "osla"], ["tictactoe", "random"], ["connect4", "random"]]) {
      await createGame(game, undefined, [{ kind: policy, seed: "7" }, { kind: policy, seed: "18446744073709551615" }]);
      ui("reduced").checked = true; ui("mute").checked = true;
      for (let i = 0; i < 4; i++) await step();
      await jump(2); ui("fork").click(); await until(() => ui("fork-dialog").open, "File-test fork did not open");
      ui("fork-controller-0").value = "human"; ui("fork-controller-0").dispatchEvent(new Event("change")); ui("fork-create").click();
      await until(() => !ui("fork-dialog").open && !ui("reset").disabled, "File-test fork did not settle");
      ui("board").querySelector("button:enabled").click(); await until(() => cursor() === 3 && !ui("ai-step").disabled, "File-test human move did not settle");
      const sources = []; for (let i = 0; i <= 3; i++) { await jump(i); sources.push(position()); }
      await jump(1); ui("ai-delay").value = "23"; ui("ai-delay").dispatchEvent(new Event("change"));
      const session = await save("session"), branch = await save("branch"), state = await save("state");
      check(JSON.parse(session).branches.length === 2, "Whole session lost a branch");
      check(JSON.parse(branch).branches.length === 1 && JSON.parse(branch).branches[0].parent === null, "Branch file retained fork relations");
      check(JSON.parse(state).kind === "tag-state" && !JSON.parse(state).branches, "State export retained history");
      await jump(3); await step(); const next = position();
      for (const text of [session, branch]) {
        await load(text); check(cursor() === 1 && ui("play-pause").textContent === "Play recording" && ui("ai-delay").value === "23", "Replay selection/delay/paused state not restored");
        for (let i = 0; i <= 3; i++) { await jump(i); check(position() === sources[i], `Loaded history differs at ${game} ${i}`); historicalPositions++; }
        await step(); check(position() === next, "Loaded controller continuation differs");
      }
      await load(state, { controllers: [{ kind: "human" }, { kind: policy === "random" ? "random" : "first_available", seed: "9007199254740993" }] });
      check(cursor() === 0 && position() === sources[1] && ui("players").textContent.includes("9007199254740993") && ui("play-pause").textContent === "Play live", "State root/seed/paused setup differs");
      ui("reset").click(); await until(() => !ui("reset").disabled && ui("branch").options.length === 2, "Loaded-state reset failed"); check(position() === sources[1], "Reset did not return to loaded root");
      const before = position();
      await load(session, { choice: "cancel" }); check(position() === before && cursor() === 0, "Replacement cancel lost current state");
      upload("{broken"); await until(() => !ui("file-progress").open && !ui("reset").disabled && ui("status").textContent !== "Loading and validating file...", "Invalid load did not settle"); check(position() === before, "Invalid file lost session");
      upload(session); ui("file-cancel").click(); await until(() => !ui("reset").disabled, "Validation cancel did not settle"); check(position() === before, "Validation cancel lost session");
      const downloadCount = downloads.length;
      await load(session, { choice: "save" });
      check(downloads.length === downloadCount + 1 && cursor() === 1, "Save and continue did not download the old session before replacement");
      // Native-generated mid-game and terminal state files use the same path.
      for (const suffix of ["", "-finished"]) {
        const text = await (await fetch(`/examples/saves/${game}${suffix}.tag-state.json`)).text();
        await load(text, { controllers: [{ kind: "human" }, { kind: "human" }] });
        check(cursor() === 0 && (suffix ? ui("result").textContent.includes("wins") : ui("details").textContent.includes("| 3 moves |")), "Native state import differs");
      }
    }
    // Save during an AI delay must preserve the unconsumed initial controller checkpoint.
    await createGame("tictactoe", undefined, [{ kind: "random", seed: "7" }, { kind: "human" }]);
    ui("ai-delay").value = "5000"; ui("ai-delay").dispatchEvent(new Event("change")); ui("play-pause").click();
    await until(() => ui("status").textContent.includes("AI delay"), "File-test AI delay did not start");
    const delayed = JSON.parse(await save("session"));
    check(delayed.branches[0].moves.length === 0 && delayed.branches[0].controllers[0].rng.word_position === "0" && ui("play-pause").textContent === "Play live", "Save consumed a pending AI choice");
    // Select a recorded match from the native batch and review its entire line.
    const batch = await (await fetch("/.scratch/browser-proof/workload.json")).json();
    for (const data of [batch[1], batch[4], batch[6], batch[7]]) {
      await load(data.replay);
      const positions = [data.initial, ...data.decisions.map(move => move.board)];
      for (let i = 0; i < positions.length; i++) {
        await jump(i);
        check(ui("details").textContent.includes(`| ${positions[i].counters[0]} moves |`), "Native match counters differ");
        const cells = [...ui("board").querySelectorAll("tbody [aria-label]")];
        for (const cell of cells) {
          const x = Number(cell.dataset.x), y = Number(cell.dataset.y); const value = positions[i].cells[x * ui("board").querySelector("tbody").rows.length + y];
          check(cell.getAttribute("aria-label").endsWith(value === null ? "empty" : data.game === "tictactoe" ? value === 0 ? "X" : "O" : `P${value}`), "Native match board differs");
        }
        historicalPositions++;
      }
    }
    await createGame("tictactoe");
    return { downloads: downloads.length, historicalPositions, checks: ["three save scopes", "historical state", "controller boundaries", "exact continuation", "paused imports", "wide seeds", "loaded-root reset", "invalid/cancelled load", "save and continue", "save during AI delay", "native state fixtures", "selected native batch matches"] };
  } finally { URL.createObjectURL = originalUrl; }
}
