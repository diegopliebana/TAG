// Graphical acceptance exercises the actual app, module worker and SVG controls.
import { createGame } from "./human-play.js";
const ui = id => document.getElementById(id);
function check(value, message) { if (!value) throw new Error(message); }
async function until(predicate, message) {
  for (let i = 0; i < 400; i++) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 25)); }
  throw new Error(`${message}: ${ui("status").textContent}`);
}
async function action(label, count) {
  const button = [...ui("board").querySelectorAll("button")].find(b => b.textContent.startsWith(label));
  check(button && !button.disabled, `Unavailable ${label} control`); button.focus(); button.click();
  await until(() => ui("details").textContent.includes(`| ${count} actions |`) && !ui("reset").disabled && ui("board").querySelector("button:enabled"), `${label} did not finish`);
}
export async function runCantStopChecks() {
  await window.gameReady;
  ui("mute").checked = true; ui("mute").dispatchEvent(new Event("change"));
  ui("reduced").checked = true; ui("reduced").dispatchEvent(new Event("change"));
  const counts = [];
  for (const n of [2, 3, 4]) {
    await createGame("cantstop", { seed: "18446744073709551615" }, Array.from({ length: n }, () => ({ kind: "human" })));
    check(ui("players").children.length === n, "Player cards do not match selected count");
    check(ui("board").querySelectorAll("[data-column]").length === 11, "Missing numbered tracks");
    check(ui("details").textContent.includes("18446744073709551615"), "Dice seed lost precision");
    await action("Roll", 1); await action("Advance", 2); await action("Stop", 3);
    check(ui("result").textContent.includes("Player 1"), "Stop did not finish the turn");
    await action("Roll", 4);
    ui("previous").click();
    await until(() => ui("status").textContent.includes("Reviewing move 3"), "History did not open");
    check(!ui("board").querySelector("button:enabled"), "Historical game accepted input");
    ui("fork").click(); await until(() => ui("fork-dialog").open, "Fork dialog did not open");
    check(ui("fork-seats").children.length === n, "Fork lost multiplayer settings");
    ui("fork-form").querySelector('[type="submit"]').click();
    await until(() => !ui("fork-dialog").open && ui("board").querySelector("button:enabled"), "Fork did not resume");
    ui("theme").value = "light"; ui("theme").dispatchEvent(new Event("change"));
    check(ui("board").querySelector("svg").getBoundingClientRect().width > 0, "SVG has no visible size");
    ui("theme").value = "dark"; ui("theme").dispatchEvent(new Event("change"));
    ui("board-display").value = "scrollable"; ui("board-display").dispatchEvent(new Event("change"));
    ui("board-zoom").value = "125"; ui("board-zoom").dispatchEvent(new Event("input"));
    ui("board-display").value = "fit"; ui("board-display").dispatchEvent(new Event("change"));
    counts.push(n);
  }
  await createGame("cantstop", { seed: "42", rules: { column_heights: Array(11).fill(3), columns_to_win: 1 } },
    [{ kind: "sampled_osla", seed: "7", repetitions_per_action: 2 }, { kind: "sampled_osla", seed: "8", repetitions_per_action: 2 }, { kind: "random", seed: "9" }]);
  ui("ai-delay").value = "0"; ui("ai-delay").dispatchEvent(new Event("change"));
  ui("ai-step").click(); await until(() => ui("details").textContent.includes("| 1 actions |") && !ui("ai-step").disabled, "AI step did not settle");
  ui("play-pause").click(); await until(() => ui("result").textContent.includes("wins"), "AI match did not finish");
  check(ui("board").querySelectorAll(".cantstop-actions button").length === 0, "Terminal actions remained");
  // Finish with a default four-player scene for a visual review/screenshot.
  await createGame("cantstop", { seed: "42" }, Array.from({ length: 4 }, () => ({ kind: "human" })));
  await action("Roll", 1); await action("Advance", 2);
  return { passed: true, playerCounts: counts, phases: ["start_turn", "allocation", "decision"], history: true, fork: true, aiMatch: true };
}
