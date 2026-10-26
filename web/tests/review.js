import { createGame } from "./human-play.js";
import { scenes } from "../presentation/scenes.js";
const ui = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(value, message) { if (!value) throw new Error(message); }
async function until(predicate, message) {
  for (let i = 0; i < 600; i++) { if (predicate()) return; await sleep(5); }
  throw new Error(`${message}; ${ui("status").textContent}; ${ui("review-position").textContent}`);
}
const cursor = () => Number(ui("review-position").textContent.match(/Move (\d+) of/)?.[1]);
const tip = () => Number(ui("review-position").textContent.match(/of (\d+)/)?.[1]);
const cells = () => [...ui("board").querySelectorAll("tbody [aria-label]")].map(cell => cell.getAttribute("aria-label"));
const settled = () => !ui("reset").disabled;
const snapshot = () => JSON.stringify({ cells: cells(), result: ui("result").textContent, details: ui("details").textContent });
async function jump(position) {
  ui("history-moves").querySelector(`[data-position="${position}"]`).click();
  await until(() => cursor() === position && settled(), "History jump did not settle");
}
async function select(branch) {
  ui("branch").value = branch; ui("branch").dispatchEvent(new Event("change"));
  await until(() => Number(ui("branch").value) === branch && settled(), "Branch switch did not settle");
}
async function move(action, count) {
  const button = "column" in action ? ui("board").querySelectorAll("thead button")[action.column] : ui("board").querySelector(`[data-x="${action.x}"][data-y="${action.y}"]`);
  check(button && !button.disabled, "Recorded test action unavailable"); button.click();
  await until(() => cursor() === count && settled() && (ui("result").textContent.includes("wins") || ui("result").textContent === "Draw" || ui("board").querySelector("button:enabled")), "Human move did not settle");
}
async function openFork() { ui("fork").click(); await until(() => ui("fork-dialog").open && !ui("fork-create").disabled, "Fork dialog did not open"); }
async function createFork() { const count = ui("branch").options.length; ui("fork-create").click(); await until(() => !ui("fork-dialog").open && ui("branch").options.length === count + 1 && settled(), "Fork did not settle"); }
async function step() {
  await until(() => !ui("ai-step").disabled, "AI step unavailable"); const before = cursor(); ui("ai-step").click();
  await until(() => cursor() === before + 1 && (ui("result").textContent.includes("wins") || !ui("ai-step").disabled || ui("board").querySelector("button:enabled")), "AI step did not settle");
}
function delay(value) { ui("ai-delay").value = value; ui("ai-delay").dispatchEvent(new Event("change")); }
export async function runReviewChecks() {
  ui("mute").checked = true; ui("mute").dispatchEvent(new Event("change"));
  ui("reduced").checked = true; ui("reduced").dispatchEvent(new Event("change")); delay(0);
  let samples = 0;
  for (const scene of scenes) {
    await createGame(scene.game, scene.parameters);
    const positions = [snapshot()];
    for (let index = 0; index < scene.moves.length; index++) { await move(scene.moves[index], index + 1); positions.push(snapshot()); }
    check(ui("history-moves").children.length === positions.length, "Accepted move list incomplete");
    for (let position = 0; position < positions.length; position++) {
      await jump(position); check(snapshot() === positions[position], `Historical facts differ: ${scene.id} ${position}`);
      if (position < positions.length - 1) check(!ui("board").querySelector("button:enabled") && ui("ai-step").disabled, "History accepts live input");
    }
    await jump(0); ui("next").click(); await until(() => cursor() === 1 && settled(), "Next did not navigate");
    ui("previous").click(); await until(() => cursor() === 0 && settled(), "Previous did not navigate");
    ui("play-pause").click(); await until(() => cursor() === scene.moves.length && ui("play-pause").textContent === "Play live", "Recording did not pause at its end");
    check(tip() === scene.moves.length, "Playback generated a move");
    // Terminal forks stay terminal and preserve the original continuation.
    await openFork(); await createFork(); check(!ui("board").querySelector("button:enabled"), "Terminal fork became playable");
    await select(0); check(snapshot() === positions.at(-1), "Original terminal continuation changed"); samples++;
  }
  // Preserve an AI's exact historical progress, then replace a seat inside an inherited prefix.
  for (const game of ["tictactoe", "connect4"]) {
    await createGame(game, undefined, [{ kind: "osla", seed: "7" }, { kind: "osla", seed: "8" }]);
    for (let index = 0; index < 4; index++) await step();
    const fourth = snapshot(); await jump(2); await openFork();
    check(ui("fork-controller-0").value === "osla" && ui("fork-seed-0").value === "7", "Fork defaults lost historical settings");
    await createFork(); check(ui("play-pause").textContent === "Play live" && cursor() === 2, "Fork did not start paused");
    await step(); await step(); check(snapshot() === fourth, "Preserved AI continuation differs");
    await jump(2); await openFork(); ui("fork-controller-0").value = "human"; ui("fork-controller-0").dispatchEvent(new Event("change"));
    await createFork(); check(ui("board").querySelector("button:enabled") && ui("players").textContent.includes("Human"), "Human takeover failed");
    ui("board").querySelector("button:enabled").click(); await until(() => cursor() === 3 && !ui("ai-step").disabled, "Fork human move failed");
    await jump(1); check(!ui("players").textContent.includes("Human"), "Inherited controller identity changed early");
    // Fork before the parent's fork boundary, taking over the other seat.
    await openFork(); ui("fork-controller-1").value = "human"; ui("fork-controller-1").dispatchEvent(new Event("change"));
    await createFork(); check(ui("board").querySelector("button:enabled") && ui("result").textContent.includes("Player 1"), "Player 1 inherited-prefix takeover failed");
    ui("board").querySelector("button:enabled").click(); await until(() => cursor() === 2 && !ui("ai-step").disabled, "Player 1 takeover move did not settle");
    await select(2);
    await jump(2); check(ui("players").textContent.includes("Human"), "Replacement missing at fork boundary");
    await select(0); check(snapshot() === fourth, "Original branch changed after takeover");
    // Replay ending at a live AI position must stop before generating another decision.
    await jump(0); ui("play-pause").click(); await until(() => cursor() === 4 && ui("play-pause").textContent === "Play live", "Live-end playback did not pause");
    await sleep(60); check(tip() === 4, "Recording continued into live AI");
    // Queue multiple requests; the last selected cursor wins.
    ui("history-moves").querySelector('[data-position="1"]').click(); ui("history-moves").querySelector('[data-position="3"]').click(); ui("history-moves").querySelector('[data-position="2"]').click();
    await until(() => cursor() === 2 && settled(), "Latest history jump lost");
    const listButton = ui("history-moves").querySelector('[data-position="2"]'); listButton.focus();
    await jump(3); check(document.activeElement === listButton, "History controls lost keyboard focus");
    for (const branch of [1, 0, 2]) { ui("branch").value = branch; ui("branch").dispatchEvent(new Event("change")); }
    await until(() => Number(ui("branch").value) === 2 && cursor() === 3 && settled(), "Latest branch switch lost");
    await select(2); check(tip() === 3 && ui("board").querySelector("button:enabled") === null && !ui("ai-step").disabled, "Nested branch tip restoration failed");
    await openFork(); ui("fork-controller-1").value = "osla"; ui("fork-controller-1").dispatchEvent(new Event("change")); ui("fork-seed-1").value = "18446744073709551616";
    const branchCount = ui("branch").options.length; ui("fork-create").click();
    check(ui("fork-error").textContent.includes("Seed must") && ui("branch").options.length === branchCount, "Invalid fork mutated collection");
    ui("fork-seed-1").value = "9007199254740993"; await createFork(); check(ui("players").textContent.includes("9007199254740993"), "Fork seed rounded");
    // Reset remains a preserved branch, then restore the original.
    ui("reset").click(); await until(() => cursor() === 0 && settled(), "Reset branch did not settle");
    check(ui("branch").selectedOptions[0].textContent.includes("Reset"), "Reset continuation label missing");
    await select(0); check(snapshot() === fourth, "Reset lost original continuation");
    delay(5000); ui("play-pause").click(); await until(() => ui("status").textContent.includes("AI delay"), "Navigation AI delay did not start");
    await jump(0); await sleep(60); check(tip() === 4 && cursor() === 0, "Pending AI advanced after navigation");
    delay(0);
  }
  // Measure end-to-end UI browsing on a larger configured public board.
  ui("board-display").value = "scrollable"; ui("board-display").dispatchEvent(new Event("change"));
  await createGame("tictactoe", { grid_size: 32 });
  for (let index = 0; index < 31; index++) await move({ x: Math.floor(index / 2), y: index % 2 }, index + 1);
  const milliseconds = [];
  for (const position of [0, 10, 20, 31, 10, 0, 31, 20]) { const start = performance.now(); await jump(position); milliseconds.push(performance.now() - start); }
  check(ui("board").scrollWidth > ui("board").clientWidth, "Large board did not scroll locally");
  await checkViewportControls();
  await createGame("tictactoe");
  return { samples, navigationMeasurement: { board: "32x32", moves: 31, milliseconds }, checks: ["historical facts", "read-only input", "previous/next", "recording end", "terminal forks", "exact AI continuation", "both-seat takeover", "nested inherited-prefix forks", "branch preservation", "rapid jumps and branch switches", "retained focus", "seed replacement", "invalid fork", "reset branches", "pending AI cancellation", "large-board navigation", "sticky controls", "independent board scrolling", "settings menu"] };
}

async function checkViewportControls() {
  check(ui("board").scrollHeight > ui("board").clientHeight, "Large board did not scroll vertically inside its viewport");
  document.querySelector(".toolbar").scrollIntoView();
  await sleep(30);
  const pagePosition = window.scrollY;
  ui("board").scrollTop = 500; ui("board").scrollLeft = 200;
  await sleep(30);
  const toolbarBounds = document.querySelector(".toolbar").getBoundingClientRect();
  check(window.scrollY === pagePosition && ui("board").scrollTop > 0 && ui("board").scrollLeft > 0, "Board scrolling moved the page");
  const column = ui("board").querySelector("thead button");
  if (column) check(column.getBoundingClientRect().top >= ui("board").getBoundingClientRect().top - 1, "Scrolled column controls are clipped");
  check(toolbarBounds.top >= -1 && toolbarBounds.bottom <= innerHeight, "Game controls left the viewport");
  check(document.querySelector(".toolbar").contains(ui("status")), "Game status is outside the sticky controls");
  const settings = document.querySelector(".presentation-settings"); settings.open = true;
  await sleep(30);
  const settingsBounds = settings.querySelector(".settings-menu").getBoundingClientRect();
  check(settingsBounds.height > 0 && settingsBounds.left >= 0 && settingsBounds.right <= innerWidth && settingsBounds.bottom <= innerHeight, "Settings menu left the viewport");
  ui("theme").focus(); check(document.activeElement === ui("theme"), "Settings controls cannot receive focus"); settings.open = false;
  const files = document.querySelector(".file-menu"); files.open = true;
  const fileBounds = files.querySelector(".settings-menu").getBoundingClientRect();
  const fileButton = files.querySelector("summary").getBoundingClientRect();
  check(fileBounds.height > 0 && Math.abs(fileBounds.right - fileButton.right) < 1 && Math.abs(fileBounds.top - fileButton.bottom - 8) < 1, "Files dropdown is not anchored to its button");
  check(fileBounds.left >= 0 && fileBounds.right <= innerWidth && fileBounds.bottom <= innerHeight, "Files menu left the viewport"); files.open = false;
}

export async function runLayoutChecks() {
  for (const [game, parameters] of [["tictactoe", { grid_size: 32 }], ["connect4", { width: 32, height: 32, winning_length: 4 }]]) {
    await createGame(game, parameters);
    ui("board-display").value = "scrollable"; ui("board-display").dispatchEvent(new Event("change"));
    await checkViewportControls();
    const button = ui("board").querySelector("button");
    const position = snapshot();
    const cell = () => ui("board").querySelector("tbody td").getBoundingClientRect().width;
    const normalSize = cell();
    document.querySelector(".presentation-settings").open = true;
    ui("board-zoom").focus();
    for (const percent of [25, 200]) {
      ui("board-zoom").value = percent; ui("board-zoom").dispatchEvent(new Event("input"));
      await sleep(30);
      check(Math.abs(cell() - normalSize * percent / 100) < 1, "Zoom did not resize board cells");
      check(snapshot() === position, "Zoom changed game state");
      check(ui("board").querySelector("button") === button, "Zoom replaced board input controls");
      check(document.activeElement === ui("board-zoom"), `Zoom lost focus: ${document.activeElement?.id}`);
    }
    ui("board-zoom-reset").click();
    check(ui("board-zoom").value === "100" && Math.abs(cell() - normalSize) < 1, "Zoom reset failed");
    ui("board-display").value = "fit"; ui("board-display").dispatchEvent(new Event("change"));
    await sleep(60);
    check(ui("board").scrollWidth <= ui("board").clientWidth + 1 && ui("board").scrollHeight <= ui("board").clientHeight + 1, "Fit board exceeds its panel viewport");
    check(ui("board").scrollTop === 0 && ui("board").scrollLeft === 0 && ui("board").querySelector("button") === button, "Display mode replaced input controls or retained scroll offset");
    check(ui("board-zoom").disabled && ui("board-zoom-reset").disabled, "Fit allows conflicting manual zoom");
    document.querySelector(".presentation-settings").open = false;
    check(document.documentElement.scrollWidth <= innerWidth, "Large board overflows the page");
  }
  await createGame("connect4");
  await sleep(60);
  check(ui("board").scrollWidth <= ui("board").clientWidth + 1 && ui("board").scrollHeight <= ui("board").clientHeight + 1, "Default Connect 4 does not fit");
  await createGame("tictactoe");
  return { viewport: [innerWidth, innerHeight], games: 3, checks: ["sticky status", "scrollable boards", "fit boards", "retained controls", "default Connect 4", "zoom range/reset", "zoom preserves state and focus"] };
}
