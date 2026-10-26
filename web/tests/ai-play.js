import { createGame } from "./human-play.js";
const ui = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(value, message) { if (!value) throw new Error(message); }
async function until(predicate, message) {
  for (let i = 0; i < 400; i++) { if (predicate()) return; await sleep(10); }
  throw new Error(`${message}: ${ui("status").textContent}; ${ui("details").textContent}`);
}
const moves = () => Number(ui("details").textContent.match(/\| (\d+) moves \|/)?.[1]);
const cells = () => [...ui("board").querySelectorAll("tbody [aria-label]")].map(cell => cell.getAttribute("aria-label"));
const finished = () => ui("result").textContent.includes("wins") || ui("result").textContent === "Draw";
function delay(value) { ui("ai-delay").value = value; ui("ai-delay").dispatchEvent(new Event("change")); }
async function reset() { ui("reset").click(); await until(() => moves() === 0 && !ui("reset").disabled && ui("play-pause").textContent === "Play live", "Reset did not settle"); }
async function step() {
  await until(() => !ui("ai-step").disabled, "AI step unavailable"); const before = moves();
  ui("ai-step").click(); await until(() => moves() === before + 1 && (finished() || !ui("ai-step").disabled || ui("board").querySelector("button:enabled")), "AI step did not finish");
  return cells();
}
export async function runAiChecks() {
  ui("mute").checked = true; ui("mute").dispatchEvent(new Event("change"));
  ui("reduced").checked = true; ui("reduced").dispatchEvent(new Event("change"));
  const max = "18446744073709551615";
  // Manual and automatic responses, including remembered Play on human turns.
  await createGame("tictactoe", undefined, [{ kind: "human" }, { kind: "osla", seed: max }], false);
  check(ui("players").textContent.includes(max), "Full-width seed was rounded");
  check(ui("play-pause").textContent === "Pause live", "New game did not enable AI play");
  ui("board").querySelector("button:enabled").click();
  await until(() => moves() === 2 && ui("board").querySelector("button:enabled"), "AI did not respond automatically after creation");
  ui("play-pause").click(); await reset();
  ui("board").querySelector("button:enabled").click(); await until(() => !ui("ai-step").disabled, "Human-to-AI handover failed");
  check(moves() === 1 && ui("play-pause").textContent === "Play live", "Reset pause failed");
  await step(); await sleep(80); check(moves() === 2, "AI step advanced multiple moves");
  delay(0); ui("play-pause").click(); ui("board").querySelector("button:enabled").click();
  await until(() => moves() === 4 && ui("board").querySelector("button:enabled"), "Play was not remembered across human turn");
  check(ui("play-pause").textContent === "Pause live", "Automatic intent was lost"); ui("play-pause").click();
  await reset(); check(ui("players").textContent.includes(max), "Reset changed initial seed");
  // Cancellation during the global delay must preserve the next seeded decision.
  await createGame("connect4", undefined, [{ kind: "osla", seed: "7" }, { kind: "first_available", seed: "0" }]);
  delay(5000); ui("play-pause").click(); await until(() => ui("status").textContent.includes("AI delay"), "Delay did not start");
  ui("play-pause").click(); await until(() => !ui("ai-step").disabled, "Pause did not discard proposal");
  check(moves() === 0, "Pause committed a pending move"); const afterCancel = await step();
  await reset(); check(JSON.stringify(await step()) === JSON.stringify(afterCancel), "Cancelled work advanced agent randomness");
  await reset(); ui("play-pause").click(); await until(() => ui("status").textContent.includes("AI delay"), "Reset delay did not start");
  await reset(); await sleep(80); check(moves() === 0, "Scheduled AI changed reset game");
  ui("play-pause").click(); await until(() => ui("status").textContent.includes("AI delay"), "New-game delay did not start");
  ui("new-game").click(); await until(() => ui("setup").open && !ui("create").disabled, "New game failed to cancel AI");
  const oldSeed = ui("seed-0").value; ui("regenerate-0").click(); check(ui("seed-0").value !== oldSeed, "Regenerate repeated its seed");
  // Invalid seeds and OSLA board limits leave the old session intact.
  const oldTable = ui("board").querySelector("table");
  ui("seed-0").value = "18446744073709551616"; ui("create").click();
  check(ui("setup-error").textContent.includes("Seed must") && ui("board").querySelector("table") === oldTable, "Invalid seed replaced session");
  ui("seed-0").value = "7"; ui("game").value = "tictactoe"; ui("game").dispatchEvent(new Event("change"));
  ui("setup-parameters").querySelector("input").value = "17"; ui("create").click();
  check(ui("setup-error").textContent.includes("256") && ui("board").querySelector("table") === oldTable, "OSLA limit was not atomic");
  ui("setup-cancel").click(); await sleep(30); check(moves() === 0, "Cancelled setup resumed AI");
  delay(0);
  let matches = 0;
  for (const [game, parameters] of [["tictactoe", undefined], ["connect4", undefined], ["tictactoe", { grid_size: 5 }], ["connect4", { width: 4, height: 3, winning_length: 3 }]]) {
    for (const first of ["first_available", "osla", "random"]) for (const second of ["first_available", "osla", "random"]) {
      await createGame(game, parameters, [{ kind: first, seed: "9007199254740993" }, { kind: second, seed: max }]);
      const trace = [];
      while (!finished()) trace.push(await step());
      const result = ui("result").textContent; matches++;
      await reset(); ui("play-pause").click();
      await until(finished, "Automatic match did not finish");
      check(ui("result").textContent === result && JSON.stringify(cells()) === JSON.stringify(trace.at(-1)), "Seeded step/automatic results differ");
      await until(() => ui("play-pause").textContent === "Play live", "Terminal play did not stop");
    }
  }
  // Normal drop animation still locks input and supports immediate Reset.
  await createGame("connect4", { width: 4, height: 3, winning_length: 3 }, [{ kind: "first_available", seed: "0" }, { kind: "osla", seed: "7" }], false);
  await sleep(60);
  check(moves() === 0 && ui("play-pause").textContent === "Play live" && !ui("ai-step").disabled, "AI-only game did not start paused");
  ui("play-pause").click();
  await until(finished, "AI/AI did not continue after Play");
  await createGame("connect4", undefined, [{ kind: "first_available", seed: "0" }, { kind: "human" }]);
  ui("reduced").checked = false; ui("reduced").dispatchEvent(new Event("change")); ui("ai-step").click();
  await until(() => moves() === 1 && ui("board").getAnimations({ subtree: true }).length > 0, "AI drop animation did not start");
  await reset(); await sleep(550); check(moves() === 0, "Old animation changed reset game");
  ui("reduced").checked = true; ui("reduced").dispatchEvent(new Event("change"));
  // Random gets the same human/AI scheduling and seeded cancellation controls.
  await createGame("tictactoe", undefined, [{ kind: "human" }, { kind: "random", seed: max }], false);
  ui("board").querySelector("button:enabled").click();
  await until(() => moves() === 2 && ui("board").querySelector("button:enabled"), "Random did not respond to a human move");
  check(ui("players").textContent.includes("Random") && ui("players").textContent.includes(max), "Random identity/seed missing");
  await createGame("tictactoe", { grid_size: 64 }, [{ kind: "random", seed: "7" }, { kind: "random", seed: "8" }]);
  await step(); await step(); check(moves() === 2, "Random large-board play failed");
  await createGame("tictactoe");
  return { matches, checks: ["editable full-width seeds", "regenerate", "automatic Human/AI start", "paused AI-only start and Play continuation", "single step", "remembered Play", "delay", "pause/discard", "cancelled RNG", "reset", "new game", "invalid seeds", "OSLA limit", "all AI combinations", "configured boards", "seeded step/automatic parity", "AI animation cancellation", "zero delay"] };
}
