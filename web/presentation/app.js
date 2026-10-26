import { BrowserClient } from "../client.js";
import { renderBoard } from "../board.js";
import { loadAssets } from "./assets.js";
import { PreviewAudio } from "./audio.js";
import { animateMove } from "./animation.js";
import { scenes, buildSample } from "./scenes.js";

const ui = Object.fromEntries(["scene", "theme", "reduced", "mute", "volume", "status", "game-name", "position-title", "result", "players", "board", "initial", "middle", "finished", "play", "step", "position-details", "moves", "sounds", "audio-status"].map(id => [id, document.getElementById(id)]));
let assets, audio, samples;
let selected = 0, cursor = 0, busy = false;
const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
ui.reduced.checked = motionPreference.matches;
motionPreference.addEventListener("change", event => { ui.reduced.checked = event.matches; });
document.getElementById("setup-open").addEventListener("click", () => document.getElementById("setup-preview").showModal());
const setupGame = document.getElementById("setup-game");
function setupParameters() {
  const target = document.getElementById("setup-parameters"); target.replaceChildren();
  const defaults = scenes.find(scene => scene.game === setupGame.value).parameters;
  for (const [name, title] of setupGame.value === "tictactoe" ? [["grid_size", "Board size"]] : [["width", "Width"], ["height", "Height"], ["winning_length", "Winning length"]]) {
    const label = document.createElement("label"); label.append(document.createTextNode(title));
    const input = document.createElement("input"); input.type = "number"; input.min = "1"; input.value = defaults[name];
    label.append(input); target.append(label);
  }
}
setupGame.addEventListener("change", setupParameters); setupParameters();
ui.reduced.addEventListener("change", () => {
  if (ui.reduced.checked) for (const animation of ui.board.getAnimations({ subtree: true })) animation.cancel();
});
const cueButtons = [["Mark placement", "sound-mark"], ["Piece landing", "sound-land"], ["Win", "sound-win"], ["Draw", "sound-draw"], ["Alternative landing", "sound-land-alternative"], ["Alternative win", "sound-win-alternative"]];

function render() {
  const sample = samples[selected]; const view = sample.views[cursor]; const terminal = view.board.outcome.kind !== "ongoing";
  ui["game-name"].textContent = view.game === "tictactoe" ? "TIC-TAC-TOE" : "CONNECT 4";
  ui["position-title"].textContent = sample.scene.title;
  ui.result.textContent = view.board.outcome.kind === "won" ? `Player ${view.board.outcome.player} wins` : terminal ? "Draw" : `Player ${view.next_player} to move`;
  ui["position-details"].textContent = `Move ${cursor} of ${sample.scene.moves.length} | ${view.board.width} x ${view.board.height} | Sample review`;
  const emphasis = terminal ? view.board.winning_cells : [];
  renderBoard(ui.board, view, () => {}, { busy: true, sprites: assets.sprites, emphasis });
  ui.players.replaceChildren();
  for (let seat = 0; seat < 2; seat++) {
    const player = document.createElement("div"); player.className = "player"; player.dataset.active = view.next_player === seat;
    const image = document.createElement("img"); image.alt = ""; image.src = view.game === "tictactoe" ? (seat === 0 ? assets.sprites.x : assets.sprites.o) : (seat === 0 ? assets.sprites.red : assets.sprites.yellow);
    const label = document.createElement("span"); label.textContent = `Player ${seat}${view.game === "tictactoe" ? seat === 0 ? " / X" : " / O" : seat === 0 ? " / red" : " / yellow"}`;
    player.append(image, label); ui.players.append(player);
  }
  ui.moves.replaceChildren();
  sample.scene.moves.forEach((move, index) => {
    const item = document.createElement("li"); item.dataset.current = index + 1 === cursor;
    const actor = sample.views[index].next_player;
    item.textContent = "column" in move ? `Player ${actor}: column ${move.column + 1}` : `Player ${actor}: row ${move.y + 1}, column ${move.x + 1}`;
    ui.moves.append(item);
  });
  for (const name of ["scene", "initial", "middle", "finished", "play", "step"]) ui[name].disabled = busy || (name === "step" && cursor === sample.scene.moves.length) || (name === "middle" && sample.scene.moves.length < 2);
}

async function sound(id) {
  try { await audio.play(id); ui["audio-status"].textContent = audio.muted || audio.volume === 0 ? "Sound is muted." : "Playing the selected cue."; }
  catch { ui["audio-status"].textContent = "Audio could not start. Try a sound button or check browser audio settings."; }
}

async function step() {
  const sample = samples[selected]; if (cursor >= sample.views.length - 1) return;
  const before = sample.views[cursor]; cursor += 1; render();
  const after = sample.views[cursor];
  await animateMove(ui.board, before, after, ui.reduced.checked);
  await sound(after.board.outcome.kind === "won" ? "sound-win" : after.board.outcome.kind === "draw" ? "sound-draw" : after.game === "tictactoe" ? "sound-mark" : "sound-land");
}

async function run(operation) {
  if (busy) return;
  busy = true; render();
  try { await operation(); } catch (error) { ui.status.textContent = error.message; }
  finally { busy = false; render(); }
}

ui.theme.addEventListener("change", () => { document.body.dataset.theme = ui.theme.value; });
ui.mute.addEventListener("change", () => audio?.setMuted(ui.mute.checked));
ui.volume.addEventListener("input", () => audio?.setVolume(Number(ui.volume.value) / 100));
ui.scene.addEventListener("change", () => { audio.stop(); selected = Number(ui.scene.value); cursor = 0; render(); });
for (const [name, position] of [["initial", () => 0], ["middle", () => Math.max(1, Math.floor(samples[selected].scene.moves.length / 2))], ["finished", () => samples[selected].scene.moves.length]]) {
  ui[name].addEventListener("click", () => { audio.stop(); cursor = position(); render(); });
}
ui.step.addEventListener("click", () => run(step));
ui.play.addEventListener("click", () => run(async () => {
  audio.stop(); cursor = 0; render();
  while (cursor < samples[selected].views.length - 1) {
    await step();
    await new Promise(resolve => setTimeout(resolve, ui.reduced.checked ? 120 : 220));
  }
}));

async function start() {
  const client = new BrowserClient();
  try {
    assets = await loadAssets(); await client.ready;
    samples = [];
    for (const scene of scenes) samples.push(await buildSample(client, scene));
    audio = new PreviewAudio(assets); audio.setMuted(ui.mute.checked); audio.setVolume(Number(ui.volume.value) / 100);
    scenes.forEach((scene, index) => { const option = document.createElement("option"); option.value = index; option.textContent = scene.title; ui.scene.append(option); });
    cueButtons.forEach(([label, id]) => { const button = document.createElement("button"); button.textContent = label; button.dataset.cue = id; button.addEventListener("click", () => sound(id)); ui.sounds.append(button); });
    ui.status.textContent = "Preview ready. Compare themes, sample moves and sound cues.";
    render();
    return { samples: samples.length, assets: assets.manifest.assets.length };
  } finally { client.worker.terminate(); }
}
window.presentationReady = start();
window.presentationReady.catch(error => { ui.status.textContent = `Preview could not start: ${error.message}`; });
