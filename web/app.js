import { BrowserClient, MAX_FILE_BYTES } from "./client.js";
import { actionResult, shouldShowActionResult, ActionResultStep } from "./loveletter-results.js";
import { renderBoard } from "./board.js";
import { playerColors, playerShapes } from "./cantstop-board.js";
import { loadAssets } from "./presentation/assets.js";
import { PreviewAudio } from "./presentation/audio.js";
import { animateMove, animateResult } from "./presentation/animation.js";
import { setupFields, readParameters, CryptoSeeds } from "./setup.js";
import { AiPlayback } from "./ai-playback.js";
import { RecordingPlayback } from "./recording-playback.js";
import { ReviewPanel } from "./review-panel.js";
import { createSeatFields, fillSeats, readControllers, controllerName, setModels, setModelLoader } from "./seat-fields.js";
import { FileControls } from "./files.js";

const ids = ["action-results-controls", "always-action-results", "action-result", "action-result-title", "action-result-body", "action-result-pause", "observer-controls", "observer-view", "reveal-controls", "reveal-hidden", "new-game", "reset", "play-pause", "ai-step", "ai-delay", "theme", "reduced", "mute", "volume", "enable-sound", "status", "game-name", "match-title", "result", "players", "board", "details", "audio-status", "setup", "setup-form", "game", "setup-seats", "setup-parameters", "defaults", "setup-error", "setup-cancel", "create", "discard-dialog", "branch", "review-position", "branch-ancestry", "history-moves", "previous", "next", "branch-end", "fork", "fork-dialog", "fork-form", "fork-title", "fork-seats", "fork-error", "fork-cancel", "fork-create"];
const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const actionResults = new ActionResultStep(ui["action-result"],ui["action-result-title"],ui["action-result-body"]);
let effectView = null;
let presentationIntent = 0;
let client = new BrowserClient(); const seeds = new CryptoSeeds();
let fileBusy = false, fileStatus = "";
let assets, audio, view = null, settings = null, errorMessage = "";
let audioReady = Promise.resolve();
let commandBusy = true, lifecyclePending = false, setupPending = false;
let animating = false, presentationVersion = 0, sessionHasMoves = false;
let displayView = null, recordedMoves = [], historyBranch = null;
let navigationPending = false, navigationIntent = 0, desiredNavigation = null, navigationTask = null;
let forkPending = false, forkSource = null;
const preference = matchMedia("(prefers-reduced-motion: reduce)");
ui.reduced.checked = preference.matches;
const playback = new AiPlayback({
  client, view: () => view,
  ready: () => !fileBusy && !commandBusy && !lifecyclePending && !setupPending && !animating && !ui.setup.open && !ui["fork-dialog"].open && !forkPending && !navigationPending && !recording.busy && !reviewing() && !client.failure,
  apply: after => {
    if (fileBusy || lifecyclePending || navigationPending) { acceptLive(after); sessionHasMoves = true; render(); return Promise.resolve(); }
    return present(after, audioReady);
  }, changed: () => render(),
  failed: error => { errorMessage = error.message; render(); },
});
const recording = new RecordingPlayback({
  position: () => ({ cursor: shown()?.cursor ?? 0, tip: shown()?.tip ?? 0 }),
  advance: recordedNext, delay: () => playback.delay, changed: () => render(),
  failed: error => { errorMessage = error.message; render(); },
});
const panel = new ReviewPanel({ branch: ui.branch, list: ui["history-moves"], position: ui["review-position"], ancestry: ui["branch-ancestry"], jump: cursor => navigate({ type: "history", branch: shown().branch, cursor }), select: branch => navigate({ type: "select_branch", branch }) });
const files = new FileControls({
  client: () => client, current: () => ({ display: shown(), delay: playback.delay }), seeds,
  settle: async () => { playback.pause(); recording.pause(); cancelPresentation(); await playback.settled(); await recording.settled(); cancelPresentation(); },
  busy: (active, message = "") => { fileBusy = active; fileStatus = message; if (active) errorMessage = ""; else setModels(client.models); if (!view) ui.status.textContent = message || errorMessage || "Choose a game to begin."; render(); },
  saved: () => { sessionHasMoves = false; }, confirmReplace: confirmDiscard,
  failed: error => { errorMessage = error.message; render(); },
  adopt: (candidate, reply) => {
    const previous = client; client = candidate; playback.client = client;
    setModels(client.models);
    resetPresentation();
    acceptLive(reply.live_view, true); acceptDisplay(reply.view); settings = reply.parameters;
    playback.setDelay(reply.delay_ms); ui["ai-delay"].value = reply.delay_ms;
    sessionHasMoves = false; errorMessage = ""; commandBusy = false;
    ui.setup.close(); ui["fork-dialog"].close(); previous.dispose(); render();
  },
});
setModelLoader(async file => {
  if (file.size > MAX_FILE_BYTES) throw new Error("Model exceeds 10 MiB.");
  commandBusy = true; playback.pause(); recording.pause(); render();
  try {
    await playback.settled(); await recording.settled();
    const target = files.candidate ?? client;
    await target.requestFile({ type: "load_model" }, await file.text());
    setModels(target.models);
  } finally { commandBusy = false; render(); }
});
function shown() { return effectView ?? displayView ?? view; }
function reviewing() { const display = shown(); return Boolean(display && (display.read_only || display.cursor !== display.tip || display.branch !== view?.branch)); }
function updateRecords(after, clear = false) {
  if (clear || historyBranch !== after.branch) recordedMoves = [];
  historyBranch = after.branch;
  if (after.moves !== null) recordedMoves = after.moves;
  else if (after.accepted_move) recordedMoves[after.accepted_move.position - 1] = after.accepted_move;
}
function acceptLive(after, clear = false) { updateRecords(after, clear); view = after; displayView = null; }
function acceptDisplay(after) {
  updateRecords(after);
  if (after.read_only) displayView = after;
  else { view = after; displayView = null; }
}

function cancelPresentation(stopAudio = true) {
  presentationVersion++;
  actionResults.cancel(); effectView = null;
  for (const animation of ui.board.getAnimations({ subtree: true })) animation.cancel();
  for (const animation of ui.result.getAnimations()) animation.cancel();
  if (stopAudio) audio?.stop(); animating = false;
}
function parameters() {
  ui["setup-parameters"].replaceChildren();
  for (const field of setupFields(ui.game.value)) {
    const label = document.createElement("label"); label.textContent = field.label;
    const input = document.createElement("input"); input.type = field.type ?? "number"; input.min = "1"; input.step = "1";
    input.name = field.name; input.value = field.value; if (field.name === "first_player") input.min = "0";
    label.append(input); ui["setup-parameters"].append(label);
    if (field.name === "players") {
      input.min = "2"; input.max = "4";
      input.addEventListener("change", () => {
        const n = Number(input.value);
        if (Number.isInteger(n) && n >= 2 && n <= 4) createSeatFields(ui["setup-seats"], seeds, "", n, ui.game.value);
      });
    }
  }
  ui["setup-error"].textContent = "";
  createSeatFields(ui["setup-seats"], seeds, "", ui.game.value === "loveletter" ? 3 : 2, ui.game.value);
}
function refreshSeeds() { for (const input of ui["setup-seats"].querySelectorAll('input[id^="seed-"]')) input.value = seeds.next(); }
function render() {
  const display = shown(), historical = reviewing();
  const locked = fileBusy || commandBusy || lifecyclePending || setupPending || forkPending;
  files.render(locked || navigationPending || ui.setup.open || ui["fork-dialog"].open, Boolean(view));
  ui["new-game"].disabled = locked || navigationPending; ui.reset.disabled = locked || navigationPending || !view;
  for (const control of ui["setup-form"].querySelectorAll("button, input, select")) control.disabled = locked;
  const terminal = display?.next_player == null;
  const hasAi = display?.controllers.some(seat => seat.kind !== "human");
  ui["play-pause"].textContent = recording.playing || recording.busy ? "Pause recording" : playback.playing || playback.busy ? "Pause live" : historical ? "Play recording" : "Play live";
  ui["play-pause"].disabled = (!recording.busy && (locked || navigationPending)) || !display || (!historical && !hasAi && !recording.busy) || (!historical && terminal && !recording.busy && !playback.busy);
  ui["ai-step"].disabled = locked || navigationPending || historical || recording.busy || playback.playing || playback.busy || animating || !playback.isAi() || ui.setup.open || ui["fork-dialog"].open;
  ui.previous.disabled = locked || !display || display.cursor === 0;
  ui.next.disabled = locked || !display || display.cursor >= display.tip;
  ui["branch-end"].disabled = locked || !display || (!historical && !navigationPending);
  ui.fork.disabled = locked || navigationPending || !display;
  for (const control of ui["fork-form"].querySelectorAll("button, input, select")) control.disabled = locked;
  ui["observer-view"].disabled = ui["reveal-hidden"].disabled = locked || navigationPending;
  ui["observer-controls"].hidden = ui["reveal-controls"].hidden = ui["action-results-controls"].hidden = display?.game !== "loveletter";
  if (display?.game === "loveletter") {
    const select=ui["observer-view"];
    if (select.options.length !== display.controllers.length+2) {
      const selected=select.value;
      select.replaceChildren();
      for (const [value,label] of [["auto","Current human"],["public","Public spectator"],...display.controllers.map((_,s)=>[String(s),`Player ${s}`])]) {
        const option=document.createElement("option");option.value=value;option.textContent=label;select.append(option);
      }
      select.value=selected||"auto";
    }
  }
  for (const [id,label] of [["save-session","Save whole session"],["save-branch","Save current branch"],["save-state","Save current state"]]) {
    document.getElementById(id).textContent=label+(display?.game==="loveletter"?" (privileged)":"");
  }
  if (!display) return;
  panel.render(display, recordedMoves, locked);
  const name = display.game === "loveletter" ? "Love Letter" : display.game === "cantstop" ? "Can't Stop" : display.game === "tictactoe" ? "Tic-Tac-Toe" : "Connect 4";
  ui["game-name"].textContent = name.toUpperCase();
  ui["match-title"].textContent = display.controllers.map(seat => controllerName(seat.kind)).join(" versus ");
  ui.result.textContent = display.board.outcome.kind === "shared_win" ? `Players ${display.board.outcome.player.join(", ")} share the win` : display.board.outcome.kind === "won" ? `Player ${display.board.outcome.player} wins` : terminal ? "Draw" : `Player ${display.next_player} to move`;
  ui.details.textContent = `${name} | ${display.board.counters[0]} moves | ${display.board.width} × ${display.board.height}${settings?.winning_length ? ` | Winning length ${settings.winning_length}` : ""}`;
  if (display.game === "loveletter") ui.details.textContent = `${name} | ${display.board.counters[0]} moves | ${display.controllers.length} players | Round ${display.board.loveletter.round}`;
  const human = !terminal && display.controllers[display.next_player].kind === "human";
  ui.status.textContent = errorMessage || (navigationPending ? "Loading the selected recorded position..." : recording.playing ? `Playing recording: move ${display.cursor} of ${display.tip}.` : historical ? `Reviewing move ${display.cursor} of ${display.tip}. Fork to continue from here.` : lifecyclePending ? "Waiting for the current engine command to finish..." : commandBusy ? "Applying your request..." : terminal ? "Game finished. Reset to play again, or choose New game." : animating ? "Showing the accepted move..." : playback.busy ? ({ thinking: "AI is thinking...", waiting: "Waiting for the AI delay...", committing: "Applying the AI move...", cancelling: "Discarding the pending AI move...", animating: "Showing the accepted move..." })[playback.phase] : human ? `Waiting for Player ${display.next_player}. ${display.game === "tictactoe" ? "Choose an outlined empty cell." : display.game === "loveletter" ? "Choose a card, target and any Guard guess." : "Choose an available column."}${hasAi ? playback.playing ? " AI play is enabled." : " AI play is paused." : ""}` : playback.playing ? "Preparing the next AI turn..." : `AI play is paused. Use Play or AI step for Player ${display.next_player}.`);
  if (fileStatus && !errorMessage) ui.status.textContent = fileStatus;
  if (display.game === "cantstop") {
    ui.status.textContent = ui.status.textContent.replace(display.game === "loveletter" ? "Choose a card, target and any Guard guess." : "Choose an available column.", "Choose one of the available actions below the tracks.");
    ui.details.textContent = `${name} | ${display.board.counters[0]} actions | ${display.board.counters[1]} completed turns | Dice seed: ${settings?.seed ?? "loaded"}`;
  }
  renderBoard(ui.board, display, action => move(action), { busy: locked || navigationPending || historical || recording.busy || playback.busy || animating, sprites: assets.sprites, emphasis: display.board.winning_cells });
  if (ui.players.dataset.game !== `${display.game}-${display.controllers.length}`) {
    ui.players.replaceChildren(); ui.players.dataset.game = `${display.game}-${display.controllers.length}`;
    for (let seat = 0; seat < display.controllers.length; seat++) {
      const card = document.createElement("div"); card.className = "player";
      const image = document.createElement(["cantstop","loveletter"].includes(display.game) ? "span" : "img"); image.alt = "";
      if (["cantstop","loveletter"].includes(display.game)) { image.className = "cantstop-player"; image.style.color = playerColors[seat]; image.textContent = ["●", "◆", "■", "▲"][seat]; image.setAttribute("aria-hidden", "true"); }
      image.src = display.game === "tictactoe" ? (seat === 0 ? assets.sprites.x : assets.sprites.o) : (seat === 0 ? assets.sprites.red : assets.sprites.yellow);
      const label = document.createElement("span"); const seed = document.createElement("span"); seed.className = "agent-seed";
      card.append(image, label, seed); ui.players.append(card);
    }
  }
  [...ui.players.children].forEach((card, seat) => {
    const controller = display.controllers[seat]; card.dataset.active = display.next_player === seat;
    card.children[1].textContent = `Player ${seat} / ${display.game === "tictactoe" ? (seat === 0 ? "X" : "O") : (seat === 0 ? "red" : "yellow")} / ${controllerName(controller.kind)}`;
    card.children[2].textContent = controller.seed === undefined || display.game === "loveletter" ? "" : `Seed: ${controller.seed}`;
    if (display.game === "loveletter") card.children[1].textContent = `Player ${seat} / ${controllerName(controller.kind)} / ${display.scores[seat]} favor tokens`;
    if (display.game === "cantstop") card.children[1].textContent = `Player ${seat} / ${playerShapes[seat]} / ${controllerName(controller.kind)} / ${display.scores[seat]} columns`;
  });
}
async function present(after, unlocked = Promise.resolve(), recorded = false) {
  const before = shown(); cancelPresentation(false); const version = presentationVersion;
  const summary = actionResult(before,after);
  const result = shouldShowActionResult(before,after,ui["always-action-results"].checked) ? summary : null;
  if (recorded) acceptDisplay(after); else { acceptLive(after); sessionHasMoves = true; }
  effectView = result ? before : null;
  animating = true; render();
  try {
    await unlocked; if (version !== presentationVersion) return;
    if (!result) await animateMove(ui.board, before, after, ui.reduced.checked);
    if (version !== presentationVersion) return;
    const outcome = after.board.outcome.kind;
    if (outcome !== "ongoing") animateResult(ui.result, ui.reduced.checked);
    try {
      await audio.play((["won","shared_win"].includes(outcome) || summary?.sections.some(s=>/^Round .* won by /.test(s.title))) ? "sound-win" : outcome === "draw" ? "sound-draw" : after.game === "tictactoe" ? "sound-mark" : "sound-land");
      if (version === presentationVersion) ui["audio-status"].textContent = "";
    } catch { if (version === presentationVersion) ui["audio-status"].textContent = "Sound could not start. Use Test sound to retry, or check browser audio settings."; }
    if (result && version === presentationVersion) {
      if (!await actionResults.show(result) || version !== presentationVersion) return;
      if (!recorded) {
        const fresh = await client.request({type:"view",observer:null});
        if (version !== presentationVersion) return;
        acceptLive(fresh.view);
      }
      effectView = null;
    }
  } finally {
    if (version === presentationVersion) { animating = false; render(); playback.kick(); }
  }
}
function resetPresentation() {
  presentationIntent++;
  ui["reveal-hidden"].checked=false;ui["observer-view"].value="auto";
}
async function changePresentation() {
  if (!view || view.game!=="loveletter" || fileBusy || lifecyclePending) return;
  if (effectView || actionResults.resolve) cancelPresentation(false);
  const intent=++presentationIntent;
  const value=ui["observer-view"].value;
  const reveal=ui["reveal-hidden"].checked;
  // Remove private DOM immediately while reading the replacement observation.
  for (const current of [view,displayView]) if (current?.board?.loveletter) {
    delete current.board.loveletter.privileged;
    current.board.loveletter.hand=[];current.board.loveletter.private_events=[];current.board.loveletter.last_private_events=[];
    current.actions=[];
  }
  render();
  try {
    const reply=await client.request({type:"presentation",observer:["auto","public"].includes(value)?null:Number(value),
      public_only:value==="public",reveal_hidden_information:reveal});
    if (intent===presentationIntent) { acceptDisplay(reply.view); render(); }
  } catch(error) { if(intent===presentationIntent) { errorMessage=error.message;render(); } }
}
ui["observer-view"].addEventListener("change",changePresentation);
ui["reveal-hidden"].addEventListener("change",changePresentation);
ui["action-result-pause"].addEventListener("click",()=>{
  playback.pause(); recording.pause(); ui["action-result"].close();
});
function move(action) {
  if (fileBusy || commandBusy || lifecyclePending || setupPending || forkPending || navigationPending || recording.busy || reviewing() || playback.busy || animating || !view) return;
  return execute({ type: "human_move", actor: view.next_player, action, ...(view.game === "loveletter" ? {observer:view.observer} : {}) }, { moved: true });
}
async function execute(command, { moved = false, newSettings = null } = {}) {
  if (commandBusy || playback.busy) return false;
  cancelPresentation(); const version = presentationVersion; errorMessage = "";
  const unlocked = moved ? audio.unlock() : Promise.resolve();
  commandBusy = true; render();
  try {
    const reply = await client.request(command, { positionVersion: view?.position_version ?? null, replace: command.type === "create" || command.type === "reset" });
    commandBusy = false;
    if (moved) await present(reply.view, unlocked);
    else {
      acceptLive(reply.view, command.type === "create");
      if (command.type === "reset" || command.type === "fork") sessionHasMoves = true;
      if (command.type === "fork") ui["fork-dialog"].close();
      if (["create","reset"].includes(command.type)) resetPresentation();
      if (command.type === "create") { settings = newSettings; sessionHasMoves = false; ui.setup.close(); }
    }
    ui["setup-error"].textContent = ""; return true;
  } catch (error) {
    errorMessage = error.code === "worker_failure" ? "Game engine stopped. Reload the page to try again." : error.message;
    if (command.type === "create") ui["setup-error"].textContent = errorMessage;
    if (command.type === "fork") ui["fork-error"].textContent = errorMessage;
    playback.pause(); return false;
  } finally {
    if (version === presentationVersion) commandBusy = Boolean(client.failure);
    render(); playback.kick();
  }
}
async function lifecycle(operation) {
  if (fileBusy || commandBusy || lifecyclePending || setupPending || forkPending || navigationPending) return;
  lifecyclePending = true; playback.pause(); recording.pause(); cancelPresentation(); render();
  try { await playback.settled(); await recording.settled(); await operation(); }
  finally { lifecyclePending = false; render(); playback.kick(); }
}
function navigate(operation) {
  if (fileBusy || commandBusy || lifecyclePending || setupPending || forkPending || !view) return;
  desiredNavigation = operation; navigationIntent++; navigationPending = true;
  playback.pause(); recording.pause(); cancelPresentation(); render();
  if (navigationTask) return navigationTask;
  navigationTask = (async () => {
    await playback.settled(); await recording.settled();
    while (desiredNavigation) {
      const next = desiredNavigation; desiredNavigation = null; const intent = navigationIntent;
      try {
        if (next.type === "history" && next.branch !== view.branch) {
          const selected = await client.request({ type: "select_branch", branch: next.branch }, { positionVersion: view.position_version });
          view = selected.view;
          if (intent !== navigationIntent) continue;
        }
        const reply = await client.request(next, { positionVersion: view.position_version });
        // A superseded selection still advanced live branch identity in Rust.
        if (next.type === "select_branch") view = reply.view;
        if (intent === navigationIntent) { acceptDisplay(reply.view); errorMessage = ""; }
      } catch (error) { if (intent === navigationIntent) errorMessage = error.message; }
    }
  })().finally(() => { navigationPending = false; navigationTask = null; render(); });
  return navigationTask;
}
async function recordedNext(cursor, current) {
  const reply = await client.request({ type: "history", branch: shown().branch, cursor }, { positionVersion: view.position_version });
  if (current() && !navigationPending && !lifecyclePending) await present(reply.view, audioReady, true);
}
ui.previous.addEventListener("click", () => navigate({ type: "history", branch: shown().branch, cursor: Math.max(0, shown().cursor - 1) }));
ui.next.addEventListener("click", () => navigate({ type: "history", branch: shown().branch, cursor: Math.min(shown().tip, shown().cursor + 1) }));
ui["branch-end"].addEventListener("click", () => navigate({ type: "history", branch: shown().branch, cursor: shown().tip }));
ui.fork.addEventListener("click", () => {
  const source = shown();
  lifecycle(async () => {
    forkSource = { branch: source.branch, cursor: source.cursor, controllers: source.controllers };
    createSeatFields(ui["fork-seats"], seeds, "fork-", source.controllers.length, source.game);
    fillSeats(document, source.controllers, "fork-");
    source.controllers.forEach((controller, seat) => { if (controller.seed === undefined) document.getElementById(`fork-seed-${seat}`).value = seeds.next(); });
    ui["fork-title"].textContent = `Fork branch ${source.branch} at move ${source.cursor}${source.next_player === null ? " (finished)" : ""}`;
    ui["fork-error"].textContent = ""; ui["fork-dialog"].showModal();
  });
});
ui["fork-form"].addEventListener("submit", async event => {
  event.preventDefault(); if (forkPending || commandBusy || lifecyclePending || navigationPending) return;
  let controllers;
  try { controllers = readControllers(document, settings, "fork-"); }
  catch (error) { ui["fork-error"].textContent = error.message; return; }
  const replacements = controllers.map((controller, seat) => JSON.stringify(controller) === JSON.stringify(forkSource.controllers[seat]) ? null : controller);
  forkPending = true; render();
  try { await execute({ type: "fork", branch: forkSource.branch, cursor: forkSource.cursor, replacements }); }
  finally { forkPending = false; render(); }
});
ui["fork-cancel"].addEventListener("click", () => ui["fork-dialog"].close());
ui["fork-dialog"].addEventListener("close", render);
async function confirmDiscard() {
  await playback.settled(); await recording.settled();
  if (!sessionHasMoves) return true;
  const dialog = ui["discard-dialog"]; dialog.returnValue = "cancel";
  const choice = await new Promise(resolve => { dialog.addEventListener("close", () => resolve(dialog.returnValue), { once: true }); dialog.showModal(); });
  if (choice === "save") {
    try { await files.exportCurrent("session"); return true; }
    catch (error) { errorMessage = error.message; ui["setup-error"].textContent = error.message; render(); return false; }
  }
  return choice === "discard";
}
ui["setup-form"].addEventListener("submit", async event => {
  event.preventDefault(); if (commandBusy || setupPending || lifecyclePending) return;
  const game = ui.game.value; let selected, controllers;
  try { selected = readParameters(game, [...ui["setup-parameters"].querySelectorAll("input")]); controllers = readControllers(document, selected); }
  catch (error) { ui["setup-error"].textContent = error.message; return; }
  audioReady = audio.unlock();
  let created = false;
  setupPending = true; playback.pause(); render();
  try {
    if (!await confirmDiscard()) return;
    await playback.settled(); created = await execute({ type: "create", game, parameters: selected, controllers }, { newSettings: selected });
  } finally {
    setupPending = false; render();
    if (created && controllers.some(controller => controller.kind === "human") && controllers.some(controller => controller.kind !== "human")) playback.play();
  }
});
ui["new-game"].addEventListener("click", () => lifecycle(async () => { ui["setup-error"].textContent = ""; refreshSeeds(); ui.setup.showModal(); }));
ui["setup-cancel"].addEventListener("click", () => ui.setup.close());
document.getElementById("setup-load").addEventListener("click", () => files.choose());
ui.setup.addEventListener("close", render);
ui.reset.addEventListener("click", () => lifecycle(() => execute({ type: "reset" })));
ui["play-pause"].addEventListener("click", () => {
  if (recording.playing || recording.busy) recording.pause();
  else if (playback.playing || playback.busy) playback.pause();
  else { errorMessage = ""; audioReady = audio.unlock(); if (reviewing()) { playback.pause(); recording.play(); } else playback.play(); }
});
ui["ai-step"].addEventListener("click", () => { errorMessage = ""; audioReady = audio.unlock(); playback.step(); });
ui["ai-delay"].addEventListener("change", () => {
  try { playback.setDelay(ui["ai-delay"].valueAsNumber); ui["ai-delay"].setCustomValidity(""); }
  catch (error) { ui["ai-delay"].setCustomValidity(error.message); ui["ai-delay"].reportValidity(); }
});
ui.game.addEventListener("change", parameters); ui.defaults.addEventListener("click", parameters);
ui.theme.addEventListener("change", () => { document.body.dataset.theme = ui.theme.value; });
ui.mute.addEventListener("change", () => audio?.setMuted(ui.mute.checked));
ui.volume.addEventListener("input", () => audio?.setVolume(Number(ui.volume.value) / 100));
ui["enable-sound"].addEventListener("click", async () => {
  try { await audio.play("sound-mark"); ui["audio-status"].textContent = audio.muted || audio.volume === 0 ? "Sound is muted." : "Sound is ready."; }
  catch { ui["audio-status"].textContent = "Sound could not start. Check browser audio settings."; }
});
function reduce() { if (ui.reduced.checked && !commandBusy && !lifecyclePending) { cancelPresentation(); render(); playback.kick(); } }
ui.reduced.addEventListener("change", reduce);
preference.addEventListener("change", event => { ui.reduced.checked = event.matches; reduce(); });
parameters(); createSeatFields(ui["fork-seats"], seeds, "fork-"); refreshSeeds(); render();
window.gameReady = (async () => {
  [assets] = await Promise.all([loadAssets(), client.ready]);
  audio = new PreviewAudio(assets); audio.setMuted(ui.mute.checked); audio.setVolume(Number(ui.volume.value) / 100);
  commandBusy = false; ui["enable-sound"].disabled = false; ui.status.textContent = "Choose a game to begin."; render(); ui.setup.showModal();
})();
window.gameReady.catch(error => { ui.status.textContent = `Game could not start: ${error.message}`; });
