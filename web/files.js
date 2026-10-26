// Download/save files in the browser and validate loads in a temporary worker before replacement.
import { BrowserClient, BridgeFailure, MAX_FILE_BYTES } from "./client.js";
import { createSeatFields, fillSeats, readControllers } from "./seat-fields.js";

export function downloadFile(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  // Keep the URL alive while the browser starts its download.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export class FileControls {
  constructor({ client, current, settle, busy, saved, confirmReplace, adopt, failed, seeds, download = downloadFile, makeClient = () => new BrowserClient() }) {
    Object.assign(this, { client, current, settle, busy, saved, confirmReplace, adopt, failed, seeds, download, makeClient });
    this.candidate = null; this.epoch = 0; this.running = false;
    this.ui = Object.fromEntries(["save-session", "save-branch", "save-state", "load-file", "file-input", "file-progress", "file-progress-status", "file-cancel", "state-setup", "state-form", "state-seats", "state-summary", "state-error", "state-cancel"].map(id => [id, document.getElementById(id)]));
    createSeatFields(this.ui["state-seats"], seeds, "state-");
    for (const scope of ["session", "branch", "state"]) this.ui[`save-${scope}`].addEventListener("click", () => this.save(scope));
    this.ui["load-file"].addEventListener("click", () => this.choose());
    this.ui["file-input"].addEventListener("change", () => { const file = this.ui["file-input"].files[0]; if (file) this.load(file); });
    this.ui["file-cancel"].addEventListener("click", () => this.cancel());
    this.ui["file-progress"].addEventListener("cancel", event => { event.preventDefault(); this.cancel(); });
    this.ui["state-cancel"].addEventListener("click", () => this.ui["state-setup"].close("cancel"));
  }
  render(locked, hasGame) {
    for (const scope of ["session", "branch", "state"]) this.ui[`save-${scope}`].disabled = locked || !hasGame;
    this.ui["load-file"].disabled = locked;
  }
  choose() {
    if (this.running) return;
    document.querySelector(".file-menu").open = false;
    this.settle().catch(error => this.failed(error));
    this.ui["file-input"].value = ""; this.ui["file-input"].click();
  }
  async exportCurrent(scope) {
    const { display, delay } = this.current();
    const reply = await this.client().requestFile({ type: "export", scope, cursor: display.cursor, delay_ms: delay });
    const date = new Date().toISOString().replaceAll(":", "-").replace(/\.\d+Z$/, "Z");
    const kind = JSON.parse(reply.text).kind;
    const extension = scope === "state" ? "tag-state.json" : kind === "tag-learned-browser" ? "tag-learned-browser.json" : "tag-replay.json";
    this.download(reply.text, `${display.game}-${scope}-branch-${display.branch}-move-${display.cursor}-${date}.${extension}`);
    if (scope === "session") this.saved();
  }
  async save(scope) {
    if (this.running) return;
    this.running = true; this.busy(true, "Preparing file download...");
    document.querySelector(".file-menu").open = false;
    try { await this.settle(); await this.exportCurrent(scope); }
    catch (error) { this.failed(error); }
    finally { this.running = false; this.busy(false); }
  }
  cancel() {
    this.epoch++; this.candidate?.dispose(); this.candidate = null;
    this.ui["file-progress"].close(); this.ui["state-setup"].close("cancel");
  }
  controllers(reply) {
    const dialog = this.ui["state-setup"];
    const gameName = reply.view.game === "loveletter" ? "Love Letter" : reply.view.game === "cantstop" ? "Can't Stop" : reply.view.game === "tictactoe" ? "Tic-Tac-Toe" : "Connect 4";
    this.ui["state-summary"].textContent = `${reply.view.game === "tictactoe" ? "Tic-Tac-Toe" : "Connect 4"} | ${reply.view.board.width} × ${reply.view.board.height} | ${reply.view.board.counters[0]} moves${reply.parameters.winning_length ? ` | Winning length ${reply.parameters.winning_length}` : ""}`;
    const defaultAi = reply.view.board.width * reply.view.board.height <= 256 ? "osla" : "first_available";
    if (["cantstop","loveletter"].includes(reply.view.game)) this.ui["state-summary"].textContent = `${gameName} | ${reply.view.controllers.length} players | ${reply.view.board.counters[0]} accepted actions`;
    createSeatFields(this.ui["state-seats"], this.seeds, "state-", reply.view.controllers.length, reply.view.game);
    fillSeats(document, reply.view.controllers.map((_, seat) => seat === 0 ? { kind: "human" } : { kind: ["cantstop","loveletter"].includes(reply.view.game) ? "sampled_osla" : defaultAi, seed: this.seeds.next(), ...(["cantstop","loveletter"].includes(reply.view.game) ? { repetitions_per_action: 1 } : {}) }), "state-");
    document.getElementById("state-seed-0").value = this.seeds.next();
    this.ui["state-error"].textContent = "";
    return new Promise(resolve => {
      let selected = null;
      const submit = event => { event.preventDefault(); try { selected = readControllers(document, reply.parameters, "state-"); dialog.close("load"); } catch (error) { this.ui["state-error"].textContent = error.message; } };
      this.ui["state-form"].addEventListener("submit", submit);
      dialog.addEventListener("close", () => { this.ui["state-form"].removeEventListener("submit", submit); resolve(dialog.returnValue === "load" ? selected : null); }, { once: true });
      dialog.returnValue = "cancel"; dialog.showModal();
    });
  }
  async load(file) {
    if (this.running) return;
    this.running = true; const epoch = ++this.epoch;
    const current = () => { if (epoch !== this.epoch) throw new BridgeFailure("cancelled", "File operation cancelled."); };
    this.busy(true, "Loading and validating file...");
    this.ui["file-progress-status"].textContent = `Validating ${file.name}...`;
    const dialog = this.ui["file-progress"]; dialog.showModal();
    try {
      await this.settle(); current();
      if (file.size > MAX_FILE_BYTES) throw new BridgeFailure("resource_limit", "File exceeds 10 MiB.");
      const text = await file.text(); current();
      this.candidate = this.makeClient();
      if (JSON.parse(text).kind !== "tag-learned-browser") {
        for (const model of this.client().modelTexts ?? []) {
          await this.candidate.requestFile({ type: "load_model" }, model); current();
        }
      }
      let reply = await this.candidate.requestFile({ type: "import", controllers: null }, text); current();
      if (reply.needs_controllers) {
        dialog.close(); const controllers = await this.controllers(reply); current();
        if (!controllers) return;
        dialog.showModal(); reply = await this.candidate.requestFile({ type: "import", controllers }, text); current();
      }
      dialog.close();
      if (!await this.confirmReplace()) return;
      current();
      const candidate = this.candidate; this.candidate = null;
      this.adopt(candidate, reply);
    } catch (error) { if (error.code !== "cancelled") this.failed(error); }
    finally {
      this.candidate?.dispose(); this.candidate = null;
      dialog.close(); this.running = false; this.busy(false);
    }
  }
}
