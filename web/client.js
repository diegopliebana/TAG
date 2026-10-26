// Match each worker reply to its request ID; ignore results for an obsolete UI operation.
export class BridgeFailure extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
export const MAX_COMMAND_BYTES = 32 * 1024;
export const MAX_FILE_BYTES = 10 * 1024 * 1024;
export class BrowserClient {
  constructor(worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" })) {
    this.worker = worker;
    this.generation = "0";
    this.nextId = 0n;
    this.intent = 0;
    this.presentationEpoch = 0;
    this.revealHiddenInformation = false;
    this.pending = new Map();
    this.failure = null;
    this.models = [];
    this.modelTexts = [];
    this.ready = new Promise((resolve, reject) => { this.readyResolve = resolve; this.readyReject = reject; });
    worker.addEventListener("message", ({ data }) => this.receive(data));
    worker.addEventListener("error", () => this.fail("Browser engine could not start."));
    worker.addEventListener("messageerror", () => this.fail("Browser engine reply could not be read.", "protocol_error"));
  }
  fail(message, code = "worker_failure") {
    const error = new BridgeFailure(code, message);
    this.failure = error;
    this.readyReject(error);
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
  receive(reply) {
    if (!reply || typeof reply !== "object") { this.fail("Invalid browser engine reply.", "protocol_error"); return; }
    if (reply.type === "ready") { this.readyResolve(); return; }
    if (reply.type === "failure") { this.fail(reply.message); return; }
    const pending = this.pending.get(reply.request_id);
    if (!pending) {
      if (reply.error && !reply.request_id) this.fail("Browser engine error has no request identity.", "protocol_error");
      return;
    }
    this.pending.delete(reply.request_id);
    if (pending.intent !== this.intent) { pending.reject(new BridgeFailure("obsolete_reply", "Request was superseded.")); return; }
    if (reply.request_generation !== pending.generation) { pending.reject(new BridgeFailure("protocol_error", "Reply lifecycle does not match request.")); return; }
    if (typeof reply.generation !== "string" || !/^\d+$/.test(reply.generation)) { pending.reject(new BridgeFailure("protocol_error", "Invalid reply generation.")); return; }
    this.generation = reply.generation;
    if (reply.error) { pending.reject(new BridgeFailure(reply.error.code, reply.error.message)); return; }
    pending.resolve(reply);
  }
  async request(command, { positionVersion = null, replace = false, fileText } = {}) {
    if (this.failure) throw this.failure;
    if (replace) this.intent += 1;
    if (["create","reset","presentation"].includes(command.type)) {
      this.presentationEpoch++;
      this.revealHiddenInformation = command.type === "presentation" && command.reveal_hidden_information;
    }
    const presentationEpoch = this.presentationEpoch;
    const intent = this.intent;
    const generation = this.generation;
    await this.ready;
    if (this.failure) throw this.failure;
    if (intent !== this.intent) throw new BridgeFailure("obsolete_reply", "Request was superseded.");
    const request_id = (++this.nextId).toString();
    const text = JSON.stringify({ request_id, generation, ...(fileText === undefined ? { position_version: positionVersion } : {}), command });
    if (new TextEncoder().encode(text).byteLength > MAX_COMMAND_BYTES) throw new BridgeFailure("resource_limit", "Browser command exceeds 32 KiB.");
    if (fileText !== undefined && new TextEncoder().encode(fileText).byteLength > MAX_FILE_BYTES) throw new BridgeFailure("resource_limit", "File exceeds 10 MiB.");
    const reply = await new Promise((resolve, reject) => {
      this.pending.set(request_id, { resolve, reject, intent, generation });
      try { this.worker.postMessage(fileText === undefined ? text : { type: "file", envelope: text, text: fileText }); }
      catch (error) { this.fail(`Browser command could not be sent: ${error.message}`); }
    });
    // Presentation changes must not invalidate committed moves or prepared AI
    // identities. Refresh only the stale display, retaining operation metadata.
    if (fileText === undefined && reply.view?.game === "loveletter" && presentationEpoch !== this.presentationEpoch) {
      const fresh = await this.request({type:"view",observer:null});
      reply.view = fresh.view;
    }
    if (!this.revealHiddenInformation) {
      for (const view of [reply.view,reply.live_view]) if (view?.board?.loveletter) delete view.board.loveletter.privileged;
    }
    return reply;
  }
  async requestFile(command, text = "") {
    const reply = await this.request(command, { fileText: text });
    const alreadyLoaded = command.type === "load_model" && this.models.some(model => model.id === reply.text);
    if (reply.models) this.models = reply.models;
    if (command.type === "load_model" && !alreadyLoaded && !this.modelTexts.includes(text)) this.modelTexts.push(text);
    // Preserve floating metadata and signed zero for later candidate workers.
    // JSON.parse/stringify would change the model document's numeric types.
    if (reply.model_sources) this.modelTexts = [...new Set(reply.model_sources)];
    return reply;
  }
  dispose() { this.worker.terminate(); this.fail("File operation cancelled.", "cancelled"); }
}
