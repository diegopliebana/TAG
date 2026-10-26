// Schedule AI decisions and remember Play/Pause. Rust calculates moves and owns agent/RNG progress.
export class AiPlayback {
  constructor({ client, view, ready, apply, changed, failed, now = () => performance.now(), setTimer = (callback, milliseconds) => setTimeout(callback, milliseconds), clearTimer = timer => clearTimeout(timer) }) {
    Object.assign(this, { client, view, ready, apply, changed, failed, now, setTimer, clearTimer });
    this.playing = false; this.busy = false; this.phase = "idle"; this.delay = 500;
    this.epoch = 0; this.timer = null; this.wake = null; this.task = null;
  }
  isAi() {
    const view = this.view();
    return view?.next_player != null && view.controllers[view.next_player].kind !== "human";
  }
  play() { this.playing = true; this.changed(); this.kick(); }
  pause() {
    this.playing = false; this.epoch++;
    if (this.timer !== null) { this.clearTimer(this.timer); this.timer = null; }
    this.wake?.(); this.changed();
  }
  async settled() { if (this.task) await this.task; }
  setDelay(value) {
    if (!Number.isInteger(value) || value < 0 || value > 5000) throw new Error("AI delay must be a whole number from 0 to 5,000 ms.");
    this.delay = value;
  }
  kick() {
    if (!this.playing || this.busy || this.timer !== null || !this.ready() || !this.isAi()) return;
    // Even at zero delay, allow input/rendering between independent moves.
    this.timer = this.setTimer(() => { this.timer = null; this.start(false); }, 0);
  }
  step() {
    if (this.busy || !this.ready() || !this.isAi()) return;
    this.pause(); this.start(true);
  }
  start(single) {
    if (this.busy || !this.ready() || !this.isAi() || (!single && !this.playing)) return;
    const epoch = this.epoch;
    this.busy = true; this.phase = "thinking"; this.changed();
    this.task = this.run(epoch, single).catch(error => {
      this.pause(); this.failed(error);
    }).finally(() => {
      this.busy = false; this.phase = "idle"; this.task = null;
      if (this.view()?.next_player == null) this.playing = false;
      this.changed(); this.kick();
    });
  }
  wait(milliseconds) {
    if (milliseconds <= 0) return Promise.resolve();
    return new Promise(resolve => {
      const timer = this.setTimer(() => { this.wake = null; resolve(); }, milliseconds);
      this.wake = () => { this.clearTimer(timer); this.wake = null; resolve(); };
    });
  }
  async run(epoch, single) {
    const view = this.view();
    const generation = this.client.generation, positionVersion = view.position_version;
    const deadline = this.now() + (single ? 0 : this.delay);
    const current = () => epoch === this.epoch && this.client.generation === generation && this.view()?.position_version === positionVersion;
    let prepared = await this.client.request({ type: "prepare_ai" }, { positionVersion });
    while (prepared.proposal?.pending && current()) {
      prepared = await this.client.request({ type: "poll_ai", proposal: prepared.proposal.id }, { positionVersion });
    }
    const discard = () => this.client.request({ type: "discard_ai", proposal: prepared.proposal.id }, { positionVersion });
    if (!current()) { this.phase = "cancelling"; this.changed(); await discard(); return; }
    const remaining = Math.max(0, deadline - this.now());
    if (remaining > 0) { this.phase = "waiting"; this.changed(); await this.wait(remaining); }
    if (!current()) { this.phase = "cancelling"; this.changed(); await discard(); return; }
    this.phase = "committing"; this.changed();
    const observer = view.game === "loveletter" && !view.board.loveletter.public_only ? { observer: view.observer } : {};
    const reply = await this.client.request({ type: "commit_ai", proposal: prepared.proposal.id, ...observer }, { positionVersion });
    // Pause cannot undo an already dispatched commitment. Always display its accepted move.
    this.phase = "animating"; this.changed(); await this.apply(reply.view);
  }
}
