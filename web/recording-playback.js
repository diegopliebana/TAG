// Visit already recorded positions at the selected delay; never ask an agent to make new moves.
export class RecordingPlayback {
  constructor({ position, advance, delay, changed, failed, setTimer = (callback, milliseconds) => setTimeout(callback, milliseconds), clearTimer = timer => clearTimeout(timer) }) {
    Object.assign(this, { position, advance, delay, changed, failed, setTimer, clearTimer });
    this.playing = false; this.busy = false; this.epoch = 0; this.wake = null; this.task = null;
  }
  pause() { this.playing = false; this.epoch++; this.wake?.(); this.changed(); }
  async settled() { if (this.task) await this.task; }
  play() {
    if (this.busy || this.position().cursor >= this.position().tip) return;
    this.playing = true; this.busy = true; const epoch = this.epoch; this.changed();
    this.task = this.run(epoch).catch(error => { this.pause(); this.failed(error); }).finally(() => {
      this.playing = false; this.busy = false; this.task = null; this.changed();
    });
  }
  wait(milliseconds) {
    return new Promise(resolve => {
      const timer = this.setTimer(() => { this.wake = null; resolve(); }, milliseconds);
      this.wake = () => { this.clearTimer(timer); this.wake = null; resolve(); };
    });
  }
  async run(epoch) {
    const current = () => this.playing && this.epoch === epoch;
    while (current() && this.position().cursor < this.position().tip) {
      // A timer yields even at zero delay, keeping Pause and navigation responsive.
      await this.wait(this.delay());
      if (!current()) return;
      await this.advance(this.position().cursor + 1, current);
    }
  }
}
