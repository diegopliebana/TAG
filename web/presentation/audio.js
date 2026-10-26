// Play existing local cues with user-controlled volume and mute.
export class PreviewAudio {
  constructor(assets, AudioType = Audio) {
    this.assets = assets;
    this.audio = new AudioType();
    this.volume = 0.35;
    this.muted = false;
  }
  setVolume(volume) {
    if (!Number.isFinite(volume) || volume < 0 || volume > 1) throw new Error("Volume must be between zero and one.");
    this.volume = volume; this.audio.volume = volume;
  }
  setMuted(muted) { this.muted = Boolean(muted); this.audio.muted = this.muted; }
  stop() { this.audio.pause(); this.audio.currentTime = 0; }
  // Prime this media element during a user gesture for accepted-move cues.
  async unlock() {
    if (this.muted || this.volume === 0) return;
    this.audio.src = this.assets.get("sound-mark"); this.audio.muted = true;
    try { await this.audio.play(); } catch { /* The accepted cue reports audio failures. */ }
    finally { this.stop(); this.audio.muted = this.muted; }
  }
  async play(id) {
    this.stop();
    if (this.muted || this.volume === 0) return;
    this.audio.src = this.assets.get(id);
    this.audio.volume = this.volume;
    await this.audio.play();
  }
}
