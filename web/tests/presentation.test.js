import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { PreviewAudio } from "../presentation/audio.js";
import { changedCell } from "../presentation/animation.js";

test("every candidate file and licence matches its recorded SHA-256 identity", () => {
  const root = new URL("../assets/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
  for (const asset of manifest.assets) {
    const data = readFileSync(new URL(asset.path, root));
    assert.equal(data.length, asset.bytes);
    assert.equal(createHash("sha256").update(data).digest("hex"), asset.sha256);
    assert.ok(manifest.sources[asset.source]);
  }
  for (const source of Object.values(manifest.sources)) {
    const data = readFileSync(new URL(source.licence_path, root));
    assert.equal(createHash("sha256").update(data).digest("hex"), source.licence_sha256);
    assert.ok(data.toString().includes("CC0"));
  }
});

test("mute, zero volume and cue changes preserve explicit audio controls", async () => {
  class AudioDouble { plays = 0; pauses = 0; play() { this.plays++; return Promise.resolve(); } pause() { this.pauses++; } }
  const audio = new PreviewAudio({ get: id => `${id}.ogg` }, AudioDouble);
  await audio.play("mark"); assert.equal(audio.audio.src, "mark.ogg"); assert.equal(audio.audio.plays, 1);
  audio.setMuted(true); await audio.play("win"); assert.equal(audio.audio.plays, 1); assert.equal(audio.audio.muted, true);
  audio.setMuted(false); audio.setVolume(0); await audio.play("land"); assert.equal(audio.audio.plays, 1);
  audio.setVolume(.25); await audio.play("land"); assert.equal(audio.audio.src, "land.ogg"); assert.equal(audio.audio.volume, .25);
  assert.throws(() => audio.setVolume(NaN)); assert.throws(() => audio.setVolume(2));
});

test("landing coordinates come from x-major differences in accepted views", () => {
  const before = { board: { cells: [null, null, null, null, null, null], height: 3 } };
  const after = { board: { cells: [null, null, null, null, null, 1], height: 3 } };
  assert.deepEqual(changedCell(before, after), { x: 1, y: 2 });
  assert.equal(changedCell(before, before), null);
});
