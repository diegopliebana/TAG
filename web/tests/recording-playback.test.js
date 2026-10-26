import test from "node:test";
import assert from "node:assert/strict";
import { RecordingPlayback } from "../recording-playback.js";
const tick = () => new Promise(resolve => setTimeout(resolve, 5));
test("recording visits only recorded positions and stops at the end", async () => {
  let cursor = 0; const visited = [], errors = [];
  const recording = new RecordingPlayback({ position: () => ({ cursor, tip: 3 }), delay: () => 0, advance: async next => { visited.push(next); cursor = next; }, changed: () => {}, failed: error => errors.push(error) });
  recording.play(); await recording.settled();
  assert.deepEqual(visited, [1, 2, 3]); assert.equal(recording.playing, false); assert.deepEqual(errors, []);
  recording.play(); assert.deepEqual(visited, [1, 2, 3]);
});
test("pause wakes a recording delay and suppresses an outstanding display reply", async () => {
  let cursor = 0, resolveReply, visits = 0;
  const recording = new RecordingPlayback({ position: () => ({ cursor, tip: 3 }), delay: () => 5000, advance: async (next, current) => {
    visits++; await new Promise(resolve => { resolveReply = resolve; }); if (current()) cursor = next;
  }, changed: () => {}, failed: error => { throw error; } });
  recording.play(); recording.pause(); await recording.settled(); assert.equal(visits, 0);
  recording.delay = () => 0; recording.play(); await tick(); assert.equal(visits, 1);
  recording.pause(); resolveReply(); await recording.settled(); assert.equal(cursor, 0);
});
test("recording errors stop playback without continuing into live execution", async () => {
  const errors = [];
  const recording = new RecordingPlayback({ position: () => ({ cursor: 0, tip: 3 }), delay: () => 0, advance: async () => { throw new Error("bad position"); }, changed: () => {}, failed: error => errors.push(error.message) });
  recording.play(); await recording.settled(); assert.deepEqual(errors, ["bad position"]); assert.equal(recording.busy, false);
});
