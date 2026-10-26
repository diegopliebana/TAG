import test from "node:test";
import assert from "node:assert/strict";
import { AiPlayback } from "../ai-playback.js";
const tick = () => new Promise(resolve => setImmediate(resolve));
const aiView = version => ({ position_version: String(version), next_player: 0, controllers: [{ kind: "osla" }] });
function fixture() {
  let view = aiView(0), ready = true;
  const requests = [], replies = [];
  const client = { generation: "1", request(command, options) { requests.push({ command, options }); return new Promise((resolve, reject) => replies.push({ resolve, reject })); } };
  const applied = [], errors = [];
  const playback = new AiPlayback({ client, view: () => view, ready: () => ready, apply: async next => { applied.push(next); view = next; }, changed: () => {}, failed: error => errors.push(error) });
  playback.setDelay(0);
  return { playback, requests, replies, applied, errors, setView: next => { view = next; }, setReady: value => { ready = value; } };
}
test("pause during calculation discards the proposal; resume prepares a fresh one", async () => {
  const f = fixture(); f.playback.step();
  assert.equal(f.requests[0].command.type, "prepare_ai");
  f.playback.pause(); f.replies.shift().resolve({ proposal: { id: "1" } }); await tick();
  assert.equal(f.requests[1].command.type, "discard_ai");
  f.replies.shift().resolve({}); await f.playback.settled();
  assert.equal(f.applied.length, 0); assert.equal(f.playback.busy, false);
  f.playback.step(); f.replies.shift().resolve({ proposal: { id: "2" } }); await tick();
  assert.deepEqual(f.requests[3], { command: { type: "commit_ai", proposal: "2" }, options: { positionVersion: "0" } });
  f.replies.shift().resolve({ view: aiView(1) }); await f.playback.settled();
  assert.equal(f.applied.length, 1); assert.equal(f.playback.playing, false);
});
test("pause during delay wakes immediately and prevents commitment", async () => {
  const f = fixture(); f.playback.setDelay(5000); f.playback.playing = true; f.playback.start(false);
  f.replies.shift().resolve({ proposal: { id: "1" } }); await tick();
  assert.equal(f.playback.phase, "waiting"); f.playback.pause(); await tick();
  assert.equal(f.requests[1].command.type, "discard_ai"); f.replies.shift().resolve({}); await f.playback.settled();
  assert.equal(f.applied.length, 0);
});

test("pause during a suspended search discards its pending calculation", async () => {
  const f = fixture(); f.playback.step();
  f.replies.shift().resolve({ proposal: { id: "9", pending: true } }); await tick();
  assert.equal(f.requests[1].command.type, "poll_ai");
  f.playback.pause();
  f.replies.shift().resolve({ proposal: { id: "9", pending: true } }); await tick();
  assert.deepEqual(f.requests[2].command, { type: "discard_ai", proposal: "9" });
  f.replies.shift().resolve({}); await f.playback.settled();
  assert.equal(f.applied.length, 0);
});
test("pause after dispatching commitment retains exactly the accepted move", async () => {
  const f = fixture(); f.playback.step(); f.replies.shift().resolve({ proposal: { id: "1" } }); await tick();
  assert.equal(f.playback.phase, "committing"); f.playback.pause();
  f.replies.shift().resolve({ view: aiView(1) }); await f.playback.settled();
  assert.equal(f.applied.length, 1); assert.equal(f.requests.length, 2); assert.equal(f.playback.playing, false);
});
test("automatic play waits for a human, respects readiness and stops on errors", async () => {
  const f = fixture(); f.setView({ ...aiView(0), controllers: [{ kind: "human" }] });
  f.playback.play(); await tick(); assert.equal(f.requests.length, 0); assert.equal(f.playback.playing, true);
  f.setView(aiView(1)); f.setReady(false); f.playback.kick(); await tick(); assert.equal(f.requests.length, 0);
  f.setReady(true); f.playback.step(); assert.equal(f.requests.length, 1);
  f.replies.shift().reject(new Error("evaluation failed")); await f.playback.settled();
  assert.equal(f.errors[0].message, "evaluation failed"); assert.equal(f.playback.playing, false);
  for (const delay of [-1, 5001, 1.5, NaN]) assert.throws(() => f.playback.setDelay(delay));
});


test("Love Letter retains the result observer and awaits acknowledgement before another AI turn",async()=>{
  const f=fixture();
  const before={...aiView(0),game:"loveletter",observer:1,board:{loveletter:{public_only:false}}};
  f.setView(before);let acknowledge;
  f.playback.apply=async next=>{f.setView(next);await new Promise(resolve=>acknowledge=resolve);};
  f.playback.step();f.replies.shift().resolve({proposal:{id:"result",pending:false}});await tick();
  assert.deepEqual(f.requests[1].command,{type:"commit_ai",proposal:"result",observer:1});
  f.replies.shift().resolve({view:{...before,position_version:"1"}});await tick();
  assert.equal(f.playback.busy,true);assert.equal(f.requests.length,2);
  f.playback.pause();acknowledge();await f.playback.settled();
  assert.equal(f.playback.busy,false);assert.equal(f.requests.length,2);
});
