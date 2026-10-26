import test from "node:test";
import assert from "node:assert/strict";
import { BrowserClient } from "../client.js";
class FakeWorker {
  listeners = {}; sent = [];
  addEventListener(type, callback) { this.listeners[type] = callback; }
  postMessage(text) { this.sent.push(JSON.parse(text)); }
  reply(data) { this.listeners.message({ data }); }
}
const turn = () => new Promise((resolve) => setImmediate(resolve));
function respond(worker, request, extras = {}) { worker.reply({ request_id: request.request_id, request_generation: request.generation, generation: "1", view: {}, error: null, ...extras }); }
test("worker readiness gates commands and wide identifiers remain strings", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker);
  const result = client.request({ type: "create", game: "tictactoe" }, { replace: true });
  assert.equal(worker.sent.length, 0); worker.reply({ type: "ready" }); await turn();
  assert.equal(worker.sent[0].generation, "0"); respond(worker, worker.sent[0]); await result;
  client.nextId = 9007199254740992n;
  const next = client.request({ type: "view" }); await turn();
  assert.equal(worker.sent[1].request_id, "9007199254740993"); respond(worker, worker.sent[1]); await next;
});
test("superseded replies cannot replace newer UI work", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker); worker.reply({ type: "ready" });
  const old = client.request({ type: "view" }); const rejected = assert.rejects(old, (e) => e.code === "obsolete_reply"); await turn();
  const next = client.request({ type: "create", game: "connect4" }, { replace: true }); await turn();
  respond(worker, worker.sent[0]); await rejected;
  respond(worker, worker.sent[1]); await next; assert.equal(client.generation, "1");
});
test("mismatched identities and worker failures reject pending work", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker); worker.reply({ type: "ready" });
  const request = client.request({ type: "view" }); const rejected = assert.rejects(request, (e) => e.code === "protocol_error"); await turn();
  respond(worker, worker.sent[0], { request_generation: "99" }); await rejected;
  const failed = client.request({ type: "view" }); const failure = assert.rejects(failed, (e) => e.code === "worker_failure"); await turn();
  worker.reply({ type: "failure", message: "test failure" }); await failure; assert.equal(client.pending.size, 0);
  await assert.rejects(client.request({ type: "view" }), (e) => e.code === "worker_failure");
});

test("oversized UTF-8 commands fail locally and leave no pending work", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker); worker.reply({ type: "ready" });
  await assert.rejects(client.request({ type: "view", padding: "\u00e9".repeat(17000) }), error => error.code === "resource_limit");
  assert.equal(worker.sent.length, 0); assert.equal(client.pending.size, 0);
});

test("worker failure while awaiting readiness prevents dispatch", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker); worker.reply({ type: "ready" });
  const request = client.request({ type: "view" });
  client.fail("Engine stopped.");
  await assert.rejects(request, error => error.code === "worker_failure");
  assert.equal(worker.sent.length, 0); assert.equal(client.pending.size, 0);
});

test("failed dispatch and uncorrelated errors settle all pending work", async () => {
  const worker = new FakeWorker(); const client = new BrowserClient(worker); worker.reply({ type: "ready" });
  worker.postMessage = () => { throw new Error("Worker disconnected."); };
  await assert.rejects(client.request({ type: "view" }), error => error.code === "worker_failure");
  assert.equal(client.pending.size, 0);
  const otherWorker = new FakeWorker(); const other = new BrowserClient(otherWorker); otherWorker.reply({ type: "ready" });
  const request = other.request({ type: "view" });
  const rejected = assert.rejects(request, error => error.code === "protocol_error");
  await turn(); otherWorker.reply({ request_id: "", error: { code: "invalid_request" } });
  await rejected; assert.equal(other.pending.size, 0);
});


test("presentation epochs refresh stale private views without discarding a prepared AI identity", async () => {
  const worker=new FakeWorker(),client=new BrowserClient(worker);worker.reply({type:"ready"});
  const makeView=secret=>({game:"loveletter",board:{loveletter:{hand:[],...(secret?{privileged:{hands:[["princess"]]}}:{})}}});
  const reveal=client.request({type:"presentation",observer:null,public_only:false,reveal_hidden_information:true});await turn();
  respond(worker,worker.sent[0],{view:makeView(true)});await reveal;
  const prepared=client.request({type:"prepare_ai"},{positionVersion:"3"});await turn();
  const off=client.request({type:"presentation",observer:null,public_only:false,reveal_hidden_information:false});await turn();
  respond(worker,worker.sent[2],{view:makeView(false)});await off;
  respond(worker,worker.sent[1],{view:makeView(true),proposal:{id:"17",pending:false,action:null}});await turn();
  assert.equal(worker.sent[3].command.type,"view");
  respond(worker,worker.sent[3],{view:makeView(false)});
  const accepted=await prepared;
  assert.equal(accepted.proposal.id,"17");assert.ok(!accepted.view.board.loveletter.privileged);
  assert.ok(worker.sent.every(r=>r.command.type!=="discard_ai"));
});
