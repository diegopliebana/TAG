import { BrowserClient } from "../client.js";
function equal(actual, expected, label) { if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${label}: ${JSON.stringify(actual)} differs from ${JSON.stringify(expected)}`); }
function board(view, expected, label) { for (const key of ["cells", "current_player", "counters", "outcome", "winning_cells"]) equal(view.board[key], expected[key], `${label} ${key}`); }
async function rejects(promise, code) { try { await promise; } catch (error) { equal(error.code, code, "error code"); return; } throw new Error(`Expected ${code}`); }
export async function runBrowserChecks() {
  const validation = new BrowserClient(); await validation.ready;
  await rejects(validation.request({ type: "unknown_command" }), "invalid_request");
  await rejects(validation.request({ type: "create", game: "tictactoe", controllers: "invalid" }), "invalid_request");
  await rejects(validation.request({ type: "view", padding: "\u00e9".repeat(17000) }), "resource_limit");
  await validation.request({ type: "create", game: "tictactoe" });
  equal(validation.pending.size, 0, "rejected requests settle"); validation.worker.terminate();
  const workloads = await (await fetch("/.scratch/browser-proof/workload.json")).json();
  let transitions = 0, historicalPositions = 0, importedPositions = 0;
  for (const data of workloads) {
    const client = new BrowserClient(); await client.ready;
    let reply = await client.request({ type: "create", game: data.game, parameters: data.parameters, controllers: data.controllers }, { replace: true });
    board(reply.view, data.initial, data.name);
    for (const move of data.decisions) {
      equal(reply.view.actions, move.actions, `${data.name} offered actions`);
      equal(reply.view.next_player, move.actor, `${data.name} actor`);
      const version = reply.view.position_version;
      const prepared = await client.request({ type: "prepare_ai" }, { positionVersion: version });
      equal(prepared.proposal.action, move.action, `${data.name} AI choice`);
      equal(prepared.view.board, reply.view.board, `${data.name} preparation preserves live state`);
      reply = await client.request({ type: "commit_ai", proposal: prepared.proposal.id }, { positionVersion: version });
      board(reply.view, move.board, data.name);
      equal(reply.view.next_player, move.next_player, `${data.name} next player`); transitions++;
    }
    equal(reply.view.actions, [], `${data.name} terminal actions`);
    const version = reply.view.position_version;
    for (let cursor = 0; cursor <= data.decisions.length; cursor++) {
      const historical = await client.request({ type: "history", branch: 0, cursor });
      board(historical.view, cursor === 0 ? data.initial : data.decisions[cursor - 1].board, `${data.name} history ${cursor}`);
      equal(historical.view.position_version, version, "history preserves live version");
      equal(historical.view.read_only, cursor < data.decisions.length, "history read-only boundary");
      historicalPositions++;
    }
    client.worker.terminate();
  }
  // Load independently produced native replay documents and compare every position.
  for (const data of workloads) {
    const imported = new BrowserClient(); await imported.ready;
    await imported.requestFile({ type: "import", controllers: null }, data.replay);
    for (let cursor = 0; cursor <= data.decisions.length; cursor++) {
      const reply = await imported.request({ type: "history", branch: 0, cursor });
      board(reply.view, cursor === 0 ? data.initial : data.decisions[cursor - 1].board, `${data.name} imported ${cursor}`);
      equal(reply.view.next_player, cursor === 0 ? 0 : data.decisions[cursor - 1].next_player, "imported next player"); importedPositions++;
    }
    imported.dispose();
  }
  // Explicit human input follows the same native recorded moves in both games.
  for (const data of workloads.filter((x) => x.name.endsWith("standard"))) {
    const client = new BrowserClient(); await client.ready;
    let reply = await client.request({ type: "create", game: data.game }, { replace: true });
    for (const move of data.decisions) {
      reply = await client.request({ type: "human_move", actor: move.actor, action: move.action }, { positionVersion: reply.view.position_version });
      board(reply.view, move.board, `human ${data.name}`); transitions++;
    }
    client.worker.terminate();
  }
  const client = new BrowserClient(); await client.ready;
  let reply = await client.request({ type: "create", game: "tictactoe", controllers: [{ kind: "osla", seed: "7" }, { kind: "human" }] }, { replace: true });
  const first = await client.request({ type: "prepare_ai" }, { positionVersion: "0" });
  await client.request({ type: "discard_ai", proposal: first.proposal.id }, { positionVersion: "0" });
  const second = await client.request({ type: "prepare_ai" }, { positionVersion: "0" });
  equal(second.proposal.action, first.proposal.action, "discard preserves RNG continuation");
  reply = await client.request({ type: "commit_ai", proposal: second.proposal.id }, { positionVersion: "0" });
  await rejects(client.request({ type: "human_move", actor: 1, action: { x: 0, y: 0 } }, { positionVersion: "0" }), "stale_position");
  const generation = client.generation;
  reply = await client.request({ type: "create", game: "connect4" }, { replace: true });
  const stale = new Promise((resolve) => {
    const listener = ({ data }) => { if (data.request_id === "999999") { client.worker.removeEventListener("message", listener); resolve(data); } };
    client.worker.addEventListener("message", listener);
  });
  client.worker.postMessage(JSON.stringify({ request_id: "999999", generation, position_version: "0", command: { type: "human_move", actor: 0, action: { x: 1, y: 1 } } }));
  equal((await stale).error.code, "stale_generation", "old lifecycle rejected");
  const unchanged = await client.request({ type: "view" }); equal(unchanged.view.board, reply.view.board, "replacement preserved");
  client.worker.terminate();
  return { workloads: workloads.length, transitions, historicalPositions, importedPositions, humanGames: 2, checks: ["ordered actions", "AI choices", "all intermediate board/progress/outcome fields", "historical native equivalence", "native file equivalence", "human input", "proposal cancellation", "stale position", "stale lifecycle", "correlated decoding errors", "UTF-8 request limits"] };
}
