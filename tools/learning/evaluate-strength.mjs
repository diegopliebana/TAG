// Run the actual browser Wasm engine on Python prediction rows and complete games.
// No browser UI, GPU or training is involved. Run after evaluate-strength.py.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { initSync, BrowserEngine } from '../../web/pkg/tag_browser.js';
import { BrowserClient } from '../../web/client.js';

initSync({ module: readFileSync(new URL('../../web/pkg/tag_browser_bg.wasm', import.meta.url)) });
const directory = resolve(process.argv[2] ?? 'target/strength-evaluation');
const python = JSON.parse(readFileSync(`${directory}/python.json`, 'utf8'));
class EngineWorker {
  listeners = {}; engine = new BrowserEngine();
  addEventListener(type, listener) { this.listeners[type] = listener; }
  postMessage(data) { this.listeners.message({ data: JSON.parse(data?.type === 'file' ? this.engine.handle_file(data.envelope, data.text) : this.engine.handle(data)) }); }
  terminate() { this.engine.free(); }
}
const worker = new EngineWorker(), client = new BrowserClient(worker);
worker.listeners.message({ data: { type: 'ready' } });
const actionId = action => action.x * 3 + action.y;
const action = id => ({ x: Math.floor(id / 3), y: id % 3 });
async function step(view) {
  const prepared = await client.request({ type: 'prepare_ai' }, { positionVersion: view.position_version });
  return (await client.request({ type: 'commit_ai', proposal: prepared.proposal.id }, { positionVersion: view.position_version })).view;
}
async function human(view, id) {
  return (await client.request({ type: 'human_move', actor: view.next_player, action: action(id) }, { positionVersion: view.position_version })).view;
}
const counts = () => ({ episodes: 0, wins: 0, draws: 0, losses: 0 });
const outcome = () => ({ combined: counts(), per_seat: { 0: counts(), 1: counts() } });
function record(total, view, seat) {
  const key = view.scores[seat] > 0 ? 'wins' : view.scores[seat] < 0 ? 'losses' : 'draws';
  for (const row of [total.combined, total.per_seat[seat]]) { row.episodes++; row[key]++; }
}
try {
  const reply = await client.requestFile({ type: 'load_model' }, readFileSync(python.model, 'utf8'));
  const modelId = reply.text;
  assert.equal(modelId, python.model_identity);
  const results = { model_identity: modelId, prediction_rows: 0, max_absolute_error: 0, trace_games: 0, trace_moves: 0, evaluations: [] };
  for (const row of python.prediction_rows) {
    let view = (await client.request({ type: 'create', game: 'tictactoe' })).view;
    for (const id of row.prefix) view = await human(view, id);
    const actual = JSON.parse((await client.requestFile({ type: 'predict_model', model_id: modelId, observer: row.observer })).text);
    assert.deepEqual(actual.inputs, row.inputs);
    for (const [i, value] of [...row.logits, row.value].entries()) {
      const got = [...actual.logits, actual.value][i], error = Math.abs(got - value);
      results.max_absolute_error = Math.max(results.max_absolute_error, error);
      assert.ok(error <= 2e-6 + 2e-5 * Math.abs(value));
    }
    results.prediction_rows++;
  }
  // For Random, replay NumPy's recorded opponent moves; for OSLA, both seats
  // execute independently in Wasm. This distinguishes action parity from RNG parity.
  for (const group of python.traces) for (const trace of group.games) {
    const controllers = [{ kind: 'human' }, { kind: 'human' }];
    controllers[trace.seat] = { kind: 'neural_osla', seed: String(trace.learner_seed), model_id: modelId };
    if (group.opponent === 'osla') controllers[1-trace.seat] = { kind: 'osla', seed: String(trace.opponent_seed) };
    let view = (await client.request({ type: 'create', game: 'tictactoe', controllers })).view;
    for (const expected of trace.actions) {
      view = group.opponent === 'random' && view.next_player !== trace.seat ? await human(view, expected) : await step(view);
      assert.equal(actionId(view.accepted_move.action), expected, `trace ${group.seed}/${trace.learner_seed}`);
      results.trace_moves++;
    }
    assert.equal(view.next_player, null);
    assert.equal(view.scores[trace.seat], { 1: 1, 2: 0, 3: -1 }[trace.result]);
    results.trace_games++;
  }
  const models = [{ kind: 'neural_osla', id: modelId }];
  let qRows;
  if (python.q_model) {
    const source = readFileSync(python.q_model, 'utf8');
    const q = await client.requestFile({ type: 'load_model' }, source);
    models.push({ kind: 'q_learning', id: q.text });
    qRows = new Map(JSON.parse(source).rows.map(row => [row.cells.join(','), row.values]));
  }
  for (const model of models) for (const opponent of ['random', 'osla']) for (const seed of python.seeds) {
    const totals = outcome(), coverage = { decisions: 0, unseen: 0, all_legal_values_equal: 0 };
    for (let logical = 0; logical < python.episodes_per_seed; logical++) {
      const seat = logical % 2, controllers = [null, null];
      controllers[seat] = { kind: model.kind, seed: String(seed+2*logical), model_id: model.id };
      controllers[1-seat] = { kind: opponent, seed: String(seed+2*logical+1) };
      let view = (await client.request({ type: 'create', game: 'tictactoe', controllers })).view;
      while (view.next_player !== null) {
        let expected;
        if (model.kind === 'q_learning' && view.next_player === seat) {
          const key = view.board.cells.map(cell => cell === null ? 0 : cell === seat ? 1 : -1).join(',');
          const row = qRows.get(key);
          coverage.decisions++; if (!row) coverage.unseen++;
          const legal = view.actions.map(actionId), values = row ?? Array(9).fill(0);
          const scores = legal.map(id => values[id]);
          if (Math.max(...scores) === Math.min(...scores)) coverage.all_legal_values_equal++;
          const prediction = JSON.parse((await client.requestFile({ type: 'predict_model', model_id: model.id, observer: seat })).text);
          assert.deepEqual(prediction.q_values, values);
          expected = { values, best: Math.max(...scores) };
        }
        view = await step(view);
        if (expected) assert.equal(expected.values[actionId(view.accepted_move.action)], expected.best);
      }
      record(totals, view, seat);
    }
    const result = { mode: model.kind, opponent, seed, ...totals, ...(model.kind === 'q_learning' ? { coverage } : {}) };
    results.evaluations.push(result); console.log(JSON.stringify(result));
  }
  writeFileSync(`${directory}/wasm.json`, JSON.stringify(results, null, 2)+'\n');
  console.log(JSON.stringify({ prediction_rows: results.prediction_rows, max_absolute_error: results.max_absolute_error, trace_games: results.trace_games, trace_moves: results.trace_moves }));
} finally { client.dispose(); }
