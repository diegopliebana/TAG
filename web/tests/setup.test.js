import test from "node:test";
import assert from "node:assert/strict";
import { setupFields, readParameters, readSeed, CryptoSeeds } from "../setup.js";

test("Can't Stop setup separates player count from rules and preserves exact dice seeds", () => {
  const inputs = setupFields("cantstop").map(f => ({ name: f.name, value: String(f.value) }));
  const set = (name, value) => { inputs.find(i => i.name === name).value = value; };
  set("seed", "18446744073709551615"); set("players", "4");
  const p = readParameters("cantstop", inputs);
  assert.equal(p.seed, "18446744073709551615"); assert.equal(p.players, undefined);
  assert.deepEqual(p.rules.column_heights, [3,5,7,9,11,13,11,9,7,5,3]);
  set("column_7", "129"); assert.throws(() => readParameters("cantstop", inputs), /128/);
  set("column_7", "13"); set("columns_to_win", "4"); assert.throws(() => readParameters("cantstop", inputs), /winning target/);
});
const inputs = values => Object.entries(values).map(([name, value]) => ({ name, value: String(value) }));
test("configured setup preserves valid values and enforces browser limits", () => {
  assert.equal(readParameters("tictactoe", inputs({ grid_size: 1 })).grid_size, 1);
  assert.equal(readParameters("tictactoe", inputs({ grid_size: 64 })).grid_size, 64);
  for (const grid_size of [0, -1, 1.5, 65, 129, "", "garbage", Infinity]) assert.throws(() => readParameters("tictactoe", inputs({ grid_size })));
  assert.deepEqual(setupFields("connect4").map(field => field.value), [7, 6, 4]);
  const p = readParameters("connect4", inputs({ width: 128, height: 32, winning_length: 128 }));
  assert.equal(p.limits.max_cells, 4096);
  for (const values of [{ width: 128, height: 33, winning_length: 4 }, { width: 4, height: 3, winning_length: 5 }, { width: 4, height: 3, winning_length: 0 }]) assert.throws(() => readParameters("connect4", inputs(values)));
});

test("seeds preserve the full u64 range and automatic seeds use both OS-random words", () => {
  for (const seed of ["0", "9007199254740993", "18446744073709551615"]) assert.equal(readSeed(seed), seed);
  assert.equal(readSeed(" 0007 "), "7");
  for (const seed of ["", "-1", "1.5", "1e3", "+1", "18446744073709551616"]) assert.throws(() => readSeed(seed));
  const values = [[0, 0], [0xffffffff, 0xffffffff], [0x200000, 1]];
  let calls = 0;
  const seeds = new CryptoSeeds(words => { words.set(values[calls++]); return words; });
  assert.equal(seeds.next(), "0");
  assert.equal(seeds.next(), "18446744073709551615");
  assert.equal(seeds.next(), "9007199254740993");
  assert.equal(calls, 3);
  assert.throws(() => new CryptoSeeds(() => { throw new Error("entropy unavailable"); }).next(), /entropy unavailable/);
});
