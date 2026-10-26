// UI defaults mirror the current Rust game defaults. Rust validates every setup.
export function setupFields(game) {
  if (game === "loveletter") return [
    { name: "players", label: "Players (2?4)", value: 3 },
    { name: "seed", label: "Game deal seed (integer or auto)", value: "auto", type: "text" },
    { name: "first_player", label: "Initial player (0-based)", value: 0 },
    ...[6,5,4].map((value,i)=>({name:`target_${i+2}`,label:`Favor tokens to win (${i+2} players)`,value})),
  ];
  if (game === "cantstop") return [
    { name: "players", label: "Players (2–4)", value: 2 },
    { name: "seed", label: "Game dice seed (integer or auto)", value: "auto", type: "text" },
    { name: "markers", label: "Temporary markers (1–11)", value: 3 },
    { name: "columns_to_win", label: "Columns to win", value: 3 },
    ...[3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3].map((value, i) => ({ name: `column_${i + 2}`, label: `Column ${i + 2} height (1–128)`, value })),
  ];
  return game === "tictactoe" ? [{ name: "grid_size", label: "Board size", value: 3 }] : [
    { name: "width", label: "Width", value: 7 }, { name: "height", label: "Height", value: 6 },
    { name: "winning_length", label: "Winning length", value: 4 },
  ];
}
export function readParameters(game, inputs) {
  if (game === "loveletter") {
    const values=Object.fromEntries(inputs.map(i=>[i.name,i.value]));
    const n=Number(values.players),first=Number(values.first_player);
    if (!Number.isInteger(n)||n<2||n>4) throw new Error("Love Letter requires 2?4 players.");
    if (!Number.isInteger(first)||first<0||first>=n) throw new Error("Initial player must be a valid seat.");
    const targets=[2,3,4].map(n=>Number(values[`target_${n}`]));
    if (targets.some(n=>!Number.isInteger(n)||n<1||n>4294967295)) throw new Error("Token targets must be positive u32 integers.");
    const raw=values.seed?.trim();
    return {rules:{winning_tokens:targets},first_player:first,seed:!raw||raw==="auto"?new CryptoSeeds().next():readSeed(raw)};
  }
  if (game === "cantstop") {
    const values = Object.fromEntries(inputs.map(input => [input.name, input.value]));
    const integer = name => { const n = Number(values[name]); if (!Number.isSafeInteger(n) || n < 1) throw new Error(`${name} must be a positive whole number.`); return n; };
    const players = integer("players"), markers = integer("markers"), columns_to_win = integer("columns_to_win");
    const column_heights = Array.from({ length: 11 }, (_, i) => integer(`column_${i + 2}`));
    if (players < 2 || players > 4) throw new Error("Can't Stop requires 2–4 players.");
    if (markers > 11 || columns_to_win > Math.ceil(11 / players)) throw new Error("Use 1–11 markers and a winning target no greater than ceil(11 / players).");
    if (column_heights.some(h => h > 128)) throw new Error("Browser column heights must be at most 128.");
    const raw = values.seed?.trim();
    const seed = !raw || raw === "auto" ? new CryptoSeeds().next() : readSeed(raw);
    return { rules: { column_heights, markers, columns_to_win }, seed };
  }
  const values = Object.fromEntries(inputs.map(input => [input.name, Number(input.value)]));
  for (const field of setupFields(game)) {
    if (!Number.isSafeInteger(values[field.name]) || values[field.name] < 1) throw new Error(`${field.label} must be a positive whole number.`);
  }
  const width = game === "tictactoe" ? values.grid_size : values.width;
  const height = game === "tictactoe" ? values.grid_size : values.height;
  if (width > 128 || height > 128 || width * height > 4096) throw new Error("Use at most 128 cells per dimension and 4,096 cells total.");
  if (game === "connect4" && values.winning_length > Math.max(width, height)) throw new Error("Winning length cannot exceed the longest board dimension.");
  return { ...values, limits: { max_cells: 4096 } };
}

const MAX_SEED = (1n << 64n) - 1n;
export function readSeed(text) {
  const value = text.trim();
  if (!/^\d+$/.test(value) || BigInt(value) > MAX_SEED) throw new Error("Seed must be a whole number from 0 to 18,446,744,073,709,551,615.");
  return BigInt(value).toString();
}
// Resolve seeds at setup through browser Web Crypto. Keep decimal strings so
// values above JavaScript's safe integer range retain every bit.
export class CryptoSeeds {
  constructor(fill = values => {
    if (!globalThis.crypto?.getRandomValues) throw new Error("Automatic seeds require Web Crypto; use localhost or HTTPS.");
    return globalThis.crypto.getRandomValues(values);
  }) { this.fill = fill; }
  next() {
    const words = new Uint32Array(2);
    this.fill(words);
    return ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString();
  }
}
