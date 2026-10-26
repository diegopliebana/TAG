// SVG draws public positions only. Rust supplies every legal action and transition.
export const playerColors = ["#df704e", "#35a9c7", "#d4aa32", "#a98ae3"];
export const playerShapes = ["circle", "diamond", "square", "triangle"];
export function cantStopActionLabel(action) {
  if (typeof action === "string") return ({ Roll: "Roll all four dice", Stop: "Stop and bank progress", Bust: "Bust — end turn" })[action] ?? action;
  const allocation = action.Allocate;
  return allocation ? `Advance ${allocation.columns.slice(0, allocation.length).join(" + ")}` : "Unknown action";
}
const NS = "http://www.w3.org/2000/svg";
const pips = [[], [4], [0, 8], [0, 4, 8], [0, 2, 6, 8], [0, 2, 4, 6, 8], [0, 2, 3, 5, 6, 8]];

export class CantStopBoard {
  constructor(container, view, sprites) {
    Object.assign(this, { container, game: view.game, width: view.board.width, height: view.board.height, sprites });
    const d = container.ownerDocument;
    this.svgNode = (tag, attrs = {}, text) => {
      const node = d.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text !== undefined) node.textContent = text;
      return node;
    };
    this.root = d.createElement("div"); this.root.className = "cantstop-board";
    this.caption = d.createElement("p"); this.caption.className = "cantstop-caption";
    this.dice = this.svgNode("svg", { viewBox: "0 0 220 52", role: "img", class: "cantstop-dice" });
    this.svg = this.svgNode("svg", { role: "img", class: "cantstop-tracks" });
    this.actions = d.createElement("div"); this.actions.className = "cantstop-actions";
    this.actions.setAttribute("role", "group"); this.actions.setAttribute("aria-label", "Available Can't Stop actions");
    this.legend = d.createElement("p"); this.legend.className = "cantstop-legend";
    this.root.append(this.caption, this.dice, this.svg, this.legend, this.actions);
    container.replaceChildren(this.root); this.buttons = new Map();
  }
  update(view, choose, busy) {
    const p = view.board.cantstop, n = view.controllers.length;
    const max = Math.max(...p.rules.column_heights), height = max * 24 + 84, bottom = height - 44;
    this.root.setAttribute("data-phase", p.phase);
    const phase = { start_turn: "Start turn: roll the dice", allocation: "Choose an allocation, or accept a bust", decision: "Bank your progress or roll again" }[p.phase];
    this.caption.textContent = view.next_player === null ? `Player ${view.board.outcome.player} wins` : `Player ${view.next_player} · ${phase}`;
    this.dice.setAttribute("aria-label", p.dice ? `Dice: ${p.dice.join(", ")}` : "Dice have not been rolled");
    this.dice.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const x = i * 54 + 4;
      this.dice.append(this.svgNode("rect", { x, y: 4, width: 44, height: 44, rx: 9, class: "cantstop-die" }));
      for (const dot of pips[p.dice?.[i] ?? 0]) this.dice.append(this.svgNode("circle", { cx: x + 11 + dot % 3 * 11, cy: 15 + Math.floor(dot / 3) * 11, r: 3.5, class: "cantstop-pip" }));
      if (!p.dice) this.dice.append(this.svgNode("text", { x: x + 22, y: 33, "text-anchor": "middle", class: "cantstop-number" }, "?"));
    }
    this.svg.setAttribute("viewBox", `0 0 748 ${height}`);
    const summary = [];
    this.svg.replaceChildren();
    const marker = (seat, x, y, temporary = false) => {
      const color = temporary ? "#faf7ef" : playerColors[seat];
      const attrs = { fill: color, stroke: "#202b33", "stroke-width": temporary ? 3 : 1.5 };
      let node;
      if (temporary || seat === 0) node = this.svgNode("circle", { ...attrs, cx: x, cy: y, r: temporary ? 9 : 6.5 });
      else if (seat === 1) node = this.svgNode("polygon", { ...attrs, points: `${x},${y - 8} ${x + 7},${y} ${x},${y + 8} ${x - 7},${y}` });
      else if (seat === 2) node = this.svgNode("rect", { ...attrs, x: x - 6, y: y - 6, width: 12, height: 12, rx: 1 });
      else node = this.svgNode("polygon", { ...attrs, points: `${x},${y - 8} ${x + 7},${y + 6} ${x - 7},${y + 6}` });
      return node;
    };
    for (let c = 0; c < 11; c++) {
      const x = 34 + c * 68, h = p.rules.column_heights[c], top = bottom - h * 24;
      const group = this.svgNode("g", { "data-column": c + 2 });
      group.append(this.svgNode("rect", { x: x - 26, y: top - 18, width: 52, height: h * 24 + 42, rx: 20, class: "cantstop-track", ...(p.owners[c] === null ? {} : { style: `fill:${playerColors[p.owners[c]]};fill-opacity:.25` }) }));
      group.append(this.svgNode("text", { x, y: top - 27, "text-anchor": "middle", class: "cantstop-number" }, c + 2));
      for (let step = 1; step <= h; step++) group.append(this.svgNode("line", { x1: x - 19, x2: x + 19, y1: bottom - step * 24, y2: bottom - step * 24, class: "cantstop-rung" }));
      const facts = [];
      for (let seat = 0; seat < n; seat++) {
        const progress = p.permanent[seat][c]; facts.push(`Player ${seat}: ${progress}/${h}`);
        if (progress || p.owners[c] === seat) group.append(marker(seat, x + (seat - (n - 1) / 2) * 11, bottom - progress * 24));
      }
      if (p.temporary[c] !== null) { group.append(marker(0, x, bottom - p.temporary[c] * 24, true)); facts.push(`temporary: ${p.temporary[c]}/${h}`); }
      if (p.owners[c] !== null) group.append(this.svgNode("text", { x, y: bottom + 19, "text-anchor": "middle", class: "cantstop-number" }, `P${p.owners[c]}`));
      const detail = `Column ${c + 2}. ${facts.join(". ")}${p.owners[c] === null ? "" : `. Claimed by Player ${p.owners[c]}`}`;
      group.append(this.svgNode("title", {}, detail)); summary.push(detail); this.svg.append(group);
    }
    this.svg.setAttribute("aria-label", summary.join(". "));
    this.legend.textContent = `${Array.from({ length: n }, (_, s) => `P${s} ${playerShapes[s]}`).join(" · ")} · White circles: temporary progress · Claim ${p.rules.columns_to_win} columns to win`;
    const enabled = !busy && view.next_player !== null && view.controllers[view.next_player].kind === "human";
    const oldFocus = this.container.ownerDocument.activeElement;
    let focusKey = null;
    for (const [key, button] of this.buttons) if (button === oldFocus) focusKey = key;
    if (focusKey) this.focusedKey = focusKey;
    const next = new Map();
    for (const action of view.actions) {
      const key = JSON.stringify(action);
      const button = this.buttons.get(key) ?? this.container.ownerDocument.createElement("button");
      button.type = "button"; button.textContent = cantStopActionLabel(action); button.disabled = !enabled;
      button.onclick = () => { if (!button.disabled) choose(action); };
      next.set(key, button);
    }
    this.buttons = next; this.actions.replaceChildren(...next.values());
    if (enabled && this.focusedKey) { (next.get(this.focusedKey) ?? next.values().next().value)?.focus(); this.focusedKey = null; }
  }
}
