import test from "node:test";
import assert from "node:assert/strict";
import { renderBoard } from "../board.js";
import { cantStopActionLabel } from "../cantstop-board.js";
class Element {
  constructor(document, tag) { Object.assign(this, { ownerDocument: document, tag, children: [], attributes: {} }); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = value; }
  focus() { this.ownerDocument.activeElement = this; }
}
function fixture(players = 4) {
  const document = { createElement: tag => new Element(document, tag), createElementNS: (_, tag) => new Element(document, tag) };
  const container = document.createElement("div");
  const view = { game: "cantstop", next_player: 0, controllers: Array.from({ length: players }, () => ({ kind: "human" })),
    actions: ["Roll"], board: { width: 11, height: 13, outcome: { kind: "ongoing" }, cantstop: {
      phase: "start_turn", dice: null, rules: { column_heights: [3,5,7,9,11,13,11,9,7,5,3], columns_to_win: 3 },
      owners: Array(11).fill(null), temporary: Array(11).fill(null), permanent: Array.from({ length: players }, () => Array(11).fill(0)) } } };
  return { container, document, view };
}
test("vector tracks, dice and shapes describe public progress and use only offered actions", () => {
  const { container, document, view } = fixture(); let selected;
  const choose = action => { selected = action; };
  renderBoard(container, view, choose);
  const [caption, dice, svg, legend, actions] = container.children[0].children;
  assert.equal(svg.children.length, 11); assert.equal(dice.children.length, 8);
  assert.match(caption.textContent, /Start turn/); assert.match(legend.textContent, /P3 triangle/);
  const roll = actions.children[0]; roll.focus(); roll.onclick(); assert.equal(selected, "Roll");
  renderBoard(container, view, choose, { busy: true }); assert.equal(actions.children[0], roll); assert.equal(roll.disabled, true);
  document.activeElement = null; // Disabling an active button may move browser focus.
  view.actions = [{ Allocate: { columns: [6, 8], length: 2 } }, { Allocate: { columns: [7, 7], length: 2 } }];
  view.board.cantstop.dice = [1,2,3,6]; view.board.cantstop.phase = "allocation";
  renderBoard(container, view, choose);
  assert.equal(document.activeElement, actions.children[0]); assert.equal(actions.children.length, 2);
  assert.match(dice.attributes["aria-label"], /1, 2, 3, 6/);
  assert.equal(actions.children[1].textContent, "Advance 7 + 7");
  actions.children[1].onclick(); assert.deepEqual(selected, view.actions[1]);
  view.board.cantstop.permanent[3][0] = 3; view.board.cantstop.owners[0] = 3;
  view.board.cantstop.temporary[5] = 8;
  view.actions = []; view.next_player = null; view.board.outcome = { kind: "won", player: 3 };
  renderBoard(container, view, choose);
  assert.match(caption.textContent, /Player 3 wins/); assert.equal(actions.children.length, 0);
  assert.match(svg.attributes["aria-label"], /Claimed by Player 3/);
  assert.match(svg.attributes["aria-label"], /temporary: 8\/13/);
});
test("Can't Stop move labels retain the repeated sum and explicit bust", () => {
  assert.equal(cantStopActionLabel("Bust"), "Bust — end turn");
  assert.equal(cantStopActionLabel({ Allocate: { columns: [7,0], length: 1 } }), "Advance 7");
});
