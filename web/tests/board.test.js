import test from "node:test";
import assert from "node:assert/strict";
import { renderBoard } from "../board.js";

// Minimal DOM double tracks element identity, focus and event handlers.
class Element {
  constructor(ownerDocument, tag) { this.ownerDocument = ownerDocument; this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {}; this.textContent = ""; }
  append(element) { this.children.push(element); }
  replaceChildren(...elements) { this.children = elements; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(event, listener) { this.listeners[event] = listener; }
  createTHead() { return this.insert("thead"); }
  createTBody() { return this.insert("tbody"); }
  insertRow() { return this.insert("tr"); }
  insertCell() { return this.insert("td"); }
  insert(tag) { const node = new Element(this.ownerDocument, tag); this.append(node); return node; }
  focus() { this.ownerDocument.activeElement = this; }
  buttons() { return [...(this.tag === "button" ? [this] : []), ...this.children.flatMap(child => child.buttons())]; }
}
function container() {
  const document = { activeElement: null, createElement(tag) { return new Element(this, tag); } };
  return new Element(document, "div");
}
function view(game) {
  return { game, next_player: 0, controllers: [{ kind: "human" }, { kind: "human" }], board: { width: 2, height: 2, cells: [null, null, null, null] }, actions: game === "tictactoe" ? [{ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 0 }, { x: 1, y: 1 }] : [{ column: 0 }, { column: 1 }] };
}

test("board updates retain controls, use current actions and preserve available focus", () => {
  const board = container(); const first = view("tictactoe"); const choices = [];
  renderBoard(board, first, action => choices.push(action));
  const table = board.children[0]; const buttons = board.buttons(); buttons[1].focus();
  renderBoard(board, first, action => choices.push(action), { busy: true });
  assert.ok(buttons.every(button => button.disabled));
  const next = view("tictactoe"); next.board.cells[0] = 0; next.next_player = 1; next.actions.shift();
  renderBoard(board, next, action => choices.push(action));
  assert.equal(board.children[0], table); assert.deepEqual(board.buttons(), buttons);
  assert.equal(buttons[0].textContent, "X"); assert.equal(buttons[0].disabled, true);
  assert.equal(board.ownerDocument.activeElement, buttons[1]);
  buttons[1].listeners.click(); assert.equal(choices[0], next.actions.find(action => action.x === 1 && action.y === 0));
  buttons[0].listeners.click(); assert.equal(choices.length, 1);
});

test("Connect 4 columns retain identity and game changes rebuild their structure", () => {
  const board = container(); const first = view("connect4"); const choices = [];
  renderBoard(board, first, action => choices.push(action));
  const buttons = board.buttons(); const next = view("connect4"); next.actions = [{ column: 1 }];
  renderBoard(board, next, action => choices.push(action));
  assert.deepEqual(board.buttons(), buttons); assert.equal(buttons[0].disabled, true);
  buttons[1].listeners.click(); assert.equal(choices[0], next.actions[0]);
  const old = board.children[0]; renderBoard(board, view("tictactoe"), () => {});
  assert.notEqual(board.children[0], old); assert.equal(board.buttons().length, 4);
});

test("candidate artwork updates existing layers and keeps empty pieces hidden", () => {
  const board = container(); const first = view("tictactoe");
  const sprites = { x: "x.svg", o: "o.png", tttCell: "cell.svg", c4Cell: "blue.svg", socket: "socket.svg", red: "red.svg", yellow: "yellow.svg" };
  renderBoard(board, first, () => {}, { busy: true, sprites });
  const button = board.buttons()[0]; const piece = button.children[1];
  assert.equal(piece.hidden, true); assert.equal(button.children[0].src, "cell.svg");
  first.board.cells[0] = 0;
  renderBoard(board, first, () => {}, { busy: true, sprites, emphasis: [[0, 0]] });
  assert.equal(board.buttons()[0], button); assert.equal(piece.hidden, false); assert.equal(piece.src, "x.svg");
  assert.equal(button.attributes["data-winning"], true);
  first.board.cells[0] = null;
  renderBoard(board, first, () => {}, { busy: true, sprites });
  assert.equal(piece.hidden, true); assert.equal(button.attributes["data-winning"], false);
});
