import { LoveLetterBoard } from "./loveletter-board.js";
import { CantStopBoard } from "./cantstop-board.js";
// Keep each board's controls and update their values and offered actions in place.
const boards = new WeakMap();

class BoardRenderer {
  constructor(container, view, sprites) {
    this.container = container;
    this.game = view.game;
    this.width = view.board.width;
    this.height = view.board.height;
    this.sprites = sprites;
    this.actions = new Map();
    this.buttons = [];
    this.cells = [];
    this.focused = null;
    const document = container.ownerDocument;
    const table = document.createElement("table");
    if (sprites) table.className = "art-board";
    table.setAttribute("aria-label", this.game === "tictactoe" ? "Tic-Tac-Toe board" : "Connect 4 board");
    const addButton = (cell, key) => {
      const button = document.createElement("button");
      button.addEventListener("click", () => {
        const action = this.actions.get(key);
        if (!button.disabled && action) this.choose(action);
      });
      cell.append(button);
      this.buttons.push({ button, key });
      return button;
    };
    if (this.game === "connect4") {
      const header = table.createTHead().insertRow();
      for (let x = 0; x < this.width; x++) addButton(header.insertCell(), x).textContent = `Column ${x + 1}`;
    }
    const body = table.createTBody();
    for (let y = 0; y < this.height; y++) {
      const row = body.insertRow();
      for (let x = 0; x < this.width; x++) {
        const cell = row.insertCell();
        const index = x * this.height + y;
        const element = this.game === "tictactoe" ? addButton(cell, index) : cell;
        const entry = { element, index, x, y };
        if (sprites) {
          element.setAttribute("data-x", x); element.setAttribute("data-y", y);
          const image = (src, className) => {
            const art = document.createElement("img"); if (src) art.src = src; art.alt = "";
            art.className = className; art.setAttribute("aria-hidden", "true"); element.append(art); return art;
          };
          image(this.game === "tictactoe" ? sprites.tttCell : sprites.c4Cell, "cell-art");
          if (this.game === "connect4") image(sprites.socket, "socket-art");
          entry.piece = image("", "piece-art"); entry.piece.hidden = true;
          if (this.game === "connect4") {
            entry.number = document.createElement("span"); entry.number.className = "piece-number"; element.append(entry.number);
          }
        }
        this.cells.push(entry);
      }
    }
    container.replaceChildren(table);
  }

  update(view, choose, busy, emphasis) {
    this.choose = choose;
    // Index each offered action once, using the board's x-major coordinate order.
    this.actions = new Map(view.actions.map(action => [this.game === "tictactoe" ? action.x * this.height + action.y : action.column, action]));
    const human = view.next_player !== null && view.controllers[view.next_player].kind === "human";
    const focused = this.container.ownerDocument.activeElement;
    if (this.buttons.some(({ button }) => button === focused)) this.focused = focused;
    for (const { button, key } of this.buttons) button.disabled = busy || !human || !this.actions.has(key);
    const winning = new Set(emphasis.map(([x, y]) => x * this.height + y));
    for (const { element, piece, number, index, x, y } of this.cells) {
      const value = view.board.cells[index];
      const label = value === null ? "-" : this.game === "tictactoe" ? (value === 0 ? "X" : "O") : `P${value}`;
      if (piece) {
        piece.hidden = value === null;
        if (value !== null) {
          const src = this.game === "tictactoe" ? (value === 0 ? this.sprites.x : this.sprites.o) : (value === 0 ? this.sprites.red : this.sprites.yellow);
          if (piece.src !== src) piece.src = src;
        }
        if (number) number.textContent = value === null ? "" : value;
        element.setAttribute("data-winning", winning.has(index));
      } else if (element.textContent !== label) element.textContent = label;
      element.setAttribute("aria-label", `Row ${y + 1}, column ${x + 1}: ${value === null ? "empty" : label}`);
    }
    // Restore board focus after busy controls become available, including the next legal cell.
    if (!busy && this.focused) {
      const target = this.focused.disabled ? this.buttons.find(({ button }) => !button.disabled)?.button : this.focused;
      target?.focus();
      this.focused = null;
    }
  }
}

// Rebuild the structure when the selected game or dimensions change.
export function renderBoard(container, view, choose, { busy = false, sprites = null, emphasis = [] } = {}) {
  let board = boards.get(container);
  if (!board || board.game !== view.game || board.width !== view.board.width || board.height !== view.board.height || board.sprites !== sprites) {
    board = view.game === "loveletter" ? new LoveLetterBoard(container,view,sprites) : view.game === "cantstop" ? new CantStopBoard(container, view, sprites) : new BoardRenderer(container, view, sprites);
    boards.set(container, board);
  }
  board.update(view, choose, busy, emphasis);
}
