// Update the displayed move selection without rebuilding existing buttons; append new accepted moves.
import { controllerName } from "./seat-fields.js";
import { loveLetterActionLabel } from "./loveletter-board.js";
import { cantStopActionLabel } from "./cantstop-board.js";
export function branchLabel(branch) {
  if (branch.parent === null) return `Original · ${branch.move_count} moves`;
  const name = branch.origin === "reset" ? `Reset ${branch.id}` : `Branch ${branch.id}`;
  return `${name} · from ${branch.parent === 0 ? "Original" : `branch ${branch.parent}`} at move ${branch.fork_position} · ${branch.move_count} moves`;
}
export function moveLabel(move) {
  const action = move.action;
  const choice = typeof action === "string" || action.Allocate ? cantStopActionLabel(action) : (action.play || action.target || action.guess) ? loveLetterActionLabel(action) : "column" in action ? `column ${action.column + 1}` : `row ${action.y + 1}, column ${action.x + 1}`;
  const source = controllerName(move.controller.kind);
  return `${move.position}. Player ${move.actor}: ${choice} · ${source}`;
}
// Retain move controls while cursor changes; only append accepted records during live play.
export class ReviewPanel {
  constructor({ branch, list, position, ancestry, jump, select }) {
    Object.assign(this, { branch, list, position, ancestry, jump, select });
    this.identity = null; this.buttons = [];
    branch.addEventListener("change", () => select(Number(branch.value)));
  }
  render(view, moves, busy) {
    const document = this.list.ownerDocument;
    while (this.branch.options.length > view.branches.length) this.branch.remove(this.branch.options.length - 1);
    for (let i = 0; i < view.branches.length; i++) {
      if (!this.branch.options[i]) this.branch.add(document.createElement("option"));
      const option = this.branch.options[i], info = view.branches[i];
      option.value = info.id; option.textContent = branchLabel(info);
    }
    this.branch.value = view.branch;
    this.position.textContent = `${view.read_only ? "Reviewing" : "Branch end"} · Move ${view.cursor} of ${view.tip}`;
    const info = view.branches.find(branch => branch.id === view.branch);
    this.ancestry.textContent = info.parent === null ? "Original continuation" : `Inherited through move ${info.fork_position}; new continuation starts here.`;
    const identity = `${view.game}:${view.branch}`;
    if (this.identity !== identity || this.buttons.length > moves.length + 1) {
      this.identity = identity; this.buttons = []; this.list.replaceChildren();
    }
    while (this.buttons.length < moves.length + 1) {
      const index = this.buttons.length;
      const item = document.createElement("li"), button = document.createElement("button");
      button.type = "button"; button.dataset.position = index; button.addEventListener("click", () => this.jump(index));
      item.append(button); this.list.append(item); this.buttons.push(button);
    }
    this.buttons.forEach((button, index) => {
      button.textContent = index === 0 ? "Initial position" : moveLabel(moves[index - 1]);
      button.disabled = busy;
      if (index === view.cursor) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
      button.dataset.inherited = info.parent !== null && index <= info.fork_position;
    });
    this.branch.disabled = busy;
  }
}
