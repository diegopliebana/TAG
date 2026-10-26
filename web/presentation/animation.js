// Animate the change between the displayed position before and after an accepted move.
export function changedCell(before, after) {
  const index = after.board.cells.findIndex((cell, index) => cell !== before.board.cells[index]);
  return index < 0 ? null : { x: Math.floor(index / after.board.height), y: index % after.board.height };
}

export function animateResult(element, reducedMotion) {
  if (reducedMotion) return;
  element.animate([{ transform: "scale(1)" }, { transform: "scale(1.08)" }, { transform: "scale(1)" }], { duration: 350, easing: "ease-out" });
}

export async function animateMove(container, before, after, reducedMotion) {
  if (reducedMotion) return;
  if (after.game === "cantstop") {
    const target = container.querySelector(before.board.cantstop?.dice?.join() !== after.board.cantstop.dice?.join() ? ".cantstop-dice" : ".cantstop-tracks");
    if (target) {
      const animation = target.animate([{ opacity: .55 }, { opacity: 1 }], { duration: 180, easing: "ease-out" });
      try { await animation.finished; } catch { /* A cancelled presentation keeps the accepted position. */ }
    }
    return;
  }
  const cell = changedCell(before, after);
  if (!cell) return;
  const piece = container.querySelector(`[data-x="${cell.x}"][data-y="${cell.y}"] .piece-art`);
  if (!piece) return;
  const fall = after.game === "connect4";
  const cellHeight = piece.parentElement.getBoundingClientRect().height;
  const animation = piece.animate(fall ? [
    { transform: `translateY(-${(cell.y + 1) * cellHeight}px)`, opacity: 0.75 },
    { transform: "translateY(0)", opacity: 1 },
  ] : [
    { transform: "scale(0.7)", opacity: 0 },
    { transform: "scale(1)", opacity: 1 },
  ], { duration: fall ? Math.min(480, 180 + cell.y * 40) : 160, easing: fall ? "cubic-bezier(.5,0,.9,.6)" : "ease-out" });
  try { await animation.finished; } catch { /* Cancellation retains the displayed Rust position. */ }
}
