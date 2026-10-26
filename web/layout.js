// Measure the available screen space again when controls wrap or player/status text changes.
const toolbar = document.querySelector(".toolbar");
const panel = document.querySelector(".game-panel");
const board = document.querySelector("#board");
const display = document.querySelector("#board-display");
const zoom = document.querySelector("#board-zoom");
const zoomValue = document.querySelector("#board-zoom-value");
const zoomReset = document.querySelector("#board-zoom-reset");
function sizeTable() {
  for (const menu of document.querySelectorAll(".toolbar-menus > details")) {
    // Right-align each dropdown with its own button; leave room at the left edge.
    menu.style.setProperty("--menu-width", `${Math.max(0, menu.getBoundingClientRect().right - 12)}px`);
  }
  const toolbarHeight = toolbar.getBoundingClientRect().height;
  document.documentElement.style.setProperty("--toolbar-height", `${toolbarHeight}px`);
  // All panel space outside the board: headings, players, captions and padding.
  const chrome = panel.getBoundingClientRect().height - board.getBoundingClientRect().height;
  panel.style.setProperty("--board-chrome", `${chrome}px`);
  board.dataset.display = display.value;
  zoom.disabled = zoomReset.disabled = display.value === "fit";
  zoomValue.value = `${zoom.value}%`;
  zoom.setAttribute("aria-valuetext", `${zoom.value} percent`);
  if (board.querySelector(".loveletter-table")) {
    const padding = getComputedStyle(board);
    const width = Math.max(1, board.clientWidth - parseFloat(padding.paddingLeft) - parseFloat(padding.paddingRight));
    // Fit reflows the card table to the viewport width. Long transcripts remain
    // vertically scrollable so cards and text retain readable sizes.
    board.style.setProperty("--letter-width", display.value === "fit" ? `${Math.min(1100, width)}px` : "1100px");
    board.style.setProperty("--letter-scale", display.value === "fit" ? "1" : String(Number(zoom.value) / 100));
    board.style.removeProperty("--cell");
    return;
  }
  if (board.querySelector(".cantstop-tracks")) {
    board.style.setProperty("--cantstop-width", display.value === "fit" ? "100%" : `${748 * Number(zoom.value) / 100}px`);
    // Controls remain visible; taller/custom boards can use scrollable mode.
    board.style.setProperty("--cantstop-height", display.value === "fit" ? `${Math.max(200, innerHeight - toolbarHeight - chrome - 230)}px` : "none");
    return;
  }
  const table = board.querySelector("table");
  if (!table?.tBodies[0]?.rows.length) {
    board.style.removeProperty("--cell");
    return;
  }
  const rows = table.tBodies[0].rows.length;
  const columns = table.tBodies[0].rows[0].cells.length;
  board.style.setProperty("--board-columns", columns);
  const normalCell = Math.min(80, Math.max(52, innerWidth * .07));
  if (display.value === "scrollable") {
    board.style.setProperty("--cell", `${normalCell * Number(zoom.value) / 100}px`);
    return;
  }
  const padding = getComputedStyle(board);
  const width = board.clientWidth - parseFloat(padding.paddingLeft) - parseFloat(padding.paddingRight);
  const height = Math.max(120, innerHeight - toolbarHeight - chrome - 40)
    - parseFloat(padding.paddingTop) - parseFloat(padding.paddingBottom);
  // Connect 4's column buttons occupy another 0.6 cell above its rows.
  const cell = Math.max(0.1, Math.min(normalCell, width / columns, height / (rows + (table.tHead ? .6 : 0))));
  board.style.setProperty("--cell", `${Math.floor(cell * 10) / 10}px`);
}
const observer = new ResizeObserver(sizeTable);
for (const element of [toolbar, panel, ...panel.children].filter(element => element !== board)) observer.observe(element);
// A newly selected game replaces the table; ordinary accepted moves update text.
new MutationObserver(sizeTable).observe(board, { childList: true, subtree: true });
display.addEventListener("change", () => { board.scrollTop = 0; board.scrollLeft = 0; sizeTable(); });
function changeZoom() {
  // Keep the same board point at the viewport center while changing its scale.
  const scaleProperty = board.querySelector(".loveletter-table") ? "--letter-scale" : "--cell";
  const previousCell = parseFloat(getComputedStyle(board).getPropertyValue(scaleProperty));
  const padding = getComputedStyle(board);
  const centerX = board.scrollLeft + board.clientWidth / 2 - parseFloat(padding.paddingLeft);
  const centerY = board.scrollTop + board.clientHeight / 2 - parseFloat(padding.paddingTop);
  sizeTable();
  const ratio = parseFloat(getComputedStyle(board).getPropertyValue(scaleProperty)) / previousCell;
  if (Number.isFinite(ratio)) {
    board.scrollLeft = centerX * ratio - board.clientWidth / 2 + parseFloat(padding.paddingLeft);
    board.scrollTop = centerY * ratio - board.clientHeight / 2 + parseFloat(padding.paddingTop);
  }
}
zoom.addEventListener("input", changeZoom);
zoomReset.addEventListener("click", () => { zoom.value = "100"; changeZoom(); });
window.addEventListener("resize", sizeTable);
sizeTable();
