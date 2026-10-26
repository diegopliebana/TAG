import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

function layout() {
  const element = () => ({
    style: { values: {}, setProperty(name, value) { this.values[name] = value; }, removeProperty(name) { delete this.values[name]; } },
    dataset: {}, listeners: {}, children: [],
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this[name] = value; },
    getBoundingClientRect() { return { height: 500, right: 800 }; },
  });
  const ids = Object.fromEntries(["board", "board-display", "board-zoom", "board-zoom-value", "board-zoom-reset"].map(id => [id, element()]));
  const board = ids.board, display = ids["board-display"], zoom = ids["board-zoom"];
  Object.assign(board, { clientWidth: 800, clientHeight: 400, scrollTop: 0, scrollLeft: 0 });
  board.querySelector = selector => selector === ".loveletter-table" ? {} : null;
  display.value = "fit"; zoom.value = "100";
  const document = {
    documentElement: element(),
    querySelector: selector => selector.startsWith("#") ? ids[selector.slice(1)] : element(),
    querySelectorAll: () => [],
  };
  const computed = node => ({
    paddingLeft: "10px", paddingRight: "10px", paddingTop: "10px", paddingBottom: "10px",
    getPropertyValue: name => node.style.values[name] ?? "",
  });
  vm.runInNewContext(readFileSync(new URL("../layout.js", import.meta.url), "utf8"), {
    document, getComputedStyle: computed, innerHeight: 1000, innerWidth: 1200,
    ResizeObserver: class { observe() {} }, MutationObserver: class { observe() {} },
    window: { addEventListener() {} },
  });
  return { ids, board, display, zoom };
}

test("Love Letter fits viewport width, zooms in scrollable mode and resets", () => {
  const { ids, board, display, zoom } = layout();
  assert.equal(board.style.values["--letter-width"], "780px");
  assert.equal(board.style.values["--letter-scale"], "1");
  assert.equal(zoom.disabled, true);
  display.value = "scrollable"; display.listeners.change();
  assert.equal(zoom.disabled, false);
  assert.equal(board.style.values["--letter-width"], "1100px");
  zoom.value = "150"; zoom.listeners.input();
  assert.equal(board.style.values["--letter-scale"], "1.5");
  assert.equal(ids["board-zoom-value"].value, "150%");
  // The same content point remains at the viewport center after scaling.
  assert.equal(board.scrollLeft, 195); assert.equal(board.scrollTop, 95);
  zoom.value = "25"; zoom.listeners.input();
  assert.equal(board.style.values["--letter-scale"], "0.25");
  ids["board-zoom-reset"].listeners.click();
  assert.equal(zoom.value, "100"); assert.equal(board.style.values["--letter-scale"], "1");
  board.clientWidth = 300;
  display.value = "fit"; display.listeners.change();
  assert.equal(board.style.values["--letter-width"], "280px");
  assert.equal(board.scrollLeft, 0); assert.equal(board.scrollTop, 0);
  assert.equal(ids["board-zoom-reset"].disabled, true);
});
