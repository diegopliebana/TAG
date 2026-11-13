import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

function layout(game = "loveletter") {
  const windowEvents = {};
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
  board.querySelector = selector => selector === (game === "sushigo" ? ".sushigo-board" : game === "thegame" ? ".thegame-board" : ".loveletter-table") ? {} : null;
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
    window: { addEventListener(name, listener) { windowEvents[name] = listener; } },
  });
  return { ids, board, display, zoom, start: game => windowEvents["tag-game-started"]({detail:{game}}) };
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

test("Sushi Go! zoom changes the card board in scrollable mode and resets", () => {
  const { ids, board, display, zoom } = layout("sushigo");
  assert.equal(board.style.values["--sushi-scale"], "1");
  display.value = "scrollable"; display.listeners.change();
  zoom.value = "150"; zoom.listeners.input();
  assert.equal(board.style.values["--sushi-scale"], "1.5");
  assert.equal(ids["board-zoom-value"].value, "150%");
  assert.equal(board.scrollLeft, 195); assert.equal(board.scrollTop, 95);
  ids["board-zoom-reset"].listeners.click();
  assert.equal(board.style.values["--sushi-scale"], "1");
});

test("new sessions use each game's starting display and leave later manual changes alone", () => {
  const { ids, board, display, zoom, start } = layout("sushigo");
  start("sushigo");
  assert.equal(display.value, "scrollable");
  assert.equal(zoom.value, "80");
  assert.equal(ids["board-zoom-value"].value, "80%");
  assert.equal(board.style.values["--sushi-scale"], "0.8");
  assert.equal(zoom.disabled, false);
  zoom.value = "120"; zoom.listeners.input();
  assert.equal(board.style.values["--sushi-scale"], "1.2");
  assert.equal(zoom.value, "120");
  start("loveletter");
  assert.equal(display.value, "fit");
  assert.equal(zoom.value, "100");
  assert.equal(zoom.disabled, true);
  start("sushigo");
  assert.equal(display.value, "scrollable");
  assert.equal(zoom.value, "80");
});

test("The Game starts scrollable at 100% and supports the shared zoom control",()=>{
  const {ids,board,display,zoom,start}=layout("thegame");
  start("thegame");
  assert.equal(display.value,"scrollable");
  assert.equal(zoom.value,"100");
  assert.equal(zoom.disabled,false);
  assert.equal(board.style.values["--thegame-scale"],"1");
  zoom.value="125";zoom.listeners.input();
  assert.equal(board.style.values["--thegame-scale"],"1.25");
  assert.equal(ids["board-zoom-value"].value,"125%");
  start("thegame");
  assert.equal(zoom.value,"100");
  assert.equal(board.style.values["--thegame-scale"],"1");
});
