const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function element() {
  const classes = new Set();
  return {
    children: [], dataset: {}, hidden: false, attributes: {},
    classList: {
      add: (...names) => names.forEach(name => classes.add(name)),
      remove: (...names) => names.forEach(name => classes.delete(name)),
      contains: name => classes.has(name),
      toggle(name, enabled) {
        if (enabled) classes.add(name);
        else classes.delete(name);
      },
    },
    set innerHTML(value) { this.children = []; this.html = value; },
    get innerHTML() { return this.html || ""; },
    appendChild(child) { this.children.push(child); },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return null; },
  };
}

function setup() {
  const elements = new Map();
  const context = vm.createContext({
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element,
      createElementNS: element,
      addEventListener() {},
    },
    console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app.js"), "utf8"), context);
  return { elements, run: code => vm.runInContext(code, context) };
}

test("hints toggle clears advice immediately and restores it without changing play state", () => {
  const { elements, run } = setup();
  run(`
    state.sanList = ["e4"];
    state.ply = 1;
    state.positions.push(initialPosition());
    state.evals = [0, 1];
    state.suggestions = [null, { best: "d4", score: 2, lost: 1 }];
    state.mateThreats = [null, [{ from: 0, to: 8 }]];
    freePlay.selected = 12;
    freePlay.legalForSelected = [{ from: 12, to: 28 }];
    renderMoveList();
    renderMoveFeedback();
    renderForkArrows();
  `);
  const list = elements.get("move-list");
  assert.match(list.children[4].title, /Better: d4/);
  assert.equal(list.children[4].children.length, 1);
  assert.equal(elements.get("move-feedback").hidden, false);
  assert.ok(elements.get("board-overlay").children.length > 0);

  const square = element();
  square.classList.add("hanging");
  elements.get("board").children.push(square);
  run("toggleHints()");
  assert.equal(run("state.hintsEnabled"), false);
  assert.equal(elements.get("btn-hints").textContent, "Hints: off");
  assert.equal(elements.get("btn-hints").attributes["aria-pressed"], "false");
  for (const id of ["move-feedback", "hanging-warn", "opening-info"]) {
    assert.equal(elements.get(id).hidden, true);
  }
  assert.equal(elements.get("board-overlay").children.length, 0);
  assert.equal(square.classList.contains("hanging"), false);
  assert.equal(list.children[4].title, undefined);
  assert.equal(list.children[4].children.length, 0);
  assert.equal(list.children[4].textContent, "e4");
  assert.equal(run("state.ply"), 1);
  assert.equal(run("freePlay.selected"), 12);
  assert.equal(run("freePlay.legalForSelected.length"), 1);

  run("toggleHints()");
  assert.equal(elements.get("btn-hints").attributes["aria-pressed"], "true");
  assert.equal(elements.get("btn-hints").textContent, "Hints: on");
  assert.equal(elements.get("move-feedback").hidden, false);
  assert.ok(elements.get("board-overlay").children.length > 0);
  assert.match(list.children[4].title, /Better: d4/);
});

test("disabled hints stay hidden when renderers run for a different position", () => {
  const { elements, run } = setup();
  run(`
    toggleHints();
    state.sanList = ["e4"];
    state.ply = 1;
    state.positions.push(initialPosition());
    state.matesAllowed = [null, [{ from: 0, to: 8 }]];
    renderMoveList();
    renderMoveFeedback();
    renderHangingWarning();
    renderOpeningInfo();
    renderForkArrows();
  `);
  assert.equal(elements.get("move-list").children[4].textContent, "e4");
  assert.equal(elements.get("board-overlay").children.length, 0);
  for (const id of ["move-feedback", "hanging-warn", "opening-info"]) {
    assert.equal(elements.get(id).hidden, true);
  }
});
