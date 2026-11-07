const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function setup(response) {
  const elements = new Map();
  const requests = [];
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, { hidden: true, classList: { toggle() {} } });
      }
      return elements.get(id);
    },
    addEventListener() {},
  };
  const context = vm.createContext({
    document, console: { error() {} },
    fetch: async (url, options) => {
      requests.push(JSON.parse(options.body));
      return { ok: !response.error, json: async () => response };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app.js"), "utf8"), context);
  vm.runInContext(`
    championPlay.active = true;
    championPlay.settings = {};
    state.positions = [initialPosition()];
  `, context);
  return { context, elements, requests };
}

test("human claim sends an explicit action and ends without adding a move", async () => {
  const { context, elements, requests } = setup({
    move: null, result: "1/2-1/2", reason: "THREEFOLD_REPETITION",
  });
  await vm.runInContext("requestChampionReply(true)", context);
  assert.equal(requests[0].claim_draw, true);
  assert.equal(vm.runInContext("state.positions.length", context), 1);
  assert.equal(vm.runInContext("championPlay.ended", context), true);
  assert.equal(elements.get("btn-champion-claim").hidden, true);
});

test("normal reply offers a claim without accepting it automatically", async () => {
  const { context, elements, requests } = setup({
    move: null, result: null, reason: null, claim_available: true,
  });
  await vm.runInContext("requestChampionReply({ type: 'click' })", context);
  assert.equal(requests[0].claim_draw, false);
  assert.equal(vm.runInContext("championPlay.ended", context), false);
  assert.equal(elements.get("btn-champion-claim").hidden, false);
});

test("failed claim reports the error and allows retrying the claim", async () => {
  const { context, elements } = setup({ error: "No draw claim is available" });
  await vm.runInContext("requestChampionReply(true)", context);
  assert.equal(elements.get("champion-status").textContent, "No draw claim is available");
  assert.equal(elements.get("btn-champion-claim").hidden, false);
  assert.equal(elements.get("btn-champion-retry").hidden, true);
  assert.equal(vm.runInContext("championPlay.busy", context), false);
});
