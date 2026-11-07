const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const context = vm.createContext({
  document: { getElementById: () => null, addEventListener: () => {} },
  console,
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../app.js"), "utf8"), context);

function run(code) {
  return vm.runInContext(code, context);
}

for (const color of ["w", "b"]) {
  for (const side of ["K", "Q"]) {
    test(`${color} ${side}-side castling moves both king and rook`, () => {
      run(`{
        const pos = initialPosition();
        pos.board.fill("");
        pos.sideToMove = "${color}";
        const rank = "${color}" === "w" ? 0 : 7;
        const king = "${color}" === "w" ? "K" : "k";
        const rook = "${color}" === "w" ? "R" : "r";
        pos.board[sq(4, rank)] = king;
        pos.board[sq(4, 7 - rank)] = "${color}" === "w" ? "k" : "K";
        pos.board[sq("${side}" === "K" ? 7 : 0, rank)] = rook;
        const castle = legalMoves(pos).find(m => m.castle === "${side}");
        if (!castle) throw new Error("Castling is absent from legal moves");
        const bare = makeBareMove(pos, castle);
        const applied = applyMove(pos, buildMoveObject(pos, castle.from, castle.to, null));
        for (const next of [bare, applied]) {
          if (next.board[castle.to] !== king) throw new Error("King did not castle");
          if (next.board[sq("${side}" === "K" ? 5 : 3, rank)] !== rook)
            throw new Error("Rook did not castle");
          if (next.castling["${color}" === "w" ? "K" : "k"]
              || next.castling["${color}" === "w" ? "Q" : "q"])
            throw new Error("Castling rights survived");
        }
      }`);
    });
  }
}

test("castling is blocked by pieces, lost rights, missing rooks, or king attacks", () => {
  for (const setup of [
    'pos.board[sq(5, 0)] = "B"',
    'pos.castling.K = false',
    'pos.board[sq(7, 0)] = ""',
    'pos.board[sq(4, 6)] = "r"',
    'pos.board[sq(5, 6)] = "r"',
    'pos.board[sq(6, 6)] = "r"',
  ]) {
    assert.equal(run(`{
      const pos = initialPosition();
      pos.board.fill("");
      pos.board[sq(4, 0)] = "K";
      pos.board[sq(7, 0)] = "R";
      pos.board[sq(0, 7)] = "k";
      ${setup};
      legalMoves(pos).some(m => m.castle === "K");
    }`), false);
  }
});

test("queenside requires b-file clearance but permits attacks on b1", () => {
  assert.equal(run(`{
    const pos = initialPosition();
    pos.board.fill("");
    pos.board[4] = "K"; pos.board[0] = "R"; pos.board[63] = "k";
    pos.board[1] = "N";
    const blocked = legalMoves(pos).some(m => m.castle === "Q");
    pos.board[1] = ""; pos.board[49] = "r";
    !blocked && legalMoves(pos).some(m => m.castle === "Q");
  }`), true);
});

test("bare rook moves and captures remove castling rights", () => {
  assert.equal(run(`{
    const pos = initialPosition();
    const next = makeBareMove(pos, {from: 0, to: 56});
    !next.castling.Q && !next.castling.q;
  }`), true);
});
