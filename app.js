/* chess.com analyzer — vanilla JS app
 *
 *   - Loads games.json (produced by download_games.py)
 *   - Renders a list of games in the left pane
 *   - On selection, parses the PGN, generates the full position list and
 *     lets the user step through plies with buttons / arrow keys / by
 *     clicking a move in the move list.
 *
 * Everything is intentionally self-contained — no build step, no
 * dependencies.  The chess logic is the minimum needed to replay a
 * legal game (SAN, captures, castling, en passant, promotion).
 */

"use strict";

// =====================================================================
//  Board model
// =====================================================================
//
// A square is encoded as an integer 0..63 with rank 0 = white's first
// rank ("1") and file 0 = a-file.  Pieces are single chars: uppercase
// = white, lowercase = black, "" = empty square.
//   P/p  pawn      N/n  knight    B/b  bishop
//   R/r  rook      Q/q  queen     K/k  king

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const PIECE_GLYPHS = {
  K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};

const sq = (file, rank) => rank * 8 + file;
const fileOf = (s) => s & 7;
const rankOf = (s) => s >> 3;
const sqName = (s) => FILES[fileOf(s)] + (rankOf(s) + 1);
const parseSquare = (name) => sq(name.charCodeAt(0) - 97, parseInt(name[1], 10) - 1);
const isWhite = (p) => p && p === p.toUpperCase();
const isBlack = (p) => p && p === p.toLowerCase();
const sameColor = (a, b) => a && b && (isWhite(a) === isWhite(b));

function initialPosition() {
  const board = new Array(64).fill("");
  const back = ["R", "N", "B", "Q", "K", "B", "N", "R"];
  for (let f = 0; f < 8; f++) {
    board[sq(f, 0)] = back[f];
    board[sq(f, 1)] = "P";
    board[sq(f, 6)] = "p";
    board[sq(f, 7)] = back[f].toLowerCase();
  }
  return {
    board,
    sideToMove: "w",
    castling: { K: true, Q: true, k: true, q: true },
    epTarget: -1,        // -1 = none
    lastMove: null,      // { from, to }
  };
}

function clonePosition(p) {
  return {
    board: p.board.slice(),
    sideToMove: p.sideToMove,
    castling: { ...p.castling },
    epTarget: p.epTarget,
    lastMove: p.lastMove,
  };
}

// ---- Attack detection ------------------------------------------------
const KNIGHT_DELTAS = [[1,2],[2,1],[2,-1],[1,-2],[-1,-2],[-2,-1],[-2,1],[-1,2]];
const ROOK_DIRS     = [[1,0],[-1,0],[0,1],[0,-1]];
const BISHOP_DIRS   = [[1,1],[1,-1],[-1,1],[-1,-1]];
const KING_DELTAS   = [...ROOK_DIRS, ...BISHOP_DIRS];

function isSquareAttackedBy(board, target, byWhite) {
  const tf = fileOf(target), tr = rankOf(target);

  // Pawns
  const pawnDir = byWhite ? -1 : 1; // attacker pawn came from this rank delta
  const pawnPiece = byWhite ? "P" : "p";
  for (const df of [-1, 1]) {
    const f = tf + df, r = tr + pawnDir;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === pawnPiece) return true;
  }

  // Knights
  const knightPiece = byWhite ? "N" : "n";
  for (const [df, dr] of KNIGHT_DELTAS) {
    const f = tf + df, r = tr + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === knightPiece) return true;
  }

  // King (adjacent)
  const kingPiece = byWhite ? "K" : "k";
  for (const [df, dr] of KING_DELTAS) {
    const f = tf + df, r = tr + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === kingPiece) return true;
  }

  // Sliding: rook/queen orthogonally, bishop/queen diagonally
  const rookLike  = byWhite ? ["R", "Q"] : ["r", "q"];
  const bishopLike = byWhite ? ["B", "Q"] : ["b", "q"];
  for (const [df, dr] of ROOK_DIRS) {
    let f = tf + df, r = tr + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      const p = board[sq(f, r)];
      if (p) { if (rookLike.includes(p)) return true; break; }
      f += df; r += dr;
    }
  }
  for (const [df, dr] of BISHOP_DIRS) {
    let f = tf + df, r = tr + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      const p = board[sq(f, r)];
      if (p) { if (bishopLike.includes(p)) return true; break; }
      f += df; r += dr;
    }
  }
  return false;
}

function findKing(board, white) {
  const k = white ? "K" : "k";
  for (let i = 0; i < 64; i++) if (board[i] === k) return i;
  return -1;
}

// ---- Pseudo-legal source candidates ---------------------------------
//
// Given a target square and a piece type for the side-to-move, return
// the list of source squares that can pseudo-legally reach the target.
function candidateSources(pos, pieceUpper, to, isCapture) {
  const white = pos.sideToMove === "w";
  const piece = white ? pieceUpper : pieceUpper.toLowerCase();
  const sources = [];
  const tf = fileOf(to), tr = rankOf(to);

  if (pieceUpper === "P") {
    const dir = white ? 1 : -1;
    if (isCapture) {
      for (const df of [-1, 1]) {
        const f = tf - df, r = tr - dir;
        if (f < 0 || f > 7 || r < 0 || r > 7) continue;
        if (pos.board[sq(f, r)] === piece) sources.push(sq(f, r));
      }
    } else {
      // single push
      const r1 = tr - dir;
      if (r1 >= 0 && r1 < 8 && pos.board[sq(tf, r1)] === piece) {
        sources.push(sq(tf, r1));
      } else {
        // double push: from rank 1 (white) / 6 (black) only
        const startRank = white ? 1 : 6;
        const r2 = tr - 2 * dir;
        if (
          r2 === startRank &&
          pos.board[sq(tf, r2)] === piece &&
          pos.board[sq(tf, r1)] === ""
        ) sources.push(sq(tf, r2));
      }
    }
    return sources;
  }

  if (pieceUpper === "N") {
    for (const [df, dr] of KNIGHT_DELTAS) {
      const f = tf + df, r = tr + dr;
      if (f >= 0 && f < 8 && r >= 0 && r < 8 && pos.board[sq(f, r)] === piece) {
        sources.push(sq(f, r));
      }
    }
    return sources;
  }

  if (pieceUpper === "K") {
    for (const [df, dr] of KING_DELTAS) {
      const f = tf + df, r = tr + dr;
      if (f >= 0 && f < 8 && r >= 0 && r < 8 && pos.board[sq(f, r)] === piece) {
        sources.push(sq(f, r));
      }
    }
    return sources;
  }

  // Sliding pieces: B / R / Q
  const dirs = pieceUpper === "B" ? BISHOP_DIRS
            : pieceUpper === "R" ? ROOK_DIRS
            : KING_DELTAS; // Q
  for (const [df, dr] of dirs) {
    let f = tf + df, r = tr + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      const here = pos.board[sq(f, r)];
      if (here) {
        if (here === piece) sources.push(sq(f, r));
        break;
      }
      f += df; r += dr;
    }
  }
  return sources;
}

// =====================================================================
//  PGN / SAN parsing
// =====================================================================

function tokenizePgnMoves(pgn) {
  // Strip the tag pair section (lines beginning with "[") at the top.
  let body = pgn.replace(/^\s*(?:\[[^\]]*\]\s*)+/m, "");

  // Strip comments {...}
  body = body.replace(/\{[^}]*\}/g, " ");

  // Strip variations (...) – iteratively to handle nesting
  let prev;
  do { prev = body; body = body.replace(/\([^()]*\)/g, " "); } while (body !== prev);

  // NAGs ($1, $14, ...)
  body = body.replace(/\$\d+/g, " ");

  // Move numbers: "12." or "12..."
  body = body.replace(/\b\d+\.(\.\.)?/g, " ");

  // Result tokens
  body = body.replace(/\b(?:1-0|0-1|1\/2-1\/2|\*)\b/g, " ");

  return body.trim().split(/\s+/).filter(Boolean);
}

function parseSan(san) {
  const move = {
    castle: null,
    piece: null,
    fromFile: -1,
    fromRank: -1,
    to: -1,
    capture: false,
    promotion: null,
    check: false,
    mate: false,
    san,
  };

  // Some PGN exporters tack on an annotation like !? or ?!
  let s = san.replace(/[!?]+$/, "");
  if (s.endsWith("#")) { move.mate = true; s = s.slice(0, -1); }
  else if (s.endsWith("+")) { move.check = true; s = s.slice(0, -1); }

  if (s === "O-O" || s === "0-0")     { move.castle = "K"; return move; }
  if (s === "O-O-O" || s === "0-0-0") { move.castle = "Q"; return move; }

  const promoMatch = s.match(/=([QRBN])$/);
  if (promoMatch) { move.promotion = promoMatch[1]; s = s.slice(0, -2); }

  const toMatch = s.match(/([a-h])([1-8])$/);
  if (!toMatch) throw new Error("Cannot parse SAN: " + san);
  move.to = parseSquare(toMatch[1] + toMatch[2]);
  s = s.slice(0, -2);

  if (s.endsWith("x")) { move.capture = true; s = s.slice(0, -1); }

  if (s.length && /[KQRBN]/.test(s[0])) { move.piece = s[0]; s = s.slice(1); }
  else                                  { move.piece = "P"; }

  for (const ch of s) {
    if (ch >= "a" && ch <= "h") move.fromFile = ch.charCodeAt(0) - 97;
    else if (ch >= "1" && ch <= "8") move.fromRank = parseInt(ch, 10) - 1;
  }
  return move;
}

// =====================================================================
//  Move application
// =====================================================================

function applyMove(prevPos, move) {
  const pos = clonePosition(prevPos);
  const white = pos.sideToMove === "w";

  // Handle castling first.
  if (move.castle) {
    const rank = white ? 0 : 7;
    const kingFrom = sq(4, rank);
    const kingTo   = move.castle === "K" ? sq(6, rank) : sq(2, rank);
    const rookFrom = move.castle === "K" ? sq(7, rank) : sq(0, rank);
    const rookTo   = move.castle === "K" ? sq(5, rank) : sq(3, rank);
    pos.board[kingTo]   = pos.board[kingFrom];
    pos.board[kingFrom] = "";
    pos.board[rookTo]   = pos.board[rookFrom];
    pos.board[rookFrom] = "";
    if (white) { pos.castling.K = false; pos.castling.Q = false; }
    else       { pos.castling.k = false; pos.castling.q = false; }
    pos.epTarget = -1;
    pos.lastMove = { from: kingFrom, to: kingTo };
    pos.sideToMove = white ? "b" : "w";
    return pos;
  }

  // Find the source square.
  const candidates = candidateSources(pos, move.piece, move.to, move.capture);
  const filtered = candidates.filter((src) => {
    if (move.fromFile !== -1 && fileOf(src) !== move.fromFile) return false;
    if (move.fromRank !== -1 && rankOf(src) !== move.fromRank) return false;
    return true;
  });

  let from = -1;
  if (filtered.length === 1) {
    from = filtered[0];
  } else if (filtered.length > 1) {
    // Disambiguate via a king-safety check (pin detection).
    for (const src of filtered) {
      if (!leavesKingInCheck(pos, src, move)) { from = src; break; }
    }
    if (from === -1) from = filtered[0];
  } else {
    throw new Error(`No source for SAN "${move.san}" at ply (${pos.sideToMove})`);
  }

  // Apply move
  const piece = pos.board[from];
  let captureSquare = move.to;

  // En passant capture: pawn moving diagonally to empty square hitting epTarget.
  if (move.piece === "P" && move.capture && pos.board[move.to] === "" && move.to === pos.epTarget) {
    captureSquare = sq(fileOf(move.to), rankOf(from));
  }

  pos.board[captureSquare] = "";
  pos.board[from] = "";
  pos.board[move.to] = move.promotion
    ? (white ? move.promotion : move.promotion.toLowerCase())
    : piece;

  // Update castling rights.
  if (piece === "K") { pos.castling.K = false; pos.castling.Q = false; }
  if (piece === "k") { pos.castling.k = false; pos.castling.q = false; }
  if (from === sq(0, 0) || captureSquare === sq(0, 0)) pos.castling.Q = false;
  if (from === sq(7, 0) || captureSquare === sq(7, 0)) pos.castling.K = false;
  if (from === sq(0, 7) || captureSquare === sq(0, 7)) pos.castling.q = false;
  if (from === sq(7, 7) || captureSquare === sq(7, 7)) pos.castling.k = false;

  // Update en-passant target.
  if (move.piece === "P" && Math.abs(rankOf(move.to) - rankOf(from)) === 2) {
    pos.epTarget = sq(fileOf(from), (rankOf(from) + rankOf(move.to)) / 2);
  } else {
    pos.epTarget = -1;
  }

  pos.lastMove = { from, to: move.to };
  pos.sideToMove = white ? "b" : "w";
  return pos;
}

function leavesKingInCheck(pos, from, move) {
  // Apply the move to a scratch board and test king safety.
  const test = clonePosition(pos);
  const piece = test.board[from];
  let captureSquare = move.to;
  if (move.piece === "P" && move.capture && test.board[move.to] === "" && move.to === test.epTarget) {
    captureSquare = sq(fileOf(move.to), rankOf(from));
  }
  test.board[captureSquare] = "";
  test.board[from] = "";
  test.board[move.to] = move.promotion
    ? (test.sideToMove === "w" ? move.promotion : move.promotion.toLowerCase())
    : piece;
  const white = test.sideToMove === "w";
  const kingSq = findKing(test.board, white);
  if (kingSq === -1) return false;
  return isSquareAttackedBy(test.board, kingSq, !white);
}

// =====================================================================
//  Tactical analysis: legal moves + fork detector
// =====================================================================
//
// Mirror of the Python helpers in chess_engine.py. Used to flag
// blunders in the move list — i.e. moves that hand the opponent a
// fork.

const PIECE_VALUE = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 100 };

function pseudoLegalMoves(pos) {
  const out = [];
  const white = pos.sideToMove === "w";
  for (let frm = 0; frm < 64; frm++) {
    const p = pos.board[frm];
    if (!p || isWhite(p) !== white) continue;
    const pt = p.toUpperCase();
    const f = fileOf(frm), r = rankOf(frm);

    if (pt === "P") {
      const dir = white ? 1 : -1;
      const startRank = white ? 1 : 6;
      const promoRank = white ? 7 : 0;
      const r1 = r + dir;
      if (r1 >= 0 && r1 < 8 && pos.board[sq(f, r1)] === "") {
        if (r1 === promoRank) {
          for (const promo of "QRBN") out.push({ from: frm, to: sq(f, r1), promotion: promo });
        } else {
          out.push({ from: frm, to: sq(f, r1) });
        }
        if (r === startRank && pos.board[sq(f, r + 2 * dir)] === "") {
          out.push({ from: frm, to: sq(f, r + 2 * dir) });
        }
      }
      for (const df of [-1, 1]) {
        const tf = f + df, tr = r + dir;
        if (tf < 0 || tf > 7 || tr < 0 || tr > 7) continue;
        const target = sq(tf, tr);
        const tp = pos.board[target];
        if (tp && isWhite(tp) !== white) {
          if (tr === promoRank) {
            for (const promo of "QRBN") out.push({ from: frm, to: target, promotion: promo });
          } else {
            out.push({ from: frm, to: target });
          }
        } else if (target === pos.epTarget) {
          out.push({ from: frm, to: target, isEp: true });
        }
      }
    } else if (pt === "N") {
      for (const [df, dr] of KNIGHT_DELTAS) {
        const tf = f + df, tr = r + dr;
        if (tf < 0 || tf > 7 || tr < 0 || tr > 7) continue;
        const target = sq(tf, tr);
        const tp = pos.board[target];
        if (!tp || isWhite(tp) !== white) out.push({ from: frm, to: target });
      }
    } else if (pt === "K") {
      for (const [df, dr] of KING_DELTAS) {
        const tf = f + df, tr = r + dr;
        if (tf < 0 || tf > 7 || tr < 0 || tr > 7) continue;
        const target = sq(tf, tr);
        const tp = pos.board[target];
        if (!tp || isWhite(tp) !== white) out.push({ from: frm, to: target });
      }
      // Castling — required only for completeness of the legal-move
      // generator. We skip it here because forks during castling are
      // not relevant in practice and the SAN replay path handles
      // castles separately.
    } else {
      const dirs = pt === "B" ? BISHOP_DIRS
                : pt === "R" ? ROOK_DIRS
                : KING_DELTAS; // Q
      for (const [df, dr] of dirs) {
        let tf = f + df, tr = r + dr;
        while (tf >= 0 && tf < 8 && tr >= 0 && tr < 8) {
          const target = sq(tf, tr);
          const tp = pos.board[target];
          if (tp) {
            if (isWhite(tp) !== white) out.push({ from: frm, to: target });
            break;
          }
          out.push({ from: frm, to: target });
          tf += df; tr += dr;
        }
      }
    }
  }
  return out;
}

function makeBareMove(pos, mv) {
  // Simplified mover used for pseudo-legal exploration only — no
  // castling-rights / EP bookkeeping beyond what the fork search needs.
  const next = clonePosition(pos);
  const piece = next.board[mv.from];
  next.board[mv.from] = "";
  if (mv.isEp) {
    next.board[sq(fileOf(mv.to), rankOf(mv.from))] = "";
    next.board[mv.to] = piece;
  } else if (mv.promotion) {
    next.board[mv.to] = isWhite(piece) ? mv.promotion : mv.promotion.toLowerCase();
  } else {
    next.board[mv.to] = piece;
  }
  next.epTarget = -1;
  if (piece.toUpperCase() === "P" && Math.abs(rankOf(mv.to) - rankOf(mv.from)) === 2) {
    next.epTarget = sq(fileOf(mv.from), (rankOf(mv.from) + rankOf(mv.to)) / 2);
  }
  next.lastMove = { from: mv.from, to: mv.to };
  next.sideToMove = pos.sideToMove === "w" ? "b" : "w";
  return next;
}

function legalMoves(pos) {
  const out = [];
  const white = pos.sideToMove === "w";
  for (const mv of pseudoLegalMoves(pos)) {
    const next = makeBareMove(pos, mv);
    const ksq = findKing(next.board, white);
    if (ksq < 0) continue;
    if (!isSquareAttackedBy(next.board, ksq, !white)) out.push(mv);
  }
  return out;
}

function attacksFrom(board, frm) {
  const p = board[frm];
  if (!p) return [];
  const pt = p.toUpperCase();
  const white = isWhite(p);
  const f = fileOf(frm), r = rankOf(frm);
  const out = [];
  if (pt === "P") {
    const dir = white ? 1 : -1;
    for (const df of [-1, 1]) {
      const tf = f + df, tr = r + dir;
      if (tf >= 0 && tf < 8 && tr >= 0 && tr < 8) out.push(sq(tf, tr));
    }
  } else if (pt === "N") {
    for (const [df, dr] of KNIGHT_DELTAS) {
      const tf = f + df, tr = r + dr;
      if (tf >= 0 && tf < 8 && tr >= 0 && tr < 8) out.push(sq(tf, tr));
    }
  } else if (pt === "K") {
    for (const [df, dr] of KING_DELTAS) {
      const tf = f + df, tr = r + dr;
      if (tf >= 0 && tf < 8 && tr >= 0 && tr < 8) out.push(sq(tf, tr));
    }
  } else {
    const dirs = pt === "B" ? BISHOP_DIRS
              : pt === "R" ? ROOK_DIRS
              : KING_DELTAS;
    for (const [df, dr] of dirs) {
      let tf = f + df, tr = r + dr;
      while (tf >= 0 && tf < 8 && tr >= 0 && tr < 8) {
        out.push(sq(tf, tr));
        if (board[sq(tf, tr)]) break;
        tf += df; tr += dr;
      }
    }
  }
  return out;
}

/** Return every legal move for `pos.sideToMove` that creates a
 *  *winning* fork: the moved piece attacks ≥ 2 enemy pieces, at least
 *  one of those targets is the king or strictly more valuable than
 *  the attacker, **and** the attacker's destination square is not
 *  defended by the opponent (so it can't simply be captured back). */
function findForks(pos) {
  const forks = [];
  const attackerIsWhite = pos.sideToMove === "w";
  for (const mv of legalMoves(pos)) {
    const next = makeBareMove(pos, mv);
    const attacker = next.board[mv.to];
    if (!attacker) continue;
    // The attacker must not be capturable.
    if (isSquareAttackedBy(next.board, mv.to, !attackerIsWhite)) continue;
    const atkValue = PIECE_VALUE[attacker.toUpperCase()];
    const targets = [];
    for (const s of attacksFrom(next.board, mv.to)) {
      const tp = next.board[s];
      if (!tp || isWhite(tp) === isWhite(attacker)) continue;
      targets.push({ square: s, piece: tp });
    }
    if (targets.length < 2) continue;
    const critical = targets.some((t) =>
      t.piece.toUpperCase() === "K" ||
      PIECE_VALUE[t.piece.toUpperCase()] > atkValue
    );
    if (critical) forks.push({ move: mv, targets, attacker });
  }
  return forks;
}

const PIECE_NAMES = {
  P: "pawn", N: "knight", B: "bishop",
  R: "rook", Q: "queen",  K: "king",
};
const pieceName = (p) => PIECE_NAMES[p.toUpperCase()] || p;

function describeFork(fork) {
  const from = sqName(fork.move.from);
  const to   = sqName(fork.move.to);
  const attackerName = fork.attacker ? pieceName(fork.attacker) : "piece";
  const tgts = fork.targets
    .map((t) => `${pieceName(t.piece)} on ${sqName(t.square)}`)
    .join(", ");
  return `${attackerName} ${from}→${to} attacks ${tgts}`;
}

// ---- Pin detection --------------------------------------------------
//
// Mirror of `chess_engine.find_pins`. A move is a *winning pin* when
// the moving piece is a bishop/rook/queen, the ray it newly controls
// pierces exactly one enemy piece (not the king) and then meets a
// second enemy piece — either the king (absolute pin) or one
// strictly more valuable than the pinned piece (relative pin) — and
// the attacker's landing square is not attacked back.

const PIN_DIRS = {
  B: BISHOP_DIRS,
  R: ROOK_DIRS,
  Q: [...BISHOP_DIRS, ...ROOK_DIRS],
};

function findPins(pos) {
  const pins = [];
  const attackerIsWhite = pos.sideToMove === "w";
  for (const mv of legalMoves(pos)) {
    const next = makeBareMove(pos, mv);
    const attacker = next.board[mv.to];
    if (!attacker) continue;
    const pt = attacker.toUpperCase();
    const dirs = PIN_DIRS[pt];
    if (!dirs) continue;
    // Same safety rule as findForks: the pinning piece must not be
    // capturable next move.
    if (isSquareAttackedBy(next.board, mv.to, !attackerIsWhite)) continue;

    const f0 = fileOf(mv.to), r0 = rankOf(mv.to);
    for (const [df, dr] of dirs) {
      let f = f0 + df, r = r0 + dr;
      let first = null; // { square, piece } of the first enemy on the ray
      while (f >= 0 && f < 8 && r >= 0 && r < 8) {
        const s = sq(f, r);
        const p = next.board[s];
        if (p) {
          const own = isWhite(p) === isWhite(attacker);
          if (first === null) {
            if (own) break;                       // blocked by friend
            if (p.toUpperCase() === "K") break;    // that's check, not pin
            first = { square: s, piece: p };
          } else {
            if (own) break;                       // friend shields the back rank
            const frontVal = PIECE_VALUE[first.piece.toUpperCase()];
            const backVal  = PIECE_VALUE[p.toUpperCase()];
            const absolute = p.toUpperCase() === "K";
            // Back piece must be the king or strictly more valuable.
            if (!absolute && backVal <= frontVal) break;
            // SEE-lite filter (mirrors `_pin_wins_material` in
            // chess_engine.py): only flag the pin if capturing the
            // pinned piece actually nets material — otherwise the UI
            // floods with cosmetic "pins" like Qh5 → f7 (pinned to
            // the king but defended, so Qxf7 just hangs the queen).
            if (!pinWinsMaterial(next.board, mv.to, first.square, attackerIsWhite)) {
              break;
            }
            pins.push({
              move: mv, attacker,
              pinnedSquare: first.square, pinnedPiece: first.piece,
              behindSquare: s,            behindPiece: p,
              absolute,
            });
            break;
          }
        }
        f += df; r += dr;
      }
    }
  }
  return pins;
}

/** Would capturing the piece on `pinnedSquare` win material for the
 *  attacker side? Mirrors `_pin_wins_material` in chess_engine.py:
 *  true when the pinned piece is undefended, or strictly more
 *  valuable than the pinning piece. */
function pinWinsMaterial(board, attackerSquare, pinnedSquare, attackerIsWhite) {
  const attackerVal = PIECE_VALUE[board[attackerSquare].toUpperCase()];
  const pinnedVal   = PIECE_VALUE[board[pinnedSquare].toUpperCase()];
  const defenders   = findAttackers(board, pinnedSquare, !attackerIsWhite);
  return defenders.length === 0 || pinnedVal > attackerVal;
}

function describePin(pin) {
  const from = sqName(pin.move.from);
  const to   = sqName(pin.move.to);
  const attackerName = pin.attacker ? pieceName(pin.attacker) : "piece";
  const kind = pin.absolute ? "absolute" : "relative";
  return `${attackerName} ${from}→${to} — ${kind} pin: `
       + `${pieceName(pin.pinnedPiece)} on ${sqName(pin.pinnedSquare)} `
       + `shields ${pieceName(pin.behindPiece)} on ${sqName(pin.behindSquare)}`;
}

// ---- Mate detection -------------------------------------------------
//
// Mirrors the Python helpers in chess_engine.py. We need three things
// for the UI:
//   * `inCheck(pos)`         — is `pos.sideToMove`'s king attacked?
//   * `findMateInOne(pos)`   — every move for `pos.sideToMove` that
//                              checkmates the opponent.
//   * `allowsMateInOne(pos, m)` — opponent's mate-in-one replies that
//                                 `m` permits. Empty ⇒ safe.

function inCheck(pos) {
  const white = pos.sideToMove === "w";
  const ksq = findKing(pos.board, white);
  if (ksq < 0) return false;
  return isSquareAttackedBy(pos.board, ksq, !white);
}

function isCheckmate(pos) {
  return inCheck(pos) && legalMoves(pos).length === 0;
}

function findMateInOne(pos) {
  const out = [];
  for (const mv of legalMoves(pos)) {
    const next = applyMove(pos, sanMoveFor(pos, mv));
    if (isCheckmate(next)) out.push(mv);
  }
  return out;
}

/** Opponent mate-in-one moves the side-to-move must address.
 *
 *  "If I (side-to-move) passed, could my opponent immediately
 *  checkmate me?" Used to *warn* the player before they walk into a
 *  forced mate, rather than only telling them after the fact.
 *
 *  Returns [] when side-to-move is already in check — the check
 *  itself is the more pressing issue and a position with both kings
 *  under attack is illegal anyway.  Also returns [] when side-to-move
 *  has no legal moves: that is stalemate (checkmate is filtered by
 *  the `inCheck` guard above) and the game is already over. */
function findMateThreats(pos) {
  if (inCheck(pos)) return [];
  if (legalMoves(pos).length === 0) return [];
  const swapped = clonePosition(pos);
  swapped.sideToMove = pos.sideToMove === "w" ? "b" : "w";
  swapped.epTarget = -1; // a null move forfeits en-passant rights
  return findMateInOne(swapped);
}

/** Convert a {from,to,promotion} move from `legalMoves` into the
 *  SAN-shaped object that `applyMove` expects. */
function sanMoveFor(pos, mv) {
  const piece = pos.board[mv.from];
  const pt = piece.toUpperCase();
  // Castling (king moving two files).
  if (pt === "K" && Math.abs(fileOf(mv.to) - fileOf(mv.from)) === 2) {
    return {
      castle: fileOf(mv.to) === 6 ? "K" : "Q",
      piece: "K", fromFile: -1, fromRank: -1, to: -1,
      capture: false, promotion: null, check: false, mate: false,
      san: fileOf(mv.to) === 6 ? "O-O" : "O-O-O",
    };
  }
  const capture = !!pos.board[mv.to] ||
    (pt === "P" && fileOf(mv.from) !== fileOf(mv.to) && mv.to === pos.epTarget);
  return {
    castle: null,
    piece: pt,
    fromFile: fileOf(mv.from),
    fromRank: rankOf(mv.from),
    to: mv.to,
    capture,
    promotion: mv.promotion || null,
    check: false, mate: false,
    san: `${pt}${sqName(mv.to)}`,
  };
}

function describeMate(mv, pos) {
  // `pos` is the board *before* the mating move was played, so we can
  // still see the moving piece on its origin square.
  const piece = pos?.board?.[mv.from];
  const pieceWord = piece ? pieceName(piece) : "piece";
  return `${pieceWord} ${sqName(mv.from)}→${sqName(mv.to)}#`;
}

// ---- Hanging-piece detection ----------------------------------------
//
// Mirror of `chess_engine.find_hanging_pieces`. A piece is "hanging"
// for the side-to-move when the opponent can win material by
// capturing it — either it's attacked and undefended, or the
// cheapest attacker is worth less than the piece itself.

function findAttackers(board, target, byWhite) {
  const out = [];
  for (let s = 0; s < 64; s++) {
    const p = board[s];
    if (!p) continue;
    if ((p === p.toUpperCase()) !== byWhite) continue;
    if (attacksFrom(board, s).includes(target)) out.push(s);
  }
  return out;
}

function findHangingPieces(pos, color) {
  if (color === undefined) color = pos.sideToMove;
  const white = color === "w";
  const out = [];
  for (let s = 0; s < 64; s++) {
    const p = pos.board[s];
    if (!p) continue;
    if ((p === p.toUpperCase()) !== white) continue;
    if (p.toUpperCase() === "K") continue;
    const attackers = findAttackers(pos.board, s, !white);
    if (!attackers.length) continue;
    const defenders = findAttackers(pos.board, s, white);
    const pieceVal = PIECE_VALUE[p.toUpperCase()];
    const cheapestAtk = Math.min(
      ...attackers.map((a) => PIECE_VALUE[pos.board[a].toUpperCase()])
    );
    if (!defenders.length || cheapestAtk < pieceVal) {
      out.push({ square: s, piece: p, attackers, defenders });
    }
  }
  return out;
}

/** Build a short human-readable description for a hanging piece,
 *  e.g. "bishop on c4 (attacked by knight d6, undefended)". */
function describeHanging(h) {
  const pieceWord = pieceName(h.piece);
  const sq = sqName(h.square);
  const atkList = h.attackers
    .map((a) => `${pieceName(state.positions[state.ply].board[a])} ${sqName(a)}`)
    .join(", ");
  const def = h.defenders.length
    ? `defended by ${h.defenders.length}`
    : "undefended";
  return `${pieceWord} on ${sq} (attacked by ${atkList}; ${def})`;
}

// =====================================================================
//  Material evaluation
// =====================================================================
//
// Mirror of `evaluate` / `score_move` / `best_move` in chess_engine.py.
// The simplest possible evaluator — sum the value of every non-king
// piece, signed by color.  Enough to catch blunders that lose material
// outright and to suggest the move that wins the most material in the
// current position.  Kings are excluded because both sides always have
// one, so they cancel out.

const EVAL_VALUES = { P: 1, N: 3, B: 3, R: 5, Q: 9 };

/** Material balance of `pos` from White's perspective, in pawn units.
 *  Positive = White is ahead, negative = Black is ahead. */
function evaluatePosition(pos) {
  let s = 0;
  for (const p of pos.board) {
    if (!p) continue;
    const v = EVAL_VALUES[p.toUpperCase()];
    if (v === undefined) continue;
    s += isWhite(p) ? v : -v;
  }
  return s;
}

/** Material gained by `move` from the perspective of the side to move.
 *  Positive = good for the mover, negative = loses material.
 *  `move` is the bare {from, to, promotion} shape produced by
 *  `legalMoves`; convert it to the SAN-shaped object `applyMove`
 *  expects before applying. */
function scoreMove(pos, move) {
  const sanMove = sanMoveFor(pos, move);
  const delta = evaluatePosition(applyMove(pos, sanMove)) - evaluatePosition(pos);
  return pos.sideToMove === "w" ? delta : -delta;
}

/** Return `{ move, score }` for the highest-scoring legal move in `pos`,
 *  or `null` if there are no legal moves (mate or stalemate). */
function bestMove(pos) {
  let best = null;
  let bestScore = -Infinity;
  for (const mv of legalMoves(pos)) {
    const s = scoreMove(pos, mv);
    if (s > bestScore) { best = mv; bestScore = s; }
  }
  return best ? { move: best, score: bestScore } : null;
}

/** Coordinate-style description of a move (e.g. "Qd1xd5", "e2-e4").
 *  Used in tooltips when we want to mention a move that was *not*
 *  played, where we don't have an existing SAN string to lean on. */
function describeMoveCoord(pos, mv) {
  const piece = pos.board[mv.from] || "";
  const sym = piece && piece.toUpperCase() !== "P" ? piece.toUpperCase() : "";
  const sep = pos.board[mv.to] || mv.isEp ? "x" : "-";
  return `${sym}${sqName(mv.from)}${sep}${sqName(mv.to)}`;
}

/** Format a pawn-units score as a signed decimal (e.g. "+1.5", "−0.3").
 *  Returns an empty string for nullish input. */
function formatEval(score) {
  if (score == null || !Number.isFinite(score)) return "";
  if (score === 0) return "0.0";
  const sign = score > 0 ? "+" : "−";
  return `${sign}${Math.abs(score).toFixed(1)}`;
}

// =====================================================================
//  Game replay
// =====================================================================

/** Build the full list of positions for a PGN, one entry per ply
 *  (positions[0] = initial, positions[i] = after the i-th half-move).
 *  Also returns a parallel `blunders` array — `blunders[i]` is null
 *  unless the move that produced positions[i] allowed the opponent
 *  a fork, in which case it holds the forks the opponent now has. */
function replayPgn(pgn) {
  const tokens = tokenizePgnMoves(pgn);
  const positions = [initialPosition()];
  const sanList = [];
  for (const tok of tokens) {
    try {
      const move = parseSan(tok);
      const next = applyMove(positions[positions.length - 1], move);
      positions.push(next);
      sanList.push(tok);
    } catch (err) {
      console.warn("Stopped replay at", tok, err.message);
      break;
    }
  }

  // Annotate moves:
  //   * blunders[i]       — non-null if the move that produced
  //                         positions[i] allowed the opponent a fresh
  //                         fork (a fork that wasn't already on the
  //                         board before the move).
  //   * goodMoves[i]      — non-null if the move that produced
  //                         positions[i] *is itself* a winning fork
  //                         the side-to-move had available before.
  //   * matesAllowed[i]   — non-null if the move that produced
  //                         positions[i] left the opponent with at
  //                         least one mate-in-one reply (and there
  //                         was *no* mate-in-one threat one ply
  //                         earlier — i.e. the move is what created
  //                         the threat, or failed to escape it).
  //   * matesDelivered[i] — non-null if the move that produced
  //                         positions[i] is itself checkmate.
  //   * mateThreats[i]    — non-null if, *at* positions[i], the
  //                         side-to-move faces an opponent
  //                         mate-in-one they must address. This is
  //                         the forward-looking warning shown to the
  //                         player who is about to move.
  const blunders       = new Array(positions.length).fill(null);
  const goodMoves      = new Array(positions.length).fill(null);
  const matesAllowed   = new Array(positions.length).fill(null);
  const matesDelivered = new Array(positions.length).fill(null);
  const mateThreats    = new Array(positions.length).fill(null);
  // Pin annotations mirror the fork ones: `pinsExecuted[i]` flags a
  // move that was itself a winning pin the side-to-move already had,
  // while `pinSuggestions[i]` lists the winning pins the side-to-move
  // can play *at* positions[i] (forward-looking, like `mateThreats`).
  // Compute pins once per position and reuse for both the executed
  // lookup (vs. the previous position's pin list) and the forward
  // suggestion array.
  const pinsExecuted   = new Array(positions.length).fill(null);
  const pinSuggestions = new Array(positions.length).fill(null);
  const pinsAtPos = positions.map((p) => findPins(p));
  for (let i = 0; i < positions.length; i++) {
    if (pinsAtPos[i].length) pinSuggestions[i] = pinsAtPos[i];
  }

  // Per-position material evaluation (White's perspective, in pawn
  // units) and per-move "you could have done better" suggestions.
  //   * evals[i]       — evaluatePosition(positions[i])
  //   * suggestions[i] — null unless the move that produced
  //                      positions[i] scored materially worse than the
  //                      best legal alternative in positions[i-1].
  //                      When set: { best: <coord SAN>, lost: <pawns> }
  const evals       = positions.map(evaluatePosition);
  const suggestions = new Array(positions.length).fill(null);
  // Only nag the player when they gave up at least this many pawns
  // compared to the best legal move.  Below this threshold the gap is
  // usually within noise of the (very crude) material-only evaluator.
  const SUGGEST_THRESHOLD = 2;

  let prevForks = findForks(positions[0]);
  for (let i = 1; i < positions.length; i++) {
    // Was the move just played one of the forks/pins that were
    // available in the previous position?
    const played = positions[i].lastMove;
    if (played) {
      const executedFork = prevForks.find(
        (f) => f.move.from === played.from && f.move.to === played.to
      );
      if (executedFork) goodMoves[i] = executedFork;
      const executedPin = pinsAtPos[i - 1].find(
        (p) => p.move.from === played.from && p.move.to === played.to
      );
      if (executedPin) pinsExecuted[i] = executedPin;
    }

    const here = findForks(positions[i]);
    if (here.length) {
      const prevSquares = new Set(
        prevForks.map((f) => `${f.move.from}-${f.move.to}`)
      );
      const fresh = here.filter(
        (f) => !prevSquares.has(`${f.move.from}-${f.move.to}`)
      );
      if (fresh.length) blunders[i] = fresh;
    }
    prevForks = here;

    // Mate annotations.
    if (isCheckmate(positions[i])) {
      // The move just played delivered checkmate.
      matesDelivered[i] = { from: played.from, to: played.to };
    } else {
      const mates = findMateInOne(positions[i]);
      if (mates.length) matesAllowed[i] = mates;
    }

    // Suggestion: did a clearly better move exist in positions[i-1]?
    // Compare the played move's material delta to bestMove()'s score.
    // Skip when the move ended the game (mate) — there's nothing left
    // to suggest at that point.
    if (played && !matesDelivered[i]) {
      const prev = positions[i - 1];
      const playerIsWhite = prev.sideToMove === "w";
      const playedScore = (evals[i] - evals[i - 1]) * (playerIsWhite ? 1 : -1);
      const best = bestMove(prev);
      if (best && best.score - playedScore >= SUGGEST_THRESHOLD) {
        suggestions[i] = {
          best: describeMoveCoord(prev, best.move),
          score: best.score,
          lost: best.score - playedScore,
        };
      }
    }
  }

  // Forward-looking threat: at each position, what mate-in-one moves
  // does the opponent threaten against the side-to-move? Computed for
  // every position (including the starting one) so the warning is
  // visible the moment the player navigates there.
  for (let i = 0; i < positions.length; i++) {
    const threats = findMateThreats(positions[i]);
    if (threats.length) mateThreats[i] = threats;
  }

  return {
    positions, sanList,
    blunders, goodMoves,
    matesAllowed, matesDelivered, mateThreats,
    pinsExecuted, pinSuggestions,
    evals, suggestions,
  };
}

// =====================================================================
//  UI
// =====================================================================

const state = {
  username: null,
  games: [],
  filteredIndices: [],
  selectedGameIdx: -1,
  positions: [initialPosition()],
  sanList: [],
  blunders: [],
  goodMoves: [],
  matesAllowed: [],
  matesDelivered: [],
  mateThreats: [],
  pinsExecuted: [],
  pinSuggestions: [],
  evals: [0],
  suggestions: [null],
  ply: 0, // index into positions
  orientation: "w", // "w" = white at the bottom, "b" = black at the bottom
};

const els = {
  form:         document.getElementById("load-form"),
  usernameIn:   document.getElementById("username-input"),
  loadBtn:      document.getElementById("load-btn"),
  cancelBtn:    document.getElementById("cancel-btn"),
  statusLabel:  document.getElementById("status-label"),
  filterInput:  document.getElementById("filter-input"),
  gameList:     document.getElementById("game-list"),
  gameListPane: document.getElementById("game-list-pane"),
  btnToggleGames: document.getElementById("btn-toggle-games"),
  gameTitle:    document.getElementById("game-title"),
  gameSubtitle: document.getElementById("game-subtitle"),
  board:        document.getElementById("board"),
  boardOverlay: document.getElementById("board-overlay"),
  rankLabels:   document.getElementById("rank-labels"),
  fileLabels:   document.getElementById("file-labels"),
  playerTop:    document.getElementById("player-top"),
  playerBottom: document.getElementById("player-bottom"),
  btnStart:     document.getElementById("btn-start"),
  btnPrev:      document.getElementById("btn-prev"),
  btnNext:      document.getElementById("btn-next"),
  btnEnd:       document.getElementById("btn-end"),
  btnFlip:      document.getElementById("btn-flip"),
  btnReset:     document.getElementById("btn-reset"),
  plyInd:       document.getElementById("ply-indicator"),
  moveList:     document.getElementById("move-list"),
  feedback:     document.getElementById("move-feedback"),
  hangingWarn:  document.getElementById("hanging-warn"),
  openingInfo:  document.getElementById("opening-info"),
};

function buildBoardSquares() {
  // Lay out the squares according to the current orientation.
  // White-at-bottom: ranks render top→bottom 8..1, files left→right a..h.
  // Black-at-bottom: ranks render top→bottom 1..8, files left→right h..a.
  const whiteBottom = state.orientation === "w";
  const ranks = whiteBottom ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  const files = whiteBottom ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];

  els.board.innerHTML = "";
  for (const r of ranks) {
    for (const f of files) {
      const cell = document.createElement("div");
      cell.className = "sq " + (((f + r) % 2 === 0) ? "dark" : "light");
      cell.dataset.idx = sq(f, r);
      els.board.appendChild(cell);
    }
  }

  els.rankLabels.innerHTML = "";
  for (const r of ranks) {
    const span = document.createElement("span");
    span.textContent = r + 1;
    els.rankLabels.appendChild(span);
  }

  els.fileLabels.innerHTML = "";
  for (const f of files) {
    const span = document.createElement("span");
    span.textContent = FILES[f];
    els.fileLabels.appendChild(span);
  }
}

function setOrientation(o) {
  if (o !== "w" && o !== "b") o = "w";
  if (state.orientation === o) return;
  state.orientation = o;
  buildBoardSquares();
}

// ---- Player bars / captures -----------------------------------------
//
// Standard starting piece counts used to display the captured pieces
// and the material balance under each player. Material values are
// shared with the tactical analysis helper (`PIECE_VALUE`, defined
// above) — kings are always present on both sides so their value
// cancels in the W-B difference regardless of what we use.
const START_COUNTS = { P: 8, N: 2, B: 2, R: 2, Q: 1, K: 1 };
// Order in which to render captured pieces (most valuable first).
const CAPTURE_ORDER = ["Q", "R", "B", "N", "P"];

function countPieces(board) {
  // Returns { w: {P,N,B,R,Q,K}, b: {...} }
  const w = { P: 0, N: 0, B: 0, R: 0, Q: 0, K: 0 };
  const b = { P: 0, N: 0, B: 0, R: 0, Q: 0, K: 0 };
  for (const p of board) {
    if (!p) continue;
    if (isWhite(p)) w[p]++;
    else            b[p.toUpperCase()]++;
  }
  return { w, b };
}

function computeCaptures(pos) {
  // Returns the pieces *missing* compared to the starting position,
  // grouped by colour of the captured piece.
  const counts = countPieces(pos.board);
  const missing = { w: {}, b: {} };
  let materialW = 0, materialB = 0;
  for (const t of CAPTURE_ORDER) {
    const lostW = Math.max(0, START_COUNTS[t] - counts.w[t]);
    const lostB = Math.max(0, START_COUNTS[t] - counts.b[t]);
    if (lostW) missing.w[t] = lostW;
    if (lostB) missing.b[t] = lostB;
    materialW += counts.w[t] * PIECE_VALUE[t];
    materialB += counts.b[t] * PIECE_VALUE[t];
  }
  return { missing, advantage: materialW - materialB };
}

function renderCaptureGlyphs(targetEl, color, missingForColor) {
  // Draws the captured pieces of `color` (i.e. pieces that the player
  // of opposite colour took). `missingForColor` is { P:n, N:n, ... }.
  targetEl.innerHTML = "";
  for (const t of CAPTURE_ORDER) {
    const n = missingForColor[t] || 0;
    for (let i = 0; i < n; i++) {
      const span = document.createElement("span");
      const piece = color === "w" ? t : t.toLowerCase();
      span.textContent = PIECE_GLYPHS[piece];
      span.className = color === "w" ? "piece-w" : "piece-b";
      targetEl.appendChild(span);
    }
  }
}

function playerInfo(g, color) {
  const side = color === "w" ? g.white : g.black;
  return {
    name: side?.username || "?",
    rating: side?.rating ?? null,
    result: side?.result || null,
  };
}

function resultClass(result) {
  // chess.com uses many strings; bucket them coarsely.
  if (!result) return "";
  if (result === "win") return "win";
  if (["agreed", "stalemate", "repetition", "insufficient",
       "50move", "timevsinsufficient"].includes(result)) return "draw";
  return "loss";
}

function resultText(result) {
  if (!result) return "";
  if (result === "win") return "1";
  if (resultClass(result) === "draw") return "½";
  return "0";
}

function fillStaticPlayerBar(barEl, info) {
  barEl.querySelector(".name").textContent = info.name;
  const rating = barEl.querySelector(".rating");
  rating.textContent = info.rating != null ? `(${info.rating})` : "";
  const badge = barEl.querySelector(".result-badge");
  const cls = resultClass(info.result);
  const text = resultText(info.result);
  badge.className = "result-badge" + (cls ? " " + cls : "");
  badge.textContent = text;
  badge.hidden = !text;
}

function renderPlayerBars() {
  const game = state.games[state.selectedGameIdx];
  if (!game) {
    for (const bar of [els.playerTop, els.playerBottom]) {
      bar.querySelector(".name").textContent = "—";
      bar.querySelector(".rating").textContent = "";
      bar.querySelector(".result-badge").textContent = "";
      bar.querySelector(".result-badge").className = "result-badge";
      bar.querySelector(".captures").innerHTML = "";
      bar.querySelector(".material").textContent = "";
    }
    return;
  }

  // Map player bar position -> color, based on board orientation.
  const bottomColor = state.orientation; // "w" or "b"
  const topColor    = bottomColor === "w" ? "b" : "w";

  fillStaticPlayerBar(els.playerTop,    playerInfo(game, topColor));
  fillStaticPlayerBar(els.playerBottom, playerInfo(game, bottomColor));

  const { missing, advantage } = computeCaptures(state.positions[state.ply]);
  // Captures shown next to a player are pieces of the *opposite* colour.
  renderCaptureGlyphs(els.playerTop.querySelector(".captures"),
                      bottomColor, missing[bottomColor]);
  renderCaptureGlyphs(els.playerBottom.querySelector(".captures"),
                      topColor,    missing[topColor]);

  // Material advantage: positive = white ahead.
  const advTop = topColor === "w" ? advantage : -advantage;
  const advBot = -advTop;
  els.playerTop.querySelector(".material")
    .textContent = advTop > 0 ? "+" + advTop : "";
  els.playerBottom.querySelector(".material")
    .textContent = advBot > 0 ? "+" + advBot : "";
}

function renderBoard(pos) {
  const cells = els.board.children;
  for (const cell of cells) {
    const idx = parseInt(cell.dataset.idx, 10);
    const piece = pos.board[idx];
    cell.classList.remove("hi");
    cell.textContent = "";
    if (piece) {
      const span = document.createElement("span");
      span.textContent = PIECE_GLYPHS[piece];
      span.className = isWhite(piece) ? "piece-w" : "piece-b";
      cell.appendChild(span);
    }
  }
  if (pos.lastMove) {
    const fromCell = els.board.querySelector(`[data-idx="${pos.lastMove.from}"]`);
    const toCell   = els.board.querySelector(`[data-idx="${pos.lastMove.to}"]`);
    fromCell?.classList.add("hi");
    toCell?.classList.add("hi");
  }
}

function renderMoveList() {
  els.moveList.innerHTML = "";
  const total = state.sanList.length;

  // Column headers — leave the move-number column blank and label the
  // two ply columns "White" / "Black" so the grid is self-explanatory.
  const blank = document.createElement("div");
  blank.className = "head";
  els.moveList.appendChild(blank);
  for (const side of ["White", "Black"]) {
    const h = document.createElement("div");
    h.className = "head";
    h.textContent = side;
    els.moveList.appendChild(h);
  }

  for (let i = 0; i < total; i += 2) {
    const num = document.createElement("div");
    num.className = "num";
    num.textContent = (i / 2 + 1) + ".";
    els.moveList.appendChild(num);

    for (const offset of [0, 1]) {
      const ply = i + offset;
      const span = document.createElement("div");
      span.className = "ply";
      if (ply < total) {
        const positionIdx = ply + 1; // positions index after this move
        const blunder       = state.blunders[positionIdx];
        const good          = state.goodMoves[positionIdx];
        const mateAllowed   = state.matesAllowed?.[positionIdx];
        const mateDelivered = state.matesDelivered?.[positionIdx];
        const pinExecuted   = state.pinsExecuted?.[positionIdx];
        const suggestion    = state.suggestions?.[positionIdx];
        const evalScore     = state.evals?.[positionIdx];
        let label = state.sanList[ply];
        // Precedence (most important first):
        //   1. mateDelivered — game-ending, definitely shown.
        //   2. mateAllowed   — usually losing, overrides fork blunder.
        //   3. blunder (fork)
        //   4. good (executed fork)
        //   5. pinExecuted   — only badged when no fork already lit
        //                      the move up.
        if (mateDelivered) {
          // Avoid double-#; some sanList entries already carry it.
          if (!label.endsWith("#")) label += "#";
          span.classList.add("good");
          span.title = "Checkmate";
        } else if (mateAllowed) {
          label += "??";
          span.classList.add("blunder");
          const detail = mateAllowed
            .map((m) => describeMate(m, state.positions[positionIdx]))
            .join("; ");
          span.title = "Allows mate-in-one: " + detail;
        } else if (blunder) {
          label += "??";
          span.classList.add("blunder");
          span.title = "Allows fork: "
            + blunder.map(describeFork).join("; ");
        } else if (good) {
          label += "!";
          span.classList.add("good");
          span.title = "Fork: " + describeFork(good);
        } else if (pinExecuted) {
          label += "!";
          span.classList.add("good");
          span.title = "Pin: " + describePin(pinExecuted);
        }
        span.textContent = label;
        // Append the post-move evaluation as a dim suffix (e.g. " +0.5").
        // Skip on mate moves — the score there is meaningless under a
        // pure material evaluator.
        if (evalScore != null && !mateDelivered && !mateAllowed) {
          const evalSpan = document.createElement("span");
          evalSpan.className = "eval";
          if (evalScore > 0) evalSpan.classList.add("eval-pos");
          else if (evalScore < 0) evalSpan.classList.add("eval-neg");
          evalSpan.textContent = " " + formatEval(evalScore);
          span.appendChild(evalSpan);
        }
        // If a clearly better move was available, mention it in the
        // tooltip without overriding existing tactical annotations.
        if (suggestion) {
          const hint = `Better: ${suggestion.best} `
            + `(${formatEval(suggestion.score)}, `
            + `lost ${suggestion.lost.toFixed(1)} pawns)`;
          span.title = span.title ? `${span.title} \u2014 ${hint}` : hint;
          if (!span.classList.contains("blunder")
              && !span.classList.contains("good")) {
            span.classList.add("suggestion");
          }
        }
        span.dataset.ply = positionIdx;
        span.addEventListener("click", () => setPly(positionIdx));
      } else {
        span.textContent = "";
      }
      els.moveList.appendChild(span);
    }
  }
  highlightActivePly();
}

function highlightActivePly() {
  for (const el of els.moveList.querySelectorAll(".ply")) {
    el.classList.toggle("active", parseInt(el.dataset.ply, 10) === state.ply);
  }
  const active = els.moveList.querySelector(".ply.active");
  active?.scrollIntoView({ block: "nearest" });
}

function setPly(p) {
  state.ply = Math.max(0, Math.min(state.positions.length - 1, p));
  renderBoard(state.positions[state.ply]);
  els.plyInd.textContent = `${state.ply} / ${state.positions.length - 1}`;
  els.btnStart.disabled = els.btnPrev.disabled = state.ply === 0;
  els.btnEnd.disabled   = els.btnNext.disabled = state.ply === state.positions.length - 1;
  highlightActivePly();
  renderPlayerBars();
  renderMoveFeedback();
  renderHangingWarning();
  renderOpeningInfo();
  renderForkArrows();
  // Selecting a different ply clears any in-progress move pickup.
  freePlay.selected = -1;
  freePlay.legalForSelected = [];
  renderSelection();
}

function renderMoveFeedback() {
  const fb = els.feedback;
  if (!fb) return;
  fb.className = "";
  fb.textContent = "";
  const idx = state.ply;
  if (idx <= 0) {
    // Even at the starting position we may want to highlight that the
    // side-to-move can immediately set up a pin. Fall through into the
    // suggestion block below instead of bailing out unconditionally.
    fb.hidden = true;
  }
  const san = idx > 0 ? state.sanList[idx - 1] : null;
  const spoken = san ? spokenSan(san) : null;
  const blunder       = idx > 0 ? state.blunders[idx]    : null;
  const good          = idx > 0 ? state.goodMoves[idx]   : null;
  const mateAllowed   = state.matesAllowed?.[idx];
  const mateDelivered = state.matesDelivered?.[idx];
  const mateThreat    = state.mateThreats?.[idx];
  const pinExecuted   = idx > 0 ? state.pinsExecuted?.[idx]   : null;
  const pinSuggestion = state.pinSuggestions?.[idx];
  if (mateDelivered) {
    fb.hidden = false;
    fb.classList.add("good");
    fb.textContent = `\u2605 ${spoken} \u2014 checkmate!`;
  } else if (mateAllowed) {
    fb.hidden = false;
    fb.classList.add("blunder");
    const detail = mateAllowed
      .map((m) => describeMate(m, state.positions[idx]))
      .join("; ");
    fb.textContent = `\u26A0 ${spoken} (??) \u2014 allows mate-in-one: ${detail}`;
  } else if (mateThreat) {
    // Forward-looking warning shown *before* the player walks into a
    // mate. The threatening side is whoever just moved \u2014 i.e. the
    // opposite of the side-to-move in the current position.
    fb.hidden = false;
    fb.classList.add("blunder");
    const threatColor =
      state.positions[idx].sideToMove === "w" ? "Black" : "White";
    const detail = mateThreat
      .map((m) => describeMate(m, state.positions[idx]))
      .join("; ");
    fb.textContent =
      `\u26A0 Watch out \u2014 ${threatColor} threatens mate-in-one: ${detail}`;
  } else if (blunder) {
    fb.hidden = false;
    fb.classList.add("blunder");
    const detail = blunder.map(describeFork).join("; ");
    fb.textContent = `\u26A0 ${spoken} (??) \u2014 allows fork: ${detail}`;
  } else if (good) {
    fb.hidden = false;
    fb.classList.add("good");
    fb.textContent = `\u2605 ${spoken} (!) \u2014 fork: ${describeFork(good)}`;
  } else if (pinExecuted) {
    fb.hidden = false;
    fb.classList.add("good");
    fb.textContent = `\u2605 ${spoken} (!) \u2014 pin: ${describePin(pinExecuted)}`;
  } else if (pinSuggestion) {
    // Forward-looking hint: the side-to-move can play a winning pin
    // right now. Only fires when no more-urgent annotation is active.
    const sideWord =
      state.positions[idx].sideToMove === "w" ? "White" : "Black";
    const detail = pinSuggestion.map(describePin).join("; ");
    fb.hidden = false;
    fb.classList.add("good");
    fb.textContent = `\u2605 ${sideWord} has a pin available: ${detail}`;
  } else {
    fb.hidden = true;
  }
}

/** Warn about loose pieces on both sides:
 *   - Side-to-move's own hanging pieces  → "you need to defend / move this"
 *   - Opponent's hanging pieces           → "you can grab this for free"
 *  Paints an amber ring on every hanging square. Suppressed when a
 *  mate alarm is already firing (those are far more urgent and the
 *  player can address the loose piece on the next ply).
 */
function renderHangingWarning() {
  const panel = els.hangingWarn;
  if (!panel) return;
  panel.innerHTML = "";
  // Clear any stale square outlines first.
  for (const cell of els.board.children) cell.classList.remove("hanging");

  const idx = state.ply;
  const pos = state.positions[idx];
  if (!pos) { panel.hidden = true; return; }

  // Don't compete with mate warnings, and skip when the side to
  // move is in check — the check itself is forcing and any "you
  // could capture" / "you should defend" hint is misleading
  // because most pieces can't legally move until the check is
  // resolved.
  if (state.matesDelivered?.[idx]
      || state.matesAllowed?.[idx]
      || state.mateThreats?.[idx]
      || inCheck(pos)) {
    panel.hidden = true;
    return;
  }

  const own       = findHangingPieces(pos, pos.sideToMove);
  const opponent  = findHangingPieces(pos, pos.sideToMove === "w" ? "b" : "w");
  if (!own.length && !opponent.length) {
    panel.hidden = true;
    return;
  }

  const sideWord = (c) => c === "w" ? "White" : "Black";

  if (own.length) {
    const sec = makeHangingSection(
      `⚠ ${sideWord(pos.sideToMove)}: `
        + (own.length === 1 ? "this piece is" : "these pieces are")
        + " hanging",
      own,
    );
    panel.appendChild(sec);
  }
  if (opponent.length) {
    const oppColor = pos.sideToMove === "w" ? "b" : "w";
    const sec = makeHangingSection(
      `★ ${sideWord(oppColor)}: `
        + (opponent.length === 1 ? "this piece is" : "these pieces are")
        + " hanging — you can capture",
      opponent,
    );
    sec.classList.add("opponent");
    panel.appendChild(sec);
  }

  panel.hidden = false;
}

/** Build one labelled section of the hanging-warn panel. Also paints
 *  the amber outline on each affected board square as a side-effect. */
function makeHangingSection(headingText, list) {
  const wrap = document.createElement("div");
  wrap.className = "hw-section";

  const heading = document.createElement("div");
  heading.className = "hw-heading";
  heading.textContent = headingText;
  wrap.appendChild(heading);

  const ul = document.createElement("ul");
  ul.className = "hw-list";
  for (const h of list) {
    const li = document.createElement("li");
    li.textContent = "• " + describeHanging(h);
    ul.appendChild(li);
    const cell = els.board.querySelector(`[data-idx="${h.square}"]`);
    cell?.classList.add("hanging");
  }
  wrap.appendChild(ul);
  return wrap;
}

/** Show the current opening name + typical replies for the side to
 *  move. Lookup is by SAN-prefix against `OPENING_BOOK`. Hidden once
 *  the game has left book — at that point the suggestions would be
 *  stale and the name no longer reflects the actual position. */
function renderOpeningInfo() {
  const panel = els.openingInfo;
  if (!panel) return;
  panel.innerHTML = "";

  // Build the SAN-token list from the moves played up to the current
  // ply, stripped of any check/mate suffix so they match book format.
  const sans = state.sanList.slice(0, state.ply).map(stripSanSuffix);
  const op = identifyOpening(sans);
  // Only show the panel while we're sitting *exactly* on a known book
  // node; once we go past it, hide everything.
  if (!op || op.moves.length !== sans.length) {
    panel.hidden = true;
    return;
  }

  const name = document.createElement("div");
  name.className = "op-name";
  name.textContent = `📖 ${op.name}`;
  panel.appendChild(name);

  const plan = document.createElement("div");
  plan.className = "op-plan";
  plan.textContent = op.plan;
  panel.appendChild(plan);

  if (op.replies?.length) {
    const sideToMove =
      state.positions[state.ply].sideToMove === "w" ? "White" : "Black";
    const heading = document.createElement("div");
    heading.className = "op-plan";
    heading.textContent = `Typical replies for ${sideToMove}:`;
    panel.appendChild(heading);

    const ul = document.createElement("ul");
    ul.className = "op-replies";
    for (const r of op.replies) {
      const li = document.createElement("li");
      const san = document.createElement("span");
      san.className = "san";
      san.textContent = r.san;
      const note = document.createElement("span");
      note.className = "note";
      note.textContent = r.note;
      li.appendChild(san);
      li.appendChild(note);
      ul.appendChild(li);
    }
    panel.appendChild(ul);
  }

  panel.hidden = false;
}

/** Turn a SAN token like "Qd5", "Nge7", "exd5", "O-O-O", "e8=Q+"
 *  into a readable English phrase used by the feedback balloon. */
function spokenSan(san) {
  if (!san) return "";
  // Strip trailing check/mate markers but remember them.
  let trail = "";
  let core = san;
  if (core.endsWith("#")) { trail = " checkmate"; core = core.slice(0, -1); }
  else if (core.endsWith("+")) { trail = " with check"; core = core.slice(0, -1); }

  if (core === "O-O" || core === "0-0") return "kingside castle" + trail;
  if (core === "O-O-O" || core === "0-0-0") return "queenside castle" + trail;

  // Promotion: split off "=Q" / "Q" suffix.
  let promo = "";
  const promoMatch = core.match(/=?([QRBN])$/);
  if (promoMatch) {
    promo = `, promote to ${PIECE_NAMES[promoMatch[1]]}`;
    core = core.slice(0, -promoMatch[0].length);
  }

  // Piece prefix (uppercase letter) or implicit pawn.
  let pieceLetter = "P";
  if (/^[KQRBN]/.test(core)) {
    pieceLetter = core[0];
    core = core.slice(1);
  }
  const pieceWord = PIECE_NAMES[pieceLetter];

  // Capture marker.
  const captures = core.includes("x");
  if (captures) core = core.replace("x", "");

  // Whatever's left is "[disambig]dest" — dest is the trailing 2 chars.
  const dest = core.slice(-2);
  const disambig = core.slice(0, -2);
  let disambigPhrase = "";
  if (disambig) {
    if (/^[a-h]$/.test(disambig))      disambigPhrase = ` (${disambig}-file)`;
    else if (/^[1-8]$/.test(disambig)) disambigPhrase = ` (rank ${disambig})`;
    else                               disambigPhrase = ` (from ${disambig})`;
  }

  const verb = captures ? "takes" : "to";
  return `${pieceWord}${disambigPhrase} ${verb} ${dest}${promo}${trail}`;
}

// ---- Opening book ---------------------------------------------------
//
// Mirror of `openings.py`. The lookup is purely string-based so the
// two implementations are kept in sync by copying the table verbatim.
// Each entry is `{ moves: [...SAN tokens], name, plan, replies: [{san, note}] }`.
// `replies` are the typical next moves for the side whose turn it is
// *after* the matched move list has been played.
const OPENING_BOOK = [
  // ---- 1.e4 ---------------------------------------------------------
  {
    moves: ["e4"],
    name:  "King's Pawn Opening",
    plan:  "White stakes out the centre and frees the king's bishop and queen.",
    replies: [
      { san: "e5", note: "Open Game — classical, symmetric centre." },
      { san: "c5", note: "Sicilian Defence — fight for the centre asymmetrically." },
      { san: "e6", note: "French Defence — solid, slightly cramped." },
      { san: "c6", note: "Caro-Kann Defence — solid pawn structure." },
    ],
  },
  {
    moves: ["e4", "e5"],
    name:  "Open Game",
    plan:  "Both sides will develop knights and contest the d4/d5 squares.",
    replies: [
      { san: "Nf3", note: "King's Knight Opening — attacks e5 and prepares O-O." },
      { san: "Nc3", note: "Vienna Game — flexible, often followed by f4." },
      { san: "Bc4", note: "Bishop's Opening — aims at f7." },
    ],
  },
  {
    moves: ["e4", "e5", "Nf3"],
    name:  "King's Knight Opening",
    plan:  "Black must defend e5 immediately.",
    replies: [
      { san: "Nc6", note: "Standard development; supports e5." },
      { san: "Nf6", note: "Petroff Defence — symmetric counter-attack on e4." },
      { san: "d6",  note: "Philidor Defence — solid but passive." },
    ],
  },
  {
    moves: ["e4", "e5", "Nf3", "Nc6"],
    name:  "King's Knight Opening (main line)",
    plan:  "White picks how to pressure the centre and Black's knight.",
    replies: [
      { san: "Bb5", note: "Ruy López — pins the c6 knight and pressures e5." },
      { san: "Bc4", note: "Italian Game — eyes f7, prepares quick castling." },
      { san: "d4",  note: "Scotch Game — opens the centre immediately." },
    ],
  },
  {
    moves: ["e4", "e5", "Nf3", "Nc6", "Bb5"],
    name:  "Ruy López (Spanish Opening)",
    plan:  "Black usually challenges the bishop or supports the centre.",
    replies: [
      { san: "a6",  note: "Morphy Defence — the main line; asks the bishop a question." },
      { san: "Nf6", note: "Berlin Defence — solid, leads to the famous endgame." },
      { san: "d6",  note: "Steinitz Defence — old, very solid." },
    ],
  },
  {
    moves: ["e4", "e5", "Nf3", "Nc6", "Bc4"],
    name:  "Italian Game",
    plan:  "Black mirrors development and contests the centre.",
    replies: [
      { san: "Bc5", note: "Giuoco Piano — quiet, symmetric setup." },
      { san: "Nf6", note: "Two Knights Defence — sharper, invites complications." },
      { san: "Be7", note: "Hungarian Defence — solid and modest." },
    ],
  },
  {
    moves: ["e4", "c5"],
    name:  "Sicilian Defence",
    plan:  "White typically prepares Nf3 and d4 to open the centre.",
    replies: [
      { san: "Nf3", note: "Open Sicilian setup — most principled." },
      { san: "Nc3", note: "Closed Sicilian — keeps the centre intact." },
      { san: "c3",  note: "Alapin Variation — prepares d4 with a strong centre." },
    ],
  },
  {
    moves: ["e4", "e6"],
    name:  "French Defence",
    plan:  "Black will challenge the centre with ...d5 next move.",
    replies: [
      { san: "d4",  note: "Main line — claims the full centre." },
      { san: "Nc3", note: "Flexible — keeps options open against ...d5." },
      { san: "d3",  note: "King's Indian Attack setup." },
    ],
  },
  {
    moves: ["e4", "c6"],
    name:  "Caro-Kann Defence",
    plan:  "Black plans ...d5 with a sound pawn structure.",
    replies: [
      { san: "d4",  note: "Main line — accepts the central challenge." },
      { san: "Nc3", note: "Two Knights setup — flexible." },
      { san: "Nf3", note: "Two Knights Attack." },
    ],
  },

  // ---- 1.d4 ---------------------------------------------------------
  {
    moves: ["d4"],
    name:  "Queen's Pawn Opening",
    plan:  "White stakes out d4 and prepares c4 to fight for the centre.",
    replies: [
      { san: "d5",  note: "Closed Game / Queen's Gambit territory." },
      { san: "Nf6", note: "Indian Defence — flexible, avoids early ...d5." },
      { san: "f5",  note: "Dutch Defence — fights for e4." },
    ],
  },
  {
    moves: ["d4", "d5"],
    name:  "Closed Game",
    plan:  "White's strongest try is to challenge d5 with c4.",
    replies: [
      { san: "c4",  note: "Queen's Gambit — the principled break." },
      { san: "Nf3", note: "Quiet developing move; avoids the gambit." },
      { san: "Bf4", note: "London System setup." },
    ],
  },
  {
    moves: ["d4", "d5", "c4"],
    name:  "Queen's Gambit",
    plan:  "Black chooses between accepting and declining the pawn.",
    replies: [
      { san: "e6",   note: "Queen's Gambit Declined — solid and classical." },
      { san: "c6",   note: "Slav Defence — solid, keeps the c8 bishop free." },
      { san: "dxc4", note: "Queen's Gambit Accepted — gives up the centre for development." },
    ],
  },
  {
    moves: ["d4", "Nf6"],
    name:  "Indian Defence",
    plan:  "White typically plays c4 to claim the centre.",
    replies: [
      { san: "c4",  note: "Main line — heads for King's/Queen's Indian territory." },
      { san: "Nf3", note: "Avoids the sharpest lines." },
      { san: "Bg5", note: "Trompowsky Attack — early pin." },
    ],
  },
  {
    moves: ["d4", "Nf6", "c4", "g6"],
    name:  "King's Indian / Grünfeld setup",
    plan:  "Black fianchettoes the king's bishop; White picks a centre.",
    replies: [
      { san: "Nc3", note: "Main line — invites the King's Indian or Grünfeld." },
      { san: "Nf3", note: "Flexible; can transpose to Fianchetto systems." },
    ],
  },
  {
    moves: ["d4", "Nf6", "c4", "e6"],
    name:  "Indian Defence (with ...e6)",
    plan:  "Black is heading for the Nimzo-Indian or Queen's Indian.",
    replies: [
      { san: "Nc3", note: "Allows the Nimzo-Indian after ...Bb4." },
      { san: "Nf3", note: "Avoids the Nimzo; invites the Queen's Indian." },
    ],
  },

  // ---- Flank openings -----------------------------------------------
  {
    moves: ["c4"],
    name:  "English Opening",
    plan:  "White fights for d5 from the side before committing centre pawns.",
    replies: [
      { san: "e5",  note: "Reversed Sicilian — Black takes the initiative." },
      { san: "Nf6", note: "Flexible; can transpose to many Indian setups." },
      { san: "c5",  note: "Symmetric English — solid and balanced." },
    ],
  },
  {
    moves: ["Nf3"],
    name:  "Réti Opening",
    plan:  "White keeps the centre flexible and may fianchetto.",
    replies: [
      { san: "d5",  note: "Classical reply — claims the centre." },
      { san: "Nf6", note: "Symmetric, very flexible." },
    ],
  },
];

/** Strip a trailing `+` or `#` from a SAN token so it can be matched
 *  against the opening book's bare-move format. */
function stripSanSuffix(san) {
  if (!san) return san;
  if (san.endsWith("+") || san.endsWith("#")) return san.slice(0, -1);
  return san;
}

/** Return the deepest opening whose move list is a prefix of `sanMoves`,
 *  or null if none matches. Mirror of `openings.identify_opening`. */
function identifyOpening(sanMoves) {
  const moves = sanMoves.map(stripSanSuffix);
  let best = null;
  for (const op of OPENING_BOOK) {
    const n = op.moves.length;
    if (n > moves.length) continue;
    let ok = true;
    for (let i = 0; i < n; i++) {
      if (op.moves[i] !== moves[i]) { ok = false; break; }
    }
    if (ok && (!best || n > best.moves.length)) best = op;
  }
  return best;
}

/** Return suggested replies for the *current* opening node, or null
 *  when we are beyond known theory or in an unknown line. Mirror of
 *  `openings.suggest_responses`. */
function suggestOpeningResponses(sanMoves) {
  const moves = sanMoves.map(stripSanSuffix);
  const op = identifyOpening(moves);
  if (!op || op.moves.length !== moves.length) return null;
  return { name: op.name, plan: op.plan, replies: op.replies };
}

// ---- Fork-arrow overlay ---------------------------------------------
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Draw arrows for whatever annotation lives on the current ply:
 *  - Allowed mate-in-one (red): one from→to arrow per mating move.
 *  - Opponent mate threat (red): same arrow drawn *before* the player
 *    blunders, as a heads-up.
 *  - Fork blunder (red): dashed from→to plus solid arrows to each target.
 *  - Executed fork (green): solid arrows to each target.
 *  Mate that was actually delivered is *not* drawn — the last-move
 *  highlight already shows it, and the king is mated where it stands.
 */
function renderForkArrows() {
  const svg = els.boardOverlay;
  if (!svg) return;
  svg.innerHTML = "";

  const idx = state.ply;
  const blunder       = state.ply > 0 ? state.blunders[idx]    : null;
  const good          = state.ply > 0 ? state.goodMoves[idx]   : null;
  const mateAllowed   = state.matesAllowed?.[idx];
  const mateDelivered = state.matesDelivered?.[idx];
  const mateThreat    = state.mateThreats?.[idx];
  const pinExecuted   = state.ply > 0 ? state.pinsExecuted?.[idx]   : null;
  const pinSuggestion = state.pinSuggestions?.[idx];

  // Build a list of "shapes" to draw. Each shape has a colour class
  // ("blunder" / "good") and one of:
  //   * a `fork` with `.move.{from,to}` + `.targets[]`,
  //   * a `mate` with `.move.{from,to}` (no targets), or
  //   * a `pin`  with `.move.{from,to}` + `.pinnedSquare` + `.behindSquare`.
  const shapes = [];
  if (mateDelivered) {
    // Don't draw anything for delivered mate — too late to learn from
    // arrows and the last-move highlight already makes it obvious.
  } else if (mateAllowed) {
    for (const m of mateAllowed) {
      shapes.push({ kind: "mate", move: m, cls: "blunder" });
    }
  } else if (mateThreat) {
    // Same visual as an allowed mate — the threat is the opponent's
    // upcoming mating move, drawn so the player can see what to defend.
    for (const m of mateThreat) {
      shapes.push({ kind: "mate", move: m, cls: "blunder" });
    }
  } else if (blunder) {
    for (const f of blunder) shapes.push({ kind: "fork", fork: f, cls: "blunder" });
  } else if (good) {
    shapes.push({ kind: "fork", fork: good, cls: "good" });
  } else if (pinExecuted) {
    shapes.push({ kind: "pin", pin: pinExecuted, cls: "good" });
  } else if (pinSuggestion) {
    // Prospective hint: dashed arrow on the suggested move + solid
    // arrow through the pinned piece to the back piece.
    for (const p of pinSuggestion) {
      shapes.push({ kind: "pin", pin: p, cls: "good", suggested: true });
    }
  }
  if (!shapes.length) return;

  // One arrowhead marker per colour, referenced by id from <line>.
  // Use userSpaceOnUse so the arrowhead size is in board-square units
  // (matching the viewBox), independent of the line's stroke-width.
  const defs = svgEl("defs");
  for (const cls of ["blunder", "good"]) {
    const marker = svgEl("marker", {
      id: `fa-head-${cls}`,
      viewBox: "0 0 10 10",
      refX: "8", refY: "5",
      markerUnits: "userSpaceOnUse",
      markerWidth: "0.45", markerHeight: "0.45",
      orient: "auto-start-reverse",
    });
    const head = svgEl("path", { d: "M0,0 L10,5 L0,10 z" });
    head.setAttribute("class", `fork-arrowhead-${cls}`);
    marker.appendChild(head);
    defs.appendChild(marker);
  }
  svg.appendChild(defs);

  const whiteBottom = state.orientation === "w";
  const center = (square) => {
    const f = fileOf(square), r = rankOf(square);
    const col = whiteBottom ? f : 7 - f;
    const row = whiteBottom ? 7 - r : r;
    return { x: col + 0.5, y: row + 0.5 };
  };
  // Shorten lines slightly so the arrowhead doesn't cover the target glyph.
  const shorten = (a, b, by) => {
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    return { x: b.x - (dx / len) * by, y: b.y - (dy / len) * by };
  };

  for (const shape of shapes) {
    const cls = shape.cls;

    if (shape.kind === "mate") {
      // Single solid arrow showing where the opponent will mate from.
      const fromC = center(shape.move.from);
      const toC   = center(shape.move.to);
      const tip   = shorten(fromC, toC, 0.32);
      const line  = svgEl("line", {
        x1: fromC.x, y1: fromC.y, x2: tip.x, y2: tip.y,
        "stroke-width": "0.13",
        "marker-end": `url(#fa-head-${cls})`,
      });
      line.setAttribute("class", `fork-arrow-${cls}`);
      svg.appendChild(line);
      continue;
    }

    if (shape.kind === "pin") {
      // For a prospective pin suggestion we also draw the dashed
      // from→to so the player can see *which* move to make. For an
      // already-played pin, the last-move highlight already shows it.
      const fromC   = center(shape.pin.move.from);
      const toC     = center(shape.pin.move.to);
      const pinnedC = center(shape.pin.pinnedSquare);
      const behindC = center(shape.pin.behindSquare);
      if (shape.suggested) {
        const tip = shorten(fromC, toC, 0.25);
        const mv = svgEl("line", {
          x1: fromC.x, y1: fromC.y, x2: tip.x, y2: tip.y,
          "stroke-width": "0.13",
          "marker-end": `url(#fa-head-${cls})`,
        });
        mv.setAttribute("class", `fork-move-${cls}`);
        svg.appendChild(mv);
      }
      // A single arrow from the attacker square straight through to
      // the back piece visualises the pin axis. The arrowhead lands
      // on the shielded back piece; the pinned piece sits along the
      // line, which the player can read directly off the board.
      const tip = shorten(toC, behindC, 0.32);
      const ray = svgEl("line", {
        x1: toC.x, y1: toC.y, x2: tip.x, y2: tip.y,
        "stroke-width": "0.13",
        "marker-end": `url(#fa-head-${cls})`,
      });
      ray.setAttribute("class", `fork-arrow-${cls}`);
      svg.appendChild(ray);
      // Suppress unused-var warning for pinnedC while leaving it
      // available for future enhancements (e.g. a midpoint marker).
      void pinnedC;
      continue;
    }

    // Fork shape — same logic as before.
    const fromC = center(shape.fork.move.from);
    const toC   = center(shape.fork.move.to);

    // Dashed "move" arrow only when the move hasn't been played yet
    // (blunders). For executed forks, the last-move highlight already
    // shows from->to and an extra arrow would just add clutter.
    // Stroke-width is set as an SVG attribute (user-space units) so it
    // scales with the board; CSS pixel widths would be re-interpreted
    // by the viewBox and render either invisible or enormous.
    if (cls === "blunder") {
      const tip = shorten(fromC, toC, 0.25);
      const mv = svgEl("line", {
        x1: fromC.x, y1: fromC.y, x2: tip.x, y2: tip.y,
        "stroke-width": "0.13",
        "marker-end": `url(#fa-head-${cls})`,
      });
      mv.setAttribute("class", `fork-move-${cls}`);
      svg.appendChild(mv);
    }

    for (const t of shape.fork.targets) {
      const targetC = center(t.square);
      const tip = shorten(toC, targetC, 0.32);
      const line = svgEl("line", {
        x1: toC.x, y1: toC.y, x2: tip.x, y2: tip.y,
        "stroke-width": "0.13",
        "marker-end": `url(#fa-head-${cls})`,
      });
      line.setAttribute("class", `fork-arrow-${cls}`);
      svg.appendChild(line);
    }
  }
}

// ---- Game list -------------------------------------------------------
function gameLabel(g) {
  const w = g.white?.username ?? "?";
  const b = g.black?.username ?? "?";
  const wr = g.white?.rating ? ` (${g.white.rating})` : "";
  const br = g.black?.rating ? ` (${g.black.rating})` : "";
  return { players: `${w}${wr} vs ${b}${br}`, game: g };
}

function resultBadge(g) {
  // From the perspective of the queried user, if known.
  const me = state.username;
  if (!me) return "";
  let myResult, oppResult;
  if (g.white?.username?.toLowerCase() === me) {
    myResult = g.white?.result; oppResult = g.black?.result;
  } else if (g.black?.username?.toLowerCase() === me) {
    myResult = g.black?.result; oppResult = g.white?.result;
  } else {
    return "";
  }
  if (myResult === "win") return { cls: "result-W", text: "W" };
  if (oppResult === "win") return { cls: "result-L", text: "L" };
  return { cls: "result-D", text: "D" };
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toISOString().slice(0, 10);
}

function renderGameList() {
  els.gameList.innerHTML = "";
  const filter = els.filterInput.value.trim().toLowerCase();
  state.filteredIndices = [];

  state.games.forEach((g, idx) => {
    const label = gameLabel(g).players.toLowerCase();
    const tc = (g.time_class ?? "") + " " + (g.time_control ?? "");
    if (filter && !label.includes(filter) && !tc.toLowerCase().includes(filter)) return;
    state.filteredIndices.push(idx);

    const li = document.createElement("li");
    li.dataset.idx = idx;

    const players = document.createElement("div");
    players.className = "players";
    players.textContent = gameLabel(g).players;
    li.appendChild(players);

    const badge = resultBadge(g);
    if (badge) {
      const span = document.createElement("div");
      span.className = badge.cls;
      span.textContent = badge.text;
      li.appendChild(span);
    }

    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = [g.time_class, g.time_control, fmtTime(g.end_time)]
      .filter(Boolean)
      .join(" • ");
    li.appendChild(sub);

    li.addEventListener("click", () => selectGame(idx));
    if (idx === state.selectedGameIdx) li.classList.add("active");
    els.gameList.appendChild(li);
  });
}

function selectGame(idx) {
  state.selectedGameIdx = idx;
  const g = state.games[idx];
  if (!g) return;

  const {
    positions, sanList,
    blunders, goodMoves,
    matesAllowed, matesDelivered, mateThreats,
    pinsExecuted, pinSuggestions,
    evals, suggestions,
  } = replayPgn(g.pgn || "");
  state.positions = positions;
  state.sanList = sanList;
  state.blunders = blunders;
  state.goodMoves = goodMoves;
  state.matesAllowed = matesAllowed;
  state.matesDelivered = matesDelivered;
  state.mateThreats = mateThreats;
  state.pinsExecuted = pinsExecuted;
  state.pinSuggestions = pinSuggestions;
  state.evals = evals;
  state.suggestions = suggestions;
  state.ply = 0;

  // Orient the board so the queried player is always at the bottom.
  // Falls back to white-at-bottom if no username is set or if the
  // queried user isn't a participant of this game.
  const me = state.username;
  const blackName = g.black?.username?.toLowerCase();
  setOrientation(me && me === blackName ? "b" : "w");

  els.gameTitle.textContent = [g.time_class, g.time_control, fmtTime(g.end_time)]
    .filter(Boolean).join(" • ") || "Game";
  els.gameSubtitle.textContent = "";
  if (g.url) {
    const a = document.createElement("a");
    a.href = g.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = g.url;
    els.gameSubtitle.appendChild(document.createTextNode("↗ "));
    els.gameSubtitle.appendChild(a);
  }

  renderMoveList();
  setPly(0);

  for (const li of els.gameList.children) {
    li.classList.toggle("active", parseInt(li.dataset.idx, 10) === idx);
  }
  updateFreePlayChrome();
}

/** Collapse or expand the games sidebar. When collapsed the panel
 *  shrinks to a thin strip and the toggle button becomes the only
 *  interactive element. */
function setGameListCollapsed(collapsed) {
  const pane = els.gameListPane;
  const btn  = els.btnToggleGames;
  if (!pane) return;
  pane.classList.toggle("collapsed", collapsed);
  document.body.classList.toggle("games-collapsed", collapsed);
  if (btn) {
    btn.textContent = collapsed ? "»" : "«";
    const label = collapsed ? "Expand games panel" : "Collapse games panel";
    btn.title = label;
    btn.setAttribute("aria-label", label);
  }
}

// ---- Loading ---------------------------------------------------------
//
// chess.com's public API supports CORS, so we can fetch directly from
// the browser. For each user we GET
//   https://api.chess.com/pub/player/<user>/games/archives
// and then walk every monthly archive, concatenating their `games`
// arrays.

const API_BASE = "https://api.chess.com/pub/player";
let activeLoad = null; // { controller: AbortController }

function setStatus(text, isError = false) {
  els.statusLabel.textContent = text;
  els.statusLabel.classList.toggle("error", !!isError);
}

function setLoading(loading) {
  els.loadBtn.disabled = loading;
  els.usernameIn.disabled = loading;
  els.cancelBtn.hidden = !loading;
}

function slimGame(g) {
  const w = g.white || {}, b = g.black || {};
  return {
    url: g.url,
    pgn: g.pgn || "",
    time_class: g.time_class,
    time_control: g.time_control,
    rated: g.rated,
    rules: g.rules,
    end_time: g.end_time,
    white: { username: w.username, rating: w.rating, result: w.result },
    black: { username: b.username, rating: b.rating, result: b.result },
  };
}

async function loadFromChessCom(username) {
  if (activeLoad) activeLoad.controller.abort();
  const controller = new AbortController();
  activeLoad = { controller };
  setLoading(true);

  // Reset current state
  state.username = username;
  state.games = [];
  state.selectedGameIdx = -1;
  renderGameList();
  els.gameTitle.textContent = "Select a game";
  els.gameSubtitle.textContent = "";
  state.positions = [initialPosition()];
  state.sanList = [];
  setPly(0);
  renderMoveList();

  try {
    setStatus(`Fetching archive list for "${username}"...`);
    const archivesResp = await fetch(
      `${API_BASE}/${encodeURIComponent(username)}/games/archives`,
      { signal: controller.signal, cache: "no-store" }
    );
    if (archivesResp.status === 404) {
      throw new Error(`User "${username}" not found on chess.com`);
    }
    if (!archivesResp.ok) {
      throw new Error(`HTTP ${archivesResp.status} fetching archives`);
    }
    const { archives = [] } = await archivesResp.json();

    if (!archives.length) {
      setStatus(`No public games for "${username}".`);
      setLoading(false);
      activeLoad = null;
      return;
    }

    const all = [];
    for (let i = 0; i < archives.length; i++) {
      setStatus(`Loading archive ${i + 1} / ${archives.length}...`);
      const r = await fetch(archives[i], {
        signal: controller.signal,
        cache: "no-store",
      });
      if (!r.ok) {
        console.warn("Skipping archive", archives[i], "HTTP", r.status);
        continue;
      }
      const data = await r.json();
      for (const g of data.games || []) all.push(slimGame(g));

      // Render incrementally so the user sees games appear.
      state.games = all.slice().sort((a, b) => (b.end_time || 0) - (a.end_time || 0));
      renderGameList();
    }

    setStatus(`Loaded ${state.games.length} games for ${username}.`);
    if (state.games.length) selectGame(0);
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Cancelled.");
    } else {
      console.error(err);
      setStatus(err.message || "Failed to load games.", true);
    }
  } finally {
    setLoading(false);
    activeLoad = null;
  }
}

/** Optional: load a pre-downloaded games.json (produced by
 *  download_games.py) if it's served alongside index.html. Used as a
 *  silent offline fallback on initial page load. */
async function tryLoadLocalGamesJson() {
  try {
    const resp = await fetch("games.json", { cache: "no-store" });
    if (!resp.ok) return false;
    const data = await resp.json();
    if (!data.games?.length) return false;
    state.username = (data.username || "").toLowerCase() || null;
    state.games = (data.games || [])
      .slice()
      .sort((a, b) => (b.end_time || 0) - (a.end_time || 0));
    if (state.username) els.usernameIn.value = state.username;
    setStatus(`Loaded ${state.games.length} cached games from games.json.`);
    renderGameList();
    if (state.games.length) selectGame(0);
    return true;
  } catch {
    return false;
  }
}

// =====================================================================
//  Free-play mode (no chess.com game loaded)
// =====================================================================
//
// When no game is selected, the user can play moves for both sides by
// clicking a piece and then a destination. Each move is fed through
// the same fork detector that powers the replay annotations, so
// blunders and tactical opportunities get the same ?? / ! markers,
// arrows, and feedback balloon.

const freePlay = {
  selected: -1,         // currently picked-up square, or -1
  legalForSelected: [], // legal {from,to,...} moves for that square
};

function freePlayActive() {
  return state.selectedGameIdx === -1;
}

function currentPosition() {
  return state.positions[state.ply];
}

/** Build a minimal SAN string for display purposes (no disambiguation,
 *  no check/mate suffix — the move-list cell shows whatever we hand it).
 *  The fork annotations still come from the structured `findForks`
 *  output, so this string is purely cosmetic. */
function simpleSan(prevPos, from, to, promotion) {
  const piece = prevPos.board[from];
  if (!piece) return "?";
  const pt = piece.toUpperCase();
  // Castling
  if (pt === "K" && Math.abs(fileOf(to) - fileOf(from)) === 2) {
    return fileOf(to) === 6 ? "O-O" : "O-O-O";
  }
  const captured = prevPos.board[to] ||
    (pt === "P" && fileOf(from) !== fileOf(to)); // en passant or normal capture
  const dest = sqName(to);
  const promo = promotion ? `=${promotion}` : "";
  if (pt === "P") {
    return captured ? `${FILES[fileOf(from)]}x${dest}${promo}` : `${dest}${promo}`;
  }
  return `${pt}${captured ? "x" : ""}${dest}`;
}

/** Build a SAN-shaped `move` object suitable for `applyMove`, given
 *  the from/to/promotion the user clicked. Returns null if the piece
 *  at `from` doesn't belong to the side-to-move. */
function buildMoveObject(pos, from, to, promotion) {
  const piece = pos.board[from];
  if (!piece) return null;
  const white = pos.sideToMove === "w";
  if (isWhite(piece) !== white) return null;
  const pt = piece.toUpperCase();
  // Castling — king moving two files.
  if (pt === "K" && Math.abs(fileOf(to) - fileOf(from)) === 2) {
    return {
      castle: fileOf(to) === 6 ? "K" : "Q",
      piece: "K", fromFile: -1, fromRank: -1, to: -1,
      capture: false, promotion: null, check: false, mate: false,
      san: fileOf(to) === 6 ? "O-O" : "O-O-O",
    };
  }
  const capture = !!pos.board[to] ||
    (pt === "P" && fileOf(from) !== fileOf(to) && to === pos.epTarget);
  return {
    castle: null,
    piece: pt,
    fromFile: fileOf(from),
    fromRank: rankOf(from),
    to,
    capture,
    promotion: promotion || null,
    check: false, mate: false,
    san: simpleSan(pos, from, to, promotion),
  };
}

/** Apply a user-initiated move. Truncates any forward history, pushes
 *  the new position and its annotation onto the parallel arrays, and
 *  jumps to the new ply so the existing rendering machinery picks it
 *  up. Returns true on success. */
function makeUserMove(from, to, promotion) {
  const prev = currentPosition();
  const moveObj = buildMoveObject(prev, from, to, promotion);
  if (!moveObj) return false;

  let next;
  try {
    next = applyMove(prev, moveObj);
  } catch (err) {
    console.warn("Illegal move", err.message);
    return false;
  }

  // Annotate: was this move a winning fork that was already available?
  const prevForks = findForks(prev);
  const executed = prevForks.find(
    (f) => f.move.from === from && f.move.to === to
  );

  // Did the move hand the opponent a fresh fork?
  const newForks = findForks(next);
  const prevKeys = new Set(prevForks.map((f) => `${f.move.from}-${f.move.to}`));
  const fresh = newForks.filter(
    (f) => !prevKeys.has(`${f.move.from}-${f.move.to}`)
  );

  // Mate annotations: did we deliver mate, or hand the opponent one?
  const delivered = isCheckmate(next)
    ? { from, to }
    : null;
  const allowed = delivered ? null : findMateInOne(next);
  // Forward-looking: does the new position contain an opponent
  // mate-in-one threat that the next side-to-move must address?
  const threats = delivered ? null : findMateThreats(next);

  // Truncate any forward history before appending.
  state.positions      = state.positions.slice(0, state.ply + 1);
  state.sanList        = state.sanList.slice(0, state.ply);
  state.blunders       = state.blunders.slice(0, state.ply + 1);
  state.goodMoves      = state.goodMoves.slice(0, state.ply + 1);
  state.matesAllowed   = state.matesAllowed.slice(0, state.ply + 1);
  state.matesDelivered = state.matesDelivered.slice(0, state.ply + 1);
  state.mateThreats    = state.mateThreats.slice(0, state.ply + 1);
  state.evals          = state.evals.slice(0, state.ply + 1);
  state.suggestions    = state.suggestions.slice(0, state.ply + 1);

  state.positions.push(next);
  state.sanList.push(moveObj.san + (delivered ? "#" : ""));
  state.blunders.push(fresh.length ? fresh : null);
  state.goodMoves.push(executed || null);
  state.matesAllowed.push(allowed && allowed.length ? allowed : null);
  state.matesDelivered.push(delivered);
  state.mateThreats.push(threats && threats.length ? threats : null);

  // Evaluation + best-move suggestion for the move just played.
  // Compare the played material delta to bestMove() in the previous
  // position; flag a suggestion when the gap is at least 2 pawns.
  const prevPos = state.positions[state.positions.length - 2];
  const prevEval = state.evals[state.evals.length - 1];
  const newEval = evaluatePosition(next);
  state.evals.push(newEval);
  let suggestion = null;
  if (!delivered) {
    const playerIsWhite = prevPos.sideToMove === "w";
    const playedScore = (newEval - prevEval) * (playerIsWhite ? 1 : -1);
    const best = bestMove(prevPos);
    if (best && best.score - playedScore >= 2) {
      suggestion = {
        best: describeMoveCoord(prevPos, best.move),
        score: best.score,
        lost: best.score - playedScore,
      };
    }
  }
  state.suggestions.push(suggestion);

  freePlay.selected = -1;
  freePlay.legalForSelected = [];
  renderMoveList();
  setPly(state.positions.length - 1);
  return true;
}

/** Highlight the currently-selected square plus all of its legal
 *  destinations. Called from `setPly` so the indicators clear whenever
 *  the user navigates away. */
function renderSelection() {
  for (const cell of els.board.children) {
    cell.classList.remove("sel", "target", "capture");
  }
  if (!freePlayActive() || freePlay.selected < 0) return;
  const selCell = els.board.querySelector(`[data-idx="${freePlay.selected}"]`);
  selCell?.classList.add("sel");
  const pos = currentPosition();
  for (const mv of freePlay.legalForSelected) {
    const cell = els.board.querySelector(`[data-idx="${mv.to}"]`);
    if (!cell) continue;
    cell.classList.add("target");
    const captured = !!pos.board[mv.to] || mv.isEp;
    if (captured) cell.classList.add("capture");
  }
}

function onBoardClick(ev) {
  if (!freePlayActive()) return;
  const cell = ev.target.closest(".sq");
  if (!cell || !els.board.contains(cell)) return;
  const idx = parseInt(cell.dataset.idx, 10);
  const pos = currentPosition();

  // Second click: try to commit a move.
  if (freePlay.selected >= 0) {
    const mv = freePlay.legalForSelected.find((m) => m.to === idx);
    if (mv) {
      // Auto-queen on promotion. (Could be replaced by a chooser.)
      makeUserMove(mv.from, mv.to, mv.promotion ? "Q" : null);
      return;
    }
    // Clicking the selected square deselects.
    if (idx === freePlay.selected) {
      freePlay.selected = -1;
      freePlay.legalForSelected = [];
      renderSelection();
      return;
    }
    // Otherwise fall through to "pick up another piece" below.
  }

  // First click (or re-selection): pick up own piece.
  const piece = pos.board[idx];
  if (!piece) {
    freePlay.selected = -1;
    freePlay.legalForSelected = [];
    renderSelection();
    return;
  }
  if (isWhite(piece) !== (pos.sideToMove === "w")) return;

  freePlay.selected = idx;
  freePlay.legalForSelected = legalMoves(pos)
    .filter((m) => m.from === idx)
    // Collapse promotion variants to a single target square — we
    // auto-queen on commit.
    .filter((m, i, arr) => !m.promotion ||
      arr.findIndex((n) => n.to === m.to) === i);
  renderSelection();
}

/** Wipe state back to a fresh starting position; used by the Reset
 *  button to either restart a free-play game or to leave a loaded
 *  chess.com game and return to free play. */
function resetFreePlay() {
  state.selectedGameIdx = -1;
  state.positions = [initialPosition()];
  state.sanList = [];
  state.blunders = [null];
  state.goodMoves = [null];
  state.matesAllowed = [null];
  state.matesDelivered = [null];
  state.mateThreats = [null];
  state.evals = [evaluatePosition(state.positions[0])];
  state.suggestions = [null];
  freePlay.selected = -1;
  freePlay.legalForSelected = [];
  els.gameTitle.textContent = "Free play — click pieces to move";
  renderGameList();        // drop the .active highlight in the sidebar
  renderMoveList();
  setPly(0);
  updateFreePlayChrome();
  // Re-open the games sidebar so the user can pick another game.
  setGameListCollapsed(false);
}

/** Show or hide the bits of UI that only make sense in free-play mode
 *  (the "interactive" cursor on board squares). The Reset button is
 *  always visible — its tooltip changes based on whether it will
 *  restart a free-play game or exit a loaded chess.com game. */
function updateFreePlayChrome() {
  const active = freePlayActive();
  els.board.classList.toggle("interactive", active);
  if (els.btnReset) {
    els.btnReset.title = active
      ? "Reset to starting position"
      : "Back to free play";
  }
}

// ---- Wire-up ---------------------------------------------------------
function init() {
  buildBoardSquares();
  // Bootstrap the parallel annotation arrays for free-play mode so the
  // existing render machinery (which indexes blunders[ply], etc.) works
  // out of the box before the user has made any moves.
  state.blunders = [null];
  state.goodMoves = [null];
  state.matesAllowed = [null];
  state.matesDelivered = [null];
  state.mateThreats = [null];
  state.evals = [evaluatePosition(state.positions[0])];
  state.suggestions = [null];
  setPly(0);
  updateFreePlayChrome();

  els.btnStart.addEventListener("click", () => setPly(0));
  els.btnPrev .addEventListener("click", () => setPly(state.ply - 1));
  els.btnNext .addEventListener("click", () => setPly(state.ply + 1));
  els.btnEnd  .addEventListener("click", () => setPly(state.positions.length - 1));
  els.btnFlip ?.addEventListener("click", () => {
    setOrientation(state.orientation === "w" ? "b" : "w");
    setPly(state.ply); // re-render board contents into the rebuilt cells
  });
  els.btnReset?.addEventListener("click", resetFreePlay);
  els.board.addEventListener("click", onBoardClick);
  els.filterInput.addEventListener("input", renderGameList);
  els.btnToggleGames?.addEventListener("click", () => {
    const collapsed = els.gameListPane?.classList.contains("collapsed");
    setGameListCollapsed(!collapsed);
  });

  els.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const username = els.usernameIn.value.trim().toLowerCase();
    if (!username) return;
    loadFromChessCom(username);
  });
  els.cancelBtn.addEventListener("click", () => {
    if (activeLoad) activeLoad.controller.abort();
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.target.tagName === "INPUT") return;
    if      (ev.key === "ArrowLeft")  { setPly(state.ply - 1); ev.preventDefault(); }
    else if (ev.key === "ArrowRight") { setPly(state.ply + 1); ev.preventDefault(); }
    else if (ev.key === "Home")       { setPly(0); ev.preventDefault(); }
    else if (ev.key === "End")        { setPly(state.positions.length - 1); ev.preventDefault(); }
  });

  // Pre-fill from URL: either /<name> in the path or ?user=<name>
  // in the query string auto-loads on page open.
  const params = new URLSearchParams(window.location.search);
  const pathUser = decodeURIComponent(window.location.pathname.replace(/^\/+|\/+$/g, ""))
    .trim().toLowerCase();
  const queryUser = (params.get("user") || "").trim().toLowerCase();
  // Ignore path segments that look like file assets (contain a dot)
  // so requests for app.js, style.css, favicon.ico, etc. don't get
  // mistaken for usernames.
  const urlUser = (pathUser && !pathUser.includes(".") ? pathUser : "")
    || queryUser;
  if (urlUser) {
    els.usernameIn.value = urlUser;
    loadFromChessCom(urlUser);
    return;
  }

  // Otherwise quietly try a pre-downloaded games.json; if missing, just
  // sit at the empty state and wait for the user to enter a name.
  tryLoadLocalGamesJson().then((loaded) => {
    if (!loaded) setStatus("Enter a chess.com username and press Load.");
  });
}

document.addEventListener("DOMContentLoaded", init);
