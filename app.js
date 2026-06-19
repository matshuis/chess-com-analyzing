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
  //   * blunders[i]  — non-null if the move that produced positions[i]
  //                    allowed the opponent a fresh fork (a fork that
  //                    wasn't already on the board before the move).
  //   * goodMoves[i] — non-null if the move that produced positions[i]
  //                    *is itself* a winning fork the side-to-move
  //                    had available in positions[i-1].
  const blunders  = new Array(positions.length).fill(null);
  const goodMoves = new Array(positions.length).fill(null);
  let prevForks = findForks(positions[0]);
  for (let i = 1; i < positions.length; i++) {
    // Was the move just played one of the forks that were available?
    const played = positions[i].lastMove;
    if (played) {
      const executed = prevForks.find(
        (f) => f.move.from === played.from && f.move.to === played.to
      );
      if (executed) goodMoves[i] = executed;
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
  }

  return { positions, sanList, blunders, goodMoves };
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
        const blunder = state.blunders[positionIdx];
        const good    = state.goodMoves[positionIdx];
        let label = state.sanList[ply];
        // Blunder takes precedence — the long-term cost outweighs the
        // immediate gain — but in practice the two are mutually
        // exclusive (you don't usually hand over a fork while making one).
        if (blunder) {
          label += "??";
          span.classList.add("blunder");
          span.title = "Allows fork: "
            + blunder.map(describeFork).join("; ");
        } else if (good) {
          label += "!";
          span.classList.add("good");
          span.title = "Fork: " + describeFork(good);
        }
        span.textContent = label;
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
  if (idx <= 0) { fb.hidden = true; return; }
  const san = state.sanList[idx - 1];
  const spoken = spokenSan(san);
  const blunder = state.blunders[idx];
  const good    = state.goodMoves[idx];
  if (blunder) {
    fb.hidden = false;
    fb.classList.add("blunder");
    const detail = blunder.map(describeFork).join("; ");
    fb.textContent = `⚠ ${spoken} (??) — allows fork: ${detail}`;
  } else if (good) {
    fb.hidden = false;
    fb.classList.add("good");
    fb.textContent = `★ ${spoken} (!) — fork: ${describeFork(good)}`;
  } else {
    fb.hidden = true;
  }
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

// ---- Fork-arrow overlay ---------------------------------------------
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Draw arrows from the forking piece's square to each of its targets.
 *  Blunders (the opponent now has a fork) draw in the "loss" colour;
 *  the player's own executed fork (good move) draws in the "win" colour. */
function renderForkArrows() {
  const svg = els.boardOverlay;
  if (!svg) return;
  svg.innerHTML = "";

  const idx = state.ply;
  if (idx <= 0) return;
  const blunder = state.blunders[idx];
  const good    = state.goodMoves[idx];
  let sets = [];
  if (blunder)   sets = blunder.map((f) => ({ fork: f, cls: "blunder" }));
  else if (good) sets = [{ fork: good, cls: "good" }];
  if (!sets.length) return;

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

  // For a blunder the attacker isn't actually on the board yet — it's
  // still on `fork.move.from` and will arrive at `fork.move.to`. Drawing
  // the arrow from the future destination is most informative for the
  // *targets* but hides where the threat comes from, so we draw both:
  //   * a dashed move arrow from .from -> .to
  //   * solid threat arrows from .to -> each target
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

  for (const { fork, cls } of sets) {
    const fromC = center(fork.move.from);
    const toC   = center(fork.move.to);

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

    for (const t of fork.targets) {
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

  const { positions, sanList, blunders, goodMoves } = replayPgn(g.pgn || "");
  state.positions = positions;
  state.sanList = sanList;
  state.blunders = blunders;
  state.goodMoves = goodMoves;
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

  // Truncate any forward history before appending.
  state.positions = state.positions.slice(0, state.ply + 1);
  state.sanList   = state.sanList.slice(0, state.ply);
  state.blunders  = state.blunders.slice(0, state.ply + 1);
  state.goodMoves = state.goodMoves.slice(0, state.ply + 1);

  state.positions.push(next);
  state.sanList.push(moveObj.san);
  state.blunders.push(fresh.length ? fresh : null);
  state.goodMoves.push(executed || null);

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
 *  button when in free-play mode. */
function resetFreePlay() {
  state.selectedGameIdx = -1;
  state.positions = [initialPosition()];
  state.sanList = [];
  state.blunders = [null];
  state.goodMoves = [null];
  freePlay.selected = -1;
  freePlay.legalForSelected = [];
  renderMoveList();
  setPly(0);
  updateFreePlayChrome();
}

/** Show or hide the bits of UI that only make sense in free-play mode
 *  (the Reset button + the "interactive" cursor on board squares). */
function updateFreePlayChrome() {
  const active = freePlayActive();
  els.board.classList.toggle("interactive", active);
  if (els.btnReset) els.btnReset.hidden = !active;
}

// ---- Wire-up ---------------------------------------------------------
function init() {
  buildBoardSquares();
  // Bootstrap the parallel annotation arrays for free-play mode so the
  // existing render machinery (which indexes blunders[ply], etc.) works
  // out of the box before the user has made any moves.
  state.blunders = [null];
  state.goodMoves = [null];
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
