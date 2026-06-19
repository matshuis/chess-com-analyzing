"""Minimal stdlib-only chess engine used by the analyzer's pytest suite.

The goal is *not* to be a full engine — we only need enough to:
  * parse a FEN,
  * generate legal moves,
  * apply a move,
  * detect tactical motifs (currently: forks).

It mirrors the JavaScript engine in app.js so positions can be reasoned
about identically on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
#  Constants & helpers
# ---------------------------------------------------------------------------

FILES = "abcdefgh"

# Approximate centipawn-style values used by the fork detector.  The king
# gets a high value so "attacks the king" always counts as critical.
PIECE_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100}

KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2),
                 (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
ROOK_DIRS   = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KING_DELTAS = ROOK_DIRS + BISHOP_DIRS


def sq(file: int, rank: int) -> int:
    return rank * 8 + file


def file_of(s: int) -> int:
    return s & 7


def rank_of(s: int) -> int:
    return s >> 3


def sq_name(s: int) -> str:
    return FILES[file_of(s)] + str(rank_of(s) + 1)


def parse_sq(name: str) -> int:
    return sq(FILES.index(name[0]), int(name[1]) - 1)


# ---------------------------------------------------------------------------
#  Move / Position dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Move:
    frm: int
    to: int
    promotion: Optional[str] = None  # "Q", "R", "B", or "N" (always upper)
    is_castle: Optional[str] = None  # "K" or "Q"
    is_ep: bool = False

    def uci(self) -> str:
        s = sq_name(self.frm) + sq_name(self.to)
        if self.promotion:
            s += self.promotion.lower()
        return s


@dataclass
class Position:
    board: list           # 64 single-char piece codes; "" = empty
    stm: str = "w"        # side to move: "w" or "b"
    castling: dict = field(default_factory=lambda: {"K": True, "Q": True,
                                                    "k": True, "q": True})
    ep: int = -1          # en-passant target square (-1 = none)

    @classmethod
    def initial(cls) -> "Position":
        return cls.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        parts = fen.split()
        board = [""] * 64
        for ri, row in enumerate(parts[0].split("/")):
            r = 7 - ri
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    board[sq(f, r)] = ch
                    f += 1
        stm = parts[1] if len(parts) > 1 else "w"
        rights = parts[2] if len(parts) > 2 else "-"
        castling = {k: (k in rights) for k in "KQkq"}
        ep = -1 if (len(parts) <= 3 or parts[3] == "-") else parse_sq(parts[3])
        return cls(board, stm, castling, ep)

    def clone(self) -> "Position":
        return Position(self.board[:], self.stm, dict(self.castling), self.ep)


# ---------------------------------------------------------------------------
#  Attack detection
# ---------------------------------------------------------------------------

def is_attacked_by(board: list, target: int, by_white: bool) -> bool:
    tf, tr = file_of(target), rank_of(target)

    # Pawns
    pawn_dir = -1 if by_white else 1
    pawn = "P" if by_white else "p"
    for df in (-1, 1):
        f, r = tf + df, tr + pawn_dir
        if 0 <= f < 8 and 0 <= r < 8 and board[sq(f, r)] == pawn:
            return True

    # Knights
    knight = "N" if by_white else "n"
    for df, dr in KNIGHT_DELTAS:
        f, r = tf + df, tr + dr
        if 0 <= f < 8 and 0 <= r < 8 and board[sq(f, r)] == knight:
            return True

    # King (adjacency)
    king = "K" if by_white else "k"
    for df, dr in KING_DELTAS:
        f, r = tf + df, tr + dr
        if 0 <= f < 8 and 0 <= r < 8 and board[sq(f, r)] == king:
            return True

    # Sliding pieces
    rook_like   = ("R", "Q") if by_white else ("r", "q")
    bishop_like = ("B", "Q") if by_white else ("b", "q")
    for df, dr in ROOK_DIRS:
        f, r = tf + df, tr + dr
        while 0 <= f < 8 and 0 <= r < 8:
            p = board[sq(f, r)]
            if p:
                if p in rook_like:
                    return True
                break
            f += df
            r += dr
    for df, dr in BISHOP_DIRS:
        f, r = tf + df, tr + dr
        while 0 <= f < 8 and 0 <= r < 8:
            p = board[sq(f, r)]
            if p:
                if p in bishop_like:
                    return True
                break
            f += df
            r += dr
    return False


def find_king(board: list, white: bool) -> int:
    target = "K" if white else "k"
    for i, p in enumerate(board):
        if p == target:
            return i
    return -1


def is_in_check(pos: Position, color: Optional[str] = None) -> bool:
    if color is None:
        color = pos.stm
    white = color == "w"
    ksq = find_king(pos.board, white)
    if ksq < 0:
        return False
    return is_attacked_by(pos.board, ksq, not white)


# ---------------------------------------------------------------------------
#  Move generation
# ---------------------------------------------------------------------------

def _pseudo_legal(pos: Position) -> list:
    moves: list = []
    white = pos.stm == "w"
    for frm in range(64):
        p = pos.board[frm]
        if not p or (p.isupper() != white):
            continue
        pt = p.upper()
        f, r = file_of(frm), rank_of(frm)

        if pt == "P":
            direction = 1 if white else -1
            start_rank = 1 if white else 6
            promo_rank = 7 if white else 0
            r1 = r + direction
            if 0 <= r1 < 8 and pos.board[sq(f, r1)] == "":
                if r1 == promo_rank:
                    for promo in "QRBN":
                        moves.append(Move(frm, sq(f, r1), promotion=promo))
                else:
                    moves.append(Move(frm, sq(f, r1)))
                if r == start_rank and pos.board[sq(f, r + 2 * direction)] == "":
                    moves.append(Move(frm, sq(f, r + 2 * direction)))
            for df in (-1, 1):
                tf = f + df
                tr = r + direction
                if not (0 <= tf < 8 and 0 <= tr < 8):
                    continue
                target = sq(tf, tr)
                tp = pos.board[target]
                if tp and (tp.isupper() != white):
                    if tr == promo_rank:
                        for promo in "QRBN":
                            moves.append(Move(frm, target, promotion=promo))
                    else:
                        moves.append(Move(frm, target))
                elif target == pos.ep:
                    moves.append(Move(frm, target, is_ep=True))

        elif pt == "N":
            for df, dr in KNIGHT_DELTAS:
                tf, tr = f + df, r + dr
                if 0 <= tf < 8 and 0 <= tr < 8:
                    target = sq(tf, tr)
                    tp = pos.board[target]
                    if not tp or (tp.isupper() != white):
                        moves.append(Move(frm, target))

        elif pt == "K":
            for df, dr in KING_DELTAS:
                tf, tr = f + df, r + dr
                if 0 <= tf < 8 and 0 <= tr < 8:
                    target = sq(tf, tr)
                    tp = pos.board[target]
                    if not tp or (tp.isupper() != white):
                        moves.append(Move(frm, target))
            # Castling
            if white and r == 0 and f == 4:
                if (pos.castling["K"]
                        and pos.board[sq(5, 0)] == ""
                        and pos.board[sq(6, 0)] == ""
                        and pos.board[sq(7, 0)] == "R"
                        and not is_attacked_by(pos.board, sq(4, 0), False)
                        and not is_attacked_by(pos.board, sq(5, 0), False)
                        and not is_attacked_by(pos.board, sq(6, 0), False)):
                    moves.append(Move(frm, sq(6, 0), is_castle="K"))
                if (pos.castling["Q"]
                        and pos.board[sq(3, 0)] == ""
                        and pos.board[sq(2, 0)] == ""
                        and pos.board[sq(1, 0)] == ""
                        and pos.board[sq(0, 0)] == "R"
                        and not is_attacked_by(pos.board, sq(4, 0), False)
                        and not is_attacked_by(pos.board, sq(3, 0), False)
                        and not is_attacked_by(pos.board, sq(2, 0), False)):
                    moves.append(Move(frm, sq(2, 0), is_castle="Q"))
            elif (not white) and r == 7 and f == 4:
                if (pos.castling["k"]
                        and pos.board[sq(5, 7)] == ""
                        and pos.board[sq(6, 7)] == ""
                        and pos.board[sq(7, 7)] == "r"
                        and not is_attacked_by(pos.board, sq(4, 7), True)
                        and not is_attacked_by(pos.board, sq(5, 7), True)
                        and not is_attacked_by(pos.board, sq(6, 7), True)):
                    moves.append(Move(frm, sq(6, 7), is_castle="K"))
                if (pos.castling["q"]
                        and pos.board[sq(3, 7)] == ""
                        and pos.board[sq(2, 7)] == ""
                        and pos.board[sq(1, 7)] == ""
                        and pos.board[sq(0, 7)] == "r"
                        and not is_attacked_by(pos.board, sq(4, 7), True)
                        and not is_attacked_by(pos.board, sq(3, 7), True)
                        and not is_attacked_by(pos.board, sq(2, 7), True)):
                    moves.append(Move(frm, sq(2, 7), is_castle="Q"))

        else:  # B / R / Q
            dirs = (BISHOP_DIRS if pt == "B"
                    else ROOK_DIRS if pt == "R"
                    else KING_DELTAS)
            for df, dr in dirs:
                tf, tr = f + df, r + dr
                while 0 <= tf < 8 and 0 <= tr < 8:
                    target = sq(tf, tr)
                    tp = pos.board[target]
                    if tp:
                        if tp.isupper() != white:
                            moves.append(Move(frm, target))
                        break
                    moves.append(Move(frm, target))
                    tf += df
                    tr += dr
    return moves


def make_move(pos: Position, move: Move) -> Position:
    new = pos.clone()
    p = new.board[move.frm]
    new.board[move.frm] = ""

    if move.is_castle:
        rank = 0 if pos.stm == "w" else 7
        if move.is_castle == "K":
            new.board[sq(6, rank)] = p
            new.board[sq(5, rank)] = new.board[sq(7, rank)]
            new.board[sq(7, rank)] = ""
        else:
            new.board[sq(2, rank)] = p
            new.board[sq(3, rank)] = new.board[sq(0, rank)]
            new.board[sq(0, rank)] = ""
    elif move.is_ep:
        new.board[move.to] = p
        new.board[sq(file_of(move.to), rank_of(move.frm))] = ""
    else:
        if move.promotion:
            new.board[move.to] = (move.promotion if pos.stm == "w"
                                  else move.promotion.lower())
        else:
            new.board[move.to] = p

    # Castling rights updates
    if p == "K":
        new.castling["K"] = False
        new.castling["Q"] = False
    elif p == "k":
        new.castling["k"] = False
        new.castling["q"] = False
    for s, key in ((sq(0, 0), "Q"), (sq(7, 0), "K"),
                   (sq(0, 7), "q"), (sq(7, 7), "k")):
        if move.frm == s or move.to == s:
            new.castling[key] = False

    # En-passant target
    if p.upper() == "P" and abs(rank_of(move.to) - rank_of(move.frm)) == 2:
        new.ep = sq(file_of(move.frm),
                    (rank_of(move.frm) + rank_of(move.to)) // 2)
    else:
        new.ep = -1

    new.stm = "b" if pos.stm == "w" else "w"
    return new


def legal_moves(pos: Position) -> list:
    out = []
    for m in _pseudo_legal(pos):
        nxt = make_move(pos, m)
        white = pos.stm == "w"
        ksq = find_king(nxt.board, white)
        if ksq >= 0 and not is_attacked_by(nxt.board, ksq, not white):
            out.append(m)
    return out


# ---------------------------------------------------------------------------
#  Tactical helpers
# ---------------------------------------------------------------------------

def attacks_from(board: list, frm: int) -> list:
    """Squares attacked / reachable by the piece on `frm`.

    For sliding pieces the first blocker square *is* included (so we
    can spot captures), but the ray stops there.
    """
    p = board[frm]
    if not p:
        return []
    pt = p.upper()
    white = p.isupper()
    f, r = file_of(frm), rank_of(frm)
    out = []
    if pt == "P":
        direction = 1 if white else -1
        for df in (-1, 1):
            tf, tr = f + df, r + direction
            if 0 <= tf < 8 and 0 <= tr < 8:
                out.append(sq(tf, tr))
    elif pt == "N":
        for df, dr in KNIGHT_DELTAS:
            tf, tr = f + df, r + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                out.append(sq(tf, tr))
    elif pt == "K":
        for df, dr in KING_DELTAS:
            tf, tr = f + df, r + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                out.append(sq(tf, tr))
    else:
        dirs = (BISHOP_DIRS if pt == "B"
                else ROOK_DIRS if pt == "R"
                else KING_DELTAS)
        for df, dr in dirs:
            tf, tr = f + df, r + dr
            while 0 <= tf < 8 and 0 <= tr < 8:
                out.append(sq(tf, tr))
                if board[sq(tf, tr)]:
                    break
                tf += df
                tr += dr
    return out


@dataclass(frozen=True)
class Fork:
    move: Move
    attacker_square: int
    targets: tuple  # tuple of (square, piece) for every attacked enemy piece

    @property
    def target_squares(self) -> list:
        return [s for s, _ in self.targets]


def find_forks(pos: Position) -> list:
    """Return every legal move for `pos.stm` that creates a *winning* fork.

    Definition used here:
      1. After the move, the moved piece must attack ≥ 2 enemy pieces.
      2. At least one of those targets must be the king or strictly
         more valuable than the moving piece.
      3. The attacker must land on a square the opponent does *not*
         attack (otherwise it can simply be captured back, and
         the "fork" is just an even trade).

    Rule (3) is what distinguishes a real tactical blunder from a
    knight check that the opponent already has covered.
    """
    forks = []
    attacker_is_white = pos.stm == "w"
    for m in legal_moves(pos):
        nxt = make_move(pos, m)
        attacker_sq = m.to
        attacker = nxt.board[attacker_sq]
        if not attacker:
            continue
        # (3) — landing square must not be attacked by the opponent.
        if is_attacked_by(nxt.board, attacker_sq, by_white=not attacker_is_white):
            continue
        atk_value = PIECE_VALUE[attacker.upper()]
        targets = []
        for s in attacks_from(nxt.board, attacker_sq):
            tp = nxt.board[s]
            if not tp or (tp.isupper() == attacker.isupper()):
                continue
            targets.append((s, tp))
        if len(targets) < 2:
            continue
        critical = any(
            tp.upper() == "K" or PIECE_VALUE[tp.upper()] > atk_value
            for _, tp in targets
        )
        if critical:
            forks.append(Fork(m, attacker_sq, tuple(targets)))
    return forks


def describe_forks(forks) -> list:
    """Cheap human-readable rendering for assertion messages."""
    return [
        {
            "move": f.move.uci(),
            "from": sq_name(f.move.frm),
            "to":   sq_name(f.move.to),
            "targets": [sq_name(s) for s in f.target_squares],
        }
        for f in forks
    ]


def is_blunder(pos: Position, move: Move) -> list:
    """Return the *winning* forks the opponent gains by playing `move`.

    Empty list ⇒ the move is not a fork-blunder. A non-empty list
    means the move hands the opponent a fork that wins material — see
    :func:`find_forks` for the precise definition of "winning".

    This is the production helper backing the "?? blunder" badges in
    the web UI. Tests use it to assert that suspicious moves are
    flagged correctly.
    """
    before = {(f.move.frm, f.move.to) for f in find_forks(_swap_side(pos))}
    after = make_move(pos, move)
    return [
        f for f in find_forks(after)
        if (f.move.frm, f.move.to) not in before
    ]


def _swap_side(pos: Position) -> Position:
    """Helper: a copy of `pos` with the side-to-move flipped.

    Used by :func:`is_blunder` to ignore forks the opponent could
    already play *before* our move (those aren't caused by us).
    """
    swapped = pos.clone()
    swapped.stm = "b" if pos.stm == "w" else "w"
    return swapped
