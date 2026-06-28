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


# ---------------------------------------------------------------------------
#  Pin detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pin:
    """A pin created by `move`.

    `attacker_square` is where the pinning piece lands. `pinned_square`
    holds the enemy piece that can't move without exposing
    `behind_square` (a more valuable enemy piece, or the king if
    `absolute` is True).
    """
    move: Move
    attacker_square: int
    pinned_square: int
    pinned_piece: str
    behind_square: int
    behind_piece: str
    absolute: bool


# Sliding directions per pinning piece type. A queen pins along any
# of the eight rays; bishop/rook only along their own.
_PIN_DIRS = {
    "B": BISHOP_DIRS,
    "R": ROOK_DIRS,
    "Q": BISHOP_DIRS + ROOK_DIRS,
}


def find_pins(pos: Position) -> list:
    """Return every legal move for `pos.stm` that creates a *winning* pin.

    Definition used here:
      1. The moving piece is a bishop, rook, or queen.
      2. After the move, a ray matching the attacker's piece type
         passes through exactly one enemy piece (the "pinned" piece —
         never the king itself, which would be check) and then hits a
         second enemy piece on the same ray with no friendly piece in
         between.
      3. The second piece is either the king (absolute pin) or
         strictly more valuable than the pinned piece (relative pin
         that wins material if the pinned piece moves).
      4. The attacker's landing square is not attacked by the
         opponent — same rule as :func:`find_forks`, so the pinning
         piece can't simply be captured back.
      5. Capturing the pinned piece next move actually wins
         material: the pinned piece must be either *undefended* or
         strictly more valuable than the pinning piece. Without this
         filter the detector flagged dozens of cosmetic pins per
         position (e.g. Qd1–h5 "pinning" f7 to the king, where the
         only follow-up is Qxf7 — losing the queen for a pawn).

    Suggestions are de-duplicated by (from, to): a queen creating
    two pins from one square is reported as a single move with one
    `Pin` entry per pin axis.
    """
    pins = []
    attacker_is_white = pos.stm == "w"
    for m in legal_moves(pos):
        # The moving piece must be a sliding pinner — a promotion to
        # B/R/Q also qualifies, so check the post-move piece.
        nxt = make_move(pos, m)
        attacker_sq = m.to
        attacker = nxt.board[attacker_sq]
        if not attacker or attacker.upper() not in _PIN_DIRS:
            continue
        # (4) — landing square must be safe.
        if is_attacked_by(nxt.board, attacker_sq, by_white=not attacker_is_white):
            continue

        f0, r0 = file_of(attacker_sq), rank_of(attacker_sq)
        for df, dr in _PIN_DIRS[attacker.upper()]:
            f, r = f0 + df, r0 + dr
            first = None  # (square, piece) of the first enemy on the ray
            while 0 <= f < 8 and 0 <= r < 8:
                s = sq(f, r)
                p = nxt.board[s]
                if p:
                    own = (p.isupper() == attacker.isupper())
                    if first is None:
                        if own:
                            break  # blocked by our own piece — no pin here
                        if p.upper() == "K":
                            break  # that's a check, not a pin
                        first = (s, p)
                    else:
                        if own:
                            break  # friendly piece shields the back rank
                        front_sq, front_p = first
                        front_val = PIECE_VALUE[front_p.upper()]
                        back_val  = PIECE_VALUE[p.upper()]
                        absolute  = p.upper() == "K"
                        # (3) — back piece must be the king or more valuable.
                        if not absolute and back_val <= front_val:
                            break
                        # (5) — taking the pinned piece must win material.
                        if not _pin_wins_material(
                                nxt.board, attacker_sq, front_sq,
                                attacker_is_white):
                            break
                        pins.append(Pin(
                            move=m, attacker_square=attacker_sq,
                            pinned_square=front_sq, pinned_piece=front_p,
                            behind_square=s, behind_piece=p,
                            absolute=absolute,
                        ))
                        break  # ray is finished either way
                f += df
                r += dr
    return pins


def _pin_wins_material(board: list, attacker_sq: int, pinned_sq: int,
                       attacker_is_white: bool) -> bool:
    """Quick SEE-lite: would capturing the pinned piece net material?

    Returns True when either:
      * the pinned piece has no defenders (free capture next move),
        or
      * the pinned piece is strictly more valuable than the pinning
        piece (so even after the recapture we come out ahead).

    This intentionally ignores deeper exchange sequences and x-ray
    reveals — a Ruy-López-style ``Bb5`` pin won't be flagged because
    it doesn't *immediately* win material, which is the whole point
    of the filter: only surface pins the player can cash in on.
    """
    attacker_val = PIECE_VALUE[board[attacker_sq].upper()]
    pinned_val   = PIECE_VALUE[board[pinned_sq].upper()]
    defenders = find_attackers(board, pinned_sq,
                               by_white=not attacker_is_white)
    return not defenders or pinned_val > attacker_val


def describe_pins(pins) -> list:
    """Cheap human-readable rendering for assertion messages."""
    return [
        {
            "move": p.move.uci(),
            "from": sq_name(p.move.frm),
            "to":   sq_name(p.move.to),
            "pinned":   sq_name(p.pinned_square),
            "behind":   sq_name(p.behind_square),
            "absolute": p.absolute,
        }
        for p in pins
    ]


# ---------------------------------------------------------------------------
#  Checkmate detection
# ---------------------------------------------------------------------------

def in_check(pos: Position) -> bool:
    """Is the side-to-move's king currently attacked?"""
    white = pos.stm == "w"
    ksq = find_king(pos.board, white)
    if ksq < 0:
        return False
    return is_attacked_by(pos.board, ksq, by_white=not white)


def is_checkmate(pos: Position) -> bool:
    """True if the side-to-move is in check and has no legal moves."""
    return in_check(pos) and not legal_moves(pos)


def find_mate_in_one(pos: Position) -> list:
    """Every legal move for `pos.stm` that delivers immediate checkmate."""
    return [m for m in legal_moves(pos) if is_checkmate(make_move(pos, m))]


def allows_mate_in_one(pos: Position, move: Move) -> list:
    """The opponent's mate-in-one replies that `move` creates.

    Returns the list of mating moves the opponent gains by us playing
    `move`. Empty list ⇒ the move is safe from mate-in-one. Mirrors
    :func:`is_blunder` but for mate threats instead of forks.
    """
    after = make_move(pos, move)
    return find_mate_in_one(after)


def find_mate_threats(pos: Position) -> list:
    """Opponent mate-in-one moves the side-to-move must address.

    "If I (side-to-move) passed, could my opponent immediately
    checkmate me?" Returns the list of such mating moves. Used to
    warn the player *before* they blunder rather than after.

    Returns an empty list when side-to-move is already in check — the
    check itself is the more urgent thing to handle, and a position
    with two kings under attack is illegal anyway.  Also returns an
    empty list when side-to-move has no legal moves at all: that is
    stalemate (checkmate is filtered by the ``in_check`` guard above),
    the game is already over, and there is nothing to warn about.
    """
    if in_check(pos):
        return []
    if not legal_moves(pos):
        return []
    swapped = _swap_side(pos)
    # A null move forfeits any en-passant right.
    swapped.ep = -1
    return find_mate_in_one(swapped)


# ---------------------------------------------------------------------------
#  Hanging-piece detection
# ---------------------------------------------------------------------------

def find_attackers(board: list, target: int, by_white: bool) -> list:
    """Squares of pieces of the requested colour that attack `target`.

    A piece "attacks" a square iff it could capture an enemy unit
    there in a single move — for pawns this is the diagonal capture
    square (not the push square). Mirrors :func:`is_attacked_by` but
    returns the source squares rather than a boolean.
    """
    out = []
    for s, p in enumerate(board):
        if not p:
            continue
        if p.isupper() != by_white:
            continue
        if target in attacks_from(board, s):
            out.append(s)
    return out


def find_hanging_pieces(pos: Position, color: Optional[str] = None) -> list:
    """Return hanging pieces of `color` (defaults to the side-to-move).

    A piece is "hanging" when at least one opponent piece attacks it
    and either:

    * it has no defenders, or
    * the opponent's cheapest attacker is worth less than the piece
      itself (so taking and being recaptured still wins material).

    Each entry is a dict ``{"square", "piece", "attackers", "defenders"}``
    where ``attackers`` and ``defenders`` are lists of source squares.

    The king is excluded — a king under attack is "check", not
    "hanging", and is handled elsewhere.

    Pass ``color="w"`` / ``"b"`` to inspect a specific side; this is
    used to warn the player both about their own loose pieces and
    about *opponent* pieces they could grab for free.
    """
    if color is None:
        color = pos.stm
    white = color == "w"
    out = []
    for s, p in enumerate(pos.board):
        if not p:
            continue
        if p.isupper() != white:
            continue
        if p.upper() == "K":
            continue
        attackers = find_attackers(pos.board, s, by_white=not white)
        if not attackers:
            continue
        defenders = find_attackers(pos.board, s, by_white=white)
        piece_val = PIECE_VALUE[p.upper()]
        cheapest_atk = min(PIECE_VALUE[pos.board[a].upper()] for a in attackers)
        if not defenders or cheapest_atk < piece_val:
            out.append({
                "square": s,
                "piece": p,
                "attackers": attackers,
                "defenders": defenders,
            })
    return out


# ---------------------------------------------------------------------------
#  Material evaluation
# ---------------------------------------------------------------------------
#
# The simplest possible position evaluator: sum the value of every
# non-king piece on the board, signed by color.  This is enough to spot
# blunders that lose material outright (hanging a queen, bad trades) and
# to suggest the move that wins the most material in the current
# position.  See ``best_move`` below.
#
# Kings are excluded because both sides always have exactly one — they
# contribute equally to every position and so cancel out.

# Values in pawn units.  Identical to ``PIECE_VALUE`` except the king
# is dropped; kept as a separate table to make that intent explicit.
_EVAL_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}


def evaluate(pos: Position) -> int:
    """Material balance of ``pos`` from White's perspective, in pawn units.

    Positive means White is ahead, negative means Black is ahead.
    """
    score = 0
    for p in pos.board:
        if not p:
            continue
        v = _EVAL_VALUES.get(p.upper())
        if v is None:
            continue
        score += v if p.isupper() else -v
    return score


def score_move(pos: Position, move: Move) -> int:
    """Material gained by ``move`` from the perspective of the side to move.

    Positive = good for the mover, negative = loses material.  This is
    simply the change in :func:`evaluate` produced by playing the move,
    flipped for Black so "higher is better for me" always holds.
    """
    delta = evaluate(make_move(pos, move)) - evaluate(pos)
    return delta if pos.stm == "w" else -delta


def best_move(pos: Position):
    """Return ``(move, score)`` for the highest-scoring legal move.

    Ties are broken by the order ``legal_moves`` produces.  Returns
    ``None`` when the side to move has no legal moves (checkmate or
    stalemate).
    """
    best = None
    best_score = None
    for mv in legal_moves(pos):
        s = score_move(pos, mv)
        if best_score is None or s > best_score:
            best, best_score = mv, s
    if best is None:
        return None
    return best, best_score

