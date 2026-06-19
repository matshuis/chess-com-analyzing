"""Very small opening-book lookup.

Each entry is a *SAN prefix* (a list of moves in standard algebraic
notation) plus the human name of the opening reached after those
moves, a one-line strategic plan, and a handful of typical replies
for the side whose turn it now is.

The table is intentionally tiny — it covers the openings a club
player meets every day, not an exhaustive ECO database. Adding new
lines is just appending more entries.

The lookup is purely string-based (it never touches the engine), so
mirroring it in JavaScript is a copy of the same array.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Optional


class Reply(NamedTuple):
    san: str       # the move, e.g. "e5"
    note: str      # one-line description / name of the resulting line


class Opening(NamedTuple):
    moves: tuple   # SAN tokens that lead to this opening, e.g. ("e4", "e5")
    name: str      # human name, e.g. "Open Game"
    plan: str      # one-line strategic idea for the side to move
    replies: tuple # tuple[Reply, ...] — typical replies for the side to move


def _op(moves, name, plan, replies):
    return Opening(
        tuple(moves),
        name,
        plan,
        tuple(Reply(s, n) for s, n in replies),
    )


# Ordered from shallow to deep; lookup picks the *longest* prefix match
# so the order here is for readability only.
OPENINGS: tuple = (
    # ---- 1.e4 ---------------------------------------------------------
    _op(
        ["e4"],
        "King's Pawn Opening",
        "White stakes out the centre and frees the king's bishop and queen.",
        [
            ("e5", "Open Game — classical, symmetric centre."),
            ("c5", "Sicilian Defence — fight for the centre asymmetrically."),
            ("e6", "French Defence — solid, slightly cramped."),
            ("c6", "Caro-Kann Defence — solid pawn structure."),
        ],
    ),
    _op(
        ["e4", "e5"],
        "Open Game",
        "Both sides will develop knights and contest the d4/d5 squares.",
        [
            ("Nf3", "King's Knight Opening — attacks e5 and prepares O-O."),
            ("Nc3", "Vienna Game — flexible, often followed by f4."),
            ("Bc4", "Bishop's Opening — aims at f7."),
        ],
    ),
    _op(
        ["e4", "e5", "Nf3"],
        "King's Knight Opening",
        "Black must defend e5 immediately.",
        [
            ("Nc6", "Standard development; supports e5."),
            ("Nf6", "Petroff Defence — symmetric counter-attack on e4."),
            ("d6", "Philidor Defence — solid but passive."),
        ],
    ),
    _op(
        ["e4", "e5", "Nf3", "Nc6"],
        "King's Knight Opening (main line)",
        "White picks how to pressure the centre and Black's knight.",
        [
            ("Bb5", "Ruy López — pins the c6 knight and pressures e5."),
            ("Bc4", "Italian Game — eyes f7, prepares quick castling."),
            ("d4",  "Scotch Game — opens the centre immediately."),
        ],
    ),
    _op(
        ["e4", "e5", "Nf3", "Nc6", "Bb5"],
        "Ruy López (Spanish Opening)",
        "Black usually challenges the bishop or supports the centre.",
        [
            ("a6",  "Morphy Defence — the main line; asks the bishop a question."),
            ("Nf6", "Berlin Defence — solid, leads to the famous endgame."),
            ("d6",  "Steinitz Defence — old, very solid."),
        ],
    ),
    _op(
        ["e4", "e5", "Nf3", "Nc6", "Bc4"],
        "Italian Game",
        "Black mirrors development and contests the centre.",
        [
            ("Bc5", "Giuoco Piano — quiet, symmetric setup."),
            ("Nf6", "Two Knights Defence — sharper, invites complications."),
            ("Be7", "Hungarian Defence — solid and modest."),
        ],
    ),
    _op(
        ["e4", "c5"],
        "Sicilian Defence",
        "White typically prepares Nf3 and d4 to open the centre.",
        [
            ("Nf3", "Open Sicilian setup — most principled."),
            ("Nc3", "Closed Sicilian — keeps the centre intact."),
            ("c3",  "Alapin Variation — prepares d4 with a strong centre."),
        ],
    ),
    _op(
        ["e4", "e6"],
        "French Defence",
        "Black will challenge the centre with ...d5 next move.",
        [
            ("d4",  "Main line — claims the full centre."),
            ("Nc3", "Flexible — keeps options open against ...d5."),
            ("d3",  "King's Indian Attack setup."),
        ],
    ),
    _op(
        ["e4", "c6"],
        "Caro-Kann Defence",
        "Black plans ...d5 with a sound pawn structure.",
        [
            ("d4",  "Main line — accepts the central challenge."),
            ("Nc3", "Two Knights setup — flexible."),
            ("Nf3", "Two Knights Attack."),
        ],
    ),

    # ---- 1.d4 ---------------------------------------------------------
    _op(
        ["d4"],
        "Queen's Pawn Opening",
        "White stakes out d4 and prepares c4 to fight for the centre.",
        [
            ("d5",  "Closed Game / Queen's Gambit territory."),
            ("Nf6", "Indian Defence — flexible, avoids early ...d5."),
            ("f5",  "Dutch Defence — fights for e4."),
        ],
    ),
    _op(
        ["d4", "d5"],
        "Closed Game",
        "White's strongest try is to challenge d5 with c4.",
        [
            ("c4",  "Queen's Gambit — the principled break."),
            ("Nf3", "Quiet developing move; avoids the gambit."),
            ("Bf4", "London System setup."),
        ],
    ),
    _op(
        ["d4", "d5", "c4"],
        "Queen's Gambit",
        "Black chooses between accepting and declining the pawn.",
        [
            ("e6",  "Queen's Gambit Declined — solid and classical."),
            ("c6",  "Slav Defence — solid, keeps the c8 bishop free."),
            ("dxc4", "Queen's Gambit Accepted — gives up the centre for development."),
        ],
    ),
    _op(
        ["d4", "Nf6"],
        "Indian Defence",
        "White typically plays c4 to claim the centre.",
        [
            ("c4",  "Main line — heads for King's/Queen's Indian territory."),
            ("Nf3", "Avoids the sharpest lines."),
            ("Bg5", "Trompowsky Attack — early pin."),
        ],
    ),
    _op(
        ["d4", "Nf6", "c4", "g6"],
        "King's Indian / Grünfeld setup",
        "Black fianchettoes the king's bishop; White picks a centre.",
        [
            ("Nc3", "Main line — invites the King's Indian or Grünfeld."),
            ("Nf3", "Flexible; can transpose to Fianchetto systems."),
        ],
    ),
    _op(
        ["d4", "Nf6", "c4", "e6"],
        "Indian Defence (with ...e6)",
        "Black is heading for the Nimzo-Indian or Queen's Indian.",
        [
            ("Nc3", "Allows the Nimzo-Indian after ...Bb4."),
            ("Nf3", "Avoids the Nimzo; invites the Queen's Indian."),
        ],
    ),

    # ---- Flank openings ---------------------------------------------
    _op(
        ["c4"],
        "English Opening",
        "White fights for d5 from the side before committing centre pawns.",
        [
            ("e5",  "Reversed Sicilian — Black takes the initiative."),
            ("Nf6", "Flexible; can transpose to many Indian setups."),
            ("c5",  "Symmetric English — solid and balanced."),
        ],
    ),
    _op(
        ["Nf3"],
        "Réti Opening",
        "White keeps the centre flexible and may fianchetto.",
        [
            ("d5",  "Classical reply — claims the centre."),
            ("Nf6", "Symmetric, very flexible."),
        ],
    ),
)


def identify_opening(san_moves: Iterable[str]) -> Optional[Opening]:
    """Return the deepest opening whose move list is a prefix of `san_moves`.

    `san_moves` should be the SAN tokens of the game so far, *without*
    check (`+`) or mate (`#`) suffixes — strip them before calling.
    Returns `None` when no entry matches.
    """
    moves = tuple(san_moves)
    best: Optional[Opening] = None
    for op in OPENINGS:
        n = len(op.moves)
        if n > len(moves):
            continue
        if moves[:n] == op.moves:
            if best is None or n > len(best.moves):
                best = op
    return best


def suggest_responses(san_moves: Iterable[str]) -> Optional[dict]:
    """Lookup the opening *at* this position and return its suggestions.

    Returns a dict with keys ``name``, ``plan``, ``replies`` (a list of
    ``Reply`` tuples) when the move list matches an opening node
    exactly, otherwise ``None``. We deliberately don't return
    suggestions for positions *beyond* the last known node — those
    would be stale and misleading.
    """
    moves = tuple(san_moves)
    op = identify_opening(moves)
    if op is None or len(op.moves) != len(moves):
        return None
    return {"name": op.name, "plan": op.plan, "replies": list(op.replies)}
