"""Mate-detection pytest cases.

The motivating bug: after ``1.e4 e5 2.c4 Qf6 3.d3 Bc5 4.Nc3?? Qxf2#``
the analyzer never warned White about the threat, even though 4.Nc3 is
*the* losing move. These tests pin down the helpers used to flag that
kind of blunder.

* :func:`test_qxf2_is_checkmate` — reproduces the exact game position
  and confirms our engine recognises 4...Qxf2 as mate.
* :func:`test_nc3_allows_mate_in_one` — :func:`allows_mate_in_one`
  must flag the Knight move that hands Black the mate.
* :func:`test_safe_developing_move_is_not_a_mate_blunder` — in the
  same position, an alternative defense (4.Nh3, protecting f2) must
  *not* be flagged.
"""
from chess_engine import (
    Move, Position,
    allows_mate_in_one,
    find_mate_in_one,
    in_check,
    is_checkmate,
    legal_moves,
    make_move,
    parse_sq,
    sq_name,
)


# Position after 1.e4 e5 2.c4 Qf6 3.d3 Bc5 4.Nc3 — Black to move.
#   row 8: R N B . K . N R  → "rnb1k1nr"
#   row 7: P P P P . P P P  → "pppp1ppp"
#   row 6: . . . . . Q . .  → "5q2"
#   row 5: . . B . P . . .  → "2b1p3"
#   row 4: . . P . P . . .  → "2P1P3"
#   row 3: . . N P . . . .  → "2NP4"
#   row 2: P P . . . P P P  → "PP3PPP"
#   row 1: R . B Q K B N R  → "R1BQKBNR"
POS_BEFORE_QXF2 = (
    "rnb1k1nr/pppp1ppp/5q2/2b1p3/2P1P3/2NP4/PP3PPP/R1BQKBNR b KQkq - 0 4"
)
# Position one ply earlier (after 3...Bc5) — used to test that 4.Nc3
# is correctly flagged as the move that *creates* the mate threat.
#   row 2: P P . . . P P P  → "PP3PPP"  (e-pawn has moved to e4)
#   row 1: R N B Q K B N R  → "RNBQKBNR" (Nb1 still on b1 before 4.Nc3)
POS_BEFORE_NC3 = (
    "rnb1k1nr/pppp1ppp/5q2/2b1p3/2P1P3/3P4/PP3PPP/RNBQKBNR w KQkq - 2 4"
)


def test_qxf2_is_checkmate():
    """4...Qxf2 must be recognised as mate by the engine."""
    pos = Position.from_fen(POS_BEFORE_QXF2)
    qxf2 = Move(parse_sq("f6"), parse_sq("f2"))

    after = make_move(pos, qxf2)
    assert in_check(after), (
        "Expected the white king to be in check after Qxf2."
    )
    assert not legal_moves(after), (
        "Expected White to have no legal replies to Qxf2; "
        f"got {[m.uci() for m in legal_moves(after)]}"
    )
    assert is_checkmate(after), "Qxf2 should be checkmate."


def test_nc3_allows_mate_in_one():
    """4.Nc3 in the Qf6/Bc5 setup must be flagged by allows_mate_in_one."""
    pos = Position.from_fen(POS_BEFORE_NC3)
    nc3 = Move(parse_sq("b1"), parse_sq("c3"))

    mates = allows_mate_in_one(pos, nc3)
    assert mates, (
        "Expected 4.Nc3 to be flagged; allows_mate_in_one() returned no mates."
    )

    qxf2 = next(
        (m for m in mates
         if sq_name(m.frm) == "f6" and sq_name(m.to) == "f2"),
        None,
    )
    assert qxf2 is not None, (
        "Expected Qxf2 to appear in the mating replies; "
        f"got {[m.uci() for m in mates]}"
    )


def test_safe_developing_move_is_not_a_mate_blunder():
    """4.Nh3 defends f2, so it must NOT be flagged as allowing mate."""
    pos = Position.from_fen(POS_BEFORE_NC3)
    nh3 = Move(parse_sq("g1"), parse_sq("h3"))

    mates = allows_mate_in_one(pos, nh3)
    assert mates == [], (
        f"Expected 4.Nh3 to be safe; got mating replies "
        f"{[m.uci() for m in mates]}"
    )


def test_find_mate_in_one_returns_qxf2():
    """From Black's POV in the bug position, Qxf2 should be the only mate."""
    pos = Position.from_fen(POS_BEFORE_QXF2)
    mates = find_mate_in_one(pos)
    assert len(mates) == 1, (
        f"Expected exactly one mate-in-one; got {[m.uci() for m in mates]}"
    )
    assert sq_name(mates[0].frm) == "f6"
    assert sq_name(mates[0].to)  == "f2"
