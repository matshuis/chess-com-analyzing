"""Fork-related pytest cases.

* :func:`test_white_can_fork_with_knight` — confirms the fork detector
  finds the knight fork Nc4–d6+ that hits Black's king and queen.

* :func:`test_bd7_is_flagged_as_blunder` — exercises the production
  :func:`chess_engine.is_blunder` helper. White plays ``Bd7??``,
  abandoning the bishop's defense of c2 (the bishop leaves the
  h7–b1 diagonal) and lets Black play ``Nb4–c2+`` forking K+R.
  The helper must report this fork.

* :func:`test_quiet_move_is_not_a_blunder` — sanity check that a
  defensive move which doesn't allow any new fork is *not* flagged.
"""
from chess_engine import (
    Move, Position,
    describe_forks, find_forks, is_blunder, make_move,
    parse_sq, sq_name,
)


# ---------------------------------------------------------------------------
#  Confirm a real fork is detected
# ---------------------------------------------------------------------------

def test_white_can_fork_with_knight():
    """White to move, knight on c4 forks Black king (e8) + queen (f5)."""
    fen = "4k3/8/8/5q2/2N5/8/8/4K3 w - - 0 1"
    pos = Position.from_fen(fen)

    forks = find_forks(pos)
    assert forks, (
        "Expected at least one fork in this position, but "
        "find_forks() returned nothing."
    )

    nd6 = next(
        (f for f in forks
         if sq_name(f.move.frm) == "c4" and sq_name(f.move.to) == "d6"),
        None,
    )
    assert nd6 is not None, (
        f"Expected the Nc4–d6 fork. Forks found: {describe_forks(forks)}"
    )

    target_squares = sorted(sq_name(s) for s in nd6.target_squares)
    assert "e8" in target_squares, (
        f"Expected fork to attack the black king on e8; got {target_squares}"
    )
    assert "f5" in target_squares, (
        f"Expected fork to attack the black queen on f5; got {target_squares}"
    )


# ---------------------------------------------------------------------------
#  Production helper: is_blunder()
# ---------------------------------------------------------------------------
#
# Position: White Ke1, Ra1, Bf5; Black Kh8, Nb4. White to move.
# The bishop on f5 sits on the h7–b1 diagonal, defending c2 and
# therefore keeping Nc2+ from being a winning fork (after Nc2+, the
# bishop simply takes back). Moving the bishop *off* that diagonal —
# e.g. Bd7 — abandons the c2 defense and lets Black fork K+R.
# Black's king is on h8 (not e8) so Bd7 doesn't accidentally also
# give check, which would prevent Black from playing the fork.
BLUNDER_FEN = "7k/8/8/5B2/1n6/8/8/R3K3 w - - 0 1"


def test_bd7_is_flagged_as_blunder():
    """The blunder detector must spot Black's resulting Nc2+ fork."""
    pos = Position.from_fen(BLUNDER_FEN)
    bd7 = Move(parse_sq("f5"), parse_sq("d7"))

    forks = is_blunder(pos, bd7)
    assert forks, "Expected Bd7 to be flagged — Black has Nc2+ forking K+R."

    fork = forks[0]
    assert sq_name(fork.move.frm) == "b4"
    assert sq_name(fork.move.to)  == "c2"
    targets = sorted(sq_name(s) for s in fork.target_squares)
    assert targets == ["a1", "e1"], (
        f"Expected the fork to hit a1 and e1; got {targets}"
    )


def test_quiet_move_is_not_a_blunder():
    """A bishop move that keeps the c2 defense should *not* be flagged."""
    pos = Position.from_fen(BLUNDER_FEN)
    # Bf5–e4 stays on the h7–b1 diagonal, so c2 remains defended.
    safe = Move(parse_sq("f5"), parse_sq("e4"))
    forks = is_blunder(pos, safe)
    assert forks == [], (
        f"Expected no blunder on Be4, got {describe_forks(forks)}"
    )


# ---------------------------------------------------------------------------
#  Sanity: a real game move applied to the same position works too
# ---------------------------------------------------------------------------

def test_make_move_after_blunder_actually_loses_material():
    """Playing the fork after Bd7?? wins the rook on a1."""
    pos = Position.from_fen(BLUNDER_FEN)
    after_blunder = make_move(pos, Move(parse_sq("f5"), parse_sq("d7")))

    nc2 = Move(parse_sq("b4"), parse_sq("c2"))
    after_fork = make_move(after_blunder, nc2)

    assert after_fork.board[parse_sq("c2")] == "n", "knight should be on c2"
    # The rook on a1 must be under attack by Black after Nc2+.
    from chess_engine import is_attacked_by
    assert is_attacked_by(after_fork.board, parse_sq("a1"), by_white=False), (
        "After Nc2+, the rook on a1 must be under attack by Black."
    )
