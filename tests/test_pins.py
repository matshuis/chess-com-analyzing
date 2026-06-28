"""Pin-detection pytest cases.

These mirror the fork tests in :mod:`tests.test_forks`. The detector
under test is :func:`chess_engine.find_pins`, which returns every
legal move for the side-to-move that creates a *winning* pin —
either absolute (the pinned piece shields the king) or relative
(the pinned piece shields a strictly more valuable enemy unit).
"""
from chess_engine import (
    Position,
    describe_pins, find_pins,
    sq_name,
)


# ---------------------------------------------------------------------------
#  Absolute pin: rook pins knight against king
# ---------------------------------------------------------------------------

def test_rook_creates_absolute_pin():
    """White to move, Rh1–e1 pins Black's knight on e4 against Ke8."""
    # Layout (white king tucked into a1 so the rook can sweep rank 1):
    #   black: Ke8, Ne4
    #   white: Ka1, Rh1
    fen = "4k3/8/8/8/4n3/8/8/K6R w - - 0 1"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    assert pins, (
        "Expected at least one pin in this position, but "
        "find_pins() returned nothing."
    )

    re1 = next(
        (p for p in pins
         if sq_name(p.move.frm) == "h1" and sq_name(p.move.to) == "e1"),
        None,
    )
    assert re1 is not None, (
        f"Expected the Rh1–e1 pin. Pins found: {describe_pins(pins)}"
    )

    assert re1.absolute is True, "rook–knight–king pin must be absolute"
    assert sq_name(re1.pinned_square) == "e4"
    assert sq_name(re1.behind_square) == "e8"
    assert re1.pinned_piece.lower() == "n"
    assert re1.behind_piece.lower() == "k"


# ---------------------------------------------------------------------------
#  Relative pin: bishop pins rook against queen, undefended rook → wins material
# ---------------------------------------------------------------------------

def test_bishop_creates_relative_pin():
    """White to move, Ba2–c4 pins Black's Rd5 against Qg8.

    Layout (chosen so the pin actually wins material — see the
    `_pin_wins_material` filter):
      black: Ka8, Rd5, Qg8
      white: Ka1, Ba2
    Rd5 is only "defended" by Qg8 along the same a2–g8 diagonal, and
    the rook is worth more than our bishop, so even if the queen
    recaptures after Bxd5 we come out ahead (exchange + rook).
    """
    fen = "k5q1/8/8/3r4/8/8/B7/K7 w - - 0 1"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    bc4 = next(
        (p for p in pins
         if sq_name(p.move.frm) == "a2" and sq_name(p.move.to) == "c4"),
        None,
    )
    assert bc4 is not None, (
        f"Expected the Ba2–c4 pin. Pins found: {describe_pins(pins)}"
    )

    assert bc4.absolute is False, "rook-vs-queen pin is relative, not absolute"
    assert sq_name(bc4.pinned_square) == "d5"
    assert sq_name(bc4.behind_square) == "g8"
    assert bc4.pinned_piece.lower() == "r"
    assert bc4.behind_piece.lower() == "q"


# ---------------------------------------------------------------------------
#  Negative case: pin square is attacked, so the "pin" loses material
# ---------------------------------------------------------------------------

def test_pin_is_not_suggested_when_attacker_can_be_captured():
    """Same absolute-pin layout, but a black bishop on a5 defends e1.

    Rh1–e1 would land on a square attacked by Ba5, so Black just
    recaptures: the rook is gone for a bishop. The detector must
    refuse to suggest it (mirrors rule (4) of :func:`find_forks`).
    """
    fen = "4k3/8/8/b7/4n3/8/8/K6R w - - 0 1"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    assert pins == [], (
        "Expected no winning pin (Re1 is attacked by Ba5), "
        f"but find_pins returned: {describe_pins(pins)}"
    )


# ---------------------------------------------------------------------------
#  Negative case: nothing behind the attacked piece is a pin
# ---------------------------------------------------------------------------

def test_attack_without_back_piece_is_not_a_pin():
    """A bishop that merely attacks a knight (with nothing behind it)
    is not a pin — make sure the detector doesn't conflate the two."""
    # Black has only a knight; the king sits off any potential pin
    # axis so there's nothing to shield.
    fen = "8/8/2k5/8/4n3/8/8/B6K w - - 0 1"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    assert pins == [], (
        f"Expected no pin (nothing behind the knight), "
        f"but got: {describe_pins(pins)}"
    )


# ---------------------------------------------------------------------------
#  Material-winning filter: pawn-defended pins must not be suggested
# ---------------------------------------------------------------------------
#
# This is the exact noise the analyzer used to surface after 1.e4 e5 —
# Qd1–h5 "pinning" f7 to the king, Bf1–b5 "pinning" d7 to the king,
# etc. Each of those pinned pieces is a pawn defended by another
# piece, so playing the "pin" actually loses the bishop/queen for a
# pawn. None of them should appear.

def test_open_game_does_not_suggest_pawn_pins_after_e4_e5():
    """In the position after 1.e4 e5 White has no material-winning pin.

    The Qh5/Qf3/Qg4/Bc4/Bb5 moves all set up *geometric* pins but the
    pinned pawn is defended in each case, so capturing it loses
    material. With the SEE filter, find_pins() must return [].
    """
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    assert pins == [], (
        "Expected no winning pin in the open-game start position; "
        f"got noise: {describe_pins(pins)}"
    )


def test_pawn_pin_is_suggested_when_pawn_is_undefended():
    """Same shape as the noisy open-game case, but engineered so the
    pinned pawn is genuinely undefended — the pin *should* be flagged.

    Layout (chosen so the back piece is a rook that doesn't defend
    the pinned pawn — a queen would defend along the same diagonal
    and undo the "undefended" property):
      black: Ka8 (off any relevant axis), Re8, Pf7
      white: Kg1 (off the e-file — otherwise Re8 pins the king and
             leaves no legal queen moves), Qd1
    White plays Qd1–h5: the queen rides the d1–h5 diagonal up to a
    square attacking f7 along the h5–e8 diagonal. f7 is pinned to
    the rook on e8 and has no defenders, so Qxf7 next move wins the
    pawn outright.
    """
    fen = "k3r3/5p2/8/8/8/8/8/3Q2K1 w - - 0 1"
    pos = Position.from_fen(fen)

    pins = find_pins(pos)
    assert pins, (
        "Expected the Qd1–h5 pin to fire (pawn is undefended), "
        f"but find_pins returned nothing."
    )
    qh5 = next(
        (p for p in pins
         if sq_name(p.move.frm) == "d1" and sq_name(p.move.to) == "h5"),
        None,
    )
    assert qh5 is not None, (
        f"Expected a Qd1–h5 pin (attacker on h5, pinned on f7); "
        f"got {describe_pins(pins)}"
    )
    assert sq_name(qh5.pinned_square) == "f7"
    assert sq_name(qh5.behind_square) == "e8"
