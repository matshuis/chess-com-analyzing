"""Tests for hanging-piece detection in :mod:`chess_engine`.

A piece is "hanging" for the side-to-move when the opponent can
capture it and end up materially ahead — either because the piece is
attacked and undefended, or because the cheapest attacker is worth
less than the piece itself.
"""
from chess_engine import (
    Position,
    find_attackers,
    find_hanging_pieces,
    parse_sq,
    sq_name,
)


def _squares_of_hanging(pos):
    return sorted(sq_name(h["square"]) for h in find_hanging_pieces(pos))


# ---------------------------------------------------------------------------
#  find_attackers
# ---------------------------------------------------------------------------

def test_find_attackers_in_starting_position_is_empty():
    """No piece attacks any opposing piece in the starting array."""
    pos = Position.from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    # The bishops, queens and kings are all hidden behind pawns; the
    # knights attack only their own colour's squares.
    for sq_label in ("e4", "d4", "e5", "d5"):
        s = parse_sq(sq_label)
        assert find_attackers(pos.board, s, by_white=True) == []
        assert find_attackers(pos.board, s, by_white=False) == []


def test_find_attackers_pawn_attacks_diagonally_only():
    """A pawn attacks its diagonal squares, not the push square."""
    # White pawn on e4 alone on the board (plus kings to keep it legal).
    pos = Position.from_fen("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    # e4 pawn attacks d5 and f5 — not e5.
    assert find_attackers(pos.board, parse_sq("d5"), by_white=True) == [parse_sq("e4")]
    assert find_attackers(pos.board, parse_sq("f5"), by_white=True) == [parse_sq("e4")]
    assert find_attackers(pos.board, parse_sq("e5"), by_white=True) == []


# ---------------------------------------------------------------------------
#  find_hanging_pieces
# ---------------------------------------------------------------------------

def test_no_hanging_pieces_in_starting_position():
    pos = Position.from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    assert find_hanging_pieces(pos) == []


def test_undefended_attacked_bishop_is_hanging():
    """Black bishop on c5 attacked by white queen on d4, no defender."""
    # White Kc1, Qd4; Black Kc8, Bc5. It's Black to move — the bishop
    # is en prise and undefended.
    pos = Position.from_fen("2k5/8/8/2b5/3Q4/8/8/2K5 b - - 0 1")
    hanging = find_hanging_pieces(pos)
    assert len(hanging) == 1
    h = hanging[0]
    assert sq_name(h["square"]) == "c5"
    assert h["piece"] == "b"
    assert sq_name(h["attackers"][0]) == "d4"
    assert h["defenders"] == []


def test_defended_piece_attacked_by_equal_value_is_not_hanging():
    """Bishop defended by a knight, attacked by another knight — fair trade, not hanging."""
    # White Kg1, Bc4 (defended by Na3); Black Kg8, Nb6 (attacks c4).
    # 8: . . . . . . k .  -> "6k1"
    # 7: . . . . . . . .  -> "8"
    # 6: . n . . . . . .  -> "1n6"
    # 5: . . . . . . . .  -> "8"
    # 4: . . B . . . . .  -> "2B5"
    # 3: N . . . . . . .  -> "N7"
    # 2: . . . . . . . .  -> "8"
    # 1: . . . . . . K .  -> "6K1"
    pos = Position.from_fen("6k1/8/1n6/8/2B5/N7/8/6K1 w - - 0 1")
    # White to move. The bishop is attacked by Nb6 (3) and defended by
    # Na3 (3); equal value with a defender ⇒ not hanging. Nothing else
    # is en prise.
    assert find_hanging_pieces(pos) == []


def test_defended_queen_attacked_by_pawn_is_hanging():
    """A queen defended by a rook is still hanging if a pawn attacks it."""
    # White Kg1, Pe4; Black Kg8, Qd5 defended by Rd8.
    # 8: . . . r . . k .  -> "3r2k1"
    # 7: . . . . . . . .  -> "8"
    # 6: . . . . . . . .  -> "8"
    # 5: . . . q . . . .  -> "3q4"
    # 4: . . . . P . . .  -> "4P3"
    # 3: . . . . . . . .  -> "8"
    # 2: . . . . . . . .  -> "8"
    # 1: . . . . . . K .  -> "6K1"
    pos = Position.from_fen("3r2k1/8/8/3q4/4P3/8/8/6K1 b - - 0 1")
    hanging = find_hanging_pieces(pos)
    sqs = [sq_name(h["square"]) for h in hanging]
    assert "d5" in sqs, f"queen should be flagged hanging; got {sqs!r}"
    h = next(h for h in hanging if sq_name(h["square"]) == "d5")
    # The defender is recorded but doesn't save it: pawn (1) < queen (9).
    assert sq_name(h["defenders"][0]) == "d8"
    assert sq_name(h["attackers"][0]) == "e4"


def test_king_is_never_reported_as_hanging():
    """King attacks are 'check', not 'hanging'."""
    # White Ke1, Qe2; Black king on e8 attacked along the e-file.
    pos = Position.from_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
    for h in find_hanging_pieces(pos):
        assert h["piece"].upper() != "K"


def test_only_side_to_move_pieces_are_reported():
    """We never warn about *opponent* pieces — only about the player's own."""
    # White Kg1, Bc4 attacked by Black knight on d6 (undefended).
    # 8: . . . . . . k .  -> "6k1"
    # 7: . . . . . . . .  -> "8"
    # 6: . . . n . . . .  -> "3n4"
    # 5: . . . . . . . .  -> "8"
    # 4: . . B . . . . .  -> "2B5"
    # 3: . . . . . . . .  -> "8"
    # 2: . . . . . . . .  -> "8"
    # 1: . . . . . . K .  -> "6K1"
    fen = "6k1/8/3n4/8/2B5/8/8/6K1"
    # When it's WHITE's turn the bishop is in danger.
    pos_w = Position.from_fen(fen + " w - - 0 1")
    sqs_w = _squares_of_hanging(pos_w)
    assert "c4" in sqs_w
    # When it's BLACK's turn we should NOT report the white bishop —
    # only black's own hanging pieces (none here).
    pos_b = Position.from_fen(fen + " b - - 0 1")
    sqs_b = _squares_of_hanging(pos_b)
    assert "c4" not in sqs_b


def test_pawn_attacked_by_pawn_with_defender_is_not_hanging():
    """Equal-value piece + at least one defender = safe."""
    # Classic Italian-ish pawn skirmish: white Pd4, black Pe5, both
    # supported by a knight.  Neither is hanging.
    # 8: . . . . . . k .  -> "6k1"
    # 7: . . . . . . . .  -> "8"
    # 6: . . . . . n . .  -> "5n2"
    # 5: . . . . p . . .  -> "4p3"
    # 4: . . . P . . . .  -> "3P4"
    # 3: . . . . . N . .  -> "5N2"
    # 2: . . . . . . . .  -> "8"
    # 1: . . . . . . K .  -> "6K1"
    pos = Position.from_fen("6k1/8/5n2/4p3/3P4/5N2/8/6K1 w - - 0 1")
    assert find_hanging_pieces(pos) == []


def test_famous_qxf2_blunder_position_shows_no_hanging_pieces_for_black():
    """After 3...Bc5 (Black just developed), Black has no hanging pieces.

    This is the position from the mate-in-one tests: the danger is a
    mate threat, not a hanging piece — they shouldn't both fire.
    """
    pos = Position.from_fen(
        "rnb1k1nr/pppp1ppp/5q2/2b1p3/2P1P3/3P4/PP3PPP/RNBQKBNR w KQkq - 2 4"
    )
    # Note: it's white to move here. Black's bishop on c5 is defended
    # by ...Bxf2+ ideas via the queen, but more importantly nothing
    # of white's is en prise either.
    assert find_hanging_pieces(pos) == []


def test_opponent_e5_pawn_after_2_nf3_bc5_is_hanging():
    """After 1.e4 e5 2.Nf3 Bc5 it's White to move and Black's e5 pawn
    is hanging — Nf3 attacks it and nothing defends (the bishop sat
    on c5 instead of developing the knight to c6). The UI should warn
    White that they can grab a free pawn.
    """
    # Position after 2...Bc5:
    #   row 8: r n b q k . n r  -> "rnbqk1nr"
    #   row 7: p p p p . p p p  -> "pppp1ppp"
    #   row 6: . . . . . . . .  -> "8"
    #   row 5: . . b . p . . .  -> "2b1p3"
    #   row 4: . . . . P . . .  -> "4P3"
    #   row 3: . . . . . N . .  -> "5N2"
    #   row 2: P P P P . P P P  -> "PPPP1PPP"
    #   row 1: R N B Q K B . R  -> "RNBQKB1R"
    pos = Position.from_fen(
        "rnbqk1nr/pppp1ppp/8/2b1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )
    # White (side-to-move) has nothing hanging.
    assert find_hanging_pieces(pos, color="w") == []
    # Black's e5 pawn is en prise to Nf3 with no defender.
    black_loose = find_hanging_pieces(pos, color="b")
    sqs = sorted(sq_name(h["square"]) for h in black_loose)
    assert "e5" in sqs, (
        f"Expected the undefended e5 pawn to be flagged for Black; got {sqs!r}"
    )
    e5 = next(h for h in black_loose if sq_name(h["square"]) == "e5")
    assert e5["piece"] == "p"
    assert sq_name(e5["attackers"][0]) == "f3"
    assert e5["defenders"] == []


def test_queen_defends_e5_pawn_so_it_is_not_hanging():
    """Counter-example matching a common misconception: after
    1.e4 e5 2.Nf3 Bc5 3.Bc4 Qf6, the queen on f6 defends the e5 pawn
    diagonally, so it is NOT hanging — Nxe5? would lose the knight."""
    # Position after 3...Qf6:
    #   row 8: r n b . k . n r  -> "rnb1k1nr"
    #   row 7: p p p p . p p p  -> "pppp1ppp"
    #   row 6: . . . . . q . .  -> "5q2"
    #   row 5: . . b . p . . .  -> "2b1p3"
    #   row 4: . . B . P . . .  -> "2B1P3"
    #   row 3: . . . . . N . .  -> "5N2"
    #   row 2: P P P P . P P P  -> "PPPP1PPP"
    #   row 1: R N B Q K . . R  -> "RNBQK2R"
    pos = Position.from_fen(
        "rnb1k1nr/pppp1ppp/5q2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    # Nothing of White's is en prise.
    assert find_hanging_pieces(pos, color="w") == []
    # And crucially, e5 is NOT flagged for Black: Qf6 defends it.
    black_loose = find_hanging_pieces(pos, color="b")
    sqs = [sq_name(h["square"]) for h in black_loose]
    assert "e5" not in sqs, (
        f"e5 should NOT be hanging when Qf6 defends it; got {sqs!r}"
    )


def test_find_hanging_pieces_color_parameter_defaults_to_side_to_move():
    """Calling without `color` must match the original behaviour."""
    pos = Position.from_fen("2k5/8/8/2b5/3Q4/8/8/2K5 b - - 0 1")
    default = find_hanging_pieces(pos)
    explicit = find_hanging_pieces(pos, color="b")
    assert default == explicit
