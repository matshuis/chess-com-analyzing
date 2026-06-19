"""Tests for the opening-book lookup in :mod:`openings`."""
from openings import OPENINGS, Opening, identify_opening, suggest_responses


# ---------------------------------------------------------------------------
#  identify_opening
# ---------------------------------------------------------------------------

def test_single_move_e4_is_kings_pawn():
    op = identify_opening(["e4"])
    assert op is not None
    assert op.name == "King's Pawn Opening"
    assert op.moves == ("e4",)


def test_e4_e5_is_open_game():
    op = identify_opening(["e4", "e5"])
    assert op is not None
    assert op.name == "Open Game"


def test_longest_prefix_wins_ruy_lopez():
    """e4 e5 Nf3 Nc6 Bb5 must resolve to the Ruy López, not a shallower entry."""
    op = identify_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"])
    assert op is not None
    assert "Ruy López" in op.name


def test_italian_game_identified():
    op = identify_opening(["e4", "e5", "Nf3", "Nc6", "Bc4"])
    assert op is not None
    assert op.name == "Italian Game"


def test_sicilian_after_one_pair():
    op = identify_opening(["e4", "c5"])
    assert op is not None
    assert op.name == "Sicilian Defence"


def test_french_and_caro_kann():
    assert identify_opening(["e4", "e6"]).name == "French Defence"
    assert identify_opening(["e4", "c6"]).name == "Caro-Kann Defence"


def test_queens_gambit_recognised():
    op = identify_opening(["d4", "d5", "c4"])
    assert op is not None
    assert op.name == "Queen's Gambit"


def test_kings_indian_setup_after_g6():
    op = identify_opening(["d4", "Nf6", "c4", "g6"])
    assert op is not None
    assert "King's Indian" in op.name


def test_english_and_reti():
    assert identify_opening(["c4"]).name == "English Opening"
    assert identify_opening(["Nf3"]).name == "Réti Opening"


def test_deeper_line_still_matches_known_prefix():
    """After ...a6 (still book in the Ruy), we still report Ruy López."""
    op = identify_opening(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    assert op is not None
    assert "Ruy López" in op.name
    assert len(op.moves) == 5  # matched the 5-move prefix, not the 6-move game


def test_unknown_first_move_returns_none():
    assert identify_opening(["Nh3"]) is None
    assert identify_opening(["a4"])  is None


def test_empty_move_list_returns_none():
    assert identify_opening([]) is None


# ---------------------------------------------------------------------------
#  suggest_responses
# ---------------------------------------------------------------------------

def test_responses_for_e4_include_main_replies():
    sug = suggest_responses(["e4"])
    assert sug is not None
    sans = [r.san for r in sug["replies"]]
    # The four classical replies to 1.e4 must all be mentioned.
    for expected in ("e5", "c5", "e6", "c6"):
        assert expected in sans, f"missing reply {expected!r} from {sans!r}"


def test_responses_for_open_game_suggest_knight_developing_moves():
    sug = suggest_responses(["e4", "e5"])
    assert sug is not None
    sans = [r.san for r in sug["replies"]]
    assert "Nf3" in sans
    assert "Nc3" in sans


def test_responses_for_ruy_lopez_include_morphy_defence():
    sug = suggest_responses(["e4", "e5", "Nf3", "Nc6", "Bb5"])
    assert sug is not None
    sans = [r.san for r in sug["replies"]]
    assert "a6" in sans, f"Morphy Defence (...a6) missing from {sans!r}"


def test_responses_for_queens_gambit_include_accept_and_decline():
    sug = suggest_responses(["d4", "d5", "c4"])
    assert sug is not None
    sans = [r.san for r in sug["replies"]]
    assert "e6"   in sans   # QGD
    assert "dxc4" in sans   # QGA
    assert "c6"   in sans   # Slav


def test_no_suggestions_beyond_known_theory():
    """Once we're past the last known node we deliberately go silent."""
    # ...a6 is matched as part of the Ruy López prefix (5 moves), so
    # passing a 6-move history must return None — we're "beyond book"
    # from the lookup's point of view.
    sug = suggest_responses(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    assert sug is None


def test_no_suggestions_for_unknown_opening():
    assert suggest_responses(["Nh3"]) is None


# ---------------------------------------------------------------------------
#  Data-integrity checks — cheap insurance against typos in the table.
# ---------------------------------------------------------------------------

def test_all_entries_have_unique_move_lists():
    seen = set()
    for op in OPENINGS:
        assert op.moves not in seen, f"duplicate entry for {op.moves!r}"
        seen.add(op.moves)


def test_all_entries_have_at_least_one_reply():
    for op in OPENINGS:
        assert isinstance(op, Opening)
        assert op.replies, f"{op.name!r} has no replies listed"


def test_replies_are_unique_within_each_entry():
    for op in OPENINGS:
        sans = [r.san for r in op.replies]
        assert len(sans) == len(set(sans)), (
            f"{op.name!r} has duplicate reply SANs: {sans!r}"
        )


def test_replies_never_use_check_or_mate_suffixes():
    """The lookup is prefix-string-based; SANs must be stripped of +/#."""
    for op in OPENINGS:
        for r in op.replies:
            assert not r.san.endswith(("+", "#")), (
                f"{op.name!r}: reply {r.san!r} carries a check/mate suffix"
            )
        for m in op.moves:
            assert not m.endswith(("+", "#")), (
                f"{op.name!r}: move {m!r} carries a check/mate suffix"
            )
