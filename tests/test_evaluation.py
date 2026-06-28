"""Tests for the material evaluator in chess_engine.py."""
from chess_engine import (
    Move, Position,
    best_move, evaluate, make_move, parse_sq, score_move, sq_name,
)


def test_evaluate_initial_position_is_balanced():
    assert evaluate(Position.initial()) == 0


def test_evaluate_signs_match_perspective():
    # Just the two kings + a white queen — White is up nine pawns.
    pos = Position.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate(pos) == 9
    # Same idea, Black's queen — score flips sign.
    pos = Position.from_fen("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert evaluate(pos) == -9


def test_evaluate_ignores_kings():
    # No pieces but the two kings — material is balanced regardless
    # of which side has the move.
    pos = Position.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert evaluate(pos) == 0


def test_score_move_rewards_capturing_a_hanging_queen():
    # Black queen sits undefended on d4; White rook on d1 can take it.
    pos = Position.from_fen("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1")
    assert score_move(pos, Move(parse_sq("d1"), parse_sq("d4"))) == 9


def test_score_move_perspective_flips_for_black():
    # Mirror of the previous test: White's queen hangs on d5 and Black's
    # rook on d8 takes it.  Score must be positive from Black's POV.
    pos = Position.from_fen("3rk3/8/8/3Q4/8/8/8/4K3 b - - 0 1")
    assert score_move(pos, Move(parse_sq("d8"), parse_sq("d5"))) == 9


def test_score_move_zero_for_a_quiet_move():
    # Pushing a pawn changes nothing material.
    pos = Position.initial()
    assert score_move(pos, Move(parse_sq("e2"), parse_sq("e4"))) == 0


def test_best_move_picks_the_hanging_queen():
    pos = Position.from_fen("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1")
    result = best_move(pos)
    assert result is not None
    mv, score = result
    assert sq_name(mv.frm) == "d1"
    assert sq_name(mv.to) == "d4"
    assert score == 9


def test_best_move_returns_none_when_no_legal_moves():
    # Fool's mate — White to move and mated.
    pos = Position.from_fen(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    assert best_move(pos) is None


def test_best_move_prefers_capture_over_quiet_move():
    # Black bishop on d5 is undefended; White can take with the queen
    # along the d-file (+3) or play any quiet move (0).
    pos = Position.from_fen("4k3/8/8/3b4/8/8/8/3QK1N1 w - - 0 1")
    result = best_move(pos)
    assert result is not None
    mv, score = result
    assert sq_name(mv.frm) == "d1"
    assert sq_name(mv.to) == "d5"
    assert score == 3
