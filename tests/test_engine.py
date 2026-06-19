"""Sanity checks for the small chess engine in chess_engine.py."""
from chess_engine import (
    Move, Position,
    legal_moves, make_move, parse_sq, sq_name,
)


def test_initial_position_has_20_legal_moves():
    pos = Position.initial()
    assert len(legal_moves(pos)) == 20


def test_e2e4_updates_board_and_side_to_move():
    pos = Position.initial()
    move = Move(parse_sq("e2"), parse_sq("e4"))
    nxt = make_move(pos, move)
    assert nxt.board[parse_sq("e4")] == "P"
    assert nxt.board[parse_sq("e2")] == ""
    assert nxt.stm == "b"


def test_legal_moves_filter_out_self_check():
    # White king on e1, white rook on e2, black queen on e8 pins the rook
    # to its king. The rook may move along the e-file but never off it.
    fen = "4q3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    pos = Position.from_fen(fen)
    rook_moves = [m for m in legal_moves(pos)
                  if sq_name(m.frm) == "e2"]
    assert rook_moves, "rook should still have e-file moves while pinned"
    for m in rook_moves:
        assert sq_name(m.to)[0] == "e", \
            f"pinned rook tried to leave the e-file: {m.uci()}"
