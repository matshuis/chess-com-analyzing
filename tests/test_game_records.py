"""Saved PGNs retain legal histories and explain repetition claims."""
import io

import chess
import chess.pgn
import pytest

import az_selfplay
import train
from az_network import ChessNet


def read_record(text):
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None
    assert not game.errors
    return game


def test_self_play_records_claimable_repetition_without_inventing_a_move(monkeypatch):
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]

    def search(self, board, *args, **kwargs):
        return {board.parse_uci(moves[len(board.move_stack)]): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    endings = []
    examples = az_selfplay.self_play_game(
        ChessNet(16, 1), simulations=1, max_plies=20, on_game_end=endings.append,
    )
    game = read_record(endings[0].pgn)
    assert len(examples) == 7
    assert [move.uci() for move in game.mainline_moves()] == moves
    assert game.headers["Result"] == "1/2-1/2"
    assert game.headers["TerminationReason"] == "threefold_repetition"
    assert game.headers["PlyCount"] == "7"
    assert game.headers["FinalFEN"] == game.end().board().fen()
    assert not game.end().board().is_repetition(3)
    assert game.end().board().can_claim_threefold_repetition()
    assert "Ng8 (f6g8)" in game.end().comment
    assert "these moves were not played" in game.end().comment
    assert "previously seen after plies 0" in list(game.mainline())[3].comment


def test_completed_repetition_is_annotated_and_original_board_is_unchanged():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(uci)
    before = board.fen(), list(board.move_stack)
    game = read_record(az_selfplay._game_result(board).pgn)
    assert (board.fen(), board.move_stack) == before
    assert "Position occurrence 3" in game.end().comment
    assert "previously seen after plies 0, 4" in game.end().comment
    assert "these moves were not played" not in game.end().comment


def test_evaluation_record_includes_opening_and_checkmate(monkeypatch):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

    def search(self, board, *args, **kwargs):
        return {board.parse_uci(moves[len(board.move_stack)]): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    endings = []
    model = ChessNet(16, 1)
    az_selfplay.evaluate_match(
        model, model, games=2, simulations=1, max_plies=1,
        openings=(tuple(moves[:3]),), on_game_end=endings.append,
    )
    for ending in endings:
        game = read_record(ending.pgn)
        assert [move.uci() for move in game.mainline_moves()] == moves
        assert game.headers["OpeningMoves"] == " ".join(moves[:3])
        assert game.headers["Result"] == "0-1"
        assert game.headers["TerminationReason"] == "checkmate"
        assert game.end().board().is_checkmate()


def test_move_cap_record_and_atomic_file_replacement(tmp_path):
    board = chess.Board()
    board.push_uci("e2e4")
    ending = az_selfplay._game_result(board)
    path = tmp_path / "games" / "selfplay-001.pgn"
    for iteration in ("1", "2"):
        train._save_game(path, ending, {"Iteration": iteration, "White": "Learner"})
        game = read_record(path.read_text())
        assert game.headers["Iteration"] == iteration
        assert game.headers["White"] == "Learner"
        assert game.headers["Termination"] == "adjudication"
        assert game.headers["TerminationReason"] == "move_cap"
        assert game.headers["Date"] != "????.??.??"
    assert not path.with_suffix(".pgn.tmp").exists()


def test_game_write_failure_is_not_silenced(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked")
    with pytest.raises(OSError):
        train._save_game(blocker / "game.pgn", az_selfplay._game_result(chess.Board()), {})
