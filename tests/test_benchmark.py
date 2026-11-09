"""Versioned tactical checks and search diagnostics."""
import chess
import pytest

from az_benchmark import TACTICS, tactical_scorecard, tactic_board
from az_mcts import MCTS
from az_network import ChessNet


@pytest.mark.parametrize("case", TACTICS, ids=[case[0] for case in TACTICS])
def test_tactical_positions_and_expected_moves_are_legal(case):
    _, category, fen, history, expected = case
    board = tactic_board(fen, history)
    assert board.is_valid()
    assert expected
    for uci in expected:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        if category in ("capture", "promotion"):
            assert board.is_capture(move)
        if category == "promotion":
            assert move.promotion == chess.QUEEN
        if category == "mate":
            board.push(move)
            assert board.is_checkmate()
            board.pop()
    if category == "defense":
        safe = set()
        for move in list(board.legal_moves):
            board.push(move)
            mate_available = False
            for reply in list(board.legal_moves):
                board.push(reply)
                mate_available |= board.is_checkmate()
                board.pop()
            board.pop()
            if not mate_available:
                safe.add(move.uci())
        assert set(expected) == safe
        assert len(safe) < board.legal_moves.count()


def test_scorecard_reports_visits_and_values_from_root_players_perspective(monkeypatch):
    def predict(self, board):
        moves = list(board.legal_moves)
        return {move: 1 / len(moves) for move in moves}, 0.75

    monkeypatch.setattr(MCTS, "_predict", predict)
    report = tactical_scorecard(ChessNet(16, 1), 1, "cpu")
    for case in report["cases"][:3]:
        assert case["mate_safeguard"]
        assert case["search_children"] == []
        assert case["search_root_value"] is None
    for case in report["cases"][3:]:
        assert not case["mate_safeguard"]
        assert sum(child["visits"] for child in case["search_children"]) == 1
        visited = [child for child in case["search_children"] if child["visits"]]
        assert len(visited) == 1
        assert visited[0]["value_for_player"] == -0.75
        assert all(
            child["value_for_player"] is None
            for child in case["search_children"] if not child["visits"]
        )


def test_search_clears_previous_root_when_using_mate_safeguard():
    search = MCTS(ChessNet(16, 1))
    search.search(chess.Board(), 1)
    assert search.root is not None
    search.search(tactic_board(TACTICS[0][2], TACTICS[0][3]), 1)
    assert search.root is None
