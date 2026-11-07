"""Game ending and training log regression tests."""
import sys

import pytest

import az_selfplay
import train
from az_network import ChessNet


@pytest.mark.parametrize("max_plies,expected_result,expected_reason", [
    (3, "1/2-1/2", "move_cap"),
    (4, "0-1", "checkmate"),
])
def test_self_play_reports_real_ending_at_cap(monkeypatch, max_plies,
                                             expected_result, expected_reason):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

    def search(self, board, *args, **kwargs):
        return {board.parse_uci(moves[len(board.move_stack)]): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    endings = []
    examples = az_selfplay.self_play_game(
        ChessNet(16, 1), simulations=1, max_plies=max_plies,
        on_game_end=endings.append,
    )
    assert len(examples) == max_plies
    assert endings[0].result == expected_result
    assert endings[0].reason == expected_reason
    assert endings[0].move_cap == (expected_reason == "move_cap")
    if expected_reason == "checkmate":
        assert [example.outcome for example in examples] == [-1, 1, -1, 1]


def test_evaluation_callback_preserves_candidate_score(monkeypatch):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

    def search(self, board, *args, **kwargs):
        return {board.parse_uci(moves[len(board.move_stack)]): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    endings = []
    model = ChessNet(16, 1)
    score = az_selfplay.evaluate_match(
        model, model, games=2, simulations=1, max_plies=4,
        on_game_end=endings.append,
        openings=((),),
    )
    assert score == 0.5
    assert len(endings) == 2
    assert all(game.result == "0-1" and not game.move_cap for game in endings)


def test_training_prints_draw_and_move_cap_summaries(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--iterations", "1",
        "--games-per-iteration", "1", "--simulations", "1",
        "--eval-games", "2", "--eval-simulations", "1",
        "--max-plies", "2", "--train-steps", "1", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(tmp_path),
    ])
    assert train.main() == 0
    log = capsys.readouterr().out
    assert "result=1/2-1/2; reason=move_cap" in log
    assert "white_wins=0, black_wins=0, draws=1 (move_cap=1)" in log
    assert "evaluation game 1/2: candidate=draw; color=white; plies=6" in log
    assert "evaluation game 2/2: candidate=draw; color=black; plies=6" in log
    assert "opening=e2e4 e7e5 g1f3 b8c6" in log
    assert "wins=0, draws=2, losses=0; move_cap=2; candidate_score=0.500" in log
    assert "promotion_threshold=0.550" in log
    assert "Kept incumbent" in log
    assert (tmp_path / "champion.pt").exists()
