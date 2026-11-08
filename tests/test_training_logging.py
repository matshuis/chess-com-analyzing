"""Game ending and training log regression tests."""
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import chess.pgn

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
    temperatures = []
    original_choose_move = az_selfplay.choose_move

    def search(self, board, *args, **kwargs):
        assert kwargs["add_root_noise"] is False
        return {board.parse_uci(moves[len(board.move_stack)]): 1}

    def choose_move(visits, temperature=0.0, rng=None):
        temperatures.append(temperature)
        return original_choose_move(visits, temperature=temperature, rng=rng)

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    monkeypatch.setattr(az_selfplay, "choose_move", choose_move)
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
    assert temperatures == [0.0] * 8


@pytest.mark.parametrize("temperature_moves", [None, 0, 30, 60])
def test_training_prints_draw_and_move_cap_summaries(
    tmp_path, monkeypatch, capsys, temperature_moves,
):
    original_self_play = train.self_play_game
    sampling_windows = []

    def self_play(*args, **kwargs):
        sampling_windows.append(kwargs["temperature_moves"])
        return original_self_play(*args, **kwargs)

    monkeypatch.setattr(train, "self_play_game", self_play)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--iterations", "1",
        "--games-per-iteration", "1", "--simulations", "1",
        "--eval-games", "2", "--eval-simulations", "1",
        "--max-plies", "2", "--train-steps", "1", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(tmp_path),
    ] + ([] if temperature_moves is None else [
        "--temperature-moves", str(temperature_moves),
    ]))
    assert train.main() == 0
    assert sampling_windows == [60 if temperature_moves is None else temperature_moves]
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
    records_dir = tmp_path / "games" / "iteration-000001"
    assert len(list(records_dir.glob("*.pgn"))) == 3
    for name, plies, white in [
        ("selfplay-001.pgn", 2, "Learner"),
        ("evaluation-001.pgn", 6, "Candidate"),
        ("evaluation-002.pgn", 6, "Champion"),
    ]:
        with (records_dir / name).open() as handle:
            game = chess.pgn.read_game(handle)
        assert game is not None and not game.errors
        assert len(list(game.mainline_moves())) == plies
        assert game.headers["White"] == white
        assert game.headers["Iteration"] == "1"


def test_training_rejects_negative_sampling_window(tmp_path, monkeypatch):
    output_dir = tmp_path / "invalid"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--temperature-moves", "-1", "--output-dir", str(output_dir),
    ])
    with pytest.raises(ValueError, match="temperature-moves must be non-negative"):
        train.main()
    assert not output_dir.exists()


def test_training_logs_mlflow_metrics_and_scorecard(
    tmp_path, monkeypatch, capsys,
):
    calls = {
        "params": None, "metrics": None, "artifact": None,
        "experiment": None, "tracking_uri": None,
    }
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.update(tracking_uri=uri),
        set_experiment=lambda name: calls.update(experiment=name),
        start_run=lambda **kwargs: nullcontext(),
        log_params=lambda params: calls.update(params=params),
        log_metrics=lambda metrics, step: calls.update(metrics=(metrics, step)),
        log_artifact=lambda path, artifact_path: calls.update(
            artifact=(path, artifact_path)
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--iterations", "1",
        "--games-per-iteration", "1", "--simulations", "1",
        "--eval-games", "2", "--eval-simulations", "1",
        "--max-plies", "2", "--train-steps", "1", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(tmp_path),
        "--mlflow",
    ])

    assert train.main() == 0

    assert calls["experiment"] == "chess-com-analyzing"
    assert calls["tracking_uri"] == "sqlite:///mlflow.db"
    assert calls["params"]["learning_rate"] == 0.001
    metrics, step = calls["metrics"]
    assert step == 1
    assert metrics["evaluation/candidate_score"] == 0.5
    assert metrics["evaluation/promoted"] == 0
    scorecard = json.loads(
        (tmp_path / "games" / "iteration-000001" / "scorecard.json").read_text()
    )
    assert metrics["tactics/passed"] == scorecard["tactics"]["passed"]
    assert metrics["tactics/total"] == 5
    assert calls["artifact"] == (
        str(tmp_path / "games" / "iteration-000001" / "scorecard.json"),
        "iterations/iteration-000001",
    )
    assert "MLflow tracking enabled" in capsys.readouterr().out


@pytest.mark.parametrize("temperature_moves", [None, 0, 2, 30, 60])
def test_self_play_sampling_cutoff(monkeypatch, temperature_moves):
    temperatures = []
    original_choose_move = az_selfplay.choose_move

    def search(self, board, *args, **kwargs):
        assert kwargs["add_root_noise"] is True
        for move in board.legal_moves:
            next_board = board.copy(stack=True)
            next_board.push(move)
            if not next_board.is_game_over(claim_draw=True):
                return {move: 1}
        pytest.fail("Scripted self-play has no non-terminal continuation")

    def choose_move(visits, temperature=0.0, rng=None):
        temperatures.append(temperature)
        return original_choose_move(visits, temperature=temperature, rng=rng)

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    monkeypatch.setattr(az_selfplay, "choose_move", choose_move)
    kwargs = {} if temperature_moves is None else {"temperature_moves": temperature_moves}
    examples = az_selfplay.self_play_game(
        ChessNet(16, 1), simulations=1, max_plies=61, **kwargs,
    )
    window = 60 if temperature_moves is None else temperature_moves
    assert len(examples) == 61
    assert temperatures == [1.0] * window + [0.0] * (61 - window)
