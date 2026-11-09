"""Draw claims are choices, never mandatory endings that hide mate."""
import io
import json
import sys
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np
import pytest
import torch

import az_selfplay
import train
from az_benchmark import TACTICS, tactic_board, tactical_scorecard
from az_mcts import MCTS, Node, choose_move, visit_policy
from az_network import ChessNet
from az_rules import CLAIM_DRAW, claim_outcome
from test_parallel import UniformModel


@pytest.mark.parametrize("case", TACTICS[:3], ids=[case[0] for case in TACTICS[:3]])
def test_recorded_claimable_positions_choose_mate(case):
    _, _, fen, history, expected = case
    board = tactic_board(fen, history)
    assert board.can_claim_threefold_repetition()
    assert board.outcome(claim_draw=False) is None
    before = board.fen(), list(board.move_stack)
    visits = MCTS(UniformModel()).search(board, simulations=128)
    assert sum(visits.values()) == 128
    move = choose_move(visits)
    assert move.uci() in expected
    assert (board.fen(), board.move_stack) == before
    board.push(move)
    assert board.is_checkmate()


@pytest.mark.parametrize("case", TACTICS[:3])
def test_evaluation_can_continue_from_claimable_opening_to_mate(monkeypatch, case):
    _, _, fen, history, _ = case
    # Keep full repetition history by substituting the starting board, not a bare FEN.
    board = tactic_board(fen, history)
    chess_proxy = SimpleNamespace(**vars(chess))
    chess_proxy.Board = lambda: board.copy(stack=True)
    monkeypatch.setattr(az_selfplay, "chess", chess_proxy)
    endings = []
    model = UniformModel()
    score = az_selfplay.evaluate_match(
        model, model, games=2, simulations=128, max_plies=1,
        openings=((),), on_game_end=endings.append,
    )
    assert score == 0.5
    assert all(ending.reason == "checkmate" for ending in endings)


@pytest.mark.parametrize("case", TACTICS[:3])
def test_selfplay_can_continue_from_claimable_position_to_mate(monkeypatch, case):
    _, _, fen, history, _ = case
    board = tactic_board(fen, history)
    chess_proxy = SimpleNamespace(**vars(chess))
    chess_proxy.Board = lambda: board.copy(stack=True)
    monkeypatch.setattr(az_selfplay, "chess", chess_proxy)
    endings = []
    examples = az_selfplay.self_play_game(
        UniformModel(), simulations=128, max_plies=1, on_game_end=endings.append,
    )
    assert len(examples) == 1
    assert endings[0].reason == "checkmate"
    assert examples[0].outcome == 1.0


def repetition_board():
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    return board


def test_optional_claim_is_search_action_not_chess_policy_action():
    board = repetition_board()
    visits = MCTS(UniformModel()).search(board, 128)
    assert CLAIM_DRAW in visits
    assert set(visits) == set(board.legal_moves) | {CLAIM_DRAW}
    assert sum(visits.values()) == 128
    move = next(iter(board.legal_moves))
    actions, probabilities = visit_policy(board, {move: 3, CLAIM_DRAW: 9})
    assert len(actions) == 1
    np.testing.assert_array_equal(probabilities, [1.0])
    with pytest.raises(ValueError):
        visit_policy(board, {CLAIM_DRAW: 1})


def test_declining_claim_preserves_mandatory_fivefold(monkeypatch):
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]

    def search(self, board, *args, **kwargs):
        return {board.parse_uci(moves[len(board.move_stack) % 4]): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    endings = []
    examples = az_selfplay.self_play_game(
        UniformModel(), simulations=1, max_plies=30, on_game_end=endings.append,
    )
    assert len(examples) == 16
    assert endings[0].reason == "fivefold_repetition"
    game = chess.pgn.read_game(io.StringIO(endings[0].pgn))
    assert game.end().board().is_fivefold_repetition()
    monkeypatch.undo()
    assert MCTS(UniformModel()).search(game.end().board(), 1) == {}


def test_fifty_move_claim_is_optional_and_seventyfive_is_mandatory():
    board = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 100 51")
    assert claim_outcome(board).termination == chess.Termination.FIFTY_MOVES
    assert CLAIM_DRAW in MCTS(UniformModel()).search(board, 4)
    board.halfmove_clock = 150
    assert board.outcome().termination == chess.Termination.SEVENTYFIVE_MOVES
    assert MCTS(UniformModel()).search(board, 4) == {}


def test_search_prefers_known_draw_to_known_loss():
    root = Node(1.0, visits=20, expanded=True)
    root.children = {
        chess.Move.from_uci("a2a8"): Node(0.5, visits=10, value_sum=10.0),
        CLAIM_DRAW: Node(0.5, visits=10, value_sum=0.0),
    }
    move, _ = MCTS(UniformModel())._select_child(root)
    assert move == CLAIM_DRAW


def test_scorecard_includes_separate_mate_and_capture_cases():
    report = tactical_scorecard(UniformModel(), 128, "cpu")
    assert report["total"] == len(TACTICS)
    assert report["suite_version"] == 2
    assert all(case["passed"] for case in report["cases"][:3])
    assert all(case["claim_available"] for case in report["cases"][:3])
    assert [case["category"] for case in report["cases"][:5]] == ["mate"] * 3 + ["capture"] * 2


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("suite", ["legacy-v1", "expanded-v2"])
def test_frozen_baseline_saved_without_mutation(tmp_path, monkeypatch, workers, suite):
    model = ChessNet(16, 1)
    baseline = tmp_path / "frozen.pt"
    train._save_checkpoint(baseline, model, 5, False, 16, 1)
    original = baseline.read_bytes()
    output = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--workers", str(workers),
        "--iterations", "2", "--games-per-iteration", "2",
        "--simulations", "1", "--eval-games", "2", "--eval-simulations", "2",
        "--max-plies", "2", "--train-steps", "1", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(output),
        "--baseline-checkpoint", str(baseline), "--baseline-eval-games", "2",
        "--evaluation-suite", suite,
    ])
    assert train.main() == 0
    assert baseline.read_bytes() == original
    hashes = []
    for iteration in (1, 2):
        folder = output / "games" / f"iteration-{iteration:06d}"
        report = json.loads((folder / "scorecard.json").read_text())
        assert report["baseline"]["score"] == 0.5
        assert report["evaluation_suite"] == suite
        assert report["evaluation_openings"] == (20 if suite == "expanded-v2" else 10)
        assert report["baseline"]["games"] == 2
        assert report["tactics"]["total"] == len(TACTICS)
        hashes.append(report["baseline"]["sha256"])
        assert len(list(folder.glob("baseline-*.pgn"))) == 2
        for path in [folder / "baseline-001.pgn", folder / "evaluation-001.pgn"]:
            with path.open() as handle:
                game = chess.pgn.read_game(handle)
            assert game.headers["EvaluationSuite"] == suite
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("arguments", [
    ["--baseline-eval-games", "1"],
    ["--baseline-eval-games", "-2"],
    ["--baseline-eval-games", "2"],
    ["--baseline-checkpoint", "runs/az/champion.pt"],
])
def test_invalid_baseline_configuration_fails_before_training(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["train.py"] + arguments)
    with pytest.raises(ValueError, match="baseline"):
        train.main()
