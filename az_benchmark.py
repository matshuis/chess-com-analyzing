"""Fixed, history-preserving tactical checks independent of promotion."""
from __future__ import annotations

import chess
import torch

from az_mcts import MCTS, choose_move
from az_chess import encode_move, encode_position
from az_rules import CLAIM_DRAW


TACTICAL_SUITE_VERSION = 2

TACTICS = (
    (
        "iteration-33-evaluation-18", "mate",
        "5k2/2p4p/r1npRP2/pp5r/nPb4P/R5p1/5q2/2NK3B b - - 15 69",
        "c6b4 a3a2 a6a7 h1g2 a7a8 g2h1 a8d8 h1g2 d8c8 g2h1 c8b8 h1g2 "
        "b8b7 g2h1 b7b6 h1g2 b6b8 g2h1 b8c8 h1g2 c8a8 g2h1 a8d8 h1g2",
        ("a4c3",),
    ),
    (
        "iteration-36-evaluation-13", "mate",
        "8/k1Np2rp/r3Qp2/5P2/4n1Pb/pRp1q2P/P1K1P3/1R3B2 b - - 4 59",
        "a6b6 b1a1 b6c6 a1b1 c6c5 b1a1 e3d4 a1b1 d4e3 b1a1 e3d4 a1b1",
        ("d4d2",),
    ),
    (
        "iteration-37-evaluation-11", "mate",
        "4k1b1/8/6pp/r4rnP/pN4p1/8/P2pn3/RKR3B1 b - - 3 76",
        "f5f7 c1c2 f7f6 c2c1 f6f5 c1c2 g8f7 c2c1 f7g8 c1c2 g8f7 c2c1",
        ("d2c1q",),
    ),
    (
        "white-hanging-queen", "capture",
        "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", "", ("d1d4",),
    ),
    (
        "black-hanging-queen", "capture",
        "3rk3/8/8/3Q4/8/8/8/4K3 b - - 0 1", "", ("d8d5",),
    ),
    (
        "knight-hanging-queen", "capture",
        "4k3/8/8/4q3/8/5N2/8/4K3 w - - 0 1", "", ("f3e5",),
    ),
    (
        "bishop-hanging-queen", "capture",
        "4k3/8/8/8/6q1/8/4B3/4K3 w - - 0 1", "", ("e2g4",),
    ),
    (
        "pawn-hanging-queen", "capture",
        "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "", ("e4d5",),
    ),
    (
        "capture-promotion", "promotion",
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "", ("a7b8q",),
    ),
    (
        "iteration-67-selfplay-18-defense", "defense",
        chess.STARTING_FEN, "b2b3 f7f5 g2g4 e7e6",
        (
            "g1h3", "g1f3", "f1h3", "f1g2", "c1a3", "c1b2", "b1c3", "b1a3",
            "g4f5", "g4g5", "b3b4", "h2h3", "e2e3", "d2d3", "c2c3", "a2a3",
            "h2h4", "e2e4", "d2d4", "c2c4", "a2a4",
        ),
    ),
    (
        "iteration-67-evaluation-4-defense", "defense",
        chess.STARTING_FEN,
        "e2e4 c7c5 g1f3 d7d6 e4e5 f7f6 h2h4 g7g5 f3g5 h7h6 d1h5 e8d7 "
        "f2f4 e7e6 b1a3",
        (
            "h8h7", "g8e7", "f8g7", "f8e7", "d8e8", "d8e7", "d8c7", "d8b6",
            "d8a5", "b8c6", "b8a6", "d7c7", "d7c6", "h6g5", "f6g5", "f6e5",
            "d6e5", "b7b6", "a7a6", "f6f5", "d6d5", "c5c4", "b7b5", "a7a5",
        ),
    ),
)


def tactic_board(fen: str, history: str) -> chess.Board:
    board = chess.Board(fen)
    for uci in history.split():
        board.push_uci(uci)
    return board


def tactical_scorecard(
    model: torch.nn.Module, simulations: int, device: torch.device | str,
) -> dict:
    search = MCTS(model, device=device)
    cases = []
    model.eval()
    for name, category, fen, history, expected in TACTICS:
        board = tactic_board(fen, history)
        visits = search.search(board, simulations)
        move = choose_move(visits)
        root = search.root
        root_children = [
            {
                "move": "claim_draw" if action == CLAIM_DRAW else action.uci(),
                "prior": child.prior, "visits": child.visits,
                "value_for_player": -child.mean_value if child.visits else None,
            }
            for action, child in (root.children.items() if root is not None else ())
        ]
        legal_moves = list(board.legal_moves)
        with torch.inference_mode():
            inputs = torch.from_numpy(encode_position(board)).unsqueeze(0).to(device)
            logits, _ = model(inputs)
            legal_logits = logits[0, [encode_move(m, board.turn) for m in legal_moves]]
            probabilities = torch.softmax(legal_logits.float(), dim=0).cpu().tolist()
        policy_move = legal_moves[max(range(len(legal_moves)), key=probabilities.__getitem__)]
        cases.append({
            "name": name, "category": category, "fen": board.fen(),
            "initial_fen": fen, "history": history.split(),
            "claim_available": board.can_claim_draw(),
            "selected": "claim_draw" if move == CLAIM_DRAW else move.uci(),
            "expected": list(expected), "passed": move.uci() in expected,
            "policy_selected": policy_move.uci(), "policy_passed": policy_move.uci() in expected,
            "expected_policy_mass": sum(
                probability for m, probability in zip(legal_moves, probabilities)
                if m.uci() in expected
            ),
            "search_root_value": root.mean_value if root is not None else None,
            "search_children": root_children,
            "mate_safeguard": root is None,
        })
    return {
        "suite_version": TACTICAL_SUITE_VERSION, "simulations": simulations,
        "passed": sum(case["passed"] for case in cases), "total": len(cases),
        "policy_passed": sum(case["policy_passed"] for case in cases),
        "cases": cases,
    }
