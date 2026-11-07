"""Load a champion and choose replies to a validated standard-chess history."""
from __future__ import annotations

from pathlib import Path

import chess
import torch

from az_mcts import MCTS, choose_move
from az_network import ChessNet
from az_rules import CLAIM_DRAW, claim_outcome


class ChampionPlayer:
    def __init__(self, checkpoint: Path, device: str = "auto", simulations: int = 64) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        weights = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model = ChessNet(weights["channels"], weights["blocks"]).to(self.device)
        self.model.load_state_dict(weights["state_dict"])
        self.model.eval()
        self.search = MCTS(self.model, self.device)
        self.simulations = simulations

    def reply(
        self, moves: list[str], human_color: str, simulations: int | None = None,
        claim_draw: bool = False,
    ) -> dict:
        if simulations is None:
            simulations = self.simulations
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if human_color not in ("w", "b"):
            raise ValueError("human_color must be w or b")
        board = chess.Board()
        for uci in moves:
            if board.is_game_over(claim_draw=False):
                raise ValueError("Move history continues after the game ended")
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move in history: {uci}")
            board.push(move)
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            return {"move": None, "result": outcome.result(), "reason": outcome.termination.name}
        if claim_draw:
            if board.turn != (human_color == "w"):
                raise ValueError("Only the player to move can claim a draw")
            outcome = claim_outcome(board)
            if outcome is None:
                raise ValueError("No draw claim is available")
            return {"move": None, "result": outcome.result(), "reason": outcome.termination.name}
        if board.turn == (human_color == "w"):
            raise ValueError("It is the human player's turn")
        move = choose_move(self.search.search(board, simulations))
        if move == CLAIM_DRAW:
            outcome = claim_outcome(board)
            if outcome is None:
                raise RuntimeError("Search returned an unavailable draw claim")
            return {"move": None, "result": outcome.result(), "reason": outcome.termination.name}
        board.push(move)
        outcome = board.outcome(claim_draw=False)
        return {
            "move": move.uci(),
            "result": outcome.result() if outcome else None,
            "reason": outcome.termination.name if outcome else None,
            "claim_available": outcome is None and claim_outcome(board) is not None,
        }
