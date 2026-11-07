"""Self-play game generation and head-to-head evaluation."""
from __future__ import annotations

import random

import chess
import numpy as np
import torch

from az_chess import encode_position
from az_mcts import MCTS, choose_move, visit_policy
from az_network import ChessNet
from az_replay import TrainingExample


def _white_result(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


def self_play_game(
    model: ChessNet,
    simulations: int,
    max_plies: int = 256,
    device: torch.device | str = "cpu",
    temperature_moves: int = 30,
    rng: random.Random | None = None,
) -> list[TrainingExample]:
    if max_plies < 1 or temperature_moves < 0:
        raise ValueError("max_plies must be positive and temperature_moves non-negative")
    random_source = rng or random.Random()
    search = MCTS(model, device=device)
    board = chess.Board()
    records: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, chess.Color]
    ] = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        visits = search.search(
            board, simulations=simulations, add_root_noise=True, rng=random_source
        )
        if not visits:
            break
        state = encode_position(board)
        actions, probabilities = visit_policy(board, visits, temperature=1.0)
        temperature = 1.0 if ply < temperature_moves else 0.0
        move = choose_move(visits, temperature=temperature, rng=random_source)
        records.append((state, actions, probabilities, board.turn))
        board.push(move)

    result = 0.0 if not board.is_game_over(claim_draw=True) else _white_result(board)
    return [
        TrainingExample(
            state=state,
            actions=actions,
            probabilities=probabilities,
            outcome=result if player == chess.WHITE else -result,
        )
        for state, actions, probabilities, player in records
    ]


def evaluate_match(
    candidate: ChessNet,
    incumbent: ChessNet,
    games: int,
    simulations: int,
    max_plies: int = 256,
    device: torch.device | str = "cpu",
) -> float:
    """Return candidate's score per game (win=1, draw=.5, loss=0)."""
    if games < 1:
        raise ValueError("games must be at least one")
    candidate_search = MCTS(candidate, device=device)
    incumbent_search = MCTS(incumbent, device=device)
    score = 0.0

    for game_index in range(games):
        candidate_is_white = game_index % 2 == 0
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            is_candidate_turn = board.turn == chess.WHITE and candidate_is_white
            is_candidate_turn |= board.turn == chess.BLACK and not candidate_is_white
            search = candidate_search if is_candidate_turn else incumbent_search
            visits = search.search(board, simulations, add_root_noise=False)
            if not visits:
                break
            board.push(choose_move(visits, temperature=0.0))

        result = _white_result(board) if board.is_game_over(claim_draw=True) else 0.0
        candidate_result = result if candidate_is_white else -result
        score += (candidate_result + 1.0) / 2.0
    return score / games
