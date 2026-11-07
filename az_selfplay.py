"""Self-play game generation and head-to-head evaluation."""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np
import torch

from az_chess import encode_position
from az_mcts import MCTS, choose_move, visit_policy
from az_replay import TrainingExample
from az_rules import CLAIM_DRAW, claim_outcome


# Short, fixed openings keep comparisons reproducible across iterations.
EVALUATION_OPENINGS: tuple[tuple[str, ...], ...] = (
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("e2e4", "e7e6", "d2d4", "d7d5"),
    ("e2e4", "c7c6", "d2d4", "d7d5"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("d2d4", "d7d5", "c2c4", "c7c6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
    ("d2d4", "g8f6", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6"),
    ("g1f3", "d7d5", "g2g3", "g8f6"),
)


@dataclass(frozen=True)
class GameResult:
    plies: int
    result: str
    reason: str
    move_cap: bool
    white_score: float
    opening: tuple[str, ...] = ()
    pgn: str = ""


def _game_result(
    board: chess.Board, opening: tuple[str, ...] = (), claimed: bool = False,
) -> GameResult:
    outcome = board.outcome(claim_draw=False)
    if claimed:
        outcome = claim_outcome(board)
        if outcome is None:
            raise ValueError("Cannot record an unavailable draw claim")
    result = outcome.result() if outcome else "1/2-1/2"
    reason = outcome.termination.name.lower() if outcome else "move_cap"
    game = chess.pgn.Game()
    position = board.root()
    game.setup(position)
    game.headers["Result"] = result
    game.headers["Termination"] = "normal" if outcome else "adjudication"
    game.headers["TerminationReason"] = reason
    game.headers["DrawClaimed"] = "true" if claimed else "false"
    game.headers["FinalFEN"] = board.fen()
    game.headers["PlyCount"] = str(len(board.move_stack))
    game.headers["OpeningMoves"] = " ".join(opening)
    seen: dict[tuple[str, ...], list[int]] = {
        tuple(position.fen().split()[:4]): [0],
    }
    node: chess.pgn.GameNode = game
    for ply, move in enumerate(board.move_stack, start=1):
        position.push(move)
        node = node.add_variation(move)
        key = tuple(position.fen().split()[:4])
        previous_plies = seen.setdefault(key, [])
        if previous_plies:
            node.comment = (
                "Position occurrence "
                f"{len(previous_plies) + 1}; previously seen after plies "
                + ", ".join(map(str, previous_plies)) + " (0 = initial position)."
            )
        previous_plies.append(ply)
    ending_note = f"Game ended: {reason}."
    if reason == "threefold_repetition" and not board.is_repetition(3):
        claim_moves = []
        for move in list(position.legal_moves):
            san = position.san(move)
            position.push(move)
            if position.is_repetition(3):
                claim_moves.append(f"{san} ({move.uci()})")
            position.pop()
        ending_note += (
            " Threefold repetition is claimable by announcing "
            + ", ".join(claim_moves) + "; these moves were not played."
        )
    node.comment = f"{node.comment} {ending_note}".strip()
    return GameResult(
        plies=len(board.move_stack),
        result=result,
        reason=reason,
        move_cap=outcome is None,
        white_score=(
            0.0 if outcome is None or outcome.winner is None
            else 1.0 if outcome.winner == chess.WHITE else -1.0
        ),
        opening=opening,
        pgn=game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=True)),
    )


def _white_result(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=False)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


def self_play_game(
    model: torch.nn.Module,
    simulations: int,
    max_plies: int = 256,
    device: torch.device | str = "cpu",
    temperature_moves: int = 60,
    rng: random.Random | None = None,
    on_game_end: Callable[[GameResult], None] | None = None,
) -> list[TrainingExample]:
    if max_plies < 1 or temperature_moves < 0:
        raise ValueError("max_plies must be positive and temperature_moves non-negative")
    random_source = rng or random.Random()
    search = MCTS(model, device=device)
    board = chess.Board()
    records: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, chess.Color]
    ] = []

    claimed = False
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=False):
            break
        visits = search.search(
            board, simulations=simulations, add_root_noise=True, rng=random_source
        )
        if not visits:
            break
        temperature = 1.0 if ply < temperature_moves else 0.0
        move = choose_move(visits, temperature=temperature, rng=random_source)
        if move == CLAIM_DRAW:
            claimed = True
            break
        state = encode_position(board)
        actions, probabilities = visit_policy(board, visits, temperature=1.0)
        records.append((state, actions, probabilities, board.turn))
        board.push(move)

    game_result = _game_result(board, claimed=claimed)
    if on_game_end is not None:
        on_game_end(game_result)
    result = game_result.white_score
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
    candidate: torch.nn.Module,
    incumbent: torch.nn.Module,
    games: int,
    simulations: int,
    max_plies: int = 256,
    device: torch.device | str = "cpu",
    on_game_end: Callable[[GameResult], None] | None = None,
    openings: tuple[tuple[str, ...], ...] = EVALUATION_OPENINGS,
) -> float:
    """Score paired openings; max_plies counts searched plies after the opening."""
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even number for color-swapped pairs")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not openings:
        raise ValueError("At least one evaluation opening is required")
    starting_boards: list[chess.Board] = []
    for opening in openings:
        board = chess.Board()
        for uci in opening:
            move = chess.Move.from_uci(uci)
            if board.is_game_over(claim_draw=False) or move not in board.legal_moves:
                raise ValueError(f"Invalid evaluation opening move: {uci}")
            board.push(move)
        if board.is_game_over(claim_draw=False):
            raise ValueError("Evaluation openings must end in a non-terminal position")
        starting_boards.append(board)
    candidate_search = MCTS(candidate, device=device)
    incumbent_search = MCTS(incumbent, device=device)
    score = 0.0

    for game_index in range(games):
        candidate_is_white = game_index % 2 == 0
        opening_index = (game_index // 2) % len(openings)
        opening = openings[opening_index]
        board = starting_boards[opening_index].copy(stack=True)
        claimed = False
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=False):
                break
            is_candidate_turn = board.turn == chess.WHITE and candidate_is_white
            is_candidate_turn |= board.turn == chess.BLACK and not candidate_is_white
            search = candidate_search if is_candidate_turn else incumbent_search
            visits = search.search(board, simulations, add_root_noise=False)
            if not visits:
                break
            move = choose_move(visits, temperature=0.0)
            if move == CLAIM_DRAW:
                claimed = True
                break
            board.push(move)

        game_result = _game_result(board, opening, claimed=claimed)
        if on_game_end is not None:
            on_game_end(game_result)
        result = game_result.white_score
        candidate_result = result if candidate_is_white else -result
        score += (candidate_result + 1.0) / 2.0
    return score / games
