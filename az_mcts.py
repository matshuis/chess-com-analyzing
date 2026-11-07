"""PUCT Monte Carlo tree search guided by a policy/value network."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from az_chess import encode_move, encode_position
from az_network import ChessNet


@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[chess.Move, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(
        self,
        model: ChessNet,
        device: torch.device | str = "cpu",
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
    ) -> None:
        if c_puct <= 0 or dirichlet_alpha <= 0:
            raise ValueError("c_puct and dirichlet_alpha must be positive")
        if not 0 <= noise_fraction <= 1:
            raise ValueError("noise_fraction must be between zero and one")
        self.model = model
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

    def _predict(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        encoded = torch.from_numpy(encode_position(board)).unsqueeze(0)
        encoded = encoded.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            logits, value = self.model(encoded)
        logits_np = logits[0].float().cpu().numpy()
        legal_moves = list(board.legal_moves)
        action_ids = [encode_move(move, board.turn) for move in legal_moves]
        legal_logits = logits_np[np.asarray(action_ids, dtype=np.int64)]
        legal_logits -= np.max(legal_logits)
        probabilities = np.exp(legal_logits)
        probabilities /= probabilities.sum()
        return dict(zip(legal_moves, probabilities.tolist())), float(value.item())

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError("Position is not terminal")
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _expand(self, node: Node, board: chess.Board) -> float:
        priors, value = self._predict(board)
        node.children = {move: Node(prior) for move, prior in priors.items()}
        node.expanded = True
        return value

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        scale = math.sqrt(max(node.visits, 1))

        def score(child: Node) -> float:
            exploration = (
                self.c_puct * child.prior * scale / (1 + child.visits)
            )
            return -child.mean_value + exploration

        return max(node.children.items(), key=lambda item: score(item[1]))

    def search(
        self,
        board: chess.Board,
        simulations: int = 64,
        add_root_noise: bool = False,
        rng: random.Random | None = None,
    ) -> dict[chess.Move, int]:
        if simulations < 1:
            raise ValueError("simulations must be at least one")
        if board.is_game_over(claim_draw=True):
            return {}

        root = Node(1.0)
        self._expand(root, board)
        if add_root_noise and root.children:
            random_source = rng or random
            noise = np.random.default_rng(random_source.randrange(2**32)).dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for child, noise_prior in zip(root.children.values(), noise):
                child.prior = (
                    (1.0 - self.noise_fraction) * child.prior
                    + self.noise_fraction * float(noise_prior)
                )

        for _ in range(simulations):
            state = board.copy(stack=True)
            node = root
            path = [node]

            while node.expanded and node.children:
                move, node = self._select_child(node)
                state.push(move)
                path.append(node)
                if state.is_game_over(claim_draw=True):
                    break

            if state.is_game_over(claim_draw=True):
                value = self._terminal_value(state)
            else:
                value = self._expand(node, state)

            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += value
                value = -value

        return {move: child.visits for move, child in root.children.items()}


def choose_move(
    visits: dict[chess.Move, int],
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> chess.Move:
    if not visits:
        raise ValueError("Cannot choose a move from an empty visit distribution")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    moves = list(visits)
    counts = np.asarray([visits[move] for move in moves], dtype=np.float64)
    if temperature == 0:
        return moves[int(np.argmax(counts))]
    weights = np.power(counts, 1.0 / temperature)
    total = float(weights.sum())
    if total == 0:
        return (rng or random).choice(moves)
    return (rng or random).choices(moves, weights=weights.tolist(), k=1)[0]


def visit_policy(
    board: chess.Board, visits: dict[chess.Move, int], temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse action IDs and normalized visit-count probabilities."""
    if not visits:
        raise ValueError("Cannot create a policy from empty visits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    actions = np.asarray(
        [encode_move(move, board.turn) for move in visits], dtype=np.int16
    )
    counts = np.asarray(list(visits.values()), dtype=np.float64)
    weights = np.power(counts, 1.0 / temperature)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Visit counts produced an invalid policy distribution")
    probabilities = (weights / weights.sum()).astype(np.float32)
    return actions, probabilities
