"""Chess state and action encoding for the AlphaZero-style engine."""
from __future__ import annotations

import numpy as np
import chess

HISTORY_LENGTH = 8
CONTEXT_FEATURES = 6
BOARD_FEATURES = HISTORY_LENGTH * 12 + CONTEXT_FEATURES
POLICY_PLANES = 73
ACTION_SIZE = 64 * POLICY_PLANES

_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_KNIGHT_DELTAS = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMOTION_DX = (-1, 0, 1)


def _canonical_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return chess.square_mirror(square) if turn == chess.BLACK else square


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    """Encode a move as ``plane * 64 + from_square`` in side-to-move view."""
    origin = _canonical_square(move.from_square, turn)
    target = _canonical_square(move.to_square, turn)
    dx = chess.square_file(target) - chess.square_file(origin)
    dy = chess.square_rank(target) - chess.square_rank(origin)

    if move.promotion in _UNDERPROMOTIONS:
        if dy != 1 or dx not in _UNDERPROMOTION_DX:
            raise ValueError(f"Invalid underpromotion move: {move.uci()}")
        plane = 64 + _UNDERPROMOTION_DX.index(dx) * 3
        plane += _UNDERPROMOTIONS.index(move.promotion)
    elif (dx, dy) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((dx, dy))
    else:
        distance = max(abs(dx), abs(dy))
        if not 1 <= distance <= 7:
            raise ValueError(f"Move is not representable: {move.uci()}")
        direction = (dx // distance, dy // distance)
        if direction not in _DIRECTIONS:
            raise ValueError(f"Move is not representable: {move.uci()}")
        plane = _DIRECTIONS.index(direction) * 7 + distance - 1

    return plane * 64 + origin


def decode_action(action: int, board: chess.Board) -> chess.Move:
    """Decode a policy index into a move, rejecting malformed indices."""
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"Action index out of range: {action}")
    turn = board.turn
    plane, origin = divmod(action, 64)
    from_square = _canonical_square(origin, turn)
    file = chess.square_file(origin)
    rank = chess.square_rank(origin)

    if plane < 56:
        direction = _DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        target_file = file + direction[0] * distance
        target_rank = rank + direction[1] * distance
        promotion = None
    elif plane < 64:
        dx, dy = _KNIGHT_DELTAS[plane - 56]
        target_file, target_rank = file + dx, rank + dy
        promotion = None
    else:
        underpromotion = plane - 64
        dx = _UNDERPROMOTION_DX[underpromotion // 3]
        target_file, target_rank = file + dx, rank + 1
        promotion = _UNDERPROMOTIONS[underpromotion % 3]

    if not (0 <= target_file < 8 and 0 <= target_rank < 8):
        raise ValueError(f"Action {action} leaves the board")
    target = chess.square(target_file, target_rank)
    target = _canonical_square(target, turn)
    if (promotion is None
            and board.piece_type_at(from_square) == chess.PAWN
            and chess.square_rank(target) == (7 if turn == chess.WHITE else 0)):
        promotion = chess.QUEEN
    return chess.Move(from_square, target, promotion=promotion)


def encode_position(board: chess.Board) -> np.ndarray:
    """Return eight positions of piece history plus six current context planes.

    Each 12-plane frame contains friendly P/N/B/R/Q/K followed by enemy
    P/N/B/R/Q/K. Frames run from current to oldest; missing history repeats
    the oldest available position. The final six planes encode current
    castling rights, en-passant target, and normalized halfmove clock.
    """
    features = np.zeros((BOARD_FEATURES, 8, 8), dtype=np.float32)
    history = [board.copy(stack=False)]
    previous = board.copy(stack=True)
    while len(history) < HISTORY_LENGTH and previous.move_stack:
        previous.pop()
        history.append(previous.copy(stack=False))
    history.extend([history[-1]] * (HISTORY_LENGTH - len(history)))

    for frame, position in enumerate(history):
        frame_offset = frame * 12
        for square, piece in position.piece_map().items():
            view_square = _canonical_square(square, board.turn)
            plane = frame_offset + piece.piece_type - 1
            if piece.color != board.turn:
                plane += 6
            features[plane, chess.square_rank(view_square),
                     chess.square_file(view_square)] = 1.0

    own_king = board.has_kingside_castling_rights(board.turn)
    own_queen = board.has_queenside_castling_rights(board.turn)
    enemy = not board.turn
    enemy_king = board.has_kingside_castling_rights(enemy)
    enemy_queen = board.has_queenside_castling_rights(enemy)
    context_offset = HISTORY_LENGTH * 12
    features[context_offset].fill(float(own_king))
    features[context_offset + 1].fill(float(own_queen))
    features[context_offset + 2].fill(float(enemy_king))
    features[context_offset + 3].fill(float(enemy_queen))

    if board.ep_square is not None:
        ep = _canonical_square(board.ep_square, board.turn)
        features[context_offset + 4, chess.square_rank(ep),
                 chess.square_file(ep)] = 1.0
    features[context_offset + 5].fill(min(board.halfmove_clock, 100) / 100.0)
    return features
