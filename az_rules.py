"""Optional draw claims, separate from mandatory chess endings."""
import chess

# Internal search action only: never push it or encode it as a chess move.
CLAIM_DRAW = chess.Move.null()


def claim_outcome(board: chess.Board) -> chess.Outcome | None:
    if board.can_claim_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if board.can_claim_threefold_repetition():
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return None
