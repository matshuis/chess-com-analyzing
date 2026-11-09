"""Paired evaluation uses varied legal histories and equal search budgets."""
import chess
import pytest

import az_selfplay
from az_network import ChessNet


def test_twenty_games_use_ten_distinct_color_swapped_openings(monkeypatch):
    candidate = ChessNet(16, 1)
    incumbent = ChessNet(16, 1)
    starts = []
    results = []

    def search(self, board, simulations, add_root_noise):
        starts.append((
            board.fen(), tuple(move.uci() for move in board.move_stack),
            self.model, simulations, add_root_noise,
        ))
        return {next(iter(board.legal_moves)): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    score = az_selfplay.evaluate_match(
        candidate, incumbent, games=20, simulations=32, max_plies=1,
        on_game_end=results.append,
    )
    assert score == 0.5
    assert len(starts) == len(results) == 20
    assert len({start[0] for start in starts}) == 10
    for pair in range(10):
        first, second = starts[pair * 2:pair * 2 + 2]
        assert first[:2] == second[:2]
        assert first[1] == az_selfplay.EVALUATION_OPENINGS[pair]
        assert first[2] is candidate
        assert second[2] is incumbent
        assert first[3:] == second[3:] == (32, False)
        assert results[pair * 2].opening == results[pair * 2 + 1].opening == first[1]
        assert results[pair * 2].plies == len(first[1]) + 1
    starts.clear()
    az_selfplay.evaluate_match(candidate, incumbent, games=20, simulations=32, max_plies=1)
    assert [start[1] for start in starts] == [
        opening for opening in az_selfplay.EVALUATION_OPENINGS for _ in range(2)
    ]


def test_odd_length_opening_preserves_turn_and_history(monkeypatch):
    candidate, incumbent = ChessNet(16, 1), ChessNet(16, 1)
    players = []

    def search(self, board, *args, **kwargs):
        assert board.turn == chess.BLACK
        assert [move.uci() for move in board.move_stack] == ["e2e4"]
        players.append(self.model)
        return {board.parse_uci("e7e5"): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    az_selfplay.evaluate_match(
        candidate, incumbent, games=2, simulations=1, max_plies=1,
        openings=(("e2e4",),),
    )
    assert players == [incumbent, candidate]


def test_expanded_suite_adds_distinct_legal_positions_and_keeps_pairs(monkeypatch):
    openings = az_selfplay.EXPANDED_EVALUATION_OPENINGS
    assert len(openings) == 20
    assert openings[::2] == az_selfplay.EVALUATION_OPENINGS
    positions = []
    for opening in openings:
        board = chess.Board()
        for uci in opening:
            board.push_uci(uci)
        assert board.is_valid() and not board.is_game_over()
        positions.append(board.fen())
    assert len(set(positions)) == 20
    starts = []

    def search(self, board, *args, **kwargs):
        starts.append(tuple(move.uci() for move in board.move_stack))
        return {next(iter(board.legal_moves)): 1}

    monkeypatch.setattr(az_selfplay.MCTS, "search", search)
    model = ChessNet(16, 1)
    az_selfplay.evaluate_match(
        model, model, games=40, simulations=1, max_plies=1, openings=openings,
    )
    assert starts == [opening for opening in openings for _ in range(2)]


@pytest.mark.parametrize("games,openings", [
    (1, ((),)), (3, ((),)), (2, ()),
    (2, (("e2e5",),)),
    (2, (("0000",),)),
    (2, (("f2f3", "e7e5", "g2g4", "d8h4"),)),
])
def test_invalid_evaluation_schedules_are_rejected(games, openings):
    model = ChessNet(16, 1)
    with pytest.raises(ValueError):
        az_selfplay.evaluate_match(model, model, games=games, simulations=1, openings=openings)
