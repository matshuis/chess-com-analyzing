"""Champion inference and the existing GUI's HTTP bridge."""
import json
from http.client import HTTPConnection
from http.server import HTTPServer
from threading import Thread

import chess
import pytest
import torch

from az_network import ChessNet
from az_play import ChampionPlayer
from server import NoCacheHandler


@pytest.fixture
def champion(tmp_path):
    model = ChessNet(channels=16, blocks=1)
    path = tmp_path / "champion.pt"
    torch.save({
        "channels": 16, "blocks": 1, "state_dict": model.state_dict(),
    }, path)
    return ChampionPlayer(path, device="cpu", simulations=2)


def test_champion_replies_legally_for_either_human_color(champion):
    for moves, color in [([], "b"), (["e2e4"], "w")]:
        result = champion.reply(moves, color)
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        assert chess.Move.from_uci(result["move"]) in board.legal_moves
        assert result["result"] is None


@pytest.mark.parametrize("moves,color", [
    ([], "w"),
    (["e2e5"], "w"),
    (["0000"], "b"),
    (["invalid"], "b"),
    ([], "invalid"),
])
def test_champion_rejects_invalid_history_or_turn(champion, moves, color):
    with pytest.raises(ValueError):
        champion.reply(moves, color)


def test_champion_reports_checkmate_without_search(champion):
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    result = champion.reply(moves, "w")
    assert result == {"move": None, "result": "0-1", "reason": "CHECKMATE"}
    with pytest.raises(ValueError, match="game ended"):
        champion.reply(moves + ["e2e4"], "w")


def test_champion_reports_claimable_draw(champion):
    moves = (["g1f3", "g8f6", "f3g1", "f6g8"] * 2)[:-1]
    result = champion.reply(moves, "w")
    assert result["move"] is None
    assert result["result"] == "1/2-1/2"


@pytest.fixture
def bridge(champion, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    class Handler(NoCacheHandler):
        pass

    Handler.champion = champion
    Handler.default_checkpoint = tmp_path / "champion.pt"
    Handler.default_simulations = 2
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server, Handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def post(bridge, body, headers=None, endpoint="/api/champion"):
    server, _ = bridge
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    try:
        connection.request("POST", endpoint, body, headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_champion_http_validates_requests_and_returns_moves(bridge):
    status, result = post(bridge, json.dumps({"moves": ["e2e4"], "human_color": "w"}))
    assert status == 200 and result["move"]
    for body in [
        "{", "[]",
        json.dumps({"moves": ["e2e5"], "human_color": "w"}),
        json.dumps({"moves": 5, "human_color": "w"}),
        json.dumps({"moves": [], "human_color": "invalid"}),
    ]:
        status, result = post(bridge, body)
        assert status == 400 and result["error"]
    status, result = post(bridge, "x" * 16_385)
    assert status == 400 and result["error"]


def test_champion_http_blocks_cross_origin_and_loads_lazily(bridge):
    body = json.dumps({"moves": [], "human_color": "b"})
    status, result = post(bridge, body, {"Origin": "https://example.com"})
    assert status == 403 and result["error"]
    _, handler = bridge
    handler.champion = None
    status, result = post(bridge, body)
    assert status == 200 and result["move"]


def test_gui_settings_load_and_control_search(bridge, monkeypatch):
    settings = {"checkpoint": "champion.pt", "device": "cpu", "simulations": 3}
    status, result = post(bridge, json.dumps(settings), endpoint="/api/champion/configure")
    assert status == 200 and result["device"] == "cpu"
    _, handler = bridge
    player = handler.champion
    original_search = player.search.search
    counts = []

    def search(board, simulations):
        counts.append(simulations)
        return original_search(board, simulations)

    monkeypatch.setattr(player.search, "search", search)
    status, result = post(bridge, json.dumps({
        **settings, "moves": [], "human_color": "b",
    }))
    assert status == 200 and result["move"]
    assert counts == [3]
    assert handler.champion is player


@pytest.mark.parametrize("setting", [
    {"checkpoint": "../champion.pt"},
    {"checkpoint": "champion.pkl"},
    {"device": "invalid"},
    {"simulations": 0},
    {"simulations": 1025},
    {"simulations": True},
])
def test_gui_rejects_invalid_settings(bridge, setting):
    status, result = post(bridge, json.dumps(setting), endpoint="/api/champion/configure")
    assert status == 400 and result["error"]


def test_gui_reports_missing_checkpoint(bridge):
    status, result = post(bridge, json.dumps({"checkpoint": "missing.pt"}),
                          endpoint="/api/champion/configure")
    assert status == 503 and "Cannot load champion" in result["error"]
