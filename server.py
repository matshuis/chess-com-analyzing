#!/usr/bin/env python3
"""Tiny static file server for the analyzer UI.

Run it from the project root:

    uv run server.py

Then open http://localhost:8000 in your browser.  The UI expects
``games.json`` to be present in the same directory (produced by
``download_games.py``).
"""
from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from az_play import ChampionPlayer


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Disable caching so iterative dev edits show up on reload."""

    def end_headers(self) -> None:  # noqa: D401 - short override
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    champion: ChampionPlayer | None = None
    champion_key: tuple[Path, str, int] | None = None
    default_checkpoint = Path("runs/az/champion.pt")
    default_device = "auto"
    default_simulations = 64

    def _player(self, data: dict) -> ChampionPlayer:
        checkpoint = data.get("checkpoint", str(self.default_checkpoint))
        device = data.get("device", self.default_device)
        simulations = data.get("simulations", self.default_simulations)
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("checkpoint must be a non-empty project-relative path")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if type(simulations) is not int or not 1 <= simulations <= 1024:
            raise ValueError("simulations must be an integer between 1 and 1024")
        path = (Path.cwd() / checkpoint).resolve()
        if not path.is_relative_to(Path.cwd().resolve()) or path.suffix != ".pt":
            raise ValueError("Checkpoint must be a .pt file inside the project")
        key = (path, device, path.stat().st_mtime_ns)
        handler = type(self)
        if handler.champion is None or handler.champion_key != key:
            from az_play import ChampionPlayer

            player = ChampionPlayer(path, device, simulations)
            handler.champion = player
            handler.champion_key = key
        return handler.champion

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        endpoint = urlsplit(self.path).path
        if endpoint not in ("/api/champion", "/api/champion/configure"):
            self._json(404, {"error": "Unknown endpoint"})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self._json(403, {"error": "Cross-origin requests are not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                raise ValueError("Request body must be between 1 and 16384 bytes")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object")
            if endpoint == "/api/champion/configure" or "checkpoint" in data or self.champion is None:
                player = self._player(data)
            else:
                player = self.champion
            if endpoint == "/api/champion/configure":
                self._json(200, {"device": str(player.device)})
                return
            moves = data.get("moves")
            color = data.get("human_color")
            if (not isinstance(moves, list) or len(moves) > 1000
                    or not all(isinstance(move, str) for move in moves)):
                raise ValueError("moves must be a list of at most 1000 UCI move strings")
            if color not in ("w", "b"):
                raise ValueError("human_color must be w or b")
            simulations = data.get("simulations", self.default_simulations)
            if type(simulations) is not int or not 1 <= simulations <= 1024:
                raise ValueError("simulations must be an integer between 1 and 1024")
            claim_draw = data.get("claim_draw", False)
            if type(claim_draw) is not bool:
                raise ValueError("claim_draw must be a boolean")
            result = player.reply(moves, color, simulations=simulations, claim_draw=claim_draw)
        except (OSError, ImportError) as exc:
            self.log_error("Cannot load champion: %s", exc)
            self._json(503, {"error": f"Cannot load champion: {exc}"})
            return
        except (ValueError, UnicodeDecodeError) as exc:
            self.log_error("Invalid champion request: %s", exc)
            self._json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self.log_error("Champion inference failed: %s", exc)
            self._json(500, {"error": "Champion inference failed; see the server terminal"})
            return
        self._json(200, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--champion", type=Path, help="Enable play against this champion checkpoint")
    parser.add_argument("--device", default="auto", help="Champion inference device (auto, cpu, cuda)")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per champion move")
    args = parser.parse_args()

    NoCacheHandler.default_checkpoint = args.champion or Path("runs/az/champion.pt")
    NoCacheHandler.default_device = args.device
    NoCacheHandler.default_simulations = args.simulations
    if args.champion is not None:
        from az_play import ChampionPlayer

        NoCacheHandler.champion = ChampionPlayer(args.champion, args.device, args.simulations)
        print(f"Loaded champion {args.champion} on {NoCacheHandler.champion.device}")
    server = HTTPServer((args.host, args.port), NoCacheHandler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
