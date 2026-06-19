#!/usr/bin/env python3
"""Download all public chess.com games for a given username.

Uses only the Python standard library (no external packages).
The chess.com public API exposes monthly archives:
    https://api.chess.com/pub/player/{username}/games/archives

Each archive returns a JSON object with a "games" array (each game
includes a full PGN plus metadata).  We concatenate them all and save
them as a single JSON file that the web UI can consume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

API_BASE = "https://api.chess.com/pub/player"
# chess.com's CDN rejects requests without a User-Agent.
HEADERS = {
    "User-Agent": "chess-com-analyzer/0.1 (+https://example.local; educational use)",
    "Accept": "application/json",
}


def fetch_json(url: str, retries: int = 3, backoff: float = 1.5) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 404 = unknown user / no archive, do not retry.
            if exc.code in (404, 410):
                raise
            last_err = exc
        except urllib.error.URLError as exc:
            last_err = exc
        time.sleep(backoff * (attempt + 1))
    assert last_err is not None
    raise last_err


def download_all_games(username: str) -> list[dict]:
    archives_url = f"{API_BASE}/{username}/games/archives"
    archives = fetch_json(archives_url).get("archives", [])
    if not archives:
        return []

    all_games: list[dict] = []
    for idx, archive_url in enumerate(archives, start=1):
        print(
            f"[{idx}/{len(archives)}] fetching {archive_url}",
            file=sys.stderr,
        )
        data = fetch_json(archive_url)
        all_games.extend(data.get("games", []))
    return all_games


def slim_game(game: dict) -> dict:
    """Keep only the fields the UI needs, to keep games.json small."""
    white = game.get("white", {}) or {}
    black = game.get("black", {}) or {}
    return {
        "url": game.get("url"),
        "pgn": game.get("pgn", ""),
        "time_class": game.get("time_class"),
        "time_control": game.get("time_control"),
        "rated": game.get("rated"),
        "rules": game.get("rules"),
        "end_time": game.get("end_time"),
        "white": {
            "username": white.get("username"),
            "rating": white.get("rating"),
            "result": white.get("result"),
        },
        "black": {
            "username": black.get("username"),
            "rating": black.get("rating"),
            "result": black.get("result"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="chess.com username (case-insensitive)")
    parser.add_argument(
        "-o",
        "--output",
        default="games.json",
        help="Output JSON file (default: games.json)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Save the full API response per game instead of a slimmed copy.",
    )
    args = parser.parse_args()

    username = args.username.strip().lower()
    if not username:
        parser.error("username must not be empty")

    print(f"Downloading games for '{username}' ...", file=sys.stderr)
    try:
        games = download_all_games(username)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error from chess.com: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error: {exc.reason}", file=sys.stderr)
        return 1

    if not args.full:
        games = [slim_game(g) for g in games]

    payload = {"username": username, "count": len(games), "games": games}
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    print(f"Saved {len(games)} games to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
