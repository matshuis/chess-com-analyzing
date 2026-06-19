# chess-com-analyzing

Analyze chess.com games — initial scaffold.

This project deliberately avoids package managers. Everything runs on a
plain Python 3 install (standard library only) and a vanilla
HTML/CSS/JS frontend.

## Files

- `download_games.py` — *optional offline cache.* Fetches every public
  game for a chess.com username via the public API and writes
  `games.json`. The UI loads it automatically if it's present.
- `server.py` — tiny static file server (uses `http.server`) so the
  browser can load the assets locally.
- `index.html`, `style.css`, `app.js` — the web UI:
  - header: chess.com username input + Load button
  - left pane: filterable list of games
  - right pane: a chess board (Unicode pieces) and a clickable move list
  - navigate plies with `◀`/`▶` buttons, the move list, or the
    `←`/`→`/`Home`/`End` keys.

## Usage

```bash
# 1. start the static server
python3 server.py

# 2. open http://localhost:8000 and type a chess.com username
#    (or open http://localhost:8000/?user=<name> to auto-load)
```

The web UI fetches archives directly from chess.com's public API
(`https://api.chess.com/pub/player/<user>/games/archives`). Games
appear in the list incrementally as each monthly archive arrives.

### Optional offline cache

If you'd rather download games once and replay them offline, run:

```bash
python3 download_games.py <username>   # writes games.json
python3 server.py
```

If `games.json` exists in the project root, the UI silently loads it
on startup so you don't need a network connection.

Pass `--full` to `download_games.py` to keep every API field instead
of the slimmed copy.

## Notes / next steps

- The PGN replay engine in `app.js` implements just enough chess
  (SAN, captures, castling, en passant, promotion, pin-aware
  disambiguation) to step through real games. It is not a full move
  generator and will not validate user-entered moves.
- `games.json` is a slimmed copy of the chess.com response by default;
  pass `--full` to `download_games.py` to keep every field.
