# chess-com-analyzing

Analyze chess.com games — initial scaffold.

The chess.com analyzer runs on the Python standard library and a vanilla
HTML/CSS/JS frontend. The optional AlphaZero-style training engine uses
PyTorch, NumPy, and python-chess (see `requirements.txt`).

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
- `pieces.svg` — original vector chess pieces for the green-and-cream
  board; no external artwork or network-loaded assets are required.

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

## AlphaZero-style training engine

The separate Python engine in `az_*.py` is a compact, complete self-play
policy/value training loop; it does not replace the analyzer's existing
`chess_engine.py`. It uses `python-chess` for legal move generation, a
PyTorch residual network, PUCT MCTS, self-play targets, a bounded replay
buffer, and candidate-vs-champion evaluation.

### Setup and first run

Use Python 3.10+ and install the dependencies in `requirements.txt`. For an
NVIDIA GPU, install a CUDA-enabled PyTorch build compatible with the installed
driver using the official PyTorch installation selector first, then install
the remaining requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --device auto --iterations 1 \
  --games-per-iteration 2 --simulations 16 --eval-games 2 \
  --eval-simulations 8 --train-steps 10 --batch-size 64
```

`--device auto` selects CUDA when the installed PyTorch build can use it and
otherwise runs on CPU. The first run writes `runs/az/champion.pt`,
`runs/az/latest.pt`, and `runs/az/replay.npz`; subsequent runs resume from
those files. Training is intentionally opt-in and does not download data.

### Play against your champion

Start the existing GUI server; no champion-specific arguments are required:

```bash
source .venv/bin/activate
python server.py
```

Open http://localhost:8000, set the **Checkpoint** project-relative path
(default `runs/az/champion.pt`), **Device** (Auto, CPU, or CUDA GPU), and
**Simulations** (1-1024). Select White or Black next to **Play as**, and
click **Play champion** to load the model. Click a piece and its destination on the existing
board. The champion replies automatically; when playing Black, it makes
the opening move. Human pawn promotions currently auto-promote to queens.
To castle, click your king and then g1/c1 (White) or g8/c8 (Black).
The destination is offered only when castling is legal; the rook moves
automatically.

The server validates the complete UCI history with python-chess, preserving
the network's position history and draw rules. The status label reports
thinking, errors, and game results. **Retry reply** retries a failed request.
You can review earlier moves, but must return to the latest move to play.
**Reset** returns to two-player free play; selecting an archived game exits
champion mode. Starting **Play champion** again begins a fresh game.

The checkpoint supplies its own network dimensions, so no `--channels` or
`--blocks` flags are needed. Use 16 simulations for faster replies or
64-128 for more search. Settings are captured when starting a game; click
**Play champion** again to apply changes. Updated checkpoint files reload
automatically on the next request. Paths must point to `.pt` files inside
the project. Dependencies load lazily, so the analyzer alone still runs
with only the standard library. The existing `--champion`, `--device`, and
`--simulations` server flags remain available for command-line use.

### Model and action representation
- **Input:** 102 x 8 x 8 planes: eight historical positions with six
  current-player and six opponent piece planes each, four relative
  castling-right planes, an en-passant plane, and a normalized halfmove-clock
  plane. Black-to-move positions are rank-flipped so the side to move always
  advances toward rank eight; missing history at game start repeats the
  earliest available position.
- **Network:** 3x3 convolutional stem, six residual blocks of 96 channels by
  default, then separate policy and value heads. The policy predicts 73 planes
  per square (56 sliding moves, 8 knight moves, 9 underpromotions), or 4,672
  action logits. The value head predicts the game result from -1 to +1 for the
  player to move.
- **Search:** PUCT MCTS with 48 simulations per move by default, legal-move
  masking, and Dirichlet root noise during self-play. Search visit counts are
  the policy targets; terminal game results are the value targets.
- **Training:** AdamW minimizes policy cross-entropy plus value MSE from
  uniformly sampled positions in the bounded replay buffer. A candidate is
  retained as champion when it scores at least 55% in a small, alternating-
  color match against the previous champion. The default four-game match is
  only a quick filter; use 20-40 evaluation games before treating promotion
  scores as meaningful.

### Code map and data

- `az_chess.py` — canonical board features and reversible 73-plane actions.
- `az_network.py` — residual policy/value network.
- `az_mcts.py` — PUCT search and visit-target generation.
- `az_selfplay.py` — game generation and candidate/champion matches.
- `az_replay.py` — bounded replay memory and compressed sparse-policy files.
- `az_training.py` — policy/value optimization.
- `train.py` — iteration loop, checkpoints, and command-line settings.

The training dataset is generated by self-play, not downloaded. Replay data
stores compact board tensors, sparse legal-action visit distributions, and
side-to-move outcomes. It is capped at 50,000 positions by default; change
that with `--buffer-size`. No PGN corpus is required. A PGN warm-start or
larger-scale batched inference server would be a later extension, not a
hidden prerequisite.

### Laptop settings

For an 8 GB VRAM / 32 GB RAM laptop, start with the defaults (6 blocks x 96
channels, batch 128, replay 50k, 48 search simulations). The network and
training batches fit comfortably in 8 GB; self-play MCTS throughput, which
currently evaluates one leaf at a time, is the likely bottleneck. If GPU
memory or thermals are tight, lower `--batch-size` to 64 and `--channels` to
64; if games take too long, lower `--simulations` to 16-32 before shrinking
the network. Increase games per iteration and evaluation games only after
measuring end-to-end games/hour. The short smoke-test command above checks
the full pipeline but is not a meaningful strength run.

Useful monitoring includes positions/second, completed games/hour, policy and
value loss, replay size, promotion score, GPU memory, and temperature.
Checkpoint/replay files can grow over time, so preserve `champion.pt` and
`replay.npz` together when backing up a run.
