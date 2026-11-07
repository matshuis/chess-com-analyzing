#!/usr/bin/env python3
"""Train the compact AlphaZero-style chess engine with self-play."""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from az_network import ChessNet
from az_replay import ReplayBuffer
from az_selfplay import evaluate_match, self_play_game
from az_training import train_network


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install")
    return selected


def _load_weights(model: ChessNet, path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["channels"] != model.channels or checkpoint["blocks"] != model.blocks:
        raise ValueError(
            "Checkpoint architecture does not match --channels/--blocks"
        )
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def _save_checkpoint(
    path: Path,
    model: ChessNet,
    iteration: int,
    promoted: bool,
    channels: int,
    blocks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "promoted": promoted,
            "channels": channels,
            "blocks": blocks,
            "state_dict": model.state_dict(),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--eval-simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--promotion-score", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/az"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    positive_values = {
        "iterations": args.iterations,
        "games-per-iteration": args.games_per_iteration,
        "simulations": args.simulations,
        "eval-games": args.eval_games,
        "eval-simulations": args.eval_simulations,
        "max-plies": args.max_plies,
        "train-steps": args.train_steps,
        "batch-size": args.batch_size,
        "buffer-size": args.buffer_size,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
    if not 0.5 <= args.promotion_score <= 1:
        raise ValueError("promotion-score must be between 0.5 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    champion_path = args.output_dir / "champion.pt"
    latest_path = args.output_dir / "latest.pt"
    replay_path = args.output_dir / "replay.npz"

    model = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    start_iteration = 0
    if champion_path.exists():
        checkpoint = _load_weights(model, champion_path, device)
        start_iteration = int(checkpoint["iteration"])
        print(f"Loaded champion from iteration {start_iteration}")
    incumbent = copy.deepcopy(model).eval()
    if replay_path.exists():
        replay = ReplayBuffer.load(replay_path, capacity=args.buffer_size)
        print(f"Loaded {len(replay):,} replay positions")
    else:
        replay = ReplayBuffer(capacity=args.buffer_size)

    print(
        f"Training on {device}; network={args.blocks}x{args.channels}, "
        f"replay capacity={args.buffer_size:,}"
    )
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        rng = random.Random(args.seed + iteration)
        for game_number in range(args.games_per_iteration):
            examples = self_play_game(
                model,
                simulations=args.simulations,
                max_plies=args.max_plies,
                device=device,
                rng=rng,
            )
            for example in examples:
                replay.add(example)
            print(
                f"iteration {iteration} game {game_number + 1}/"
                f"{args.games_per_iteration}: {len(examples)} positions; "
                f"replay={len(replay):,}"
            )

        effective_batch = min(args.batch_size, len(replay))
        policy_loss, value_loss = train_network(
            model,
            replay,
            steps=args.train_steps,
            batch_size=effective_batch,
            device=device,
            learning_rate=args.learning_rate,
            rng=rng,
        )
        score = evaluate_match(
            model,
            incumbent,
            games=args.eval_games,
            simulations=args.eval_simulations,
            max_plies=args.max_plies,
            device=device,
        )
        promoted = score >= args.promotion_score
        if promoted:
            incumbent.load_state_dict(model.state_dict())
            print(f"Promoted candidate (score {score:.3f})")
        else:
            model.load_state_dict(incumbent.state_dict())
            print(f"Kept incumbent (candidate score {score:.3f})")

        _save_checkpoint(
            champion_path, incumbent, iteration, promoted, args.channels, args.blocks
        )
        _save_checkpoint(
            latest_path, model, iteration, promoted, args.channels, args.blocks
        )
        replay.save(replay_path)
        print(
            f"iteration {iteration}: policy_loss={policy_loss:.4f}, "
            f"value_loss={value_loss:.4f}, candidate_score={score:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
