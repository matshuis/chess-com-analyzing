#!/usr/bin/env python3
"""Train the compact AlphaZero-style chess engine with self-play."""
from __future__ import annotations

import argparse
import copy
import io
import hashlib
import json
import random
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import chess.pgn
import torch

from az_network import ChessNet
from az_benchmark import tactical_scorecard
from az_parallel import ParallelGames
from az_replay import ReplayBuffer
from az_selfplay import GameResult, evaluate_match, self_play_game
from az_training import train_network


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install")
    return selected


def _load_weights(
    model: ChessNet, path: Path, device: torch.device, checkpoint: dict | None = None,
) -> dict:
    if checkpoint is None:
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
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scaler_state_dict: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "iteration": iteration,
        "promoted": promoted,
        "channels": channels,
        "blocks": blocks,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None and scaler is not None:
        checkpoint.update({
            "learner_version": 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": (
                scaler.state_dict() if scaler_state_dict is None else scaler_state_dict
            ),
        })
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="CPU game workers sharing batched inference (default: 1, sequential)",
    )
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--eval-simulations", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument(
        "--temperature-moves", type=int, default=60,
        help="Self-play plies sampled at temperature 1 before deterministic moves "
             "(default: 60; 0 disables sampling; evaluation is unchanged)",
    )
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
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        help="Frozen checkpoint for independent evaluation (never promoted or modified)",
    )
    parser.add_argument(
        "--baseline-eval-games", type=int, default=0,
        help="Additional paired baseline games per iteration (default: 0, disabled)",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log training parameters, per-iteration metrics, and scorecards to MLflow",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (default: MLflow's local ./mlruns store)",
    )
    parser.add_argument(
        "--mlflow-experiment", default="chess-com-analyzing",
        help="MLflow experiment name (default: chess-com-analyzing)",
    )
    return parser


def _save_game(
    path: Path, result: GameResult, headers: dict[str, str],
) -> None:
    game = chess.pgn.read_game(io.StringIO(result.pgn))
    if game is None or game.errors:
        raise ValueError(f"Invalid game record for {path}")
    game.headers.update(headers)
    now = datetime.now(timezone.utc)
    game.headers["Date"] = now.strftime("%Y.%m.%d")
    game.headers["UTCTime"] = now.strftime("%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pgn.tmp")
    temporary.write_text(
        game.accept(chess.pgn.StringExporter(
            headers=True, variations=False, comments=True,
        )) + "\n\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    with ExitStack() as stack:
        return _main(stack)


def _main(stack: ExitStack) -> int:
    args = build_parser().parse_args()
    positive_values = {
        "iterations": args.iterations,
        "games-per-iteration": args.games_per_iteration,
        "workers": args.workers,
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
    if args.temperature_moves < 0:
        raise ValueError("temperature-moves must be non-negative")
    if args.eval_games % 2:
        raise ValueError("eval-games must be even for color-swapped opening pairs")
    if args.baseline_eval_games < 0 or args.baseline_eval_games % 2:
        raise ValueError("baseline-eval-games must be non-negative and even")
    if args.baseline_eval_games and args.baseline_checkpoint is None:
        raise ValueError("baseline-eval-games requires --baseline-checkpoint")
    if not 0.5 <= args.promotion_score <= 1:
        raise ValueError("promotion-score must be between 0.5 and 1")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    champion_path = args.output_dir / "champion.pt"
    latest_path = args.output_dir / "latest.pt"
    replay_path = args.output_dir / "replay.npz"
    if args.baseline_checkpoint is not None and args.baseline_checkpoint.resolve() in (
        champion_path.resolve(), latest_path.resolve(),
    ):
        raise ValueError("Frozen baseline must be a separate copy, not the active checkpoint")

    model = ChessNet(channels=args.channels, blocks=args.blocks).to(device)
    start_iteration = 0
    if champion_path.exists():
        checkpoint = _load_weights(model, champion_path, device)
        start_iteration = int(checkpoint["iteration"])
        print(f"Loaded champion from iteration {start_iteration}")
    incumbent = copy.deepcopy(model).eval()
    baseline = None
    baseline_hash = None
    if args.baseline_checkpoint is not None:
        baseline_hash = hashlib.sha256(args.baseline_checkpoint.read_bytes()).hexdigest()
        weights = torch.load(args.baseline_checkpoint, map_location=device, weights_only=True)
        baseline = ChessNet(weights["channels"], weights["blocks"]).to(device)
        baseline.load_state_dict(weights["state_dict"])
        baseline.eval()
        print(f"Loaded frozen baseline {args.baseline_checkpoint}; sha256={baseline_hash}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    saved_scaler = {}
    if latest_path.exists():
        learner_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if "learner_version" in learner_checkpoint:
            if learner_checkpoint["learner_version"] != 1:
                raise ValueError("Unsupported learner checkpoint version")
            if not champion_path.exists():
                raise ValueError("Learner checkpoint exists without champion.pt; restore both files")
            learner_iteration = int(learner_checkpoint["iteration"])
            if learner_iteration != start_iteration:
                raise ValueError("Learner and champion iterations differ; restore matching checkpoints")
            learner_checkpoint = _load_weights(
                model, latest_path, torch.device("cpu"), checkpoint=learner_checkpoint,
            )
            start_iteration = learner_iteration
            optimizer.load_state_dict(learner_checkpoint["optimizer_state_dict"])
            # Respect a requested learning rate while retaining Adam's accumulated moments.
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
            saved_scaler = learner_checkpoint["scaler_state_dict"]
            if scaler.is_enabled() and saved_scaler:
                scaler.load_state_dict(saved_scaler)
            elif scaler.is_enabled():
                print("CPU learner checkpoint has no AMP state; initializing CUDA scaler")
            elif saved_scaler:
                print("AMP disabled on CPU; preserving saved CUDA scaler state")
            print(f"Resumed learner and optimizer from iteration {start_iteration}")
        else:
            print("Legacy latest.pt has no learner state; initializing learner from champion")
    else:
        print("Initializing learner from champion" if champion_path.exists()
              else "Initializing learner and champion with random weights")
    if replay_path.exists():
        replay = ReplayBuffer.load(replay_path, capacity=args.buffer_size)
        print(f"Loaded {len(replay):,} replay positions")
    else:
        replay = ReplayBuffer(capacity=args.buffer_size)

    print(
        f"Training on {device}; network={args.blocks}x{args.channels}, "
        f"replay capacity={args.buffer_size:,}; workers={args.workers}"
    )
    mlflow = None
    if args.mlflow:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking was requested; install it with "
                "`uv sync --group tracking`"
            ) from exc
        mlflow = mlflow_module
        tracking_uri = args.mlflow_tracking_uri or "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        stack.enter_context(mlflow.start_run(
            run_name=f"{args.output_dir.name}-iter-{start_iteration + 1}"
        ))
        mlflow.log_params({
            "iterations": args.iterations,
            "start_iteration": start_iteration,
            "games_per_iteration": args.games_per_iteration,
            "workers": args.workers,
            "simulations": args.simulations,
            "eval_games": args.eval_games,
            "eval_simulations": args.eval_simulations,
            "max_plies": args.max_plies,
            "temperature_moves": args.temperature_moves,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "learning_rate": args.learning_rate,
            "channels": args.channels,
            "blocks": args.blocks,
            "promotion_score": args.promotion_score,
            "device": str(device),
            "seed": args.seed,
            "baseline_eval_games": args.baseline_eval_games,
            "baseline_checkpoint_sha256": baseline_hash or "none",
        })
        if args.baseline_checkpoint is not None:
            mlflow.log_artifact(str(args.baseline_checkpoint), artifact_path="baseline")
        for path in (champion_path, latest_path, replay_path):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="initial_state")
        print(
            f"MLflow tracking enabled: experiment={args.mlflow_experiment!r}; "
            f"tracking_uri={tracking_uri}"
        )
    pool = (
        stack.enter_context(ParallelGames(model, incumbent, args.workers, device))
        if args.workers > 1 else None
    )
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        if baseline is not None:
            current_hash = hashlib.sha256(args.baseline_checkpoint.read_bytes()).hexdigest()
            if current_hash != baseline_hash:
                raise ValueError("Frozen baseline file changed during training")
        games_dir = args.output_dir / "games" / f"iteration-{iteration:06d}"
        print(f"Saving game records to {games_dir}")
        rng = random.Random(args.seed + iteration)
        self_play_results: list[GameResult] = []
        parallel_results = (
            pool.self_play(
                args.games_per_iteration, args.simulations, args.max_plies,
                args.temperature_moves, (args.seed + iteration) * args.games_per_iteration,
            ) if pool is not None else iter(())
        )
        for game_number in range(args.games_per_iteration):
            if pool is None:
                examples = self_play_game(
                    model,
                    simulations=args.simulations,
                    max_plies=args.max_plies,
                    device=device,
                    temperature_moves=args.temperature_moves,
                    rng=rng,
                    on_game_end=self_play_results.append,
                )
            else:
                examples, ending = next(parallel_results)
                self_play_results.append(ending)
            for example in examples:
                replay.add(example)
            _save_game(
                games_dir / f"selfplay-{game_number + 1:03d}.pgn",
                self_play_results[-1],
                {
                    "Event": "Learner self-play", "White": "Learner", "Black": "Learner",
                    "Round": f"{iteration}.{game_number + 1}",
                    "Iteration": str(iteration), "Phase": "selfplay",
                    "Simulations": str(args.simulations),
                    "TemperatureMoves": str(args.temperature_moves),
                },
            )
            print(
                f"iteration {iteration} game {game_number + 1}/"
                f"{args.games_per_iteration}: {len(examples)} positions; "
                f"replay={len(replay):,}; result={self_play_results[-1].result}; "
                f"reason={self_play_results[-1].reason}"
            )
        print(
            f"iteration {iteration} self-play summary: "
            f"white_wins={sum(game.white_score > 0 for game in self_play_results)}, "
            f"black_wins={sum(game.white_score < 0 for game in self_play_results)}, "
            f"draws={sum(game.white_score == 0 for game in self_play_results)} "
            f"(move_cap={sum(game.move_cap for game in self_play_results)})"
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
            optimizer=optimizer,
            scaler=scaler,
        )
        evaluation_results: list[float] = []
        evaluation_move_caps = 0

        def log_evaluation(game: GameResult) -> None:
            nonlocal evaluation_move_caps
            candidate_is_white = len(evaluation_results) % 2 == 0
            candidate_result = game.white_score if candidate_is_white else -game.white_score
            evaluation_results.append(candidate_result)
            _save_game(
                games_dir / f"evaluation-{len(evaluation_results):03d}.pgn", game,
                {
                    "Event": "Candidate vs champion",
                    "White": "Candidate" if candidate_is_white else "Champion",
                    "Black": "Champion" if candidate_is_white else "Candidate",
                    "Round": f"{iteration}.{len(evaluation_results)}",
                    "Iteration": str(iteration), "Phase": "evaluation",
                    "Simulations": str(args.eval_simulations),
                },
            )
            evaluation_move_caps += int(game.move_cap)
            verdict = "win" if candidate_result > 0 else "loss" if candidate_result < 0 else "draw"
            print(
                f"iteration {iteration} evaluation game {len(evaluation_results)}/"
                f"{args.eval_games}: candidate={verdict}; "
                f"color={'white' if candidate_is_white else 'black'}; "
                f"plies={game.plies}; result={game.result}; reason={game.reason}"
                f"; opening={' '.join(game.opening) or 'start position'}"
            )

        if pool is None:
            score = evaluate_match(
                model,
                incumbent,
                games=args.eval_games,
                simulations=args.eval_simulations,
                max_plies=args.max_plies,
                device=device,
                on_game_end=log_evaluation,
            )
        else:
            score = pool.evaluate(
                args.eval_games, args.eval_simulations, args.max_plies, log_evaluation,
            )
        print(
            f"iteration {iteration} evaluation summary: "
            f"wins={sum(result > 0 for result in evaluation_results)}, "
            f"draws={sum(result == 0 for result in evaluation_results)}, "
            f"losses={sum(result < 0 for result in evaluation_results)}; "
            f"move_cap={evaluation_move_caps}; candidate_score={score:.3f}; "
            f"promotion_threshold={args.promotion_score:.3f}"
        )
        tactics = tactical_scorecard(model, args.eval_simulations, device)
        print(f"iteration {iteration} tactics: {tactics['passed']}/{tactics['total']}")
        baseline_score = None
        baseline_results: list[GameResult] = []
        if args.baseline_eval_games and baseline is not None:
            def log_baseline(game: GameResult) -> None:
                baseline_results.append(game)
                number = len(baseline_results)
                _save_game(
                    games_dir / f"baseline-{number:03d}.pgn", game,
                    {
                        "Event": "Candidate vs frozen baseline", "Phase": "baseline",
                        "White": "Candidate" if number % 2 else "Frozen baseline",
                        "Black": "Frozen baseline" if number % 2 else "Candidate",
                        "Iteration": str(iteration), "Round": f"{iteration}.{number}",
                        "Simulations": str(args.eval_simulations),
                        "BaselineSHA256": str(baseline_hash),
                    },
                )

            if pool is None:
                baseline_score = evaluate_match(
                    model, baseline, games=args.baseline_eval_games,
                    simulations=args.eval_simulations, max_plies=args.max_plies,
                    device=device, on_game_end=log_baseline,
                )
            else:
                original_models = pool.models
                pool.models = (model, baseline)
                try:
                    baseline_score = pool.evaluate(
                        args.baseline_eval_games, args.eval_simulations,
                        args.max_plies, log_baseline,
                    )
                finally:
                    pool.models = original_models
            print(f"iteration {iteration} frozen_baseline_score={baseline_score:.3f}")
        report = {
            "iteration": iteration, "draw_rules": "optional-claims-v1",
            "candidate_score": score, "tactics": tactics,
            "baseline": {
                "checkpoint": str(args.baseline_checkpoint) if baseline is not None else None,
                "sha256": baseline_hash, "score": baseline_score,
                "games": len(baseline_results),
                "results": [
                    {
                        "candidate_result": game.white_score if index % 2 == 0 else -game.white_score,
                        "reason": game.reason,
                    }
                    for index, game in enumerate(baseline_results)
                ],
            },
        }
        report_path = games_dir / "scorecard.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary_report.replace(report_path)
        promoted = score >= args.promotion_score
        if promoted:
            incumbent.load_state_dict(model.state_dict())
            print(f"Promoted candidate (score {score:.3f})")
        else:
            print(f"Kept incumbent (candidate score {score:.3f}); learner updates retained")

        _save_checkpoint(
            champion_path, incumbent, iteration, promoted, args.channels, args.blocks
        )
        if scaler.is_enabled():
            saved_scaler = scaler.state_dict()
        _save_checkpoint(
            latest_path, model, iteration, promoted, args.channels, args.blocks,
            optimizer=optimizer, scaler=scaler, scaler_state_dict=saved_scaler,
        )
        print(f"Saved learner weights and optimizer to {latest_path}")
        replay.save(replay_path)
        if mlflow is not None:
            metrics = {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "train/replay_positions": len(replay),
                "train/self_play_games": len(self_play_results),
                "train/self_play_white_wins": sum(
                    game.white_score > 0 for game in self_play_results
                ),
                "train/self_play_black_wins": sum(
                    game.white_score < 0 for game in self_play_results
                ),
                "train/self_play_draws": sum(
                    game.white_score == 0 for game in self_play_results
                ),
                "evaluation/candidate_score": score,
                "evaluation/wins": sum(result > 0 for result in evaluation_results),
                "evaluation/draws": sum(result == 0 for result in evaluation_results),
                "evaluation/losses": sum(result < 0 for result in evaluation_results),
                "evaluation/move_cap_games": evaluation_move_caps,
                "evaluation/promoted": int(promoted),
                "tactics/passed": tactics["passed"],
                "tactics/total": tactics["total"],
                "tactics/policy_passed": tactics["policy_passed"],
            }
            if baseline_score is not None:
                metrics.update({
                    "baseline/candidate_score": baseline_score,
                    "baseline/wins": sum(
                        (game.white_score if index % 2 == 0 else -game.white_score) > 0
                        for index, game in enumerate(baseline_results)
                    ),
                    "baseline/draws": sum(
                        game.white_score == 0 for game in baseline_results
                    ),
                    "baseline/losses": sum(
                        (game.white_score if index % 2 == 0 else -game.white_score) < 0
                        for index, game in enumerate(baseline_results)
                    ),
                })
            mlflow.log_metrics(metrics, step=iteration)
            mlflow.log_artifact(
                str(report_path),
                artifact_path=f"iterations/iteration-{iteration:06d}",
            )
            for path in (champion_path, latest_path, replay_path):
                mlflow.log_artifact(
                    str(path),
                    artifact_path=f"iterations/iteration-{iteration:06d}/checkpoints",
                )
        print(
            f"iteration {iteration}: policy_loss={policy_loss:.4f}, "
            f"value_loss={value_loss:.4f}, candidate_score={score:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
