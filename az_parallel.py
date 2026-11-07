"""CPU game workers sharing batched inference in the trainer process."""
from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait

import numpy as np
import torch
from torch import nn

from az_replay import TrainingExample
from az_selfplay import EVALUATION_OPENINGS, GameResult, evaluate_match, self_play_game


@dataclass(frozen=True)
class _Job:
    index: int
    evaluation: bool
    simulations: int
    max_plies: int
    temperature_moves: int = 60
    seed: int = 1
    opening: tuple[str, ...] = ()


@dataclass
class _Result:
    index: int
    examples: list[TrainingExample]
    endings: list[GameResult]
    score: float = 0.0


class _RemoteModel(nn.Module):
    def __init__(self, connection: Connection, model_index: int) -> None:
        super().__init__()
        self.connection = connection
        self.model_index = model_index

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.connection.send(("predict", self.model_index, inputs.numpy()))
        logits, values = self.connection.recv()
        return torch.from_numpy(logits), torch.from_numpy(values)


def _worker(connection: Connection) -> None:
    torch.set_num_threads(1)
    models = [_RemoteModel(connection, index) for index in range(2)]
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            endings: list[GameResult] = []
            if job.evaluation:
                score = evaluate_match(
                    models[0], models[1], games=2, simulations=job.simulations,
                    max_plies=job.max_plies, openings=(job.opening,),
                    on_game_end=endings.append,
                )
                result = _Result(job.index, [], endings, score)
            else:
                examples = self_play_game(
                    models[0], simulations=job.simulations, max_plies=job.max_plies,
                    temperature_moves=job.temperature_moves,
                    rng=random.Random(job.seed), on_game_end=endings.append,
                )
                result = _Result(job.index, examples, endings)
            connection.send(("result", result))
    except Exception:
        # Propagate worker failures with their original traceback; never lose a game.
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class ParallelGames:
    """Keep CPU workers alive between iterations; GPU weights stay in the parent."""

    def __init__(
        self, candidate: nn.Module, incumbent: nn.Module,
        workers: int, device: torch.device | str,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.models = (candidate, incumbent)
        self.device = torch.device(device)
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        context = mp.get_context("spawn")
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child,))
                self.connections.append(parent)
                process.start()
                self.processes.append(process)
                child.close()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ParallelGames:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.processes.clear()

    def _run(self, jobs: list[_Job]) -> Iterator[_Result]:
        if not self.processes or any(not process.is_alive() for process in self.processes):
            raise RuntimeError("A game worker exited unexpectedly")
        for model in self.models:
            model.eval()
        pending = iter(jobs)
        active: dict[Connection, int] = {}
        completed: dict[int, _Result] = {}
        next_result = 0
        for connection in self.connections:
            job = next(pending, None)
            if job is None:
                break
            connection.send(job)
            active[connection] = job.index

        while active:
            # Include sentinels so an abruptly exited worker cannot hang the trainer.
            ready = wait(
                list(active) + [process.sentinel for process in self.processes]
            )
            if any(process.sentinel in ready for process in self.processes):
                raise RuntimeError("A game worker exited unexpectedly")
            others = [connection for connection in active if connection not in ready]
            if others:
                ready.extend(wait(others, timeout=0.002))
            requests: list[tuple[Connection, int, np.ndarray]] = []
            for connection in list(active):
                if connection not in ready:
                    continue
                message = connection.recv()
                if message[0] == "error":
                    raise RuntimeError(f"Game worker failed:\n{message[1]}")
                if message[0] == "predict":
                    requests.append((connection, message[1], message[2]))
                elif message[0] == "result":
                    result = message[1]
                    completed[result.index] = result
                    job = next(pending, None)
                    if job is None:
                        del active[connection]
                    else:
                        connection.send(job)
                        active[connection] = job.index
                else:
                    raise RuntimeError(f"Unexpected worker message: {message[0]}")

            if requests:
                self._predict(requests)

            while next_result in completed:
                yield completed.pop(next_result)
                next_result += 1

    def _predict(self, requests: list[tuple[Connection, int, np.ndarray]]) -> None:
        for index, model in enumerate(self.models):
            batch = [request for request in requests if request[1] == index]
            if not batch:
                continue
            inputs = torch.from_numpy(np.concatenate([request[2] for request in batch]))
            with torch.inference_mode():
                logits, values = model(inputs.to(self.device))
            policies = logits.float().cpu().numpy()
            estimates = values.float().cpu().numpy()
            for row, (connection, _, _) in enumerate(batch):
                connection.send((policies[row:row + 1], estimates[row:row + 1]))

    def self_play(
        self, games: int, simulations: int, max_plies: int,
        temperature_moves: int, seed: int,
    ) -> Iterator[tuple[list[TrainingExample], GameResult]]:
        if games < 1:
            raise ValueError("games must be positive")
        jobs = [
            _Job(index, False, simulations, max_plies, temperature_moves, seed + index)
            for index in range(games)
        ]
        for result in self._run(jobs):
            yield result.examples, result.endings[0]

    def evaluate(
        self, games: int, simulations: int, max_plies: int,
        on_game_end: Callable[[GameResult], None],
    ) -> float:
        if games < 2 or games % 2:
            raise ValueError("games must be a positive even number")
        jobs = [
            _Job(
                index, True, simulations, max_plies,
                opening=EVALUATION_OPENINGS[index % len(EVALUATION_OPENINGS)],
            )
            for index in range(games // 2)
        ]
        score = 0.0
        for result in self._run(jobs):
            for ending in result.endings:
                on_game_end(ending)
            score += result.score
        return score / len(jobs)
