"""Parallel games preserve search rules and propagate worker failures."""
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

import train
from az_chess import ACTION_SIZE
from az_parallel import ParallelGames
from az_replay import ReplayBuffer, TrainingExample
from az_selfplay import evaluate_match, self_play_game


class UniformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        return (
            torch.zeros(len(inputs), ACTION_SIZE, device=inputs.device),
            torch.zeros(len(inputs), device=inputs.device),
        )


def test_parallel_games_match_independent_sequential_searches():
    model = UniformModel()
    with ParallelGames(model, model, workers=2, device="cpu") as pool:
        processes = list(pool.processes)
        actual = list(pool.self_play(
            games=4, simulations=4, max_plies=8, temperature_moves=3, seed=12,
        ))
        assert max(model.batch_sizes) == 2
        for index, (examples, ending) in enumerate(actual):
            endings = []
            expected = self_play_game(
                model, simulations=4, max_plies=8, temperature_moves=3,
                rng=random.Random(12 + index), on_game_end=endings.append,
            )
            assert ending == endings[0]
            assert len(examples) == len(expected)
            for first, second in zip(examples, expected):
                np.testing.assert_array_equal(first.state, second.state)
                np.testing.assert_array_equal(first.actions, second.actions)
                np.testing.assert_array_equal(first.probabilities, second.probabilities)
                assert first.outcome == second.outcome
        actual_endings, expected_endings = [], []
        actual_score = pool.evaluate(4, 4, 4, actual_endings.append)
        expected_score = evaluate_match(
            model, model, games=4, simulations=4, max_plies=4,
            on_game_end=expected_endings.append,
        )
        assert actual_score == expected_score
        assert actual_endings == expected_endings
    assert all(not process.is_alive() for process in processes)


def test_worker_failure_is_reported_and_workers_are_stopped():
    model = UniformModel()
    with pytest.raises(RuntimeError, match="simulations must be at least one"):
        with ParallelGames(model, model, workers=2, device="cpu") as pool:
            processes = list(pool.processes)
            list(pool.self_play(2, 0, 2, 60, 1))
    assert all(not process.is_alive() for process in processes)


def test_abrupt_worker_exit_does_not_hang():
    model = UniformModel()
    with ParallelGames(model, model, workers=2, device="cpu") as pool:
        pool.processes[0].terminate()
        pool.processes[0].join()
        with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
            list(pool.self_play(2, 1, 2, 60, 1))


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_invalid_worker_count_is_rejected(monkeypatch, workers):
    monkeypatch.setattr(sys, "argv", ["train.py", "--workers", workers])
    with pytest.raises(ValueError, match="workers"):
        train.main()


def test_parallel_training_saves_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--workers", "2", "--iterations", "2",
        "--games-per-iteration", "2", "--simulations", "2",
        "--eval-games", "4", "--eval-simulations", "2",
        "--max-plies", "2", "--train-steps", "2", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(tmp_path),
    ])
    assert train.main() == 0
    assert train.main() == 0
    log = capsys.readouterr().out
    assert "workers=2" in log
    assert "Resumed learner and optimizer from iteration 2" in log
    assert "iteration 4 evaluation game 4/4" in log
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert checkpoint["iteration"] == 4
    assert checkpoint["optimizer_state_dict"]
    assert len(ReplayBuffer.load(tmp_path / "replay.npz")) == 16


def test_replay_sampling_cache_is_invalidated_on_eviction():
    from az_chess import BOARD_FEATURES

    replay = ReplayBuffer(capacity=1)
    state = np.zeros((BOARD_FEATURES, 8, 8), dtype=np.float32)
    replay.add(TrainingExample(state, np.array([0]), np.array([1.0]), 1))
    assert replay.sample(1)[0].outcome == 1
    replay.add(TrainingExample(state, np.array([0]), np.array([1.0]), -1))
    assert replay.sample(1)[0].outcome == -1
