"""Bounded replay storage with sparse policy targets."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from az_chess import ACTION_SIZE, BOARD_FEATURES


@dataclass(frozen=True)
class TrainingExample:
    state: np.ndarray
    actions: np.ndarray
    probabilities: np.ndarray
    outcome: float


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._examples: deque[TrainingExample] = deque(maxlen=capacity)
        self._sample_pool: list[TrainingExample] | None = None

    def __len__(self) -> int:
        return len(self._examples)

    def add(self, example: TrainingExample) -> None:
        state = np.asarray(example.state, dtype=np.float16)
        actions = np.asarray(example.actions, dtype=np.int16)
        probabilities = np.asarray(example.probabilities, dtype=np.float32)
        if state.shape != (BOARD_FEATURES, 8, 8):
            raise ValueError(f"Invalid state shape: {state.shape}")
        if actions.ndim != 1 or probabilities.shape != actions.shape or not len(actions):
            raise ValueError("Policy action IDs and probabilities must be non-empty vectors")
        if np.any(actions < 0) or np.any(actions >= ACTION_SIZE):
            raise ValueError("Policy contains an out-of-range action")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Policy probabilities must be finite and non-negative")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("Policy probabilities must have positive mass")
        if not np.isfinite(example.outcome) or not -1 <= example.outcome <= 1:
            raise ValueError("Outcome must be finite and between -1 and 1")
        self._examples.append(
            TrainingExample(
                state.copy(),
                actions.copy(),
                probabilities / total,
                float(example.outcome),
            )
        )
        self._sample_pool = None

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[TrainingExample]:
        if not 1 <= batch_size <= len(self):
            raise ValueError(f"batch_size must be between 1 and {len(self)}")
        if self._sample_pool is None:
            self._sample_pool = list(self._examples)
        return (rng or random).sample(self._sample_pool, batch_size)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        examples = list(self._examples)
        if examples:
            states = np.stack([example.state for example in examples])
            outcomes = np.asarray([example.outcome for example in examples], dtype=np.float32)
            actions = np.concatenate([example.actions for example in examples])
            probabilities = np.concatenate(
                [example.probabilities for example in examples]
            ).astype(np.float32)
            offsets = np.zeros(len(examples) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(example.actions) for example in examples])
        else:
            states = np.empty((0, BOARD_FEATURES, 8, 8), dtype=np.float16)
            outcomes = np.empty((0,), dtype=np.float32)
            actions = np.empty((0,), dtype=np.int16)
            probabilities = np.empty((0,), dtype=np.float32)
            offsets = np.zeros(1, dtype=np.int64)
        np.savez_compressed(
            target,
            states=states,
            outcomes=outcomes,
            actions=actions,
            probabilities=probabilities,
            offsets=offsets,
        )

    @classmethod
    def load(cls, path: str | Path, capacity: int = 50_000) -> "ReplayBuffer":
        buffer = cls(capacity)
        with np.load(path, allow_pickle=False) as data:
            states = data["states"]
            outcomes = data["outcomes"]
            actions = data["actions"]
            probabilities = data["probabilities"]
            offsets = data["offsets"]
            if states.ndim != 4 or states.shape[1:] != (BOARD_FEATURES, 8, 8):
                raise ValueError("Replay file has an invalid state tensor shape")
            count = len(states)
            if outcomes.shape != (count,) or offsets.shape != (count + 1,):
                raise ValueError("Replay file has inconsistent example metadata")
            if (offsets[0] != 0 or offsets[-1] != len(actions)
                    or len(actions) != len(probabilities)
                    or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("Replay file has invalid sparse policy offsets")
            entries = [
                TrainingExample(
                    states[i],
                    actions[offsets[i]:offsets[i + 1]],
                    probabilities[offsets[i]:offsets[i + 1]],
                    float(outcomes[i]),
                )
                for i in range(count)
            ]
        for example in entries[-capacity:]:
            buffer.add(example)
        return buffer

    def batch(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        examples = self.sample(batch_size, rng)
        states = np.stack([example.state for example in examples]).astype(np.float32)
        targets = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        for row, example in enumerate(examples):
            targets[row, example.actions] = example.probabilities
        outcomes = np.asarray([example.outcome for example in examples], dtype=np.float32)
        return states, targets, outcomes
