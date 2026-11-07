"""Optimization step helpers for the policy/value network."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from az_network import ChessNet
from az_replay import ReplayBuffer


def train_network(
    model: ChessNet,
    replay: ReplayBuffer,
    steps: int,
    batch_size: int,
    device: torch.device | str,
    learning_rate: float = 1e-3,
    rng: random.Random | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if len(replay) < batch_size:
        raise ValueError(
            f"Need at least {batch_size} replay examples, found {len(replay)}"
        )

    target_device = torch.device(device)
    model.to(target_device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
    if scaler is None:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=target_device.type == "cuda"
        )
    totals = torch.zeros(2, device=target_device, dtype=torch.float64)

    for _ in range(steps):
        states, policy_targets, outcomes = replay.batch(batch_size, rng)
        inputs = torch.from_numpy(states).to(target_device)
        target_policy = torch.from_numpy(policy_targets).to(target_device)
        target_value = torch.from_numpy(outcomes).to(target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=target_device.type == "cuda",
        ):
            logits, values = model(inputs)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, target_value)
            loss = policy_loss + value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals += torch.stack((policy_loss.detach(), value_loss.detach())).to(torch.float64)

    model.eval()
    policy_mean, value_mean = (totals / steps).tolist()
    return policy_mean, value_mean
