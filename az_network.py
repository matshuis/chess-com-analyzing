"""Compact residual policy/value network."""
from __future__ import annotations

import torch
from torch import nn

from az_chess import ACTION_SIZE, BOARD_FEATURES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ChessNet(nn.Module):
    """Predict policy logits for 4,672 actions and a side-to-move value."""

    def __init__(self, channels: int = 96, blocks: int = 6) -> None:
        super().__init__()
        if channels < 1 or blocks < 1:
            raise ValueError("channels and blocks must both be positive")
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_FEATURES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(self.stem(inputs))
        policy = self.policy_fc(torch.flatten(self.policy_conv(trunk), start_dim=1))
        value = self.value_fc(torch.flatten(self.value_conv(trunk), start_dim=1))
        return policy, value.squeeze(-1)
