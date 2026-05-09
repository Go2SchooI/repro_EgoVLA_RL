from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, output_activation=None) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError(f"MLP dimensions must be positive, got input={input_dim}, output={output_dim}.")
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dims must be positive, got {hidden_dim}.")
        layers.extend([nn.Linear(last_dim, int(hidden_dim)), nn.ReLU()])
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """TD3+BC actor: deterministic tanh policy, no log_std/log_prob/SAC bits."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (1024, 1024, 512)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = build_mlp(self.obs_dim, hidden_dims, self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Actor expected obs dim {self.obs_dim}, got {obs.shape}.")
        return torch.tanh(self.net(obs))

