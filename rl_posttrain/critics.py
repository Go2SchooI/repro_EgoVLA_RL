from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from rl_posttrain.actors import build_mlp


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (1024, 1024, 512)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        input_dim = self.obs_dim + self.action_dim
        self.q1 = build_mlp(input_dim, hidden_dims, 1)
        self.q2 = build_mlp(input_dim, hidden_dims, 1)

    def _concat(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Critic expected obs dim {self.obs_dim}, got {obs.shape}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Critic expected action dim {self.action_dim}, got {action.shape}.")
        return torch.cat([obs, action], dim=-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._concat(obs, action)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(self._concat(obs, action))
