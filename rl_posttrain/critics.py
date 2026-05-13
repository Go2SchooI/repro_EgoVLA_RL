from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from rl_posttrain.actors import build_mlp
from rl_posttrain.h_summary import HObsProcessor, HSummaryConfig


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (1024, 1024, 512),
        h_summary: HSummaryConfig | None = None,
        actor_obs_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.actor_obs_dim = int(actor_obs_dim) if actor_obs_dim is not None else self.obs_dim
        if self.actor_obs_dim <= 0 or self.actor_obs_dim > self.obs_dim:
            raise ValueError(
                f"actor_obs_dim must be in [1, obs_dim], got actor_obs_dim={self.actor_obs_dim}, "
                f"obs_dim={self.obs_dim}."
            )
        self.h_summary = h_summary or HSummaryConfig(mode="full_h")
        self.q1_obs_processor = HObsProcessor(self.actor_obs_dim, self.h_summary)
        self.q2_obs_processor = HObsProcessor(self.actor_obs_dim, self.h_summary)
        self.processed_obs_dim = (
            int(self.q1_obs_processor.processed_obs_dim) + self.obs_dim - self.actor_obs_dim
        )
        input_dim = self.processed_obs_dim + self.action_dim
        self.q1 = build_mlp(input_dim, hidden_dims, 1)
        self.q2 = build_mlp(input_dim, hidden_dims, 1)

    def _process_obs_for_q(self, obs: torch.Tensor, processor: HObsProcessor) -> torch.Tensor:
        actor_obs = obs[..., : self.actor_obs_dim]
        env_state = obs[..., self.actor_obs_dim :]
        actor_obs = processor(actor_obs)
        if env_state.shape[-1] == 0:
            return actor_obs
        return torch.cat([actor_obs, env_state], dim=-1)

    def _concat(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Critic expected obs dim {self.obs_dim}, got {obs.shape}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Critic expected action dim {self.action_dim}, got {action.shape}.")
        obs = self._process_obs_for_q(obs, self.q1_obs_processor)
        return torch.cat([obs, action], dim=-1)

    def _concat_q2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Critic expected obs dim {self.obs_dim}, got {obs.shape}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Critic expected action dim {self.action_dim}, got {action.shape}.")
        obs = self._process_obs_for_q(obs, self.q2_obs_processor)
        return torch.cat([obs, action], dim=-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(self._concat(obs, action)), self.q2(self._concat_q2(obs, action))

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(self._concat(obs, action))
