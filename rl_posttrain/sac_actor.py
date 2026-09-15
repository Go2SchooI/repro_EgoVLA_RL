"""Squashed Gaussian policy with an exactly transferable deterministic mean."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from rl_posttrain.actors import DeterministicActor


class SquashedGaussianActor(DeterministicActor):
    def __init__(self, *args, init_log_std=-5.8, log_std_min=-10.0,
                 log_std_max=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.parameterization != "direct_tanh":
            raise ValueError("SAC conversion requires a direct_tanh source actor")
        if not log_std_min <= init_log_std <= log_std_max:
            raise ValueError("Initial log std must lie within its bounds")
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.init_log_std = float(init_log_std)
        self.action_dim = self.net[-1].out_features
        self.log_std_head = nn.Linear(self.net[-1].in_features, self.action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def distribution_parameters(self, obs):
        features = self.net[:-1](self.obs_processor(obs))
        return self.net[-1](features), self.log_std_head(features).clamp(
            self.log_std_min, self.log_std_max)

    def sample(self, obs, *, generator=None):
        mean, log_std = self.distribution_parameters(obs)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                            generator=generator)
        pre_tanh = mean + log_std.exp() * noise
        action = pre_tanh.tanh()
        # Stable change-of-variables correction, including saturated actions.
        log_density = -0.5 * (noise.square() + 2 * log_std + math.log(2 * math.pi))
        correction = 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
        log_prob = (log_density - correction).sum(-1, keepdim=True)
        return action, log_prob, mean.tanh()

    def forward(self, obs):
        # Retain the original operation order for exact initial mean matching.
        return super().forward(obs)

    def policy_config(self):
        return dict(init_log_std=self.init_log_std, log_std_min=self.log_std_min,
                    log_std_max=self.log_std_max)
