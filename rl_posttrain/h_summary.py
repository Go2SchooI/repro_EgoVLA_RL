from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass(frozen=True)
class HSummaryConfig:
    mode: str = "full_h"
    h_dim: int = 1536
    out_dim: int | None = None
    trainable: bool = True
    layernorm: bool = True
    requested_mode: str | None = None

    def __post_init__(self) -> None:
        requested_mode = str(self.requested_mode or self.mode)
        mode = str(self.mode)
        out_dim = self.out_dim
        if mode == "h_proj256":
            mode = "h_proj"
            out_dim = 256
        elif mode == "h_proj128":
            mode = "h_proj"
            out_dim = 128
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "out_dim", out_dim)
        object.__setattr__(self, "requested_mode", requested_mode)
        if mode not in ("full_h", "h_zero", "h_proj"):
            raise ValueError(
                "h_summary mode must be one of full_h, h_zero, h_proj, h_proj256, h_proj128; "
                f"got {self.mode!r}."
            )
        if int(self.h_dim) < 0:
            raise ValueError(f"h_dim must be >= 0, got {self.h_dim}.")
        if mode == "h_proj":
            if out_dim is None or int(out_dim) <= 0:
                raise ValueError("h_proj requires a positive out_dim.")
            if int(self.h_dim) <= 0:
                raise ValueError("h_proj requires h_dim > 0.")

    @property
    def is_projection(self) -> bool:
        return self.mode == "h_proj"

    def processed_h_dim(self) -> int:
        if self.mode == "h_proj":
            assert self.out_dim is not None
            return int(self.out_dim)
        return int(self.h_dim)

    def processed_obs_dim(self, raw_obs_dim: int) -> int:
        raw_obs_dim = int(raw_obs_dim)
        if self.mode == "full_h" or int(self.h_dim) == 0:
            return raw_obs_dim
        if int(self.h_dim) > raw_obs_dim:
            raise ValueError(f"h_dim={self.h_dim} exceeds raw obs dim={raw_obs_dim}.")
        if self.mode == "h_zero":
            return raw_obs_dim
        return raw_obs_dim - int(self.h_dim) + self.processed_h_dim()

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any] | None) -> "HSummaryConfig":
        if not state:
            return cls()
        return cls(
            mode=str(state.get("mode", "full_h")),
            h_dim=int(state.get("h_dim", 1536)),
            out_dim=None if state.get("out_dim") is None else int(state.get("out_dim")),
            trainable=bool(state.get("trainable", True)),
            layernorm=bool(state.get("layernorm", True)),
            requested_mode=None if state.get("requested_mode") is None else str(state.get("requested_mode")),
        )


class HSummaryProcessor(nn.Module):
    def __init__(self, cfg: HSummaryConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "h_proj":
            hidden_dim = 512
            layers: list[nn.Module] = []
            if cfg.layernorm:
                layers.append(nn.LayerNorm(int(cfg.h_dim)))
            layers.extend(
                [
                    nn.Linear(int(cfg.h_dim), hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, int(cfg.out_dim)),
                ]
            )
            if cfg.layernorm:
                layers.append(nn.LayerNorm(int(cfg.out_dim)))
            self.net = nn.Sequential(*layers)
            if not cfg.trainable:
                for param in self.net.parameters():
                    param.requires_grad_(False)
        else:
            self.net = None

    @property
    def has_trainable_parameters(self) -> bool:
        return any(param.requires_grad for param in self.parameters())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode == "full_h":
            return h
        if self.cfg.mode == "h_zero":
            return torch.zeros_like(h)
        if self.net is None:
            raise AssertionError("h_proj expected a projection network.")
        return self.net(h)


class HObsProcessor(nn.Module):
    """Process the h_summary prefix inside an already-normalized observation."""

    def __init__(self, raw_obs_dim: int, cfg: HSummaryConfig):
        super().__init__()
        self.raw_obs_dim = int(raw_obs_dim)
        self.cfg = cfg
        self.processed_obs_dim = cfg.processed_obs_dim(self.raw_obs_dim)
        self.h_processor = HSummaryProcessor(cfg)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.raw_obs_dim:
            raise ValueError(f"HObsProcessor expected raw obs dim {self.raw_obs_dim}, got {obs.shape}.")
        if self.cfg.mode == "full_h" or int(self.cfg.h_dim) == 0:
            return obs
        h_dim = int(self.cfg.h_dim)
        h = obs[..., :h_dim]
        rest = obs[..., h_dim:]
        h_processed = self.h_processor(h)
        return torch.cat([h_processed, rest], dim=-1)


def module_param_norm(module: nn.Module) -> float:
    total = torch.zeros((), device=next(module.parameters(), torch.zeros((), device="cpu")).device)
    has_param = False
    with torch.no_grad():
        for param in module.parameters():
            has_param = True
            total = total + param.detach().pow(2).sum()
    if not has_param:
        return 0.0
    return float(total.sqrt().detach().cpu())


def module_grad_norm(module: nn.Module) -> float:
    total = None
    for param in module.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().detach().cpu())


def module_num_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for param in module.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total
