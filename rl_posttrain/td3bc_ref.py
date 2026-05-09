from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.critics import DoubleQCritic
from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.replay_buffer import OfflineReplayBuffer


@dataclass
class TD3BCConfig:
    actor_hidden_dims: tuple[int, ...] = (1024, 1024, 512)
    critic_hidden_dims: tuple[int, ...] = (1024, 1024, 512)
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    td3bc_alpha: float = 0.0
    td3bc_bc_weight: float = 1.0
    td3bc_use_q_abs_norm: bool = True
    lr_actor: float = 3.0e-4
    lr_critic: float = 3.0e-4
    batch_size: int = 256
    obs_normalize: bool = True
    action_norm_clip: float = 1.0


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


class PreparedReplay:
    def __init__(self, replay: OfflineReplayBuffer, cfg: TD3BCConfig):
        self.raw = replay
        if replay.action_normalizer is None:
            raise ValueError(
                "Replay is missing action_normalizer state. Re-collect or migrate the replay so "
                "action_norm/bc_target_norm are canonical normalized fields, not raw actions."
            )
        self.action_normalizer = replay.action_normalizer
        action_dim = int(replay.arrays["bc_target_norm"].shape[-1])
        actor_obs_raw = replay.arrays["actor_obs"].copy()
        next_actor_obs_raw = replay.arrays["next_actor_obs"].copy()
        critic_obs_raw = replay.arrays["critic_obs"].copy()
        next_critic_obs_raw = replay.arrays["next_critic_obs"].copy()
        actor_obs_dim = int(actor_obs_raw.shape[-1])
        if critic_obs_raw.shape[-1] < actor_obs_dim or next_critic_obs_raw.shape[-1] < actor_obs_dim:
            raise ValueError(
                f"Replay critic_obs dim must be at least actor_obs_dim={actor_obs_dim}: "
                f"critic_obs={critic_obs_raw.shape}, next_critic_obs={next_critic_obs_raw.shape}."
            )
        if actor_obs_raw.shape[-1] < action_dim or next_actor_obs_raw.shape[-1] < action_dim:
            raise ValueError(
                f"Replay actor_obs dim is too small for action_dim={action_dim}: "
                f"actor_obs={actor_obs_raw.shape}, next_actor_obs={next_actor_obs_raw.shape}."
            )
        actor_tail_error = float(np.max(np.abs(actor_obs_raw[:, -action_dim:] - replay.arrays["bc_target_norm"])))
        next_actor_tail_error = float(
            np.max(np.abs(next_actor_obs_raw[:, -action_dim:] - replay.arrays["next_bc_target_norm"]))
        )
        if actor_tail_error > 1.0e-5 or next_actor_tail_error > 1.0e-5:
            raise ValueError(
                "Replay actor_obs action tail is not in the same canonical normalized action space "
                f"as bc_target_norm: current={actor_tail_error:.8g} next={next_actor_tail_error:.8g}."
            )
        critic_prefix_error = float(np.max(np.abs(critic_obs_raw[:, :actor_obs_dim] - actor_obs_raw)))
        next_critic_prefix_error = float(
            np.max(np.abs(next_critic_obs_raw[:, :actor_obs_dim] - next_actor_obs_raw))
        )
        if critic_prefix_error > 1.0e-5 or next_critic_prefix_error > 1.0e-5:
            raise ValueError(
                "Replay critic_obs must start with actor_obs so actor/critic action context stays aligned: "
                f"current={critic_prefix_error:.8g} next={next_critic_prefix_error:.8g}."
            )

        if cfg.obs_normalize:
            self.actor_obs_normalizer = AffineNormalizer.fit_standard(actor_obs_raw)
            self.critic_obs_normalizer = AffineNormalizer.fit_standard(critic_obs_raw)
        else:
            self.actor_obs_normalizer = AffineNormalizer.identity(actor_obs_raw.shape[-1])
            self.critic_obs_normalizer = AffineNormalizer.identity(critic_obs_raw.shape[-1])

        self.arrays = dict(replay.arrays)
        self.arrays["actor_obs"] = self.actor_obs_normalizer.normalize(actor_obs_raw)
        self.arrays["critic_obs"] = self.critic_obs_normalizer.normalize(critic_obs_raw)
        self.arrays["next_actor_obs"] = self.actor_obs_normalizer.normalize(next_actor_obs_raw)
        self.arrays["next_critic_obs"] = self.critic_obs_normalizer.normalize(next_critic_obs_raw)
        self.arrays["action_norm"] = replay.arrays["action_norm"].astype(np.float32, copy=True)
        self.arrays["bc_target_norm"] = replay.arrays["bc_target_norm"].astype(np.float32, copy=True)
        self.arrays["next_bc_target_norm"] = replay.arrays["next_bc_target_norm"].astype(np.float32, copy=True)
        for key in ("action_norm", "bc_target_norm", "next_bc_target_norm"):
            if np.max(np.abs(self.arrays[key])) > cfg.action_norm_clip + 1.0e-5:
                raise AssertionError(f"{key} exceeded configured normalized range.")

    @property
    def size(self) -> int:
        return int(self.arrays["actor_obs"].shape[0])

    @property
    def actor_obs_dim(self) -> int:
        return int(self.arrays["actor_obs"].shape[-1])

    @property
    def critic_obs_dim(self) -> int:
        return int(self.arrays["critic_obs"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.arrays["action_norm"].shape[-1])

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "actor_obs": _to_tensor(self.arrays["actor_obs"][idx], device),
            "critic_obs": _to_tensor(self.arrays["critic_obs"][idx], device),
            "action_norm": _to_tensor(self.arrays["action_norm"][idx], device),
            "bc_target_norm": _to_tensor(self.arrays["bc_target_norm"][idx], device),
            "reward": _to_tensor(self.arrays["reward"][idx], device),
            "done": _to_tensor(self.arrays["done"][idx], device),
            "next_actor_obs": _to_tensor(self.arrays["next_actor_obs"][idx], device),
            "next_critic_obs": _to_tensor(self.arrays["next_critic_obs"][idx], device),
        }


class TD3BCTrainer:
    def __init__(
        self,
        prepared: PreparedReplay,
        cfg: TD3BCConfig,
        device: str | torch.device = "cuda",
        seed: int = 0,
    ):
        self.prepared = prepared
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.rng = np.random.default_rng(seed)

        self.actor = DeterministicActor(
            prepared.actor_obs_dim,
            prepared.action_dim,
            cfg.actor_hidden_dims,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = DoubleQCritic(
            prepared.critic_obs_dim,
            prepared.action_dim,
            cfg.critic_hidden_dims,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.total_it = 0

    def train_step(self) -> Dict[str, float]:
        self.total_it += 1
        batch = self.prepared.sample(self.cfg.batch_size, self.device, self.rng)

        with torch.no_grad():
            next_action = self.actor_target(batch["next_actor_obs"])
            noise = torch.randn_like(next_action) * self.cfg.target_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(batch["next_critic_obs"], next_action)
            target_q = torch.minimum(target_q1, target_q2)
            y = batch["reward"] + self.cfg.gamma * (1.0 - batch["done"]) * target_q

        current_q1, current_q2 = self.critic(batch["critic_obs"], batch["action_norm"])
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        logs = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_target_mean": float(y.detach().mean().cpu()),
            "target_min_q_mean": float(target_q.detach().mean().cpu()),
            "q1_mean": float(current_q1.detach().mean().cpu()),
            "q2_mean": float(current_q2.detach().mean().cpu()),
            "actor_loss": 0.0,
            "bc_loss": 0.0,
            "q_abs": 0.0,
            "lambda_q": 0.0,
            "mean_abs_actor_minus_ref_norm": 0.0,
            "max_abs_actor_minus_ref_norm": 0.0,
        }

        if self.total_it % self.cfg.policy_delay == 0:
            action_pi = self.actor(batch["actor_obs"])
            q_pi = self.critic.q1_value(batch["critic_obs"], action_pi)
            bc_loss = F.mse_loss(action_pi, batch["bc_target_norm"])
            q_abs = q_pi.abs().mean().detach()
            if self.cfg.td3bc_alpha == 0.0:
                lambda_q = torch.zeros((), device=self.device)
            elif self.cfg.td3bc_use_q_abs_norm:
                lambda_q = torch.as_tensor(self.cfg.td3bc_alpha, device=self.device) / q_abs.clamp_min(1.0e-6)
            else:
                lambda_q = torch.as_tensor(self.cfg.td3bc_alpha, device=self.device)
            actor_loss = -lambda_q * q_pi.mean() + self.cfg.td3bc_bc_weight * bc_loss
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            _soft_update(self.actor, self.actor_target, self.cfg.tau)
            _soft_update(self.critic, self.critic_target, self.cfg.tau)
            diff = (action_pi.detach() - batch["bc_target_norm"]).abs()
            logs.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "q_abs": float(q_abs.cpu()),
                    "lambda_q": float(lambda_q.detach().cpu()),
                    "mean_abs_actor_minus_ref_norm": float(diff.mean().cpu()),
                    "max_abs_actor_minus_ref_norm": float(diff.max().cpu()),
                }
            )

        return logs

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "td3bc_ref_actor_v1",
                "config": asdict(self.cfg),
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_obs_dim": self.prepared.actor_obs_dim,
                "critic_obs_dim": self.prepared.critic_obs_dim,
                "action_dim": self.prepared.action_dim,
                "actor_hidden_dims": tuple(self.cfg.actor_hidden_dims),
                "critic_hidden_dims": tuple(self.cfg.critic_hidden_dims),
                "action_normalizer": self.prepared.action_normalizer.state_dict(),
                "actor_obs_normalizer": self.prepared.actor_obs_normalizer.state_dict(),
                "critic_obs_normalizer": self.prepared.critic_obs_normalizer.state_dict(),
            },
            path,
        )
        return path


def load_actor_policy(path: str | Path, device: str | torch.device = "cuda"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != "td3bc_ref_actor_v1":
        raise ValueError(f"Unsupported actor checkpoint format: {checkpoint.get('format')!r}")
    actor = DeterministicActor(
        int(checkpoint["actor_obs_dim"]),
        int(checkpoint["action_dim"]),
        tuple(checkpoint["actor_hidden_dims"]),
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return {
        "actor": actor,
        "action_normalizer": AffineNormalizer.from_state_dict(checkpoint["action_normalizer"]),
        "actor_obs_normalizer": AffineNormalizer.from_state_dict(checkpoint["actor_obs_normalizer"]),
        "checkpoint": checkpoint,
    }


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TD3+BC-style offline correction actor.")
    parser.add_argument("--replay", required=True, help="Replay .npz file or a directory containing replay .npz files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--td3bc_alpha", type=float, default=0.0)
    parser.add_argument("--td3bc_bc_weight", type=float, default=1.0)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--lr_actor", type=float, default=3.0e-4)
    parser.add_argument("--lr_critic", type=float, default=3.0e-4)
    parser.add_argument("--action_norm_clip", type=float, default=1.0)
    parser.add_argument("--no_obs_normalize", action="store_true")
    parser.add_argument("--no_q_abs_norm", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--actor_hidden_dims", default="1024,1024,512")
    parser.add_argument("--critic_hidden_dims", default="1024,1024,512")
    parser.add_argument("--replay_filter", default="base_only", choices=("base_only", "all"))
    args = parser.parse_args()

    replay = OfflineReplayBuffer.load(args.replay, replay_filter=args.replay_filter)
    cfg = TD3BCConfig(
        actor_hidden_dims=parse_hidden_dims(args.actor_hidden_dims),
        critic_hidden_dims=parse_hidden_dims(args.critic_hidden_dims),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        tau=float(args.tau),
        target_noise=float(args.target_noise),
        noise_clip=float(args.noise_clip),
        td3bc_alpha=float(args.td3bc_alpha),
        td3bc_bc_weight=float(args.td3bc_bc_weight),
        td3bc_use_q_abs_norm=not bool(args.no_q_abs_norm),
        lr_actor=float(args.lr_actor),
        lr_critic=float(args.lr_critic),
        policy_delay=int(args.policy_delay),
        obs_normalize=not bool(args.no_obs_normalize),
        action_norm_clip=float(args.action_norm_clip),
    )
    prepared = PreparedReplay(replay, cfg)
    trainer = TD3BCTrainer(prepared, cfg, device=args.device, seed=args.seed)
    print(
        "[td3bc] "
        f"replay_size={prepared.size} actor_obs_dim={prepared.actor_obs_dim} "
        f"critic_obs_dim={prepared.critic_obs_dim} action_dim={prepared.action_dim} "
        f"alpha={cfg.td3bc_alpha} bc_weight={cfg.td3bc_bc_weight}"
    )
    if replay.metadata.get("num_replays"):
        print(
            "[td3bc] "
            f"merged_replays={replay.metadata['num_replays']} "
            f"source_paths={replay.metadata.get('source_paths', [])}"
        )

    last_logs: Dict[str, float] = {}
    for step in range(1, int(args.steps) + 1):
        last_logs = trainer.train_step()
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            log_text = " ".join(f"{key}={value:.6f}" for key, value in sorted(last_logs.items()))
            print(f"[td3bc] step={step} {log_text}")

    path = trainer.save(args.output)
    print(f"[td3bc] saved_checkpoint={path}")


if __name__ == "__main__":
    main()
