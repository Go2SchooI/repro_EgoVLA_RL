from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

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
    path = resolve_actor_checkpoint_path(path)
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


def resolve_actor_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_dir():
        preferred = [path / "actor.pt", path / "checkpoint.pt"]
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        pt_files = sorted(path.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        if not pt_files:
            raise FileNotFoundError(f"Actor checkpoint directory contains no .pt file: {path}")
        raise ValueError(
            f"Actor checkpoint directory contains multiple .pt files and no actor.pt: {path}"
        )
    return path


def resolve_training_output_paths(
    output: str | Path,
    config_output: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    output_path = Path(output).expanduser()
    if output_path.suffix in (".pt", ".pth"):
        output_dir = output_path.parent
        checkpoint_path = output_path
        default_config_path = output_path.with_suffix(".yaml")
    else:
        output_dir = output_path
        checkpoint_path = output_dir / "actor.pt"
        default_config_path = output_dir / "config.yaml"
    config_path = Path(config_output).expanduser() if config_output else default_config_path
    return output_dir, checkpoint_path, config_path


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _yaml_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _write_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    value = _yaml_safe(value)
    if isinstance(value, dict):
        if not value:
            return [prefix + "{}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_write_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [prefix + "[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(prefix + "-")
                lines.extend(_write_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
        return lines
    return [prefix + _format_yaml_scalar(value)]


def write_training_yaml(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(_write_yaml_lines(payload)) + "\n"
    path.write_text(text)
    return path


def build_training_config_payload(
    args: argparse.Namespace,
    cfg: TD3BCConfig,
    prepared: PreparedReplay,
    replay: OfflineReplayBuffer,
) -> Dict[str, Any]:
    return {
        "format": "td3bc_ref_training_config_v1",
        "command": " ".join(sys.argv),
        "output": str(args.output),
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(args.checkpoint_output),
        "config_path": str(args.config_output),
        "replay": str(args.replay),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "device": str(args.device),
        "replay_filter": str(args.replay_filter),
        "td3bc_config": asdict(cfg),
        "prepared_replay": {
            "size": prepared.size,
            "actor_obs_dim": prepared.actor_obs_dim,
            "critic_obs_dim": prepared.critic_obs_dim,
            "action_dim": prepared.action_dim,
        },
        "replay_metadata": replay.metadata,
        "wandb": {
            "enabled": bool(args.wandb_project),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
            "group": args.wandb_group,
            "tags": [tag for tag in args.wandb_tags.split(",") if tag.strip()],
            "mode": args.wandb_mode,
        },
    }


def init_wandb(args: argparse.Namespace, payload: Dict[str, Any]):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed in this environment. Install wandb or omit --wandb_project."
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        tags=tags or None,
        mode=args.wandb_mode or None,
        dir=args.wandb_dir or None,
        config=_yaml_safe(payload),
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TD3+BC-style offline correction actor.")
    parser.add_argument("--replay", required=True, help="Replay .npz file or a directory containing replay .npz files.")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory, or a legacy .pt checkpoint path. Directory mode writes "
            "actor.pt and config.yaml inside the directory."
        ),
    )
    parser.add_argument(
        "--config_output",
        default=None,
        help="YAML path for training hyperparameters. Defaults to config.yaml inside output directory.",
    )
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
    parser.add_argument("--wandb_project", default=None, help="If set, log TD3+BC metrics to this wandb project.")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", default="", help="Comma-separated wandb tags.")
    parser.add_argument("--wandb_mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb_dir", default=None)
    args = parser.parse_args()
    output_dir, checkpoint_path, config_path = resolve_training_output_paths(args.output, args.config_output)
    args.output_dir = str(output_dir)
    args.checkpoint_output = str(checkpoint_path)
    args.config_output = str(config_path)

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
    config_payload = build_training_config_payload(args, cfg, prepared, replay)
    config_path = write_training_yaml(config_path, config_payload)
    wandb_run = init_wandb(args, config_payload)
    print(
        "[td3bc] "
        f"replay_size={prepared.size} actor_obs_dim={prepared.actor_obs_dim} "
        f"critic_obs_dim={prepared.critic_obs_dim} action_dim={prepared.action_dim} "
        f"alpha={cfg.td3bc_alpha} bc_weight={cfg.td3bc_bc_weight}"
    )
    print(f"[td3bc] output_dir={output_dir}")
    print(f"[td3bc] saved_training_config={config_path}")
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
            if wandb_run is not None:
                wandb_run.log({f"train/{key}": value for key, value in last_logs.items()}, step=step)

    path = trainer.save(checkpoint_path)
    print(f"[td3bc] saved_checkpoint={path}")
    if wandb_run is not None:
        wandb_run.summary["checkpoint_path"] = str(path)
        wandb_run.summary["config_path"] = str(config_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
