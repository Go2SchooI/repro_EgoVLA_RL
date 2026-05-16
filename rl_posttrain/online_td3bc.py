from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.critics import DoubleQCritic
from rl_posttrain.h_summary import HSummaryConfig, module_grad_norm, module_num_parameters, module_param_norm
from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.paired_eval import (
    _aggregate_scene_summaries as paired_aggregate_scene_summaries,
    _compare as paired_compare,
    _run_eval as paired_run_eval,
    _safe_label as paired_safe_label,
)
from rl_posttrain.replay_buffer import FAST_FIELDS, OfflineReplayBuffer
from rl_posttrain.td3bc_ref import (
    TD3BCConfig,
    action_group_errors,
    resolve_actor_checkpoint_path,
    _normalize_action_spec,
    write_training_yaml,
)


DEFAULT_ONLINE_CONFIG: Dict[str, Any] = {
    "online": {
        "enabled": True,
        "task": "Humanoid-Open-Laptop-v0",
        "model_path": "checkpoints/ego_vla_checkpoint/checkpoint-3000",
        "init_checkpoint": "h_proj128_alpha0001",
        "base_replay": "playground_eval/replays/open_laptop_v4_checkpoint3000_20260509_174957",
        "output_root": "playground_eval/online_td3bc/h_proj128_alpha0001_online_v1",
        "freeze_egovla": True,
        "train_scenes": ["room1_table1", "room1_table2", "room2_table1", "room2_table2"],
        "unseen_eval_scenes": ["room3_table1", "room3_table2"],
        "total_online_episodes": 300,
        "critic_only_episodes": 30,
        "scene_sampling": "balanced",
        "max_eval_steps": 0,
        "seed": 0,
        "device": "cuda",
    },
    "reward": {
        "type": "sparse_final_success",
        "success_reward": 1.0,
        "failure_reward": 0.0,
        "done_on_success": True,
        "done_on_timeout": True,
    },
    "replay_mix": {
        "assert_base_replay_has_no_unseen_scenes": True,
        "strict_resume_manifest_match": False,
        "min_online_transitions_for_training": 1024,
        "critic_only": {"base_ratio": 0.5, "online_ratio": 0.5},
        "joint": {"base_ratio": 0.25, "online_ratio": 0.75},
    },
    "exploration": {
        "enabled": True,
        "enabled_after_episodes": 10,
        "noise_std": 0.003,
        "noise_clip": 0.01,
        "eval_noise": False,
    },
    "td3": {
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "target_noise": 0.1,
        "target_noise_clip": 0.2,
    },
    "optimization": {
        "actor_lr": 1.0e-4,
        "critic_lr": 3.0e-4,
        "batch_size": 256,
        "utd_ratio": 4,
        "max_updates_per_episode": 800,
    },
    "td3bc": {
        "alpha": 0.001,
        "bc_weight": 1.0,
        "bc_weight_schedule": None,
    },
    "gates": {
        "residual_scale": 1.0,
        "safety_gate": False,
        "q_gate": False,
    },
    "eval": {
        "every_episodes": 50,
        "num_episodes": 8,
        "num_trials": 2,
        "include_baseline": True,
        "include_identity": True,
        "include_offline_init": True,
        "cache_static": True,
        "report_seen_unseen_split": True,
        "no_save_video": True,
    },
    "wandb": {
        "enabled": True,
        "project": "egovla-online-rl",
        "entity": None,
        "group": "open_laptop_online_td3bc",
        "run_name": "h_proj128_alpha0001_online_v1",
        "mode": None,
        "dir": None,
        "tags": ["open_laptop", "online_td3bc", "h_proj128", "alpha0001"],
        "strict": False,
        "log_every_updates": 100,
        "log_episode": True,
        "log_eval": True,
    },
}

CHECKPOINT_ALIASES = {
    "h_proj128_alpha0001": "playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj128_alpha0001",
}

SCENE_RE = re.compile(r"^room(?P<room>\d+)_table(?P<table>\d+)$")


def _deep_copy_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_ONLINE_CONFIG)


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading --config requires PyYAML in this environment.") from exc
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping, got {type(payload).__name__}.")
    return payload


def _safe_name_token(value: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _float_name_token(prefix: str, value: float | int | str | None) -> str:
    if value is None:
        return f"{prefix}none"
    numeric = float(value)
    if numeric == 0.0:
        return f"{prefix}0"
    text = f"{numeric:.8g}"
    if "e" in text or "E" in text:
        text = f"{numeric:.8f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "")
    return f"{prefix}{text}"


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _canonical_path(path: str | Path) -> str:
    resolved = _resolve_path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return str(resolved.resolve())


def _canonical_model_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = str(path).strip()
    if not value:
        return None
    return _canonical_path(value)


def _online_config_model_path(online_config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(online_config, Mapping):
        return None
    online_cfg = online_config.get("online")
    if not isinstance(online_cfg, Mapping):
        return None
    value = online_cfg.get("model_path_resolved") or online_cfg.get("model_path")
    return _canonical_model_path(value)


def _checkpoint_online_config_payload(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(cfg))
    online_cfg = payload.setdefault("online", {})
    if isinstance(online_cfg, dict):
        online_cfg["model_path_resolved"] = _canonical_model_path(online_cfg.get("model_path"))
    return payload


def resolve_init_checkpoint(path_or_alias: str | Path) -> Path:
    value = str(path_or_alias)
    value = CHECKPOINT_ALIASES.get(value, value)
    return resolve_actor_checkpoint_path(value)


def _init_checkpoint_label(path_or_alias: str | Path) -> str:
    value = str(path_or_alias)
    if value in CHECKPOINT_ALIASES:
        return _safe_name_token(value)
    path = Path(value)
    name = path.name or path.parent.name
    for prefix in ("open_laptop_hsummary_", "open_laptop_", "checkpoint_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return _safe_name_token(name)


def _model_path_label(path: str | Path) -> str:
    name = Path(str(path)).expanduser().name
    if name.startswith("checkpoint-"):
        name = "ckpt" + name[len("checkpoint-") :]
    return _safe_name_token(name)


def _derive_run_name(cfg: Mapping[str, Any], suffix: str | None = None) -> str:
    online_cfg = cfg["online"]
    td3bc_cfg = cfg["td3bc"]
    parts = [
        _init_checkpoint_label(online_cfg["init_checkpoint"]),
        "online_v1",
        _model_path_label(online_cfg["model_path"]),
        _float_name_token("td3alpha", float(td3bc_cfg.get("alpha", 0.001))),
        _float_name_token("bc", float(td3bc_cfg.get("bc_weight", 1.0))),
    ]
    if suffix:
        parts.append(_safe_name_token(suffix))
    return "_".join(part for part in parts if part)


def _should_auto_name(args: argparse.Namespace) -> bool:
    explicit = bool(getattr(args, "auto_name", False))
    command_changed_name_inputs = any(
        getattr(args, key, None) is not None
        for key in (
            "init_checkpoint",
            "model_path",
            "td3bc_alpha",
            "bc_weight",
            "noise_std",
            "noise_clip",
            "critic_only_base_ratio",
            "critic_only_online_ratio",
            "joint_base_ratio",
            "joint_online_ratio",
            "name_suffix",
            "output_root_base",
            "wandb_run_name",
        )
    )
    return explicit or command_changed_name_inputs


def _apply_derived_naming(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if not _should_auto_name(args):
        return
    run_name = args.wandb_run_name or _derive_run_name(cfg, args.name_suffix)
    if not args.output_root:
        output_root_base = args.output_root_base or "playground_eval/online_td3bc"
        cfg["online"]["output_root"] = str(_resolve_path(output_root_base) / run_name)
    cfg["wandb"]["run_name"] = run_name
    if args.wandb_group:
        cfg["wandb"]["group"] = args.wandb_group
    tags = list(cfg["wandb"].get("tags") or [])
    for tag in (
        _init_checkpoint_label(cfg["online"]["init_checkpoint"]),
        _model_path_label(cfg["online"]["model_path"]),
        _float_name_token("td3alpha", float(cfg["td3bc"].get("alpha", 0.001))),
        _float_name_token("bc", float(cfg["td3bc"].get("bc_weight", 1.0))),
    ):
        if tag and tag not in tags:
            tags.append(tag)
    cfg["wandb"]["tags"] = tags


def scene_to_room_table(scene: str) -> tuple[int, int]:
    match = SCENE_RE.match(scene)
    if not match:
        raise ValueError(f"Scene must look like room1_table2, got {scene!r}.")
    return int(match.group("room")), int(match.group("table"))


def room_table_to_scene(room_idx: int, table_idx: int) -> str:
    return f"room{int(room_idx)}_table{int(table_idx)}"


def _scene_sequence(train_scenes: Sequence[str], total: int) -> list[str]:
    if not train_scenes:
        raise ValueError("online.train_scenes must not be empty.")
    return [train_scenes[index % len(train_scenes)] for index in range(int(total))]


def _task_key(task_name: str) -> str:
    if task_name.startswith("Humanoid-") and task_name.endswith("-v0"):
        return task_name[len("Humanoid-") : -len("-v0")]
    return task_name


def _num_task_init_episodes(task_name: str) -> int:
    utils_path = Path(__file__).resolve().parents[1] / "human_plan" / "ego_bench_eval" / "utils.py"
    module = ast.parse(utils_path.read_text())
    task_init_episode = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "TASK_INIT_EPISODE" for target in node.targets):
            task_init_episode = ast.literal_eval(node.value)
            break
    if task_init_episode is None:
        raise KeyError(f"Could not find TASK_INIT_EPISODE in {utils_path}.")
    key = _task_key(task_name)
    if key not in task_init_episode:
        raise KeyError(f"Task {task_name!r} is not present in TASK_INIT_EPISODE.")
    return len(task_init_episode[key])


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _torch_load(path: str | Path, device: str | torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _replay_source_counts(replay: OfflineReplayBuffer | MutableNormalizedReplay) -> Dict[str, int]:
    sources = replay.arrays.get("source")
    if sources is None or len(sources) == 0:
        return {}
    values, counts = np.unique(np.asarray(sources).astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values.tolist(), counts.tolist())}


def _build_replay_manifest(
    base_replay_path: str | Path,
    base_replay: OfflineReplayBuffer,
    online_replay_dir: str | Path,
    online_store: MutableNormalizedReplay,
    model_path: str | Path,
) -> Dict[str, Any]:
    model_path_resolved = _canonical_model_path(model_path)
    base_payload: Dict[str, Any] = {
        "path": str(_resolve_path(base_replay_path)),
        "size": int(base_replay.size),
        "scenes": sorted(replay_scenes(base_replay)),
        "source_counts": _replay_source_counts(base_replay),
        "metadata": base_replay.metadata,
    }
    base_payload["fingerprint"] = _stable_fingerprint(base_payload)

    online_replay_dir = Path(online_replay_dir)
    shard_paths = sorted(str(path) for path in online_replay_dir.glob("*.npz"))
    online_payload: Dict[str, Any] = {
        "dir": str(online_replay_dir),
        "shards": shard_paths,
        "size": int(online_store.size),
        "scene_counts": dict(sorted(online_store.scene_counts.items())),
        "source_counts": _replay_source_counts(online_store),
    }
    online_payload["fingerprint"] = _stable_fingerprint(online_payload)

    return {
        "format": "online_td3bc_replay_manifest_v1",
        "model_path": model_path_resolved,
        "base_replay": base_payload,
        "online_replay": online_payload,
    }


def _resume_manifest_mismatches(
    checkpoint_manifest: Mapping[str, Any] | None,
    current_manifest: Mapping[str, Any],
) -> list[str]:
    if not checkpoint_manifest:
        return ["checkpoint is missing replay_manifest"]

    mismatches: list[str] = []
    checkpoint_base = checkpoint_manifest.get("base_replay", {})
    current_base = current_manifest.get("base_replay", {})
    if checkpoint_base.get("fingerprint") != current_base.get("fingerprint"):
        mismatches.append("base_replay.fingerprint")

    checkpoint_online = checkpoint_manifest.get("online_replay", {})
    current_online = current_manifest.get("online_replay", {})
    for key in ("shards", "size", "scene_counts", "source_counts"):
        if _json_safe(checkpoint_online.get(key)) != _json_safe(current_online.get(key)):
            mismatches.append(f"online_replay.{key}")
    if checkpoint_manifest.get("model_path") != current_manifest.get("model_path"):
        mismatches.append("model_path")
    return mismatches


def _validate_resume_model_path(
    checkpoint_manifest: Mapping[str, Any] | None,
    checkpoint_online_config: Mapping[str, Any] | None,
    checkpoint_payload_model_path: str | Path | None,
    current_model_path: str | Path,
    *,
    checkpoint_online_episode: int,
) -> None:
    current = _canonical_model_path(current_model_path)
    config_model = _online_config_model_path(checkpoint_online_config)
    manifest_model = None
    if isinstance(checkpoint_manifest, Mapping):
        manifest_model = _canonical_model_path(checkpoint_manifest.get("model_path"))
    payload_model = _canonical_model_path(checkpoint_payload_model_path)

    missing: list[str] = []
    if config_model is None:
        missing.append("checkpoint.online_config.online.model_path")
    if manifest_model is None:
        missing.append("checkpoint.replay_manifest.model_path")
    if int(checkpoint_online_episode) > 0 and missing:
        raise AssertionError(
            "Resume checkpoint is missing frozen EgoVLA model_path metadata after collecting "
            f"{int(checkpoint_online_episode)} online episodes: {', '.join(missing)}. "
            "Start a clean run with the intended online.model_path."
        )

    mismatches: list[str] = []
    for label, value in (
        ("checkpoint.online_config.online.model_path", config_model),
        ("checkpoint.replay_manifest.model_path", manifest_model),
        ("checkpoint.online_model_path", payload_model),
    ):
        if value is not None and value != current:
            mismatches.append(f"{label}={value}")
    if mismatches:
        raise AssertionError(
            "Resume checkpoint frozen EgoVLA model_path does not match current config "
            f"online.model_path={current}: " + ", ".join(mismatches)
        )


def _validate_resume_replay_manifest(
    checkpoint_manifest: Mapping[str, Any] | None,
    current_manifest: Mapping[str, Any],
    *,
    strict: bool,
) -> None:
    mismatches = _resume_manifest_mismatches(checkpoint_manifest, current_manifest)
    if not mismatches:
        print("[online-td3bc][resume-manifest] checkpoint manifest matches current replay state", flush=True)
        return
    message = (
        "Resume checkpoint replay_manifest does not match current output_root/online_replay: "
        + ", ".join(mismatches)
    )
    if strict:
        raise AssertionError(message)
    print(f"[online-td3bc][resume-manifest] WARNING {message}", flush=True)


def _existing_online_replay_paths(output_root: str | Path) -> list[Path]:
    return sorted(Path(output_root).expanduser().joinpath("online_replay").glob("*.npz"))


def _validate_online_replay_reuse(
    output_root: str | Path,
    *,
    resume: bool,
    allow_reuse_online_replay: bool,
) -> list[Path]:
    existing_online_replay_paths = _existing_online_replay_paths(output_root)
    if existing_online_replay_paths:
        print(
            "[online-td3bc] "
            f"found_existing_online_replay_shards={len(existing_online_replay_paths)} "
            f"resume={resume} allow_reuse_online_replay={allow_reuse_online_replay}",
            flush=True,
        )
        if not resume and not allow_reuse_online_replay:
            raise FileExistsError(
                "Found existing online replay under output_root/online_replay. "
                "Use --resume to continue a run or --allow_reuse_online_replay to explicitly reuse it."
            )
    return existing_online_replay_paths


def _normalizers_close(lhs: AffineNormalizer, rhs: AffineNormalizer, atol: float = 1.0e-5) -> bool:
    return (
        lhs.mean.shape == rhs.mean.shape
        and np.allclose(lhs.mean, rhs.mean, atol=atol, rtol=atol)
        and np.allclose(lhs.scale, rhs.scale, atol=atol, rtol=atol)
    )


class NormalizedReplayView:
    def __init__(
        self,
        replay: OfflineReplayBuffer,
        actor_obs_normalizer: AffineNormalizer,
        critic_obs_normalizer: AffineNormalizer,
        checkpoint_action_normalizer: AffineNormalizer,
        *,
        require_checkpoint_action_normalizer: bool = True,
    ):
        if replay.action_normalizer is None:
            raise ValueError("Replay is missing action_normalizer state.")
        if require_checkpoint_action_normalizer and not _normalizers_close(
            replay.action_normalizer, checkpoint_action_normalizer
        ):
            raise ValueError(
                "Replay action normalizer does not match the warm-start checkpoint. "
                "Use the same base replay used by the offline checkpoint or recollect online shards "
                "with the checkpoint normalizer."
            )

        actor_obs_raw = np.asarray(replay.arrays["actor_obs"], dtype=np.float32)
        next_actor_obs_raw = np.asarray(replay.arrays["next_actor_obs"], dtype=np.float32)
        critic_obs_raw = np.asarray(replay.arrays["critic_obs"], dtype=np.float32)
        next_critic_obs_raw = np.asarray(replay.arrays["next_critic_obs"], dtype=np.float32)
        action_dim = int(replay.arrays["bc_target_norm"].shape[-1])
        actor_obs_dim = int(actor_obs_raw.shape[-1])
        next_actor_obs_dim = int(next_actor_obs_raw.shape[-1])
        if actor_obs_dim < action_dim or next_actor_obs_dim < action_dim:
            raise ValueError(f"Replay actor_obs dim is too small for action_dim={action_dim}.")
        actor_tail_error = float(np.max(np.abs(actor_obs_raw[:, -action_dim:] - replay.arrays["bc_target_norm"])))
        next_actor_tail_error = float(
            np.max(np.abs(next_actor_obs_raw[:, -action_dim:] - replay.arrays["next_bc_target_norm"]))
        )
        if actor_tail_error > 1.0e-5 or next_actor_tail_error > 1.0e-5:
            raise ValueError(
                "Replay actor_obs action tail is not aligned with bc_target_norm: "
                f"current={actor_tail_error:.8g} next={next_actor_tail_error:.8g}."
            )
        critic_prefix_error = float(np.max(np.abs(critic_obs_raw[:, :actor_obs_dim] - actor_obs_raw)))
        next_critic_prefix_error = float(
            np.max(np.abs(next_critic_obs_raw[:, :next_actor_obs_dim] - next_actor_obs_raw))
        )
        if critic_prefix_error > 1.0e-5 or next_critic_prefix_error > 1.0e-5:
            raise ValueError(
                "Replay critic_obs must start with actor_obs: "
                f"current={critic_prefix_error:.8g} next={next_critic_prefix_error:.8g}."
            )

        self.raw = replay
        self.arrays = dict(replay.arrays)
        self.arrays["actor_obs"] = actor_obs_normalizer.normalize(actor_obs_raw, clip=None)
        self.arrays["critic_obs"] = critic_obs_normalizer.normalize(critic_obs_raw, clip=None)
        self.arrays["next_actor_obs"] = actor_obs_normalizer.normalize(next_actor_obs_raw, clip=None)
        self.arrays["next_critic_obs"] = critic_obs_normalizer.normalize(next_critic_obs_raw, clip=None)
        for key in ("action_norm", "bc_target_norm", "next_bc_target_norm", "reward", "done"):
            self.arrays[key] = np.asarray(replay.arrays[key], dtype=np.float32)
        self.size = int(self.arrays["actor_obs"].shape[0])
        self.actor_obs_dim = int(self.arrays["actor_obs"].shape[-1])
        self.critic_obs_dim = int(self.arrays["critic_obs"].shape[-1])
        self.action_dim = int(self.arrays["action_norm"].shape[-1])

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {field: _to_tensor(self.arrays[field][idx], device) for field in FAST_FIELDS}


class MutableNormalizedReplay:
    def __init__(self) -> None:
        self.arrays: Dict[str, np.ndarray] = {}
        self.size = 0
        self.scene_counts: Dict[str, int] = {}

    def append(self, replay: NormalizedReplayView) -> None:
        if self.size == 0:
            for field in FAST_FIELDS:
                self.arrays[field] = np.asarray(replay.arrays[field], dtype=np.float32).copy()
            for field in ("source", "scene"):
                if field in replay.arrays:
                    self.arrays[field] = replay.arrays[field].copy()
        else:
            for field in FAST_FIELDS:
                self.arrays[field] = np.concatenate([self.arrays[field], replay.arrays[field]], axis=0)
            for field in ("source", "scene"):
                if field in replay.arrays and field in self.arrays:
                    self.arrays[field] = np.concatenate([self.arrays[field], replay.arrays[field]], axis=0)
        self.size = int(self.arrays["actor_obs"].shape[0])
        scenes = replay.arrays.get("scene")
        if scenes is not None:
            for scene in scenes.astype(str).tolist():
                self.scene_counts[scene] = self.scene_counts.get(scene, 0) + 1

    @property
    def actor_obs_dim(self) -> int:
        return int(self.arrays["actor_obs"].shape[-1]) if self.size else 0

    @property
    def critic_obs_dim(self) -> int:
        return int(self.arrays["critic_obs"].shape[-1]) if self.size else 0

    @property
    def action_dim(self) -> int:
        return int(self.arrays["action_norm"].shape[-1]) if self.size else 0

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty online replay.")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {field: _to_tensor(self.arrays[field][idx], device) for field in FAST_FIELDS}


class MixedReplaySampler:
    def __init__(
        self,
        base: NormalizedReplayView,
        online: MutableNormalizedReplay,
        device: torch.device,
        rng: np.random.Generator,
    ):
        self.base = base
        self.online = online
        self.device = device
        self.rng = rng

    def sample(self, batch_size: int, base_ratio: float, online_ratio: float) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        batch_size = int(batch_size)
        total_ratio = float(base_ratio) + float(online_ratio)
        if total_ratio <= 0.0:
            raise ValueError("At least one replay ratio must be positive.")
        requested_online = int(round(batch_size * float(online_ratio) / total_ratio))
        requested_base = batch_size - requested_online
        if self.online.size <= 0:
            requested_base = batch_size
            requested_online = 0
        if requested_base <= 0 and self.base.size > 0:
            requested_base = 1
            requested_online = batch_size - 1

        parts = []
        if requested_base > 0:
            parts.append(self.base.sample(requested_base, self.device, self.rng))
        if requested_online > 0:
            parts.append(self.online.sample(requested_online, self.device, self.rng))
        if not parts:
            raise ValueError("Mixed sampler produced an empty batch.")

        batch = {}
        for field in FAST_FIELDS:
            batch[field] = torch.cat([part[field] for part in parts], dim=0)
        metrics = {
            "base_batch_size": float(requested_base),
            "online_batch_size": float(requested_online),
            "base_sample_ratio": float(requested_base) / float(batch_size),
            "online_sample_ratio": float(requested_online) / float(batch_size),
            "online_sample_with_replacement": float(requested_online > self.online.size and requested_online > 0),
        }
        return batch, metrics


class OnlineTD3BCAgent:
    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: Dict[str, Any],
        device: str | torch.device = "cuda",
        resume_state: Mapping[str, Any] | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        if resume_state is None:
            checkpoint = _torch_load(checkpoint_path, self.device)
        else:
            checkpoint = dict(resume_state)

        if checkpoint.get("format") not in ("td3bc_ref_actor_v1", "online_td3bc_checkpoint_v1"):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')!r}")

        h_summary = HSummaryConfig.from_state_dict(checkpoint.get("h_summary"))
        actor_hidden_dims = tuple(checkpoint["actor_hidden_dims"])
        critic_hidden_dims = tuple(checkpoint["critic_hidden_dims"])
        self.actor = DeterministicActor(
            int(checkpoint["actor_obs_dim"]),
            int(checkpoint["action_dim"]),
            actor_hidden_dims,
            h_summary=h_summary,
        ).to(self.device)
        self.critic = DoubleQCritic(
            int(checkpoint["critic_obs_dim"]),
            int(checkpoint["action_dim"]),
            critic_hidden_dims,
            h_summary=h_summary,
            actor_obs_dim=int(checkpoint["actor_obs_dim"]),
        ).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        if "actor_target_state_dict" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        if "critic_target_state_dict" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        self.action_normalizer = AffineNormalizer.from_state_dict(checkpoint["action_normalizer"])
        self.actor_obs_normalizer = AffineNormalizer.from_state_dict(checkpoint["actor_obs_normalizer"])
        self.critic_obs_normalizer = AffineNormalizer.from_state_dict(checkpoint["critic_obs_normalizer"])
        self.h_summary = h_summary
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.actor_obs_dim = int(checkpoint["actor_obs_dim"])
        self.critic_obs_dim = int(checkpoint["critic_obs_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.action_spec = _normalize_action_spec(checkpoint.get("action_spec"), self.action_dim)
        self.offline_config = checkpoint.get("config", {})
        self.checkpoint_online_config = checkpoint.get("online_config", {})
        self.checkpoint_online_model_path = checkpoint.get("online_model_path")
        self.replay_manifest = checkpoint.get("replay_manifest")

        opt_cfg = cfg["optimization"]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(opt_cfg["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(opt_cfg["critic_lr"]))
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.global_update = int(checkpoint.get("global_update", checkpoint.get("total_it", 0)))
        self.critic_updates = int(checkpoint.get("critic_updates", 0))
        self.actor_updates = int(checkpoint.get("actor_updates", 0))
        self.online_episode = int(checkpoint.get("online_episode", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        cfg: Dict[str, Any],
        *,
        update_actor: bool,
        update_actor_target: bool,
    ) -> Dict[str, float]:
        self.global_update += 1
        td3 = cfg["td3"]
        td3bc = cfg["td3bc"]
        tau = float(td3["tau"])
        with torch.no_grad():
            next_action = self.actor_target(batch["next_actor_obs"])
            noise = torch.randn_like(next_action) * float(td3["target_noise"])
            noise = noise.clamp(-float(td3["target_noise_clip"]), float(td3["target_noise_clip"]))
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(batch["next_critic_obs"], next_action)
            target_q = torch.minimum(target_q1, target_q2)
            y = batch["reward"] + float(td3["gamma"]) * (1.0 - batch["done"]) * target_q

        current_q1, current_q2 = self.critic(batch["critic_obs"], batch["action_norm"])
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_h_grad_norm = (
            module_grad_norm(self.critic.q1_obs_processor.h_processor)
            + module_grad_norm(self.critic.q2_obs_processor.h_processor)
        )
        self.critic_opt.step()
        self.critic_updates += 1
        _soft_update(self.critic, self.critic_target, tau)

        with torch.no_grad():
            q_ref = self.critic.q1_value(batch["critic_obs"], batch["bc_target_norm"])
            q_exec = self.critic.q1_value(batch["critic_obs"], batch["action_norm"])
            q_adv = q_exec - q_ref

        logs = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": 0.0,
            "bc_loss": 0.0,
            "q_abs": 0.0,
            "lambda_q": 0.0,
            "q1_mean": float(current_q1.detach().mean().cpu()),
            "q2_mean": float(current_q2.detach().mean().cpu()),
            "target_q_mean": float(target_q.detach().mean().cpu()),
            "td_target_mean": float(y.detach().mean().cpu()),
            "q_ref": float(q_ref.mean().detach().cpu()),
            "q_exec": float(q_exec.mean().detach().cpu()),
            "q_adv": float(q_adv.mean().detach().cpu()),
            "mean_abs_actor_minus_ref_norm": 0.0,
            "max_abs_actor_minus_ref_norm": 0.0,
            "h_actor_param_norm": module_param_norm(self.actor.obs_processor.h_processor),
            "h_actor_projector_num_params": float(module_num_parameters(self.actor.obs_processor.h_processor)),
            "h_actor_grad_norm": 0.0,
            "h_critic_param_norm": (
                module_param_norm(self.critic.q1_obs_processor.h_processor)
                + module_param_norm(self.critic.q2_obs_processor.h_processor)
            ),
            "h_critic_projector_num_params": float(
                module_num_parameters(self.critic.q1_obs_processor.h_processor)
                + module_num_parameters(self.critic.q2_obs_processor.h_processor)
            ),
            "h_critic_grad_norm": float(critic_h_grad_norm),
        }

        policy_delay = int(td3["policy_delay"])
        if update_actor and self.global_update % policy_delay == 0:
            action_pi = self.actor(batch["actor_obs"])
            q_pi = self.critic.q1_value(batch["critic_obs"], action_pi)
            bc_loss = F.mse_loss(action_pi, batch["bc_target_norm"])
            q_abs = q_pi.abs().mean().detach().clamp_min(1.0e-6)
            lambda_q = torch.as_tensor(float(td3bc["alpha"]), device=self.device) / q_abs
            actor_loss = -lambda_q * q_pi.mean() + float(td3bc["bc_weight"]) * bc_loss
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_h_grad_norm = module_grad_norm(self.actor.obs_processor.h_processor)
            self.actor_opt.step()
            self.actor_updates += 1
            if update_actor_target:
                _soft_update(self.actor, self.actor_target, tau)
            diff = (action_pi.detach() - batch["bc_target_norm"]).abs()
            logs.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "q_abs": float(q_abs.detach().cpu()),
                    "lambda_q": float(lambda_q.detach().cpu()),
                    "mean_abs_actor_minus_ref_norm": float(diff.mean().cpu()),
                    "max_abs_actor_minus_ref_norm": float(diff.max().cpu()),
                    "h_actor_param_norm": module_param_norm(self.actor.obs_processor.h_processor),
                    "h_actor_grad_norm": float(actor_h_grad_norm),
                    **action_group_errors(diff, self.action_spec),
                }
            )
        return logs

    def actor_checkpoint_payload(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "format": "td3bc_ref_actor_v1",
            "online_format": "online_td3bc_actor_export_v1",
            "config": self.offline_config,
            "online_config": _checkpoint_online_config_payload(cfg),
            "online_model_path": _canonical_model_path(cfg["online"].get("model_path")),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "action_dim": self.action_dim,
            "actor_hidden_dims": tuple(self.actor_hidden_dims),
            "critic_hidden_dims": tuple(self.critic_hidden_dims),
            "h_summary": self.h_summary.state_dict(),
            "actor_processed_obs_dim": int(self.actor.processed_obs_dim),
            "critic_processed_obs_dim": int(self.critic.processed_obs_dim),
            "action_normalizer": self.action_normalizer.state_dict(),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict(),
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict(),
            "online_episode": int(self.online_episode),
            "global_update": int(self.global_update),
            "critic_updates": int(self.critic_updates),
            "actor_updates": int(self.actor_updates),
            "env_steps": int(self.env_steps),
        }
        if self.action_spec is not None:
            payload["action_spec"] = self.action_spec
        if self.replay_manifest is not None:
            payload["replay_manifest"] = self.replay_manifest
        return payload

    def save_actor_checkpoint(self, path: str | Path, cfg: Dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor_checkpoint_payload(cfg), path)
        return path

    def save_training_checkpoint(self, path: str | Path, cfg: Dict[str, Any]) -> Path:
        payload = self.actor_checkpoint_payload(cfg)
        payload.update(
            {
                "format": "online_td3bc_checkpoint_v1",
                "actor_optimizer_state_dict": self.actor_opt.state_dict(),
                "critic_optimizer_state_dict": self.critic_opt.state_dict(),
            }
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path


class SafeWandbLogger:
    def __init__(self, cfg: Dict[str, Any], payload: Dict[str, Any]):
        self.run = None
        wandb_cfg = cfg.get("wandb", {})
        if not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb

            self.run = wandb.init(
                project=wandb_cfg.get("project"),
                entity=wandb_cfg.get("entity") or None,
                group=wandb_cfg.get("group") or None,
                name=wandb_cfg.get("run_name") or None,
                tags=wandb_cfg.get("tags") or None,
                mode=wandb_cfg.get("mode") or None,
                dir=wandb_cfg.get("dir") or None,
                config=payload,
            )
        except Exception as exc:  # pragma: no cover - depends on local wandb setup
            if wandb_cfg.get("strict", False):
                raise
            print(f"[online-td3bc][wandb] disabled after init failure: {exc}", flush=True)
            self.run = None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def summary(self, key: str, value: Any) -> None:
        if self.run is not None:
            self.run.summary[key] = value

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _extract_scenes_from_metadata(metadata: Mapping[str, Any]) -> set[str]:
    scenes: set[str] = set()
    if "scene" in metadata:
        scenes.add(str(metadata["scene"]))
    if "room_idx" in metadata and "table_idx" in metadata:
        scenes.add(room_table_to_scene(int(metadata["room_idx"]), int(metadata["table_idx"])))
    for child in metadata.get("child_metadata", []) or []:
        if isinstance(child, Mapping):
            scenes.update(_extract_scenes_from_metadata(child))
    return scenes


def replay_scenes(replay: OfflineReplayBuffer) -> set[str]:
    scenes = set()
    if "scene" in replay.arrays:
        scenes.update(str(item) for item in replay.arrays["scene"].astype(str).tolist())
    scenes.update(_extract_scenes_from_metadata(replay.metadata))
    return scenes


def load_online_replay_dir(
    replay_dir: str | Path,
    agent: OnlineTD3BCAgent,
) -> MutableNormalizedReplay:
    store = MutableNormalizedReplay()
    replay_dir = Path(replay_dir)
    if not replay_dir.exists():
        return store
    for path in sorted(replay_dir.rglob("*.npz")):
        replay = OfflineReplayBuffer.load(path, replay_filter="all")
        view = NormalizedReplayView(
            replay,
            agent.actor_obs_normalizer,
            agent.critic_obs_normalizer,
            agent.action_normalizer,
            require_checkpoint_action_normalizer=True,
        )
        store.append(view)
    return store


def _episode_info(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        info: Dict[str, Any] = {
            "transitions": int(data["actor_obs"].shape[0]),
            "success": float(np.asarray(data["success"]).reshape(-1).max()) if "success" in data.files else 0.0,
            "timeout": float(np.asarray(data["timeout"]).reshape(-1).max()) if "timeout" in data.files else 0.0,
            "reward": float(np.asarray(data["reward"]).reshape(-1).sum()),
            "length": int(data["actor_obs"].shape[0]),
            "mean_abs_actor_minus_ref_norm": float(
                np.asarray(data["mean_abs_actor_minus_ref_norm"]).reshape(-1).mean()
            )
            if "mean_abs_actor_minus_ref_norm" in data.files
            else 0.0,
            "max_abs_actor_minus_ref_norm": float(
                np.asarray(data["max_abs_actor_minus_ref_norm"]).reshape(-1).max()
            )
            if "max_abs_actor_minus_ref_norm" in data.files
            else 0.0,
            "num_clipped_dims": float(np.asarray(data["num_clipped_dims"]).reshape(-1).sum())
            if "num_clipped_dims" in data.files
            else 0.0,
        }
    return info


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def collect_online_episode(
    cfg: Dict[str, Any],
    agent: OnlineTD3BCAgent,
    scene: str,
    episode_idx: int,
    scene_episode_idx: int,
    output_root: Path,
    actor_checkpoint: Path,
    noise_std: float,
    noise_clip: float,
) -> tuple[Path, Dict[str, Any]]:
    room_idx, table_idx = scene_to_room_table(scene)
    num_init_episodes = _num_task_init_episodes(cfg["online"]["task"])
    episode_start_idx = int(scene_episode_idx % num_init_episodes)
    unit_name = f"episode_{episode_idx:04d}_{scene}"
    unit_dir = output_root / "rollouts" / unit_name
    replay_path = output_root / "online_replay" / f"{unit_name}.npz"
    result_path = unit_dir / "results_local_eval.txt"
    run_dir = unit_dir / "run"
    log_path = unit_dir / "run_local_eval.log"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    unit_dir.mkdir(parents=True, exist_ok=True)
    if replay_path.exists():
        replay_path.unlink()
    if result_path.exists():
        result_path.unlink()
    if log_path.exists():
        log_path.unlink()

    total_online_episodes = max(1, int(cfg["online"]["total_online_episodes"]))
    num_train_scenes = max(1, len(cfg["online"]["train_scenes"]))
    randomize_total_trials = max(1, (total_online_episodes + num_train_scenes - 1) // num_train_scenes)

    env = os.environ.copy()
    env.update(
        {
            "TASK": str(cfg["online"]["task"]),
            "MODEL_PATH": str(_resolve_path(cfg["online"]["model_path"])),
            "ROOM_IDX": str(room_idx),
            "TABLE_IDX": str(table_idx),
            "NUM_EPISODES": "1",
            "NUM_TRIALS": "1",
            "EPISODE_START_IDX": str(episode_start_idx),
            "TRIAL_START_IDX": str(scene_episode_idx),
            "RANDOMIZE_TOTAL_EPISODES": str(num_init_episodes),
            "RANDOMIZE_TOTAL_TRIALS": str(randomize_total_trials),
            "MAX_EVAL_STEPS": str(int(cfg["online"].get("max_eval_steps", 0) or 0)),
            "SAVE_VIDEO": "0",
            "SAVE_FRAMES": "0",
            "PROJECT_TRAJS": "0",
            "RL_MODE": "actor",
            "RL_ACTOR_CHECKPOINT": str(actor_checkpoint),
            "RL_COLLECT_REPLAY_PATH": str(replay_path),
            "RL_COLLECT_SOURCE": "online_actor",
            "RL_COLLECT_SAVE_RAW": "0",
            "RL_EXPLORATION_NOISE_STD": str(float(noise_std)),
            "RL_EXPLORATION_NOISE_CLIP": str(float(noise_clip)),
            "RL_EXPLORATION_NOISE_SEED": str(int(cfg["online"].get("seed", 0)) + int(episode_idx)),
            "RESULT_PATH": str(result_path),
            "RUN_DIR": str(run_dir),
        }
    )
    print(
        "[online-td3bc][collect] "
        f"episode={episode_idx} scene={scene} episode_start_idx={episode_start_idx} "
        f"noise_std={noise_std} replay={replay_path}",
        flush=True,
    )
    with log_path.open("w") as log_file:
        try:
            subprocess.run(
                ["./run_local_eval.sh"],
                env=env,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Online collector subprocess failed.\n"
                f"episode={episode_idx} scene={scene}\n"
                f"returncode={exc.returncode}\n"
                f"log_path={log_path}\n"
                f"log_tail:\n{_tail_text(log_path)}"
            ) from exc
    if not replay_path.exists():
        raise RuntimeError(
            "Online collector finished but did not write the expected replay shard.\n"
            f"episode={episode_idx} scene={scene}\n"
            f"expected_replay={replay_path}\n"
            f"result_path={result_path} exists={result_path.exists()}\n"
            f"run_dir={run_dir}\n"
            f"log_path={log_path}\n"
            f"log_tail:\n{_tail_text(log_path)}"
        )
    return replay_path, _episode_info(replay_path)


def _phase_for_episode(cfg: Dict[str, Any], episode_idx: int) -> str:
    return "critic_only" if int(episode_idx) < int(cfg["online"]["critic_only_episodes"]) else "joint"


def _exploration_for_episode(cfg: Dict[str, Any], episode_idx: int) -> tuple[float, float]:
    exploration = cfg["exploration"]
    if not exploration.get("enabled", True):
        return 0.0, 0.0
    if int(episode_idx) < int(exploration["enabled_after_episodes"]):
        return 0.0, 0.0
    return float(exploration["noise_std"]), float(exploration["noise_clip"])


def _run_updates(
    cfg: Dict[str, Any],
    agent: OnlineTD3BCAgent,
    sampler: MixedReplaySampler,
    online_store: MutableNormalizedReplay,
    episode_steps: int,
    phase: str,
    wandb_logger: SafeWandbLogger | None = None,
) -> tuple[int, Dict[str, float]]:
    min_online = int(cfg["replay_mix"]["min_online_transitions_for_training"])
    if online_store.size < min_online:
        return 0, {
            "skip_updates": 1.0,
            "skip_reason_min_online_transitions": float(min_online),
            "online_buffer_size": float(online_store.size),
        }

    opt = cfg["optimization"]
    updates = min(int(episode_steps) * int(opt["utd_ratio"]), int(opt["max_updates_per_episode"]))
    ratios = cfg["replay_mix"]["critic_only" if phase == "critic_only" else "joint"]
    update_actor = phase == "joint"
    update_actor_target = phase == "joint"
    last_logs: Dict[str, float] = {}
    for _ in range(updates):
        batch, sample_logs = sampler.sample(
            int(opt["batch_size"]),
            float(ratios["base_ratio"]),
            float(ratios["online_ratio"]),
        )
        train_logs = agent.train_step(
            batch,
            cfg,
            update_actor=update_actor,
            update_actor_target=update_actor_target,
        )
        last_logs = {**train_logs, **sample_logs, "skip_updates": 0.0}
        log_every = int(cfg["wandb"].get("log_every_updates", 0) or 0)
        if wandb_logger is not None and log_every > 0 and agent.global_update % log_every == 0:
            wandb_logger.log(
                {
                    "online/global_step": float(agent.global_update),
                    "online/critic_updates": float(agent.critic_updates),
                    "online/actor_updates": float(agent.actor_updates),
                    "online/base_sample_ratio": float(last_logs.get("base_sample_ratio", 0.0)),
                    "online/online_sample_ratio": float(last_logs.get("online_sample_ratio", 0.0)),
                    "loss/critic_loss": float(last_logs.get("critic_loss", 0.0)),
                    "loss/actor_loss": float(last_logs.get("actor_loss", 0.0)),
                    "loss/bc_loss": float(last_logs.get("bc_loss", 0.0)),
                    "loss/q_abs": float(last_logs.get("q_abs", 0.0)),
                    "loss/lambda_q": float(last_logs.get("lambda_q", 0.0)),
                    "q/q1_mean": float(last_logs.get("q1_mean", 0.0)),
                    "q/q2_mean": float(last_logs.get("q2_mean", 0.0)),
                    "q/target_q_mean": float(last_logs.get("target_q_mean", 0.0)),
                    "q/q_ref": float(last_logs.get("q_ref", 0.0)),
                    "q/q_exec": float(last_logs.get("q_exec", 0.0)),
                    "q/q_adv": float(last_logs.get("q_adv", 0.0)),
                    "action/mean_abs_actor_minus_ref_norm": float(
                        last_logs.get("mean_abs_actor_minus_ref_norm", 0.0)
                    ),
                    "action/max_abs_actor_minus_ref_norm": float(
                        last_logs.get("max_abs_actor_minus_ref_norm", 0.0)
                    ),
                },
                step=agent.global_update,
            )
    return updates, last_logs


def _comparison_aliases(compare: Mapping[str, Any]) -> Dict[str, float]:
    regress = int(compare.get("baseline_success_actor_fail", compare.get("regress", 0)))
    recover = int(compare.get("baseline_fail_actor_success", compare.get("recover", 0)))
    return {
        "regress": float(regress),
        "recover": float(recover),
        "net": float(recover - regress),
    }


def summarize_paired_eval(
    summary_path: Path,
    current_actor_checkpoint: Path,
    seen_scenes: Sequence[str],
    unseen_scenes: Sequence[str],
) -> Dict[str, float]:
    summary = json.loads(summary_path.read_text())
    current_actor_name = paired_safe_label(current_actor_checkpoint.stem)
    scene_records = []
    for item in summary.get("scenes", []):
        scene_summary_path = Path(item["summary_path"])
        scene_summary = json.loads(scene_summary_path.read_text())
        scene = scene_summary["scene"]
        scene_label = room_table_to_scene(scene["room_idx"], scene["table_idx"])
        runs = scene_summary["runs"]
        if "actors" in runs:
            actor_run = runs["actors"][current_actor_name]
            compare = scene_summary["comparisons"][current_actor_name]
        else:
            actor_run = runs["actor"]
            compare = scene_summary["comparison"]
        aliases = _comparison_aliases(compare)
        metric_means = actor_run.get("metric_means", {})
        scene_records.append(
            {
                "scene": scene_label,
                "results": list(actor_run["results"]),
                "success_rate": float(actor_run["success_rate"]),
                "mean_abs_actor_minus_ref_norm": float(
                    metric_means.get("rl_actor_mean_abs_actor_minus_ref_norm", 0.0)
                ),
                "max_abs_actor_minus_ref_norm": float(
                    metric_means.get("rl_actor_max_abs_actor_minus_ref_norm", 0.0)
                ),
                "episode_length": float(metric_means.get("episode_length", 0.0)),
                "num_clipped_dims": float(metric_means.get("rl_actor_num_clipped_dims", 0.0)),
                **aliases,
            }
        )

    def aggregate(selected: Iterable[str]) -> Dict[str, float]:
        selected_set = set(selected)
        records = [item for item in scene_records if item["scene"] in selected_set]
        results = [bool(value) for item in records for value in item["results"]]
        if not results:
            return {
                "success_rate": 0.0,
                "regress": 0.0,
                "recover": 0.0,
                "net": 0.0,
                "mean_abs_actor_minus_ref_norm": 0.0,
                "max_abs_actor_minus_ref_norm": 0.0,
                "episode_length": 0.0,
                "num_clipped_dims": 0.0,
            }
        denom = max(len(records), 1)
        return {
            "success_rate": float(sum(results) / len(results)),
            "regress": float(sum(item["regress"] for item in records)),
            "recover": float(sum(item["recover"] for item in records)),
            "net": float(sum(item["net"] for item in records)),
            "mean_abs_actor_minus_ref_norm": float(
                sum(item["mean_abs_actor_minus_ref_norm"] for item in records) / denom
            ),
            "max_abs_actor_minus_ref_norm": float(max(item["max_abs_actor_minus_ref_norm"] for item in records)),
            "episode_length": float(sum(item["episode_length"] for item in records) / denom),
            "num_clipped_dims": float(sum(item["num_clipped_dims"] for item in records) / denom),
        }

    all_scenes = list(seen_scenes) + list(unseen_scenes)
    metrics: Dict[str, float] = {}
    for prefix, values in (
        ("eval_seen", aggregate(seen_scenes)),
        ("eval_unseen", aggregate(unseen_scenes)),
        ("eval_all", aggregate(all_scenes)),
    ):
        for key, value in values.items():
            metrics[f"{prefix}/{key}"] = value
    for item in scene_records:
        metrics[f"eval_scene/{item['scene']}_success_rate"] = item["success_rate"]
        metrics[f"eval_scene/{item['scene']}_regress"] = item["regress"]
        metrics[f"eval_scene/{item['scene']}_recover"] = item["recover"]
        metrics[f"eval_scene/{item['scene']}_net"] = item["net"]
        metrics[f"eval_scene/{item['scene']}_mean_abs_actor_minus_ref_norm"] = item[
            "mean_abs_actor_minus_ref_norm"
        ]
        metrics[f"eval_scene/{item['scene']}_max_abs_actor_minus_ref_norm"] = item[
            "max_abs_actor_minus_ref_norm"
        ]
        metrics[f"eval_scene/{item['scene']}_episode_length"] = item["episode_length"]
        metrics[f"eval_scene/{item['scene']}_num_clipped_dims"] = item["num_clipped_dims"]
    return metrics


def _static_eval_cache_descriptor(
    cfg: Dict[str, Any],
    init_checkpoint: Path,
    scenes: Sequence[str],
) -> Dict[str, Any]:
    eval_cfg = cfg["eval"]
    static_actor_checkpoint = init_checkpoint.parent if init_checkpoint.name == "actor.pt" else init_checkpoint
    return {
        "format": "online_td3bc_static_eval_cache_descriptor_v1",
        "task": str(cfg["online"]["task"]),
        "model_path": _canonical_model_path(cfg["online"]["model_path"]),
        "scenes": list(scenes),
        "num_episodes": int(eval_cfg.get("num_episodes", 1)),
        "num_trials": int(eval_cfg.get("num_trials", 1)),
        "include_identity": bool(eval_cfg.get("include_identity", True)),
        "include_offline_init": bool(eval_cfg.get("include_offline_init", True)),
        "offline_init_checkpoint": str(_resolve_path(static_actor_checkpoint)),
    }


def _static_eval_cache_path(output_root: Path, descriptor: Mapping[str, Any]) -> Path:
    fingerprint = _stable_fingerprint(descriptor)[:16]
    return output_root / "eval" / "static_cache" / f"{fingerprint}.json"


def _write_static_eval_cache(summary_path: Path, cache_path: Path, descriptor: Mapping[str, Any]) -> None:
    summary = json.loads(summary_path.read_text())
    scene_summaries = []
    for item in summary.get("scenes", []):
        scene_summary_path = Path(item["summary_path"])
        scene_summaries.append(
            {
                "summary_path": str(scene_summary_path),
                "summary": json.loads(scene_summary_path.read_text()),
            }
        )
    payload = {
        "format": "online_td3bc_static_eval_cache_v1",
        "descriptor": dict(descriptor),
        "source_summary_path": str(summary_path),
        "scene_summaries": scene_summaries,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(cache_path)


def _load_static_eval_cache(cache_path: Path, descriptor: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text())
    if payload.get("format") != "online_td3bc_static_eval_cache_v1":
        raise ValueError(f"Unexpected static eval cache format in {cache_path}: {payload.get('format')!r}")
    if _json_safe(payload.get("descriptor")) != _json_safe(dict(descriptor)):
        raise ValueError(f"Static eval cache descriptor mismatch: {cache_path}")
    return payload


def _paired_eval_args_for_online(cfg: Dict[str, Any]) -> argparse.Namespace:
    eval_cfg = cfg["eval"]
    return argparse.Namespace(
        actor_checkpoint=[],
        task=str(cfg["online"]["task"]),
        model_path=str(_resolve_path(cfg["online"]["model_path"])),
        room_idx=None,
        table_idx=None,
        scene=None,
        max_eval_steps=None,
        num_episodes=int(eval_cfg.get("num_episodes", 1)),
        num_trials=int(eval_cfg.get("num_trials", 1)),
        output_root=None,
        no_save_video=bool(eval_cfg.get("no_save_video", True)),
        skip_identity=not bool(eval_cfg.get("include_identity", True)),
    )


def _actor_runs_from_scene_summary(scene_summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    runs = scene_summary["runs"]
    if "actors" in runs:
        return copy.deepcopy(dict(runs["actors"]))
    if "actor" in runs:
        return {"actor": copy.deepcopy(runs["actor"])}
    return {}


def _run_paired_eval_with_static_cache(
    cfg: Dict[str, Any],
    output: Path,
    current_actor_checkpoint: Path,
    cache_payload: Mapping[str, Any],
) -> Path:
    current_actor_name = paired_safe_label(current_actor_checkpoint.stem)
    scene_summaries = []
    base_args = _paired_eval_args_for_online(cfg)
    for cached_item in cache_payload.get("scene_summaries", []):
        cached_summary = copy.deepcopy(cached_item["summary"])
        scene = cached_summary["scene"]
        room_idx = int(scene["room_idx"])
        table_idx = int(scene["table_idx"])
        scene_args = argparse.Namespace(**vars(base_args))
        scene_args.room_idx = room_idx
        scene_args.table_idx = table_idx
        scene_root = output / f"room{room_idx}_table{table_idx}"
        scene_root.mkdir(parents=True, exist_ok=True)
        current_actor_run = paired_run_eval(
            "actor",
            scene_args,
            scene_root,
            actor_checkpoint=current_actor_checkpoint,
            run_name=f"actor_{current_actor_name}",
        )

        cached_runs = cached_summary["runs"]
        runs: Dict[str, Any] = {"baseline": copy.deepcopy(cached_runs["baseline"])}
        if "identity" in cached_runs:
            runs["identity"] = copy.deepcopy(cached_runs["identity"])

        actor_runs = _actor_runs_from_scene_summary(cached_summary)
        actor_runs = {
            name: run
            for name, run in actor_runs.items()
            if name != current_actor_name
            and paired_safe_label(Path(str(run.get("actor_checkpoint", ""))).stem) != current_actor_name
        }
        actor_runs[current_actor_name] = current_actor_run
        runs["actors"] = actor_runs

        identity_results = None if "identity" not in runs else list(runs["identity"]["results"])
        comparisons = {
            actor_name: paired_compare(
                list(runs["baseline"]["results"]),
                identity_results,
                list(actor_run["results"]),
            )
            for actor_name, actor_run in actor_runs.items()
        }
        scene_summary = {
            "runs": runs,
            "comparisons": comparisons,
            "scene": scene,
            "model_path": str(_resolve_path(cfg["online"]["model_path"])),
        }
        scene_summary_path = scene_root / "paired_summary.json"
        scene_summary_path.write_text(json.dumps(scene_summary, indent=2))
        print(f"[online-td3bc][eval-cache] scene_summary={scene_summary_path}", flush=True)
        scene_summaries.append({"summary_path": str(scene_summary_path), "summary": scene_summary})

    summary = paired_aggregate_scene_summaries(scene_summaries)
    summary["static_eval_cache"] = {
        "reused_static": True,
        "source_summary_path": cache_payload.get("source_summary_path"),
    }
    summary_path = output / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def run_paired_eval(
    cfg: Dict[str, Any],
    output_root: Path,
    init_checkpoint: Path,
    current_actor_checkpoint: Path,
    episode_idx: int,
) -> tuple[Path, Dict[str, float]]:
    eval_cfg = cfg["eval"]
    scenes = list(cfg["online"]["train_scenes"]) + list(cfg["online"]["unseen_eval_scenes"])
    output = output_root / "eval" / f"episode_{episode_idx:04d}"
    cache_descriptor = _static_eval_cache_descriptor(cfg, init_checkpoint, scenes)
    cache_path = _static_eval_cache_path(output_root, cache_descriptor)
    cache_payload = (
        _load_static_eval_cache(cache_path, cache_descriptor)
        if bool(eval_cfg.get("cache_static", True))
        else None
    )
    if cache_payload is not None:
        print(
            f"[online-td3bc][eval-cache] reusing static baseline/offline eval cache={cache_path}",
            flush=True,
        )
        summary_path = _run_paired_eval_with_static_cache(
            cfg,
            output,
            current_actor_checkpoint,
            cache_payload,
        )
        metrics = summarize_paired_eval(
            summary_path,
            current_actor_checkpoint,
            cfg["online"]["train_scenes"],
            cfg["online"]["unseen_eval_scenes"],
        )
        metrics["eval/static_cache_used"] = 1.0
        return summary_path, metrics

    cmd = [
        sys.executable,
        "-m",
        "rl_posttrain.paired_eval",
        "--actor_checkpoint",
    ]
    if eval_cfg.get("include_offline_init", True):
        cmd.append(str(init_checkpoint.parent if init_checkpoint.name == "actor.pt" else init_checkpoint))
    cmd.append(str(current_actor_checkpoint))
    cmd.extend(
        [
            "--task",
            str(cfg["online"]["task"]),
            "--model_path",
            str(_resolve_path(cfg["online"]["model_path"])),
            "--output_root",
            str(output),
            "--num_episodes",
            str(int(eval_cfg.get("num_episodes", 1))),
            "--num_trials",
            str(int(eval_cfg.get("num_trials", 1))),
        ]
    )
    if eval_cfg.get("no_save_video", True):
        cmd.append("--no_save_video")
    if not eval_cfg.get("include_identity", True):
        cmd.append("--skip_identity")
    for scene in scenes:
        room_idx, table_idx = scene_to_room_table(scene)
        cmd.extend(["--scene", str(room_idx), str(table_idx)])
    print(f"[online-td3bc][eval] running paired eval output={output}", flush=True)
    subprocess.run(cmd, check=True)
    summary_path = output / "paired_summary.json"
    if bool(eval_cfg.get("cache_static", True)):
        _write_static_eval_cache(summary_path, cache_path, cache_descriptor)
        print(f"[online-td3bc][eval-cache] wrote static eval cache={cache_path}", flush=True)
    metrics = summarize_paired_eval(
        summary_path,
        current_actor_checkpoint,
        cfg["online"]["train_scenes"],
        cfg["online"]["unseen_eval_scenes"],
    )
    metrics["eval/static_cache_used"] = 0.0
    return summary_path, metrics


def _validate_config(cfg: Dict[str, Any]) -> None:
    if not cfg["online"].get("enabled", False):
        raise ValueError("online.enabled must be true for rl_posttrain.online_td3bc.")
    if not cfg["online"].get("freeze_egovla", True):
        raise ValueError("online.freeze_egovla=false is outside online v1 scope.")
    gates = cfg["gates"]
    if float(gates.get("residual_scale", 1.0)) != 1.0:
        raise ValueError("online v1 requires gates.residual_scale=1.0.")
    if bool(gates.get("safety_gate", False)) or bool(gates.get("q_gate", False)):
        raise ValueError("online v1 requires safety_gate=false and q_gate=false.")
    train_scenes = set(cfg["online"]["train_scenes"])
    unseen_scenes = set(cfg["online"]["unseen_eval_scenes"])
    leaked = train_scenes.intersection(unseen_scenes)
    if leaked:
        raise ValueError(f"Training scenes overlap unseen eval scenes: {sorted(leaked)}")
    for scene in train_scenes.union(unseen_scenes):
        scene_to_room_table(scene)
    model_path = _resolve_path(cfg["online"].get("model_path", ""))
    if not model_path.exists():
        raise FileNotFoundError(f"online.model_path does not exist: {model_path}")
    reward = cfg["reward"]
    if reward.get("type") != "sparse_final_success":
        raise ValueError("online v1 only supports sparse_final_success reward.")
    if bool(cfg["exploration"].get("eval_noise", False)):
        raise ValueError("online v1 requires exploration.eval_noise=false.")
    eval_cfg = cfg["eval"]
    if not bool(eval_cfg.get("include_baseline", True)):
        raise ValueError("online v1 eval requires eval.include_baseline=true.")
    if not bool(eval_cfg.get("report_seen_unseen_split", True)):
        raise ValueError("online v1 eval requires eval.report_seen_unseen_split=true.")


def _build_config_payload(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": "online_td3bc_config_v1",
        "command": " ".join(sys.argv),
        "config": cfg,
        "args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Online off-policy TD3+BC fine-tuning for EgoVLA.")
    parser.add_argument("--config", default=None, help="YAML config. Defaults to online_td3bc_v1 in code.")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--output_root_base", default=None, help="Base directory used when auto-generating output_root.")
    parser.add_argument("--auto_name", action="store_true", help="Derive output_root and W&B run_name from command overrides.")
    parser.add_argument("--name_suffix", default=None, help="Optional suffix appended to the derived run name.")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--base_replay", default=None)
    parser.add_argument("--total_online_episodes", type=int, default=None)
    parser.add_argument("--critic_only_episodes", type=int, default=None)
    parser.add_argument("--min_online_transitions_for_training", type=int, default=None)
    parser.add_argument("--td3bc_alpha", "--alpha", dest="td3bc_alpha", type=float, default=None)
    parser.add_argument("--bc_weight", type=float, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--noise_clip", type=float, default=None)
    parser.add_argument("--critic_only_base_ratio", type=float, default=None)
    parser.add_argument("--critic_only_online_ratio", type=float, default=None)
    parser.add_argument("--joint_base_ratio", type=float, default=None)
    parser.add_argument("--joint_online_ratio", type=float, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rollout_only", action="store_true", help="Collect online actor replay without updates.")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume from an online_td3bc_checkpoint_v1 path.")
    parser.add_argument(
        "--allow_reuse_online_replay",
        action="store_true",
        help="Reuse an existing output_root/online_replay directory without treating it as an accidental resume.",
    )
    parser.add_argument(
        "--strict_resume_manifest_match",
        action="store_true",
        help="Fail resume when checkpoint replay_manifest differs from output_root/online_replay.",
    )
    parser.add_argument("--wandb_mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = _deep_copy_config()
    if args.config:
        _deep_update(cfg, _load_yaml(args.config))
    if args.output_root:
        cfg["online"]["output_root"] = args.output_root
    if args.init_checkpoint:
        cfg["online"]["init_checkpoint"] = args.init_checkpoint
    if args.model_path:
        cfg["online"]["model_path"] = args.model_path
    if args.base_replay:
        cfg["online"]["base_replay"] = args.base_replay
    if args.total_online_episodes is not None:
        cfg["online"]["total_online_episodes"] = int(args.total_online_episodes)
    if args.critic_only_episodes is not None:
        cfg["online"]["critic_only_episodes"] = int(args.critic_only_episodes)
    if args.min_online_transitions_for_training is not None:
        cfg["replay_mix"]["min_online_transitions_for_training"] = int(args.min_online_transitions_for_training)
    if args.td3bc_alpha is not None:
        cfg["td3bc"]["alpha"] = float(args.td3bc_alpha)
    if args.bc_weight is not None:
        cfg["td3bc"]["bc_weight"] = float(args.bc_weight)
    if args.noise_std is not None:
        cfg["exploration"]["noise_std"] = float(args.noise_std)
    if args.noise_clip is not None:
        cfg["exploration"]["noise_clip"] = float(args.noise_clip)
    if args.critic_only_base_ratio is not None:
        cfg["replay_mix"]["critic_only"]["base_ratio"] = float(args.critic_only_base_ratio)
    if args.critic_only_online_ratio is not None:
        cfg["replay_mix"]["critic_only"]["online_ratio"] = float(args.critic_only_online_ratio)
    if args.joint_base_ratio is not None:
        cfg["replay_mix"]["joint"]["base_ratio"] = float(args.joint_base_ratio)
    if args.joint_online_ratio is not None:
        cfg["replay_mix"]["joint"]["online_ratio"] = float(args.joint_online_ratio)
    if args.max_eval_steps is not None:
        cfg["online"]["max_eval_steps"] = int(args.max_eval_steps)
    if args.device:
        cfg["online"]["device"] = args.device
    if args.seed is not None:
        cfg["online"]["seed"] = int(args.seed)
    if args.no_eval:
        cfg["eval"]["every_episodes"] = 0
    if args.no_wandb:
        cfg["wandb"]["enabled"] = False
    if args.wandb_mode:
        cfg["wandb"]["mode"] = args.wandb_mode
    if args.wandb_project:
        cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_group:
        cfg["wandb"]["group"] = args.wandb_group
    if args.wandb_run_name:
        cfg["wandb"]["run_name"] = args.wandb_run_name
    if args.strict_resume_manifest_match:
        cfg["replay_mix"]["strict_resume_manifest_match"] = True
    _apply_derived_naming(cfg, args)

    _validate_config(cfg)
    output_root = _resolve_path(cfg["online"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_online_replay_reuse(
        output_root,
        resume=bool(args.resume),
        allow_reuse_online_replay=bool(args.allow_reuse_online_replay),
    )
    payload = _build_config_payload(args, cfg)
    config_path = write_training_yaml(output_root / "online_config.yaml", payload)
    init_checkpoint = resolve_init_checkpoint(cfg["online"]["init_checkpoint"])
    resume_checkpoint = Path(args.resume).expanduser() if args.resume else None

    agent = OnlineTD3BCAgent(
        resume_checkpoint or init_checkpoint,
        cfg,
        device=cfg["online"]["device"],
    )
    if agent.actor_obs_dim <= 0 or agent.critic_obs_dim <= 0 or agent.action_dim <= 0:
        raise AssertionError("Warm-start checkpoint produced invalid network dimensions.")
    checkpoint_replay_manifest = agent.replay_manifest
    current_model_path = _canonical_model_path(cfg["online"]["model_path"])

    base_replay = OfflineReplayBuffer.load(cfg["online"]["base_replay"], replay_filter="base_only")
    base_scenes = replay_scenes(base_replay)
    print(f"[online-td3bc] base_replay_size={base_replay.size} scenes={sorted(base_scenes)}", flush=True)
    if cfg["replay_mix"].get("assert_base_replay_has_no_unseen_scenes", True):
        unseen = set(cfg["online"]["unseen_eval_scenes"])
        leaked = unseen.intersection(base_scenes)
        if leaked:
            raise AssertionError(f"Base replay contains unseen eval scenes: {sorted(leaked)}")

    base_view = NormalizedReplayView(
        base_replay,
        agent.actor_obs_normalizer,
        agent.critic_obs_normalizer,
        agent.action_normalizer,
        require_checkpoint_action_normalizer=True,
    )
    online_replay_dir = output_root / "online_replay"
    online_store = load_online_replay_dir(online_replay_dir, agent)
    replay_manifest = _build_replay_manifest(
        cfg["online"]["base_replay"],
        base_replay,
        online_replay_dir,
        online_store,
        current_model_path,
    )
    if resume_checkpoint is not None:
        _validate_resume_model_path(
            checkpoint_replay_manifest,
            agent.checkpoint_online_config,
            agent.checkpoint_online_model_path,
            current_model_path,
            checkpoint_online_episode=agent.online_episode,
        )
        _validate_resume_replay_manifest(
            checkpoint_replay_manifest,
            replay_manifest,
            strict=bool(cfg["replay_mix"].get("strict_resume_manifest_match", False)),
        )
    agent.replay_manifest = replay_manifest
    rng = np.random.default_rng(int(cfg["online"]["seed"]))
    sampler = MixedReplaySampler(base_view, online_store, agent.device, rng)
    wandb_logger = SafeWandbLogger(cfg, payload)
    wandb_logger.summary("base_replay_fingerprint", replay_manifest["base_replay"]["fingerprint"])
    wandb_logger.summary("online_replay_fingerprint", replay_manifest["online_replay"]["fingerprint"])
    wandb_logger.summary("online_replay_shard_count", float(len(replay_manifest["online_replay"]["shards"])))
    wandb_logger.summary("online_replay_scene_count", float(len(replay_manifest["online_replay"]["scene_counts"])))

    print(
        "[online-td3bc] "
        f"init_checkpoint={init_checkpoint} output_root={output_root} config={config_path} "
        f"actor_obs_dim={agent.actor_obs_dim} critic_obs_dim={agent.critic_obs_dim} action_dim={agent.action_dim} "
        f"h_summary_mode={agent.h_summary.requested_mode} rollout_only={args.rollout_only}",
        flush=True,
    )
    print(
        "[online-td3bc] "
        f"online_replay_size={online_store.size} online_replay_scene_counts={replay_manifest['online_replay']['scene_counts']}",
        flush=True,
    )

    latest_actor = output_root / "checkpoints" / "latest_actor.pt"
    latest_state = output_root / "checkpoints" / "latest_online.pt"
    agent.save_actor_checkpoint(latest_actor, cfg)
    agent.save_training_checkpoint(latest_state, cfg)

    if args.eval_only:
        summary_path, eval_metrics = run_paired_eval(
            cfg,
            output_root,
            init_checkpoint,
            latest_actor,
            int(agent.online_episode),
        )
        print(f"[online-td3bc][eval] summary={summary_path} metrics={eval_metrics}", flush=True)
        if cfg["wandb"].get("log_eval", True) is not False:
            wandb_logger.log(eval_metrics, step=agent.global_update)
        wandb_logger.finish()
        return

    scene_counts = {scene: 0 for scene in cfg["online"]["train_scenes"]}
    episode_plan = _scene_sequence(cfg["online"]["train_scenes"], int(cfg["online"]["total_online_episodes"]))
    log_every_updates = int(cfg["wandb"].get("log_every_updates", 100) or 100)
    should_log_eval = cfg["wandb"].get("log_eval", True) is not False
    best_eval_success = -1.0

    for episode_idx, scene in enumerate(episode_plan):
        if episode_idx < agent.online_episode:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            continue
        if scene in cfg["online"]["unseen_eval_scenes"]:
            raise AssertionError(f"Refusing to train-rollout unseen eval scene {scene}.")

        phase = _phase_for_episode(cfg, episode_idx)
        noise_std, noise_clip = _exploration_for_episode(cfg, episode_idx)
        rollout_actor = output_root / "checkpoints" / "rollout_actor.pt"
        agent.save_actor_checkpoint(rollout_actor, cfg)
        replay_path, episode_info = collect_online_episode(
            cfg,
            agent,
            scene,
            episode_idx,
            scene_counts.get(scene, 0),
            output_root,
            rollout_actor,
            noise_std,
            noise_clip,
        )
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        shard = OfflineReplayBuffer.load(replay_path, replay_filter="all")
        shard_scenes = replay_scenes(shard)
        if shard_scenes and not shard_scenes.issubset(set(cfg["online"]["train_scenes"])):
            raise AssertionError(f"Online shard contains non-training scenes: {sorted(shard_scenes)}")
        online_view = NormalizedReplayView(
            shard,
            agent.actor_obs_normalizer,
            agent.critic_obs_normalizer,
            agent.action_normalizer,
            require_checkpoint_action_normalizer=True,
        )
        online_store.append(online_view)
        agent.replay_manifest = _build_replay_manifest(
            cfg["online"]["base_replay"],
            base_replay,
            online_replay_dir,
            online_store,
            current_model_path,
        )
        agent.env_steps += int(episode_info["transitions"])

        updates = 0
        train_logs: Dict[str, float] = {}
        if not args.rollout_only:
            updates, train_logs = _run_updates(
                cfg,
                agent,
                sampler,
                online_store,
                int(episode_info["transitions"]),
                phase,
                wandb_logger=wandb_logger,
            )

        agent.online_episode = episode_idx + 1
        agent.save_actor_checkpoint(latest_actor, cfg)
        agent.save_training_checkpoint(latest_state, cfg)
        wandb_logger.summary("online_replay_fingerprint", agent.replay_manifest["online_replay"]["fingerprint"])
        wandb_logger.summary("online_replay_shard_count", float(len(agent.replay_manifest["online_replay"]["shards"])))
        wandb_logger.summary("online_replay_scene_count", float(len(agent.replay_manifest["online_replay"]["scene_counts"])))
        wandb_logger.summary("latest_checkpoint", str(latest_state))
        wandb_logger.summary("latest_actor_checkpoint", str(latest_actor))
        eval_every = int(cfg["eval"].get("every_episodes", 0) or 0)
        if eval_every > 0 and agent.online_episode % eval_every == 0:
            periodic_actor = output_root / "checkpoints" / f"episode_{agent.online_episode:04d}_actor.pt"
            periodic_state = output_root / "checkpoints" / f"episode_{agent.online_episode:04d}_online.pt"
            agent.save_actor_checkpoint(periodic_actor, cfg)
            agent.save_training_checkpoint(periodic_state, cfg)
            wandb_logger.summary("latest_periodic_checkpoint", str(periodic_state))
            wandb_logger.summary("latest_periodic_actor_checkpoint", str(periodic_actor))

        episode_metrics = {
            "online/global_step": float(agent.global_update),
            "online/env_steps": float(agent.env_steps),
            "online/episode": float(agent.online_episode),
            "online/online_buffer_size": float(online_store.size),
            "online/base_buffer_size": float(base_view.size),
            "online/gradient_updates": float(updates),
            "online/critic_updates": float(agent.critic_updates),
            "online/actor_updates": float(agent.actor_updates),
            "online/actual_utd_ratio": float(updates) / max(float(episode_info["transitions"]), 1.0),
            "episode/success": float(episode_info["success"]),
            "episode/reward": float(episode_info["reward"]),
            "episode/length": float(episode_info["length"]),
            "episode/scene_id": float(list(cfg["online"]["train_scenes"]).index(scene)),
            "action/mean_abs_actor_minus_ref_norm": float(episode_info["mean_abs_actor_minus_ref_norm"]),
            "action/max_abs_actor_minus_ref_norm": float(episode_info["max_abs_actor_minus_ref_norm"]),
            "action/num_clipped_dims": float(episode_info["num_clipped_dims"]),
            "exploration/noise_std": float(noise_std),
            "exploration/noise_clip": float(noise_clip),
            "online/phase_is_joint": float(phase == "joint"),
            **{f"online_scene_count/{key}": float(value) for key, value in scene_counts.items()},
        }
        if train_logs:
            episode_metrics.update(
                {
                    "online/base_sample_ratio": float(train_logs.get("base_sample_ratio", 0.0)),
                    "online/online_sample_ratio": float(train_logs.get("online_sample_ratio", 0.0)),
                    "loss/critic_loss": float(train_logs.get("critic_loss", 0.0)),
                    "loss/actor_loss": float(train_logs.get("actor_loss", 0.0)),
                    "loss/bc_loss": float(train_logs.get("bc_loss", 0.0)),
                    "loss/q_abs": float(train_logs.get("q_abs", 0.0)),
                    "loss/lambda_q": float(train_logs.get("lambda_q", 0.0)),
                    "q/q1_mean": float(train_logs.get("q1_mean", 0.0)),
                    "q/q2_mean": float(train_logs.get("q2_mean", 0.0)),
                    "q/target_q_mean": float(train_logs.get("target_q_mean", 0.0)),
                    "q/q_ref": float(train_logs.get("q_ref", 0.0)),
                    "q/q_exec": float(train_logs.get("q_exec", 0.0)),
                    "q/q_adv": float(train_logs.get("q_adv", 0.0)),
                }
            )
        if cfg["wandb"].get("log_episode", True):
            wandb_logger.log(episode_metrics, step=agent.global_update)
        if train_logs and agent.global_update % log_every_updates == 0:
            wandb_logger.log({f"train/{key}": value for key, value in train_logs.items()}, step=agent.global_update)

        print(
            "[online-td3bc][episode] "
            f"episode={agent.online_episode} scene={scene} phase={phase} "
            f"success={episode_info['success']} reward={episode_info['reward']:.3f} "
            f"length={episode_info['length']} online_buffer={online_store.size} updates={updates} "
            f"actor_updates={agent.actor_updates} critic_updates={agent.critic_updates} "
            f"scene_counts={scene_counts}",
            flush=True,
        )

        if (
            not args.rollout_only
            and eval_every > 0
            and agent.online_episode % eval_every == 0
        ):
            summary_path, eval_metrics = run_paired_eval(
                cfg,
                output_root,
                init_checkpoint,
                latest_actor,
                agent.online_episode,
            )
            if should_log_eval:
                wandb_logger.log(eval_metrics, step=agent.global_update)
                wandb_logger.summary("latest_eval_summary", str(summary_path))
            current_success = eval_metrics.get("eval_all/success_rate", 0.0)
            if current_success > best_eval_success:
                best_eval_success = float(current_success)
                best_actor = output_root / "checkpoints" / "best_actor.pt"
                best_state = output_root / "checkpoints" / "best_online.pt"
                agent.save_actor_checkpoint(best_actor, cfg)
                agent.save_training_checkpoint(best_state, cfg)
                wandb_logger.summary("best_checkpoint", str(best_state))
                wandb_logger.summary("best_actor_checkpoint", str(best_actor))
                if should_log_eval:
                    wandb_logger.summary("best_eval_all_success_rate", best_eval_success)

    print(f"[online-td3bc] done scene_counts={scene_counts}", flush=True)
    final_actor = output_root / "checkpoints" / "final_actor.pt"
    final_state = output_root / "checkpoints" / "final_online.pt"
    agent.save_actor_checkpoint(final_actor, cfg)
    agent.save_training_checkpoint(final_state, cfg)
    wandb_logger.summary("final_checkpoint", str(final_state))
    wandb_logger.summary("final_actor_checkpoint", str(final_actor))
    wandb_logger.finish()


if __name__ == "__main__":
    main()
