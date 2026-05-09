from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from rl_posttrain.normalizer import AffineNormalizer


FAST_FIELDS = (
    "actor_obs",
    "critic_obs",
    "action_norm",
    "bc_target_norm",
    "reward",
    "done",
    "next_actor_obs",
    "next_critic_obs",
    "next_bc_target_norm",
)

RAW_ACTION_FIELDS = (
    "action_raw",
    "bc_target_raw",
    "next_bc_target_raw",
)

WRITER_REQUIRED_FIELDS = (
    "actor_obs",
    "critic_obs",
    "reward",
    "done",
    "next_actor_obs",
    "next_critic_obs",
    *RAW_ACTION_FIELDS,
)

BASE_REPLAY_SOURCES = ("base", "identity")
ACTOR_REPLAY_SOURCES = ("td3bc_actor",)
OPTIONAL_SCALAR_FIELDS = (
    "success",
    "timeout",
    "episode_id",
    "episode_step",
    "episode_length",
    "episode_success",
    "episode_timeout",
)


@dataclass
class ReplayBufferWriter:
    path: str | Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    save_raw: bool = False
    allowed_sources: tuple[str, ...] = BASE_REPLAY_SOURCES
    action_normalizer_mode: str = "fit_minmax"
    action_norm_clip: float = 1.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.steps: List[Dict[str, Any]] = []
        self.raw_steps: List[Dict[str, Any]] = []
        self._dims: Dict[str, tuple[int, ...]] = {}

    def add(self, transition: Mapping[str, Any], raw: Optional[Mapping[str, Any]] = None) -> None:
        missing = [field for field in WRITER_REQUIRED_FIELDS if field not in transition]
        if missing:
            raise KeyError(f"Replay transition missing writer fields: {missing}")
        if "source" not in transition:
            raise KeyError("Replay transition missing required 'source' field.")

        clean: Dict[str, Any] = {}
        for field in WRITER_REQUIRED_FIELDS:
            array = np.asarray(transition[field], dtype=np.float32)
            if field in ("reward", "done"):
                array = array.reshape(1)
            else:
                array = array.reshape(-1)
            if field not in self._dims:
                self._dims[field] = tuple(array.shape)
            elif self._dims[field] != tuple(array.shape):
                raise ValueError(
                    f"Replay field {field!r} shape changed from {self._dims[field]} to {tuple(array.shape)}."
                )
            clean[field] = array

        source = str(transition["source"])
        if source in ACTOR_REPLAY_SOURCES and source not in self.allowed_sources:
            raise ValueError(
                f"Replay source {source!r} is actor-generated data and is not allowed by this writer. "
                f"Allowed sources are {self.allowed_sources}; base replay collection must stay base/identity only."
            )
        if source not in self.allowed_sources:
            raise ValueError(f"Unsupported replay source {source!r}; allowed sources are {self.allowed_sources}.")
        clean["source"] = source
        for optional in OPTIONAL_SCALAR_FIELDS:
            if optional in transition:
                clean[optional] = np.asarray(transition[optional], dtype=np.float32).reshape(1)
        self.steps.append(clean)

        if self.save_raw and raw is not None:
            self.raw_steps.append(dict(raw))

    def __len__(self) -> int:
        return len(self.steps)

    def set_episode_result(
        self,
        episode_id: int | float,
        episode_length: int | float,
        episode_success: int | float,
        timeout: int | float,
    ) -> None:
        episode_id_int = int(episode_id)
        for step in self.steps:
            if "episode_id" not in step:
                continue
            if int(step["episode_id"].reshape(-1)[0]) != episode_id_int:
                continue
            step["episode_length"] = np.asarray([episode_length], dtype=np.float32)
            step["episode_success"] = np.asarray([episode_success], dtype=np.float32)
            step["episode_timeout"] = np.asarray([timeout], dtype=np.float32)

    def save(self) -> Path:
        if not self.steps:
            raise RuntimeError("No replay transitions were collected; refusing to write an empty replay.")
        if self.action_normalizer_mode != "fit_minmax":
            raise ValueError(f"Unsupported action_normalizer_mode={self.action_normalizer_mode!r}.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        actor_obs = np.stack([step["actor_obs"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        critic_obs = np.stack([step["critic_obs"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        next_actor_obs = np.stack([step["next_actor_obs"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        next_critic_obs = np.stack(
            [step["next_critic_obs"] for step in self.steps], axis=0
        ).astype(np.float32, copy=False)
        action_raw = np.stack([step["action_raw"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        bc_target_raw = np.stack(
            [step["bc_target_raw"] for step in self.steps], axis=0
        ).astype(np.float32, copy=False)
        next_bc_target_raw = np.stack(
            [step["next_bc_target_raw"] for step in self.steps], axis=0
        ).astype(np.float32, copy=False)

        action_dim = int(bc_target_raw.shape[-1])
        if actor_obs.shape[-1] < action_dim or next_actor_obs.shape[-1] < action_dim:
            raise ValueError(
                f"actor_obs dim is too small for action tail rewrite: "
                f"actor_obs={actor_obs.shape}, action_dim={action_dim}."
            )
        if critic_obs.shape[-1] < actor_obs.shape[-1] or next_critic_obs.shape[-1] < next_actor_obs.shape[-1]:
            raise ValueError(
                "critic_obs must begin with actor_obs so action tail can be canonicalized consistently."
            )
        actor_tail_error = float(np.max(np.abs(actor_obs[:, -action_dim:] - bc_target_raw)))
        next_actor_tail_error = float(np.max(np.abs(next_actor_obs[:, -action_dim:] - next_bc_target_raw)))
        if actor_tail_error > 1.0e-5 or next_actor_tail_error > 1.0e-5:
            raise ValueError(
                "actor_obs action tail did not match raw BC targets before canonicalization: "
                f"current={actor_tail_error:.8g} next={next_actor_tail_error:.8g}."
            )

        action_normalizer = AffineNormalizer.fit_minmax(bc_target_raw)
        action_norm_unclipped = action_normalizer.normalize(action_raw, clip=None)
        bc_target_norm_unclipped = action_normalizer.normalize(bc_target_raw, clip=None)
        next_bc_target_norm_unclipped = action_normalizer.normalize(next_bc_target_raw, clip=None)
        action_norm = np.clip(
            action_norm_unclipped, -self.action_norm_clip, self.action_norm_clip
        ).astype(np.float32, copy=False)
        bc_target_norm = np.clip(
            bc_target_norm_unclipped, -self.action_norm_clip, self.action_norm_clip
        ).astype(np.float32, copy=False)
        next_bc_target_norm = np.clip(
            next_bc_target_norm_unclipped, -self.action_norm_clip, self.action_norm_clip
        ).astype(np.float32, copy=False)

        actor_obs = actor_obs.copy()
        next_actor_obs = next_actor_obs.copy()
        critic_obs = critic_obs.copy()
        next_critic_obs = next_critic_obs.copy()
        actor_obs[:, -action_dim:] = bc_target_norm
        next_actor_obs[:, -action_dim:] = next_bc_target_norm
        actor_obs_dim = int(actor_obs.shape[-1])
        next_actor_obs_dim = int(next_actor_obs.shape[-1])
        critic_obs[:, :actor_obs_dim] = actor_obs
        next_critic_obs[:, :next_actor_obs_dim] = next_actor_obs

        data["actor_obs"] = actor_obs
        data["critic_obs"] = critic_obs
        data["action_norm"] = action_norm
        data["bc_target_norm"] = bc_target_norm
        data["reward"] = np.stack([step["reward"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        data["done"] = np.stack([step["done"] for step in self.steps], axis=0).astype(np.float32, copy=False)
        data["next_actor_obs"] = next_actor_obs
        data["next_critic_obs"] = next_critic_obs
        data["next_bc_target_norm"] = next_bc_target_norm
        data["action_raw"] = action_raw
        data["bc_target_raw"] = bc_target_raw
        data["next_bc_target_raw"] = next_bc_target_raw
        data["source"] = np.asarray([step["source"] for step in self.steps])
        for optional in OPTIONAL_SCALAR_FIELDS:
            if optional in self.steps[0]:
                data[optional] = np.stack([step[optional] for step in self.steps], axis=0).astype(np.float32)
        normalizer_state = action_normalizer.state_dict()
        data["action_normalizer_mean"] = normalizer_state["mean"]
        data["action_normalizer_scale"] = normalizer_state["scale"]
        data["action_normalizer_eps"] = np.asarray([normalizer_state["eps"]], dtype=np.float32)
        metadata = dict(self.metadata)
        field_shapes = {field: list(np.asarray(value).shape[1:]) for field, value in data.items() if field in FAST_FIELDS}
        metadata.update(
            {
                "num_transitions": len(self.steps),
                "field_shapes": field_shapes,
                "raw_field_shapes": {key: list(self._dims[key]) for key in RAW_ACTION_FIELDS},
                "format": "rl_posttrain_npz_v2",
                "action_fields_are_canonical_normalized": True,
                "action_normalizer": {
                    "mode": self.action_normalizer_mode,
                    "fit_field": "bc_target_raw",
                    "clip": float(self.action_norm_clip),
                    "dim": action_dim,
                },
                "action_num_clipped_dims": int(np.sum(np.abs(action_norm_unclipped) > self.action_norm_clip)),
                "bc_target_num_clipped_dims": int(np.sum(np.abs(bc_target_norm_unclipped) > self.action_norm_clip)),
                "next_bc_target_num_clipped_dims": int(
                    np.sum(np.abs(next_bc_target_norm_unclipped) > self.action_norm_clip)
                ),
            }
        )
        data["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(self.path, **data)

        if self.save_raw and self.raw_steps:
            raw_path = self.path.with_suffix(".raw.pkl")
            with raw_path.open("wb") as f:
                pickle.dump({"metadata": metadata, "steps": self.raw_steps}, f)
        return self.path


class OfflineReplayBuffer:
    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        action_normalizer: Optional[AffineNormalizer] = None,
    ):
        self.arrays = dict(arrays)
        self.metadata = metadata or {}
        self.action_normalizer = action_normalizer
        missing = [field for field in FAST_FIELDS if field not in self.arrays]
        if missing:
            raise KeyError(f"Replay file missing fast fields: {missing}")
        if "source" not in self.arrays:
            raise KeyError("Replay file missing required 'source' field.")
        self.size = int(self.arrays["actor_obs"].shape[0])
        for field in FAST_FIELDS:
            if int(self.arrays[field].shape[0]) != self.size:
                raise ValueError(f"Replay field {field!r} has inconsistent length {self.arrays[field].shape[0]}.")
        for field in ("action_norm", "bc_target_norm", "next_bc_target_norm"):
            value = np.asarray(self.arrays[field], dtype=np.float32)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Replay field {field!r} contains non-finite values.")
            max_abs = float(np.max(np.abs(value)))
            if max_abs > 1.0 + 1.0e-5:
                raise ValueError(
                    f"Replay field {field!r} is outside canonical normalized range [-1, 1]: max_abs={max_abs}."
                )

    @classmethod
    def load(cls, path: str | Path, replay_filter: str = "all") -> "OfflineReplayBuffer":
        path = Path(path)
        if path.is_dir():
            paths = sorted(item for item in path.rglob("*.npz") if item.is_file())
            return cls.load_many(paths, replay_filter=replay_filter)

        data = np.load(path, allow_pickle=False)
        action_normalizer = None
        if {"action_normalizer_mean", "action_normalizer_scale", "action_normalizer_eps"}.issubset(data.files):
            action_normalizer = AffineNormalizer.from_state_dict(
                {
                    "mean": data["action_normalizer_mean"],
                    "scale": data["action_normalizer_scale"],
                    "eps": float(np.asarray(data["action_normalizer_eps"]).reshape(-1)[0]),
                }
            )
        arrays = {
            key: data[key]
            for key in data.files
            if key
            not in (
                "metadata_json",
                "action_normalizer_mean",
                "action_normalizer_scale",
                "action_normalizer_eps",
            )
        }
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        replay = cls(arrays, metadata, action_normalizer=action_normalizer)
        return replay._apply_filter(replay_filter)

    @classmethod
    def load_many(cls, paths: Sequence[str | Path], replay_filter: str = "all") -> "OfflineReplayBuffer":
        replay_paths = [Path(path) for path in paths]
        if not replay_paths:
            raise ValueError("No replay .npz files were found to load.")
        if len(replay_paths) == 1:
            return cls.load(replay_paths[0], replay_filter=replay_filter)

        replays = [cls.load(path, replay_filter="all") for path in replay_paths]
        required_raw = set(RAW_ACTION_FIELDS)
        for path, replay in zip(replay_paths, replays):
            missing_raw = sorted(required_raw.difference(replay.arrays))
            if missing_raw:
                raise ValueError(
                    f"Cannot merge replay {path}: missing raw action fields {missing_raw}. "
                    "Multiple replay files are merged by re-fitting one global action normalizer "
                    "from bc_target_raw, then rewriting canonical *_norm fields."
                )

        reference_shapes = {
            field: tuple(np.asarray(replays[0].arrays[field]).shape[1:])
            for field in (*FAST_FIELDS, *RAW_ACTION_FIELDS)
        }
        for path, replay in zip(replay_paths, replays):
            for field, expected in reference_shapes.items():
                got = tuple(np.asarray(replay.arrays[field]).shape[1:])
                if got != expected:
                    raise ValueError(
                        f"Cannot merge replay {path}: field {field!r} shape suffix {got} "
                        f"does not match expected {expected}."
                    )

        arrays: Dict[str, np.ndarray] = {}
        for field in (*FAST_FIELDS, *RAW_ACTION_FIELDS):
            arrays[field] = np.concatenate(
                [np.asarray(replay.arrays[field]) for replay in replays], axis=0
            ).astype(np.float32, copy=False)
        arrays["source"] = np.concatenate([replay.arrays["source"].astype(str) for replay in replays], axis=0)

        for optional in OPTIONAL_SCALAR_FIELDS:
            if all(optional in replay.arrays for replay in replays):
                arrays[optional] = np.concatenate(
                    [np.asarray(replay.arrays[optional], dtype=np.float32) for replay in replays], axis=0
                ).astype(np.float32, copy=False)

        action_dim = int(arrays["bc_target_raw"].shape[-1])
        actor_obs_dim = int(arrays["actor_obs"].shape[-1])
        next_actor_obs_dim = int(arrays["next_actor_obs"].shape[-1])
        if actor_obs_dim < action_dim or next_actor_obs_dim < action_dim:
            raise ValueError(
                f"Cannot merge replay files: actor_obs dims are too small for action_dim={action_dim}: "
                f"actor_obs_dim={actor_obs_dim}, next_actor_obs_dim={next_actor_obs_dim}."
            )
        if arrays["critic_obs"].shape[-1] < actor_obs_dim or arrays["next_critic_obs"].shape[-1] < next_actor_obs_dim:
            raise ValueError("Cannot merge replay files: critic_obs must start with actor_obs.")

        action_normalizer = AffineNormalizer.fit_minmax(arrays["bc_target_raw"])
        action_norm_unclipped = action_normalizer.normalize(arrays["action_raw"], clip=None)
        bc_target_norm_unclipped = action_normalizer.normalize(arrays["bc_target_raw"], clip=None)
        next_bc_target_norm_unclipped = action_normalizer.normalize(arrays["next_bc_target_raw"], clip=None)
        arrays["action_norm"] = np.clip(action_norm_unclipped, -1.0, 1.0).astype(np.float32, copy=False)
        arrays["bc_target_norm"] = np.clip(bc_target_norm_unclipped, -1.0, 1.0).astype(np.float32, copy=False)
        arrays["next_bc_target_norm"] = np.clip(
            next_bc_target_norm_unclipped, -1.0, 1.0
        ).astype(np.float32, copy=False)

        arrays["actor_obs"] = arrays["actor_obs"].copy()
        arrays["next_actor_obs"] = arrays["next_actor_obs"].copy()
        arrays["critic_obs"] = arrays["critic_obs"].copy()
        arrays["next_critic_obs"] = arrays["next_critic_obs"].copy()
        arrays["actor_obs"][:, -action_dim:] = arrays["bc_target_norm"]
        arrays["next_actor_obs"][:, -action_dim:] = arrays["next_bc_target_norm"]
        arrays["critic_obs"][:, :actor_obs_dim] = arrays["actor_obs"]
        arrays["next_critic_obs"][:, :next_actor_obs_dim] = arrays["next_actor_obs"]

        metadata = {
            "format": "rl_posttrain_npz_merged_v1",
            "num_replays": len(replays),
            "num_transitions": int(arrays["actor_obs"].shape[0]),
            "source_paths": [str(path) for path in replay_paths],
            "action_fields_are_canonical_normalized": True,
            "action_normalizer": {
                "mode": "fit_minmax",
                "fit_field": "merged_bc_target_raw",
                "clip": 1.0,
                "dim": action_dim,
            },
            "action_num_clipped_dims": int(np.sum(np.abs(action_norm_unclipped) > 1.0)),
            "bc_target_num_clipped_dims": int(np.sum(np.abs(bc_target_norm_unclipped) > 1.0)),
            "next_bc_target_num_clipped_dims": int(np.sum(np.abs(next_bc_target_norm_unclipped) > 1.0)),
            "child_metadata": [replay.metadata for replay in replays],
        }
        merged = cls(arrays, metadata, action_normalizer=action_normalizer)
        return merged._apply_filter(replay_filter)

    def _apply_filter(self, replay_filter: str) -> "OfflineReplayBuffer":
        if replay_filter == "all":
            return self
        if replay_filter != "base_only":
            raise ValueError(f"Unsupported replay_filter={replay_filter!r}.")
        if "source" not in self.arrays:
            raise AssertionError("replay_filter=base_only requires replay source field.")
        sources = self.arrays["source"].astype(str)
        mask = np.isin(sources, np.asarray(["base", "identity"]))
        if not np.all(mask):
            bad = sorted(set(sources[~mask].tolist()))
            raise AssertionError(f"replay_filter=base_only found non-base sources: {bad}")
        return self

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            field: self.arrays[field][idx]
            for field in FAST_FIELDS
        }
