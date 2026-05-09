from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from rl_posttrain.action_utils import ActionSpec, make_exec_action_dict, pack_action


PROPRIO_OBS_KEYS = (
    "proprio_input_3d",
    "proprio_input_rot",
    "proprio_input_handdof",
    "proprio_input_hand_finger_tip",
)

CRITIC_ENV_KEYS = (
    "qpos",
    "qvel",
    "left_ee_pose",
    "right_ee_pose",
    "left_target_ee_pose",
    "right_target_ee_pose",
    "left_finger_tip_pos",
    "right_finger_tip_pos",
    "left_hand_contact_force",
    "right_hand_contact_force",
    "action",
    "success",
)

OPTIONAL_CRITIC_ENV_KEYS = (
    "object_pose",
    "reach_success",
    "lift_success",
    "insert_success",
    "unload_success",
    "sort_success",
    "move_lid_success",
    "flip_mug_pose_success",
)

SCALAR_OBS_KEYS = (
    "success",
    "reach_success",
    "lift_success",
    "insert_success",
    "unload_success",
    "sort_success",
    "move_lid_success",
    "flip_mug_pose_success",
)


@dataclass
class ObsBuildReport:
    actor_obs_dim: int = 0
    critic_obs_dim: int = 0
    h_in_shape: Optional[Tuple[int, ...]] = None
    proprio_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    proprio_devices: Dict[str, str] = field(default_factory=dict)
    critic_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    critic_devices: Dict[str, str] = field(default_factory=dict)
    optional_missing: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"h_in_shape={self.h_in_shape} proprio_shapes={self.proprio_shapes} "
            f"proprio_devices={self.proprio_devices} critic_shapes={self.critic_shapes} "
            f"critic_devices={self.critic_devices} optional_missing={self.optional_missing}"
        )


def to_numpy(value: object, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def value_device(value: object) -> str:
    if hasattr(value, "device"):
        return str(value.device)
    return "numpy"


def flatten_value(value: object, dtype=np.float32) -> np.ndarray:
    return to_numpy(value, dtype=dtype).reshape(-1)


def _assert_finite_numeric(key: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"critic_obs field {key!r} must be numeric, got dtype={value.dtype}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"critic_obs field {key!r} contains non-finite values.")


def _assert_num_envs_one(key: str, value: np.ndarray) -> None:
    if value.ndim == 0:
        raise ValueError(f"critic_obs field {key!r} must include num_envs dimension, got scalar.")
    if int(value.shape[0]) != 1:
        raise ValueError(f"critic_obs field {key!r} expected num_envs=1, got shape {value.shape}.")


def validate_critic_field(key: str, value: np.ndarray) -> None:
    _assert_finite_numeric(key, value)
    _assert_num_envs_one(key, value)
    if key in ("qpos", "qvel", "action"):
        if value.ndim != 2 or int(value.shape[1]) <= 0:
            raise ValueError(f"critic_obs field {key!r} expected shape [1, D], got {value.shape}.")
    elif key in ("left_ee_pose", "right_ee_pose", "left_target_ee_pose", "right_target_ee_pose"):
        if tuple(value.shape) != (1, 7):
            raise ValueError(f"critic_obs field {key!r} expected shape [1, 7], got {value.shape}.")
    elif key in ("left_finger_tip_pos", "right_finger_tip_pos"):
        if value.ndim == 3:
            if int(value.shape[-1]) != 3 or int(value.shape[1]) <= 0:
                raise ValueError(f"critic_obs field {key!r} expected shape [1, N, 3], got {value.shape}.")
        elif value.ndim == 2:
            if int(value.shape[1]) <= 0 or int(value.shape[1]) % 3 != 0:
                raise ValueError(
                    f"critic_obs field {key!r} expected flattened [1, 3*N] or [1, N, 3], got {value.shape}."
                )
        else:
            raise ValueError(f"critic_obs field {key!r} expected rank 2 or 3, got {value.shape}.")
    elif key in ("left_hand_contact_force", "right_hand_contact_force"):
        if value.ndim == 3:
            if int(value.shape[-1]) != 3 or int(value.shape[1]) <= 0:
                raise ValueError(f"critic_obs field {key!r} expected shape [1, N, 3], got {value.shape}.")
        elif value.ndim == 2:
            if int(value.shape[1]) <= 0:
                raise ValueError(f"critic_obs field {key!r} expected shape [1, D], got {value.shape}.")
        else:
            raise ValueError(f"critic_obs field {key!r} expected rank 2 or 3, got {value.shape}.")
    elif key in SCALAR_OBS_KEYS:
        if value.reshape(-1).size != 1:
            raise ValueError(f"critic_obs field {key!r} expected one scalar for num_envs=1, got {value.shape}.")
    elif key == "object_pose":
        if value.ndim == 3:
            if int(value.shape[1]) <= 0 or int(value.shape[2]) not in (7, 13):
                raise ValueError(
                    f"optional critic_obs field {key!r} expected [1, N, 7] or [1, N, 13], got {value.shape}."
                )
        elif value.ndim == 2:
            dim = int(value.shape[1])
            if dim <= 0 or (dim not in (7, 13) and dim % 7 != 0 and dim % 13 != 0):
                raise ValueError(
                    f"optional critic_obs field {key!r} expected [1, 7], [1, 13], "
                    f"or flattened [1, K*7]/[1, K*13], got {value.shape}."
                )
        else:
            raise ValueError(f"optional critic_obs field {key!r} expected rank 2 or 3, got {value.shape}.")
    else:
        if value.size <= 0:
            raise ValueError(f"critic_obs field {key!r} is empty with shape {value.shape}.")


def latent_summary(h_in: object | None) -> tuple[np.ndarray, Optional[Tuple[int, ...]]]:
    if h_in is None:
        return np.zeros((0,), dtype=np.float32), None
    array = to_numpy(h_in, dtype=np.float32)
    shape = tuple(array.shape)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32), shape
    if array.ndim == 1:
        return array.reshape(-1).astype(np.float32, copy=False), shape
    return array.reshape(-1, array.shape[-1]).mean(axis=0).astype(np.float32, copy=False), shape


def flatten_proprio(
    proprio_pack: Mapping[str, object],
) -> tuple[np.ndarray, Dict[str, Tuple[int, ...]], Dict[str, str]]:
    parts = []
    shapes: Dict[str, Tuple[int, ...]] = {}
    devices: Dict[str, str] = {}
    missing = [key for key in PROPRIO_OBS_KEYS if key not in proprio_pack]
    if missing:
        raise KeyError(
            f"Missing deployable proprio keys for actor_obs: {missing}. "
            f"Available keys: {sorted(proprio_pack.keys())}"
        )
    for key in PROPRIO_OBS_KEYS:
        devices[key] = value_device(proprio_pack[key])
        value = to_numpy(proprio_pack[key], dtype=np.float32)
        shapes[key] = tuple(value.shape)
        parts.append(value.reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False), shapes, devices


def pack_base_chunk_summary(
    action_dict: Mapping[str, object],
    spec: ActionSpec,
) -> np.ndarray:
    left_ee = to_numpy(action_dict["left_ee_pose"], dtype=np.float32)
    right_ee = to_numpy(action_dict["right_ee_pose"], dtype=np.float32)
    left_qpos = to_numpy(action_dict["left_qpos_multi_step"], dtype=np.float32)
    right_qpos = to_numpy(action_dict["right_qpos_multi_step"], dtype=np.float32)
    horizon = int(left_ee.shape[0])
    if horizon <= 0:
        raise ValueError("Cannot summarize an empty action chunk.")
    for name, value in (
        ("right_ee_pose", right_ee),
        ("left_qpos_multi_step", left_qpos),
        ("right_qpos_multi_step", right_qpos),
    ):
        if int(value.shape[0]) != horizon:
            raise ValueError(f"Action chunk field {name} has horizon {value.shape[0]}, expected {horizon}.")

    def pack_step(index: int) -> np.ndarray:
        return pack_action(
            make_exec_action_dict(left_ee[index], right_ee[index], left_qpos[index], right_qpos[index]),
            spec,
        )

    first = pack_step(0)
    last = pack_step(horizon - 1)
    mean = np.stack([pack_step(i) for i in range(horizon)], axis=0).mean(axis=0)
    return np.concatenate([first, last, last - first, mean], axis=0).astype(np.float32, copy=False)


def build_actor_obs(
    h_in: object | None,
    proprio_pack: Mapping[str, object],
    action_dict: Mapping[str, object],
    action_spec: ActionSpec,
    a_ref_norm: np.ndarray,
) -> tuple[np.ndarray, ObsBuildReport]:
    h_summary, h_shape = latent_summary(h_in)
    proprio, proprio_shapes, proprio_devices = flatten_proprio(proprio_pack)
    chunk_summary = pack_base_chunk_summary(action_dict, action_spec)
    a_ref_norm = to_numpy(a_ref_norm, dtype=np.float32).reshape(-1)
    if a_ref_norm.shape != (action_spec.dim,):
        raise ValueError(f"a_ref_norm shape mismatch: got {a_ref_norm.shape}, expected {(action_spec.dim,)}.")
    actor_obs = np.concatenate([h_summary, proprio, chunk_summary, a_ref_norm], axis=0)
    actor_obs = actor_obs.astype(np.float32, copy=False)
    report = ObsBuildReport(
        actor_obs_dim=int(actor_obs.shape[0]),
        h_in_shape=h_shape,
        proprio_shapes=proprio_shapes,
        proprio_devices=proprio_devices,
    )
    return actor_obs, report


def build_critic_obs(
    env_obs: Mapping[str, object],
    actor_obs: np.ndarray,
    optional_warned: Optional[set[str]] = None,
) -> tuple[np.ndarray, ObsBuildReport, List[str]]:
    parts = [to_numpy(actor_obs, dtype=np.float32).reshape(-1)]
    shapes: Dict[str, Tuple[int, ...]] = {"actor_obs": tuple(actor_obs.shape)}
    devices: Dict[str, str] = {"actor_obs": "numpy"}
    missing_required = [key for key in CRITIC_ENV_KEYS if key not in env_obs]
    if missing_required:
        raise KeyError(
            f"Missing required critic_obs env keys: {missing_required}. "
            f"Available keys: {sorted(env_obs.keys())}"
        )

    for key in CRITIC_ENV_KEYS:
        devices[key] = value_device(env_obs[key])
        value = to_numpy(env_obs[key], dtype=np.float32)
        validate_critic_field(key, value)
        shapes[key] = tuple(value.shape)
        parts.append(value.reshape(-1))

    newly_missing = []
    for key in OPTIONAL_CRITIC_ENV_KEYS:
        if key in env_obs:
            devices[key] = value_device(env_obs[key])
            value = to_numpy(env_obs[key], dtype=np.float32)
            validate_critic_field(key, value)
            shapes[key] = tuple(value.shape)
            parts.append(value.reshape(-1))
        elif optional_warned is None or key not in optional_warned:
            newly_missing.append(key)

    critic_obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    report = ObsBuildReport(
        actor_obs_dim=int(actor_obs.shape[0]),
        critic_obs_dim=int(critic_obs.shape[0]),
        critic_shapes=shapes,
        critic_devices=devices,
        optional_missing=newly_missing,
    )
    return critic_obs, report, newly_missing


def success_value(env_obs: Mapping[str, object]) -> float:
    if "success" not in env_obs:
        raise KeyError("env_obs does not contain final success key 'success'.")
    success = to_numpy(env_obs["success"], dtype=np.float32).reshape(-1)
    if success.size != 1:
        raise AssertionError(f"Expected scalar success for num_envs=1, got shape {success.shape}.")
    return float(success[0])


def subtask_success_metrics(env_obs: Mapping[str, object]) -> Dict[str, float]:
    metrics = {}
    for key in OPTIONAL_CRITIC_ENV_KEYS:
        if key in env_obs and "success" in key:
            value = to_numpy(env_obs[key], dtype=np.float32).reshape(-1)
            if value.size == 1:
                metrics[key] = float(value[0])
    return metrics
