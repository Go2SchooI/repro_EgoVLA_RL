from __future__ import annotations

import numpy as np
import pytest

from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.online_td3bc import (
    MutableNormalizedReplay,
    NormalizedReplayView,
    _build_replay_manifest,
    _resume_manifest_mismatches,
    _validate_online_replay_reuse,
    _validate_resume_model_path,
    _validate_resume_replay_manifest,
)
from rl_posttrain.replay_buffer import OfflineReplayBuffer, OnlineReplayBufferWriter, ReplayBufferWriter


def _base_transition(action=None, next_action=None):
    action = np.asarray(action if action is not None else [1.0, 2.0, 3.0], dtype=np.float32)
    next_action = np.asarray(next_action if next_action is not None else [3.0, 4.0, 5.0], dtype=np.float32)
    actor_obs = np.concatenate([np.asarray([10.0, 20.0], dtype=np.float32), action])
    critic_obs = np.concatenate([actor_obs, np.asarray([30.0, 40.0], dtype=np.float32)])
    next_actor_obs = np.concatenate([np.asarray([11.0, 21.0], dtype=np.float32), next_action])
    next_critic_obs = np.concatenate([next_actor_obs, np.asarray([31.0, 41.0], dtype=np.float32)])
    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "action_raw": action.copy(),
        "bc_target_raw": action.copy(),
        "reward": np.array([0.0], dtype=np.float32),
        "done": np.array([1.0], dtype=np.float32),
        "next_actor_obs": next_actor_obs,
        "next_critic_obs": next_critic_obs,
        "next_bc_target_raw": next_action.copy(),
        "source": "base",
    }


def _online_transition():
    a_ref_norm = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    a_exec_norm = np.asarray([0.2, -0.1, 0.25], dtype=np.float32)
    next_ref_norm = np.asarray([0.0, -0.3, 0.4], dtype=np.float32)
    actor_obs = np.concatenate([np.asarray([1.0, 2.0], dtype=np.float32), a_ref_norm])
    critic_obs = np.concatenate([actor_obs, np.asarray([3.0, 4.0], dtype=np.float32)])
    next_actor_obs = np.concatenate([np.asarray([1.5, 2.5], dtype=np.float32), next_ref_norm])
    next_critic_obs = np.concatenate([next_actor_obs, np.asarray([3.5, 4.5], dtype=np.float32)])
    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "action_norm": a_exec_norm,
        "bc_target_norm": a_ref_norm,
        "reward": np.asarray([1.0], dtype=np.float32),
        "done": np.asarray([1.0], dtype=np.float32),
        "next_actor_obs": next_actor_obs,
        "next_critic_obs": next_critic_obs,
        "next_bc_target_norm": next_ref_norm,
        "a_exec_norm": a_exec_norm,
        "a_ref_norm": a_ref_norm,
        "next_a_ref_norm": next_ref_norm,
        "source": "online_actor",
        "scene": "room1_table1",
        "episode_id": 7,
        "trial": 2,
        "env_step": 3,
        "success": 1.0,
        "timeout": 0.0,
        "mean_abs_actor_minus_ref_norm": float(np.abs(a_exec_norm - a_ref_norm).mean()),
        "max_abs_actor_minus_ref_norm": float(np.abs(a_exec_norm - a_ref_norm).max()),
        "num_clipped_dims": 0.0,
    }


def test_validate_online_replay_reuse_requires_explicit_override(tmp_path):
    output_root = tmp_path / "run"
    online_dir = output_root / "online_replay"
    online_dir.mkdir(parents=True)
    (online_dir / "episode_0000_room1_table1.npz").write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        _validate_online_replay_reuse(output_root, resume=False, allow_reuse_online_replay=False)

    paths = _validate_online_replay_reuse(output_root, resume=False, allow_reuse_online_replay=True)
    assert len(paths) == 1


def test_build_replay_manifest_records_shards_and_scene_counts(tmp_path):
    model_path = tmp_path / "checkpoint-3000"
    model_path.mkdir()
    base_path = tmp_path / "base.npz"
    base_writer = ReplayBufferWriter(base_path, metadata={"task": "unit", "scene": "room1_table1"})
    base_writer.add(_base_transition())
    base_writer.save()
    base_replay = OfflineReplayBuffer.load(base_path, replay_filter="base_only")

    online_dir = tmp_path / "online"
    online_dir.mkdir()
    online_writer = OnlineReplayBufferWriter(
        online_dir / "episode_0000_room1_table1.npz",
        action_normalizer=AffineNormalizer.identity(3),
        metadata={"task": "unit", "scene": "room1_table1"},
    )
    online_writer.add(_online_transition())
    online_writer.save()
    online_replay = OfflineReplayBuffer.load(online_dir, replay_filter="all")

    online_store = MutableNormalizedReplay()
    online_store.append(
        NormalizedReplayView(
            online_replay,
            AffineNormalizer.identity(5),
            AffineNormalizer.identity(7),
            online_replay.action_normalizer,
        )
    )
    manifest = _build_replay_manifest(base_path, base_replay, online_dir, online_store, model_path)

    assert manifest["model_path"] == str(model_path.resolve())
    assert manifest["base_replay"]["fingerprint"]
    assert manifest["base_replay"]["scenes"] == ["room1_table1"]
    assert manifest["online_replay"]["shards"] == [str(online_dir / "episode_0000_room1_table1.npz")]
    assert manifest["online_replay"]["scene_counts"] == {"room1_table1": 1}
    assert manifest["online_replay"]["source_counts"] == {"online_actor": 1}


def test_resume_manifest_validation_detects_replay_mismatch():
    checkpoint_manifest = {
        "base_replay": {"fingerprint": "base-a"},
        "online_replay": {
            "shards": ["episode_0000_room1_table1.npz"],
            "size": 1,
            "scene_counts": {"room1_table1": 1},
            "source_counts": {"online_actor": 1},
        },
    }
    current_manifest = {
        "base_replay": {"fingerprint": "base-a"},
        "online_replay": {
            "shards": ["episode_0001_room1_table2.npz"],
            "size": 1,
            "scene_counts": {"room1_table2": 1},
            "source_counts": {"online_actor": 1},
        },
    }

    mismatches = _resume_manifest_mismatches(checkpoint_manifest, current_manifest)
    assert "online_replay.shards" in mismatches
    assert "online_replay.scene_counts" in mismatches
    _validate_resume_replay_manifest(checkpoint_manifest, current_manifest, strict=False)
    with pytest.raises(AssertionError):
        _validate_resume_replay_manifest(checkpoint_manifest, current_manifest, strict=True)


def test_resume_model_path_validation_rejects_old_online_checkpoint(tmp_path):
    current_model = tmp_path / "checkpoint-3000"
    current_model.mkdir()
    checkpoint_manifest = {
        "base_replay": {"fingerprint": "base-a"},
        "online_replay": {"shards": ["episode_0000_room1_table1.npz"], "size": 1},
    }

    with pytest.raises(AssertionError, match="missing frozen EgoVLA model_path"):
        _validate_resume_model_path(
            checkpoint_manifest,
            {"online": {}},
            None,
            current_model,
            checkpoint_online_episode=57,
        )


def test_resume_model_path_validation_detects_model_mismatch(tmp_path):
    checkpoint_model = tmp_path / "ckpt-old"
    current_model = tmp_path / "checkpoint-3000"
    checkpoint_model.mkdir()
    current_model.mkdir()
    checkpoint_manifest = {
        "model_path": str(checkpoint_model),
        "base_replay": {"fingerprint": "base-a"},
        "online_replay": {"shards": [], "size": 0},
    }

    with pytest.raises(AssertionError, match="does not match current config"):
        _validate_resume_model_path(
            checkpoint_manifest,
            {"online": {"model_path": str(checkpoint_model)}},
            str(checkpoint_model),
            current_model,
            checkpoint_online_episode=1,
        )

    _validate_resume_model_path(
        {
            **checkpoint_manifest,
            "model_path": str(current_model),
        },
        {"online": {"model_path": str(current_model)}},
        str(current_model),
        current_model,
        checkpoint_online_episode=1,
    )
