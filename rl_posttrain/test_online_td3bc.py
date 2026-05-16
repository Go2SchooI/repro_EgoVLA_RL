from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from rl_posttrain import online_td3bc as online_mod
from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.online_td3bc import (
    MutableNormalizedReplay,
    NormalizedReplayView,
    _apply_derived_naming,
    _build_replay_manifest,
    _deep_copy_config,
    _derive_run_name,
    _load_static_eval_cache,
    _resume_manifest_mismatches,
    _run_paired_eval_with_static_cache,
    _static_eval_cache_descriptor,
    _static_eval_cache_path,
    _validate_online_replay_reuse,
    _validate_resume_model_path,
    _validate_resume_replay_manifest,
    _write_static_eval_cache,
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


def test_derive_run_name_uses_command_relevant_fields():
    cfg = _deep_copy_config()
    cfg["online"]["init_checkpoint"] = "playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj128_alpha0"
    cfg["online"]["model_path"] = "checkpoints/ego_vla_checkpoint/checkpoint-3000"
    cfg["td3bc"]["alpha"] = 0.003
    cfg["td3bc"]["bc_weight"] = 0.1

    assert _derive_run_name(cfg) == "h_proj128_alpha0_online_v1_ckpt3000_td3alpha0003_bc01"
    assert (
        _derive_run_name(cfg, suffix="debug smoke")
        == "h_proj128_alpha0_online_v1_ckpt3000_td3alpha0003_bc01_debug_smoke"
    )


def test_apply_derived_naming_updates_output_root_and_wandb_fields(tmp_path):
    cfg = _deep_copy_config()
    cfg["online"]["init_checkpoint"] = "h_proj128_alpha0001"
    cfg["online"]["model_path"] = "checkpoints/ego_vla_checkpoint/checkpoint-3000"
    cfg["td3bc"]["alpha"] = 0.001
    cfg["td3bc"]["bc_weight"] = 0.3
    args = argparse.Namespace(
        auto_name=True,
        init_checkpoint="h_proj128_alpha0001",
        model_path=None,
        td3bc_alpha=None,
        bc_weight=0.3,
        noise_std=None,
        noise_clip=None,
        critic_only_base_ratio=None,
        critic_only_online_ratio=None,
        joint_base_ratio=None,
        joint_online_ratio=None,
        name_suffix="trial_a",
        output_root_base=str(tmp_path),
        output_root=None,
        wandb_run_name=None,
        wandb_group="unit_group",
    )

    _apply_derived_naming(cfg, args)

    assert cfg["wandb"]["run_name"] == "h_proj128_alpha0001_online_v1_ckpt3000_td3alpha0001_bc03_trial_a"
    assert cfg["wandb"]["group"] == "unit_group"
    assert cfg["online"]["output_root"] == str(tmp_path / cfg["wandb"]["run_name"])
    assert "td3alpha0001" in cfg["wandb"]["tags"]
    assert "bc03" in cfg["wandb"]["tags"]


def test_apply_derived_naming_respects_explicit_output_root(tmp_path):
    cfg = _deep_copy_config()
    explicit_output = tmp_path / "explicit"
    cfg["online"]["output_root"] = str(explicit_output)
    args = argparse.Namespace(
        auto_name=True,
        init_checkpoint=None,
        model_path=None,
        td3bc_alpha=None,
        bc_weight=None,
        noise_std=None,
        noise_clip=None,
        critic_only_base_ratio=None,
        critic_only_online_ratio=None,
        joint_base_ratio=None,
        joint_online_ratio=None,
        name_suffix=None,
        output_root_base=str(tmp_path),
        output_root=str(explicit_output),
        wandb_run_name="manual_run",
        wandb_group=None,
    )

    _apply_derived_naming(cfg, args)

    assert cfg["wandb"]["run_name"] == "manual_run"
    assert cfg["online"]["output_root"] == str(explicit_output)


def test_static_eval_cache_reuses_baseline_and_static_actor(tmp_path, monkeypatch):
    cfg = _deep_copy_config()
    model_path = tmp_path / "checkpoint-3000"
    model_path.mkdir()
    init_dir = tmp_path / "offline_init"
    init_dir.mkdir()
    init_checkpoint = init_dir / "actor.pt"
    init_checkpoint.write_bytes(b"placeholder")
    cfg["online"]["model_path"] = str(model_path)
    cfg["eval"]["num_episodes"] = 1
    cfg["eval"]["num_trials"] = 1
    cfg["eval"]["include_identity"] = False
    scenes = ["room1_table1"]

    source_root = tmp_path / "source_eval" / "room1_table1"
    source_root.mkdir(parents=True)
    source_scene_summary_path = source_root / "paired_summary.json"
    source_scene_summary = {
        "runs": {
            "baseline": {
                "mode": "off",
                "results": [False],
                "success_rate": 0.0,
                "records": [{"success": False}],
                "metric_means": {},
            },
            "actors": {
                "offline_init": {
                    "mode": "actor",
                    "results": [False],
                    "success_rate": 0.0,
                    "records": [{"success": False}],
                    "metric_means": {},
                },
                "latest_actor": {
                    "mode": "actor",
                    "results": [False],
                    "success_rate": 0.0,
                    "records": [{"success": False}],
                    "metric_means": {},
                },
            },
        },
        "comparisons": {},
        "scene": {"room_idx": 1, "table_idx": 1},
        "model_path": str(model_path),
    }
    source_scene_summary_path.write_text(json.dumps(source_scene_summary))
    source_summary_path = tmp_path / "source_eval" / "paired_summary.json"
    source_summary_path.write_text(json.dumps({"scenes": [{"summary_path": str(source_scene_summary_path)}]}))

    descriptor = _static_eval_cache_descriptor(cfg, init_checkpoint, scenes)
    cache_path = _static_eval_cache_path(tmp_path, descriptor)
    _write_static_eval_cache(source_summary_path, cache_path, descriptor)
    cache_payload = _load_static_eval_cache(cache_path, descriptor)

    def fake_run_eval(mode, args, output_root, actor_checkpoint=None, run_name=None):
        assert mode == "actor"
        assert run_name == "actor_latest_actor"
        assert args.room_idx == 1
        assert args.table_idx == 1
        return {
            "mode": "actor",
            "actor_checkpoint": str(actor_checkpoint),
            "model_path": args.model_path,
            "room_idx": 1,
            "table_idx": 1,
            "run_dir": str(Path(output_root) / run_name),
            "result_path": str(Path(output_root) / run_name / "results_local_eval.txt"),
            "results": [True],
            "success_rate": 1.0,
            "records": [{"success": True, "episode_length": 7.0}],
            "metric_means": {"episode_length": 7.0},
        }

    monkeypatch.setattr(online_mod, "paired_run_eval", fake_run_eval)
    summary_path = _run_paired_eval_with_static_cache(
        cfg,
        tmp_path / "cached_eval",
        tmp_path / "checkpoints" / "latest_actor.pt",
        cache_payload,
    )

    summary = json.loads(summary_path.read_text())
    scene_summary = json.loads(Path(summary["scenes"][0]["summary_path"]).read_text())
    assert scene_summary["runs"]["baseline"]["results"] == [False]
    assert scene_summary["runs"]["actors"]["offline_init"]["results"] == [False]
    assert scene_summary["runs"]["actors"]["latest_actor"]["results"] == [True]
    assert scene_summary["comparisons"]["latest_actor"]["recover"] == 1
    assert summary["aggregate"]["actors"]["latest_actor"]["success_rate"] == 1.0
