import argparse
import numpy as np
import pytest
import torch

from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.replay_buffer import OfflineReplayBuffer
from rl_posttrain.h_summary import HSummaryConfig
from rl_posttrain.td3bc_ref import (
    PreparedReplay,
    TD3BCConfig,
    TD3BCTrainer,
    action_group_errors,
    build_training_config_payload,
    load_actor_policy,
    resolve_actor_checkpoint_path,
    resolve_training_output_paths,
    write_training_yaml,
)


def _synthetic_replay(size=32):
    rng = np.random.default_rng(7)
    action = rng.uniform(-0.5, 0.5, size=(size, 3)).astype(np.float32)
    actor_prefix = rng.normal(size=(size, 3)).astype(np.float32)
    actor_obs = np.concatenate([actor_prefix, action], axis=-1).astype(np.float32)
    critic_suffix = rng.normal(size=(size, 3)).astype(np.float32)
    critic_obs = np.concatenate([actor_obs, critic_suffix], axis=-1).astype(np.float32)
    arrays = {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "action_norm": action,
        "bc_target_norm": action.copy(),
        "reward": np.zeros((size, 1), dtype=np.float32),
        "done": np.ones((size, 1), dtype=np.float32),
        "next_actor_obs": actor_obs.copy(),
        "next_critic_obs": critic_obs.copy(),
        "next_bc_target_norm": action.copy(),
        "source": np.asarray(["base"] * size),
    }
    return OfflineReplayBuffer(
        arrays,
        metadata={"task": "unit", "action_fields_are_canonical_normalized": True},
        action_normalizer=AffineNormalizer.identity(3),
    )


def test_alpha_zero_actor_loss_is_pure_bc():
    replay = _synthetic_replay()
    cfg = TD3BCConfig(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        batch_size=8,
        policy_delay=1,
        td3bc_alpha=0.0,
        td3bc_bc_weight=1.0,
    )
    prepared = PreparedReplay(replay, cfg)
    trainer = TD3BCTrainer(prepared, cfg, device="cpu", seed=0)

    logs = trainer.train_step()

    assert logs["lambda_q"] == 0.0
    assert abs(logs["actor_loss"] - logs["bc_loss"]) < 1.0e-6
    assert logs["mean_abs_actor_minus_ref_norm"] >= 0.0


def test_action_group_errors_use_action_spec_metadata():
    diff = torch.tensor([[1.0, 2.0, 10.0, 14.0]], dtype=torch.float32)
    action_spec = {
        "dim": 4,
        "slices": [
            {"name": "left_ee_pose", "start": 0, "end": 2, "shape": [2]},
            {"name": "right_qpos", "start": 2, "end": 4, "shape": [2]},
        ],
    }

    logs = action_group_errors(diff, action_spec)

    assert logs["actor_ref_error_left_ee"] == pytest.approx(1.5)
    assert logs["actor_ref_error_right_qpos"] == pytest.approx(12.0)


def test_prepared_replay_rejects_missing_action_normalizer():
    replay = _synthetic_replay()
    replay.action_normalizer = None
    cfg = TD3BCConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), batch_size=8)
    with pytest.raises(ValueError, match="missing action_normalizer"):
        PreparedReplay(replay, cfg)


def test_training_yaml_records_hyperparameters(tmp_path):
    replay = _synthetic_replay()
    cfg = TD3BCConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), batch_size=8, td3bc_alpha=0.03)
    prepared = PreparedReplay(replay, cfg)
    args = argparse.Namespace(
        output=str(tmp_path / "actor.pt"),
        output_dir=str(tmp_path),
        checkpoint_output=str(tmp_path / "actor.pt"),
        config_output=str(tmp_path / "actor.yaml"),
        replay="synthetic.npz",
        steps=12,
        seed=3,
        device="cpu",
        replay_filter="base_only",
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_group=None,
        wandb_tags="",
        wandb_mode=None,
    )

    payload = build_training_config_payload(args, cfg, prepared, replay)
    path = write_training_yaml(tmp_path / "actor.yaml", payload)
    text = path.read_text()

    assert "td3bc_alpha: 0.03" in text
    assert "actor_obs_dim: 6" in text
    assert "critic_obs_dim: 9" in text
    assert "action_dim: 3" in text


def test_output_directory_resolves_actor_and_config_paths(tmp_path):
    output_dir, checkpoint_path, config_path = resolve_training_output_paths(tmp_path / "run_a")

    assert output_dir == tmp_path / "run_a"
    assert checkpoint_path == tmp_path / "run_a" / "actor.pt"
    assert config_path == tmp_path / "run_a" / "config.yaml"


def test_actor_checkpoint_directory_prefers_actor_pt(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    actor_path = run_dir / "actor.pt"
    actor_path.write_bytes(b"placeholder")

    assert resolve_actor_checkpoint_path(run_dir) == actor_path


def test_h_zero_keeps_raw_actor_input_dim():
    replay = _synthetic_replay()
    cfg = TD3BCConfig(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        batch_size=8,
        h_summary=HSummaryConfig(mode="h_zero", h_dim=2),
    )
    prepared = PreparedReplay(replay, cfg)
    trainer = TD3BCTrainer(prepared, cfg, device="cpu", seed=0)

    assert trainer.actor.processed_obs_dim == prepared.actor_obs_dim
    assert trainer.critic.processed_obs_dim == prepared.critic_obs_dim
    logs = trainer.train_step()
    assert "h_actor_param_norm" in logs


def test_h_proj_changes_processed_dims_and_roundtrips_checkpoint(tmp_path):
    replay = _synthetic_replay()
    cfg = TD3BCConfig(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        batch_size=8,
        policy_delay=1,
        h_summary=HSummaryConfig(mode="h_proj", h_dim=2, out_dim=4),
    )
    prepared = PreparedReplay(replay, cfg)
    trainer = TD3BCTrainer(prepared, cfg, device="cpu", seed=0)

    assert trainer.actor.processed_obs_dim == prepared.actor_obs_dim - 2 + 4
    assert trainer.critic.processed_obs_dim == prepared.critic_obs_dim - 2 + 4
    logs = trainer.train_step()
    assert logs["h_actor_param_norm"] > 0.0

    checkpoint_path = trainer.save(tmp_path / "actor.pt")
    bundle = load_actor_policy(checkpoint_path, device="cpu")
    assert bundle["h_summary"].mode == "h_proj"
    assert bundle["h_summary"].requested_mode == "h_proj"
    assert bundle["h_summary"].out_dim == 4
    assert bundle["actor"].processed_obs_dim == prepared.actor_obs_dim - 2 + 4


def test_td3bc_checkpoint_preserves_action_spec_metadata(tmp_path):
    replay = _synthetic_replay()
    replay.metadata["action_spec"] = {
        "dim": 3,
        "slices": [
            {"name": "left_ee_pose", "start": 0, "end": 1, "shape": [1]},
            {"name": "right_ee_pose", "start": 1, "end": 2, "shape": [1]},
            {"name": "left_qpos", "start": 2, "end": 3, "shape": [1]},
        ],
    }
    cfg = TD3BCConfig(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        batch_size=8,
        policy_delay=1,
        h_summary=HSummaryConfig(mode="h_proj", h_dim=2, out_dim=4),
    )
    prepared = PreparedReplay(replay, cfg)
    trainer = TD3BCTrainer(prepared, cfg, device="cpu", seed=0)

    checkpoint_path = trainer.save(tmp_path / "actor_with_action_spec.pt")
    bundle = load_actor_policy(checkpoint_path, device="cpu")

    assert bundle["action_spec"]["dim"] == 3
    assert bundle["checkpoint"]["action_spec"]["slices"][0]["name"] == "left_ee_pose"


def test_h_proj_alias_preserves_requested_mode():
    cfg = HSummaryConfig(mode="h_proj256", h_dim=8)

    assert cfg.mode == "h_proj"
    assert cfg.requested_mode == "h_proj256"
    assert cfg.out_dim == 256
