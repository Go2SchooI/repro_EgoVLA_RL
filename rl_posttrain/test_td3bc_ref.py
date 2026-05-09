import numpy as np
import pytest

from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.replay_buffer import OfflineReplayBuffer
from rl_posttrain.td3bc_ref import PreparedReplay, TD3BCConfig, TD3BCTrainer


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


def test_prepared_replay_rejects_missing_action_normalizer():
    replay = _synthetic_replay()
    replay.action_normalizer = None
    cfg = TD3BCConfig(actor_hidden_dims=(16,), critic_hidden_dims=(16,), batch_size=8)
    with pytest.raises(ValueError, match="missing action_normalizer"):
        PreparedReplay(replay, cfg)
