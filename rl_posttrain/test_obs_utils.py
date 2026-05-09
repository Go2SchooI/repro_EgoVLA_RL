import numpy as np
import pytest

from rl_posttrain.obs_utils import build_critic_obs


def _env_obs():
    return {
        "qpos": np.zeros((1, 50), dtype=np.float32),
        "qvel": np.zeros((1, 50), dtype=np.float32),
        "left_ee_pose": np.zeros((1, 7), dtype=np.float32),
        "right_ee_pose": np.zeros((1, 7), dtype=np.float32),
        "left_target_ee_pose": np.zeros((1, 7), dtype=np.float32),
        "right_target_ee_pose": np.zeros((1, 7), dtype=np.float32),
        "left_finger_tip_pos": np.zeros((1, 5, 3), dtype=np.float32),
        "right_finger_tip_pos": np.zeros((1, 5, 3), dtype=np.float32),
        "left_hand_contact_force": np.zeros((1, 1), dtype=np.float32),
        "right_hand_contact_force": np.zeros((1, 1), dtype=np.float32),
        "action": np.zeros((1, 50), dtype=np.float32),
        "success": np.zeros((1,), dtype=np.float32),
        "move_lid_success": np.zeros((1,), dtype=np.float32),
    }


def test_build_critic_obs_validates_current_schema():
    actor_obs = np.zeros(8, dtype=np.float32)
    critic_obs, report, missing = build_critic_obs(_env_obs(), actor_obs)
    assert critic_obs.shape[0] > actor_obs.shape[0]
    assert report.critic_shapes["left_ee_pose"] == (1, 7)
    assert "object_pose" in missing


def test_build_critic_obs_rejects_bad_ee_shape():
    env_obs = _env_obs()
    env_obs["left_ee_pose"] = np.zeros((1, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="left_ee_pose"):
        build_critic_obs(env_obs, np.zeros(8, dtype=np.float32))


def test_build_critic_obs_accepts_multi_object_pose():
    env_obs = _env_obs()
    env_obs["object_pose"] = np.zeros((1, 2, 7), dtype=np.float32)
    critic_obs, report, missing = build_critic_obs(env_obs, np.zeros(8, dtype=np.float32))
    assert critic_obs.shape[0] > 8
    assert report.critic_shapes["object_pose"] == (1, 2, 7)
    assert "object_pose" not in missing
