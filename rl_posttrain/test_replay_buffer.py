import numpy as np
import pytest

from rl_posttrain.replay_buffer import OfflineReplayBuffer, ReplayBufferWriter


def _transition(source="base", action=None, next_action=None):
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
        "source": source,
        "success": np.array([0.0], dtype=np.float32),
        "timeout": np.array([1.0], dtype=np.float32),
    }


def test_replay_writer_roundtrip(tmp_path):
    path = tmp_path / "base_replay.npz"
    writer = ReplayBufferWriter(path, metadata={"task": "unit"})
    writer.add(_transition("base", action=[1.0, 2.0, 3.0], next_action=[3.0, 4.0, 5.0]))
    writer.add(_transition("identity", action=[3.0, 4.0, 5.0], next_action=[1.0, 2.0, 3.0]))
    writer.save()

    replay = OfflineReplayBuffer.load(path, replay_filter="base_only")
    assert replay.size == 2
    assert replay.action_normalizer is not None
    assert replay.arrays["actor_obs"].shape == (2, 5)
    assert replay.arrays["critic_obs"].shape == (2, 7)
    assert replay.arrays["action_norm"].shape == (2, 3)
    assert replay.arrays["source"].astype(str).tolist() == ["base", "identity"]
    assert replay.metadata["task"] == "unit"
    np.testing.assert_allclose(replay.arrays["bc_target_norm"][0], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(replay.arrays["bc_target_norm"][1], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(replay.arrays["actor_obs"][:, -3:], replay.arrays["bc_target_norm"])
    np.testing.assert_allclose(replay.arrays["critic_obs"][:, :5], replay.arrays["actor_obs"])
    np.testing.assert_allclose(replay.arrays["bc_target_raw"][0], [1.0, 2.0, 3.0])


def test_replay_writer_rejects_source_outside_base_identity(tmp_path):
    writer = ReplayBufferWriter(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="actor-generated data"):
        writer.add(_transition("td3bc_actor"))


def test_replay_writer_rejects_shape_changes(tmp_path):
    writer = ReplayBufferWriter(tmp_path / "bad_shape.npz")
    writer.add(_transition("base"))
    changed = _transition("base")
    changed["actor_obs"] = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="shape changed"):
        writer.add(changed)


def test_load_directory_merges_replays_with_global_action_normalizer(tmp_path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()

    writer_a = ReplayBufferWriter(replay_dir / "a.npz", metadata={"room_idx": 1})
    writer_a.add(_transition("base", action=[0.0, 0.0, 0.0], next_action=[2.0, 2.0, 2.0]))
    writer_a.add(_transition("base", action=[2.0, 2.0, 2.0], next_action=[0.0, 0.0, 0.0]))
    writer_a.save()

    writer_b = ReplayBufferWriter(replay_dir / "b.npz", metadata={"room_idx": 2})
    writer_b.add(_transition("base", action=[10.0, 10.0, 10.0], next_action=[12.0, 12.0, 12.0]))
    writer_b.add(_transition("base", action=[12.0, 12.0, 12.0], next_action=[10.0, 10.0, 10.0]))
    writer_b.save()

    replay = OfflineReplayBuffer.load(replay_dir, replay_filter="base_only")

    assert replay.size == 4
    assert replay.metadata["num_replays"] == 2
    assert replay.action_normalizer is not None
    np.testing.assert_allclose(replay.action_normalizer.mean, [6.0, 6.0, 6.0])
    np.testing.assert_allclose(replay.action_normalizer.scale, [6.0, 6.0, 6.0])
    np.testing.assert_allclose(
        replay.arrays["bc_target_norm"][:, 0],
        [-1.0, -2.0 / 3.0, 2.0 / 3.0, 1.0],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(replay.arrays["actor_obs"][:, -3:], replay.arrays["bc_target_norm"])
    np.testing.assert_allclose(replay.arrays["critic_obs"][:, :5], replay.arrays["actor_obs"])
