import copy
import random

import numpy as np
import pytest
import torch

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.h_summary import HSummaryConfig
from rl_posttrain.initialize_reference_actor import initialize
from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.online_td3bc import OnlineTD3BCAgent, _deep_copy_config
from rl_posttrain.replay_buffer import OfflineReplayBuffer
from rl_posttrain.td3bc_ref import PreparedReplay, TD3BCConfig, load_actor_policy


def fixture():
    rng = np.random.default_rng(7)
    obs = rng.uniform(-0.8, 0.8, (32, 8)).astype(np.float32)
    critic = np.concatenate([obs, rng.normal(size=(32, 2)).astype(np.float32)], axis=1)
    ref = obs[:, -3:].copy()
    arrays = dict(actor_obs=obs, critic_obs=critic, next_actor_obs=obs.copy(),
                  next_critic_obs=critic.copy(), action_norm=ref.copy(), bc_target_norm=ref,
                  next_bc_target_norm=ref.copy(), reward=np.ones((32, 1), np.float32),
                  done=np.zeros((32, 1), np.float32), source=np.full(32, 'base'))
    replay = OfflineReplayBuffer(arrays, metadata={'scene': 'room1_table1'},
                                 action_normalizer=AffineNormalizer.identity(3))
    config = TD3BCConfig(actor_hidden_dims=(16, 16), critic_hidden_dims=(16, 16),
                         h_summary=HSummaryConfig(h_dim=0))
    return replay, config


def test_identity_uses_no_weights_and_roundtrips(monkeypatch, tmp_path):
    replay, config = fixture()
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Initialization loaded weights'))
        checkpoint = initialize(replay, 0, config)
    path = tmp_path / 'identity.pt'
    torch.save(checkpoint, path)
    policy = load_actor_policy(path, 'cpu')
    obs = policy['actor_obs_normalizer'].normalize(replay.arrays['actor_obs'])
    torch.testing.assert_close(policy['actor'](torch.from_numpy(obs)),
                               torch.from_numpy(replay.arrays['bc_target_norm']), atol=1e-7, rtol=0)
    assert checkpoint['global_update'] == checkpoint['actor_updates'] == checkpoint['critic_updates'] == 0
    other = initialize(replay, 1, config)
    assert any(not torch.equal(v, other['critic_state_dict'][k]) for k, v in checkpoint['critic_state_dict'].items())


def test_trainable_identity_warmup_and_exact_resume(tmp_path):
    replay, config = fixture()
    initial = initialize(replay, 0, config)
    cfg = _deep_copy_config()
    cfg['optimization']['batch_size'] = 8
    agent = OnlineTD3BCAgent('unused', cfg, 'cpu', resume_state=initial)
    rng = np.random.default_rng(0)
    agent.attach_replay_rng(rng)
    prepared = PreparedReplay(replay, config)
    before = copy.deepcopy(agent.actor.state_dict())
    agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=False, update_actor_target=False)
    assert agent.actor_updates == 0
    assert all(torch.equal(v, agent.actor.state_dict()[k]) for k, v in before.items())
    agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=True, update_actor_target=True)
    assert agent.actor_updates == 1
    assert not torch.equal(before['net.4.weight'], agent.actor.state_dict()['net.4.weight'])
    for _ in range(2):
        agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=True, update_actor_target=True)
    assert agent.actor.net[0].weight.grad.abs().sum() > 0
    checkpoint = agent.save_training_checkpoint(tmp_path / 'resume.pt', cfg)

    def continuation(model, generator):
        logs = []
        for _ in range(2):
            batch = prepared.sample(8, model.device, generator)
            logs.append(model.train_step(batch, cfg, update_actor=True, update_actor_target=True))
        return logs, torch.rand(5), np.random.rand(5), random.random()

    expected = continuation(agent, rng)
    resumed = OnlineTD3BCAgent(checkpoint, cfg, 'cpu')
    rng2 = np.random.default_rng(999)
    resumed.attach_replay_rng(rng2)
    actual = continuation(resumed, rng2)
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    np.testing.assert_array_equal(actual[2], expected[2])
    assert actual[3] == expected[3]
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, resumed.actor.state_dict()[key], atol=0, rtol=0)


def test_legacy_actor_state_dict_and_invalid_modes():
    actor = DeterministicActor(8, 3, (16,), h_summary=HSummaryConfig(h_dim=0))
    assert not any('reference_' in key for key in actor.state_dict())
    obs = torch.randn(5, 8)
    torch.testing.assert_close(actor(obs), torch.tanh(actor.net(actor.obs_processor(obs))), atol=0, rtol=0)
    with pytest.raises(ValueError):
        DeterministicActor(8, 3, parameterization='unknown')


def test_reject_heldout_initialization():
    replay, config = fixture()
    replay.metadata['scene'] = 'room3_table1'
    with pytest.raises(ValueError, match='training scenes'):
        initialize(replay, config=config)
