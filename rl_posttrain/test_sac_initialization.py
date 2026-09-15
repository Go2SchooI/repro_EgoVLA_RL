import copy

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform

from rl_posttrain.initialize_sac_ablation import initialize
from rl_posttrain.sac import OnlineSACAgent
from rl_posttrain.td3bc_ref import load_actor_policy
from rl_posttrain.test_sac import setup


def sources():
    a, _, cfg, replay = setup()
    source = a.actor_checkpoint_payload(cfg)
    source["format"] = "td3bc_ref_actor_v1"
    source.pop("algorithm")
    source["actor_state_dict"] = {k:v for k,v in source["actor_state_dict"].items()
                                 if not k.startswith("log_std_head.")}
    return source, cfg, replay


def test_paired_fresh_critics_and_offline_actor_transfer():
    source, cfg, _ = sources()
    before = torch.get_rng_state().clone()
    b, c = initialize(source, "B", 0), initialize(source, "C", 0)
    assert torch.equal(before, torch.get_rng_state())
    for key, value in b["critic_state_dict"].items():
        assert torch.equal(value, c["critic_state_dict"][key])
        assert torch.equal(value, b["critic_target_state_dict"][key])
    assert any(not torch.equal(v, source["critic_state_dict"][k]) for k,v in b["critic_state_dict"].items())
    b1 = initialize(source, "B", 1)
    assert any(not torch.equal(v, b1["critic_state_dict"][k]) for k,v in b["critic_state_dict"].items())
    assert all(torch.equal(v, source["actor_state_dict"][k]) for k,v in b["actor_state_dict"].items())
    for p in [b,c]:
        assert not any("optimizer" in k or "rng" in k for k in p)
        agent = OnlineSACAgent("unused", cfg, "cpu", resume_state=p)
        assert agent.initialization == p["sac_initialization"]
        assert agent.actor_updates == agent.critic_updates == agent.env_steps == 0


def test_reference_gaussian_mean_density_and_saturation():
    source, cfg, _ = sources()
    c = OnlineSACAgent("unused", cfg, "cpu", resume_state=initialize(source,"C",0))
    x = torch.randn(32, c.actor_obs_dim)
    reference = torch.linspace(-0.8,0.8,96).reshape(32,3)
    x[:,-3:] = (reference-c.actor.reference_mean)/c.actor.reference_scale
    torch.testing.assert_close(c.actor(x), reference, atol=2e-6,rtol=0)
    action,logp,mean_action = c.actor.sample(x)
    torch.testing.assert_close(mean_action,c.actor(x),atol=0,rtol=0)
    mu,ls = c.actor.distribution_parameters(x)
    distribution = TransformedDistribution(Normal(mu,ls.exp()),[TanhTransform()])
    torch.testing.assert_close(logp,distribution.log_prob(action).sum(-1,keepdim=True),atol=2e-4,rtol=2e-4)
    x[:,-3:] = (torch.tensor([-1.,1.,2.])-c.actor.reference_mean)/c.actor.reference_scale
    action,logp,_ = c.actor.sample(x)
    logp.mean().backward()
    assert torch.isfinite(action).all() and torch.isfinite(logp).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in c.actor.parameters())


@pytest.mark.parametrize("variant",["B","C"])
def test_ablation_warmup_export_and_exact_resume(variant,tmp_path):
    source,cfg,replay = sources()
    agent = OnlineSACAgent("unused",cfg,"cpu",resume_state=initialize(source,variant,0))
    rng = np.random.default_rng(1);agent.attach_replay_rng(rng)
    before = copy.deepcopy(agent.actor.state_dict());alpha=agent.log_alpha.detach().clone()
    agent.train_step(replay.sample(8,agent.device,rng),cfg,update_actor=False,update_actor_target=False)
    assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items())
    assert torch.equal(alpha,agent.log_alpha)
    agent.train_step(replay.sample(8,agent.device,rng),cfg,update_actor=True,update_actor_target=False)
    assert any(not torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items())
    export=agent.save_actor_checkpoint(tmp_path/"actor.pt",cfg)
    loaded=load_actor_policy(export,"cpu")["actor"]
    x=torch.randn(10,agent.actor_obs_dim)
    torch.testing.assert_close(loaded(x),agent.actor(x),rtol=0,atol=0)
    path=agent.save_training_checkpoint(tmp_path/"resume.pt",cfg)
    expected=agent.train_step(replay.sample(8,agent.device,rng),cfg,update_actor=True,update_actor_target=False)
    restored=OnlineSACAgent(path,cfg,"cpu");rng2=np.random.default_rng(9);restored.attach_replay_rng(rng2)
    actual=restored.train_step(replay.sample(8,restored.device,rng2),cfg,update_actor=True,update_actor_target=False)
    assert expected == actual
    assert restored.initialization == agent.initialization
    assert all(torch.equal(v,restored.actor.state_dict()[k]) for k,v in agent.actor.state_dict().items())
