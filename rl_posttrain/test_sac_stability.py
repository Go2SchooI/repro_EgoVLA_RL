import copy, math
import numpy as np
import pytest
import torch
from rl_posttrain.sac import OnlineSACAgent
from rl_posttrain.test_sac_initialization import sources
from rl_posttrain.td3bc_ref import load_actor_policy


def make_agent(anchor=0.1, low=False):
    source,cfg,replay=sources()
    cfg["sac"]["mean_anchor_weight"]=anchor
    if low:cfg["sac"].update(log_std_max=math.log(.03),target_entropy=-12.)
    return OnlineSACAgent("unused",cfg,"cpu",resume_state=source),cfg,replay


def test_anchor_is_frozen_initial_policy_and_added_to_objective():
    a,cfg,replay=make_agent();rng=np.random.default_rng(3)
    teacher=copy.deepcopy(a.mean_anchor.state_dict())
    with torch.no_grad():a.actor.net[-1].bias.add_(.25)
    batch=replay.sample(16,a.device,rng)
    with torch.no_grad():expected=torch.nn.functional.mse_loss(a.actor(batch["actor_obs"]),a.mean_anchor(batch["actor_obs"])).item()
    assert expected>0
    logs=a.train_step(batch,cfg,update_actor=True,update_actor_target=False)
    assert logs["sac/mean_anchor_loss"]==pytest.approx(expected)
    assert logs["actor_loss"]==pytest.approx(logs["sac/unregularized_actor_loss"]+.1*expected,abs=1e-7)
    assert all(torch.equal(v,a.mean_anchor.state_dict()[k]) for k,v in teacher.items())
    assert all(not p.requires_grad and p.grad is None for p in a.mean_anchor.parameters())
    assert not any(id(p)==id(q) for group in a.actor_opt.param_groups for p in group["params"] for q in a.mean_anchor.parameters())


@pytest.mark.parametrize("anchor,low",[(.1,False),(0.,True),(.1,True)])
def test_stability_warmup_exact_resume_and_inference_export(anchor,low,tmp_path):
    a,cfg,replay=make_agent(anchor,low);rng=np.random.default_rng(6);a.attach_replay_rng(rng)
    initial=copy.deepcopy(a.actor.state_dict());temperature=a.log_alpha.detach().clone()
    a.train_step(replay.sample(16,a.device,rng),cfg,update_actor=False,update_actor_target=False)
    assert all(torch.equal(v,a.actor.state_dict()[k]) for k,v in initial.items())
    assert torch.equal(temperature,a.log_alpha)
    for _ in range(4):a.train_step(replay.sample(16,a.device,rng),cfg,update_actor=True,update_actor_target=False)
    if low:
        with torch.no_grad():a.actor.log_std_head.bias.fill_(1.)
        assert a.actor.distribution_parameters(torch.randn(4,a.actor_obs_dim))[1].exp().max()<=.030001
    if anchor:assert any(not torch.equal(v,a.actor.state_dict()[k]) for k,v in a.mean_anchor.state_dict().items())
    p=a.save_training_checkpoint(tmp_path/"resume.pt",cfg)
    expected=a.train_step(replay.sample(16,a.device,rng),cfg,update_actor=True,update_actor_target=False)
    restored=OnlineSACAgent(p,cfg,"cpu");r2=np.random.default_rng(111);restored.attach_replay_rng(r2)
    actual=restored.train_step(replay.sample(16,restored.device,r2),cfg,update_actor=True,update_actor_target=False)
    assert expected==actual
    assert all(torch.equal(v,restored.actor.state_dict()[k]) for k,v in a.actor.state_dict().items())
    if anchor:assert all(torch.equal(v,restored.mean_anchor.state_dict()[k]) for k,v in a.mean_anchor.state_dict().items())
    export=a.save_actor_checkpoint(tmp_path/"actor.pt",cfg);loaded=load_actor_policy(export,"cpu")["actor"]
    x=torch.randn(5,a.actor_obs_dim);torch.testing.assert_close(loaded(x),a.actor(x),rtol=0,atol=0)
    changed=copy.deepcopy(cfg);changed["sac"]["target_entropy"]-=1
    with pytest.raises(ValueError,match="target entropy"):OnlineSACAgent(p,changed,"cpu")


def test_resume_rejects_missing_teacher_or_changed_anchor_weight(tmp_path):
    a,cfg,_=make_agent();p=a.actor_checkpoint_payload(cfg);p.pop("mean_anchor_state_dict")
    with pytest.raises(ValueError,match="missing its frozen"):OnlineSACAgent("unused",cfg,"cpu",resume_state=p)
    p=a.actor_checkpoint_payload(cfg);cfg["sac"]["mean_anchor_weight"]=.2
    with pytest.raises(ValueError,match="anchor weight"):OnlineSACAgent("unused",cfg,"cpu",resume_state=p)
