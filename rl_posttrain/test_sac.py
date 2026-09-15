import copy
import random

import numpy as np
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.h_summary import HSummaryConfig
from rl_posttrain.initialize_reference_actor import initialize
from rl_posttrain.online_td3bc import _deep_copy_config
from rl_posttrain.sac import OnlineSACAgent
from rl_posttrain.sac_actor import SquashedGaussianActor
from rl_posttrain.td3bc_ref import PreparedReplay, load_actor_policy
from rl_posttrain.test_reference_initialization import fixture


def setup():
    replay, config = fixture()
    payload = initialize(replay, 0, config)
    actor = DeterministicActor(8, 3, (16, 16), h_summary=HSummaryConfig(h_dim=0))
    payload['actor_state_dict'] = actor.state_dict()
    payload.pop('actor_target_state_dict', None)
    payload['actor_parameterization'] = 'direct_tanh'
    cfg = _deep_copy_config()
    cfg['algorithm'] = 'sac'
    cfg['sac'] = dict(init_log_std=-5.8, log_std_min=-10., log_std_max=0.,
                      initial_temperature=1e-4, temperature_lr=3e-4, target_entropy=-3.)
    agent = OnlineSACAgent('unused', cfg, 'cpu', resume_state=payload)
    return agent, actor, cfg, PreparedReplay(replay, config)


def test_initial_mean_and_checkpoint_roundtrip(tmp_path):
    agent, original, cfg, _ = setup()
    x = torch.randn(12, 8)
    torch.testing.assert_close(agent.actor(x), original(x), rtol=0, atol=0)
    assert set(agent.actor.state_dict()) - set(original.state_dict()) == {
        'log_std_head.weight', 'log_std_head.bias'}
    p = agent.save_actor_checkpoint(tmp_path/'actor.pt', cfg)
    loaded = load_actor_policy(p, 'cpu')['actor']
    torch.testing.assert_close(loaded(x), original(x), rtol=0, atol=0)
    assert agent.global_update == agent.actor_updates == agent.critic_updates == 0


def test_sampling_seed_is_reproducible_and_does_not_consume_global_rng():
    agent, _, _, _ = setup(); x = torch.randn(20, 8)
    state = torch.get_rng_state().clone()
    a = agent.actor.sample(x, generator=torch.Generator().manual_seed(21))[0]
    b = agent.actor.sample(x, generator=torch.Generator().manual_seed(21))[0]
    c = agent.actor.sample(x, generator=torch.Generator().manual_seed(22))[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(state, torch.get_rng_state())
    assert a.abs().max() <= 1


def test_log_prob_matches_distribution_and_handles_saturation():
    actor = SquashedGaussianActor(8, 3, (16,), h_summary=HSummaryConfig(h_dim=0), init_log_std=-1)
    x = torch.randn(32, 8)
    action, logp, _ = actor.sample(x)
    mean, logstd = actor.distribution_parameters(x)
    reference = TransformedDistribution(Normal(mean, logstd.exp()), [TanhTransform()])
    torch.testing.assert_close(logp, reference.log_prob(action).sum(-1, keepdim=True), atol=1e-5, rtol=1e-5)
    with torch.no_grad(): actor.net[-1].bias.fill_(50)
    _, logp, _ = actor.sample(x)
    logp.mean().backward()
    assert torch.isfinite(logp).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


def test_warmup_freezes_policy_then_mean_std_and_temperature_update():
    agent, _, cfg, prepared = setup(); rng = np.random.default_rng(3)
    before = copy.deepcopy(agent.actor.state_dict()); temp = agent.log_alpha.detach().clone()
    critic = copy.deepcopy(agent.critic.state_dict())
    agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=False, update_actor_target=False)
    assert all(torch.equal(v, agent.actor.state_dict()[k]) for k,v in before.items())
    assert torch.equal(temp, agent.log_alpha)
    assert any(not torch.equal(v, agent.critic.state_dict()[k]) for k,v in critic.items())
    agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=True, update_actor_target=False)
    assert agent.actor_updates == 1 and agent.critic_updates == 2
    assert not torch.equal(before['net.4.weight'], agent.actor.net[-1].weight)
    assert not torch.equal(before['log_std_head.weight'], agent.actor.log_std_head.weight)
    assert not torch.equal(temp, agent.log_alpha)


def test_exact_resume_includes_temperature_optimizers_and_rng(tmp_path):
    agent, _, cfg, prepared = setup(); rng = np.random.default_rng(3); agent.attach_replay_rng(rng)
    for _ in range(3): agent.train_step(prepared.sample(8, agent.device, rng), cfg, update_actor=True, update_actor_target=False)
    agent.online_episode = 4
    path = agent.save_training_checkpoint(tmp_path/'resume.pt', cfg)
    def step(a, r):
        logs = a.train_step(prepared.sample(8, a.device, r), cfg, update_actor=True, update_actor_target=False)
        return logs, random.random(), np.random.rand(), torch.rand(4)
    expected = step(agent, rng)
    restored = OnlineSACAgent(path, cfg, 'cpu'); rng2 = np.random.default_rng(900);restored.attach_replay_rng(rng2)
    actual = step(restored, rng2)
    assert expected[:3] == actual[:3]
    assert torch.equal(expected[3], actual[3])
    for k,v in agent.actor.state_dict().items(): assert torch.equal(v, restored.actor.state_dict()[k])
    assert torch.equal(agent.log_alpha, restored.log_alpha)


def test_terminal_mask_and_entropy_backup(monkeypatch):
    agent, _, cfg, prepared = setup()
    batch = prepared.sample(8, agent.device, np.random.default_rng(3))
    batch['reward'].fill_(1);batch['done'][:4].fill_(1);batch['done'][4:].zero_()
    monkeypatch.setattr(agent.actor, 'sample', lambda obs: (torch.zeros(8,3), torch.full((8,1),0.4), torch.zeros(8,3)))
    monkeypatch.setattr(agent.critic_target, 'forward', lambda obs,act: (torch.full((8,1),2.),torch.full((8,1),3.)))
    logs=agent.train_step(batch,cfg,update_actor=False,update_actor_target=False)
    expected=1+0.5*cfg['td3']['gamma']*(2-1e-4*0.4)
    assert abs(logs['td_target_mean']-expected)<1e-6


def test_dual_evaluation_orchestration_and_verified_scene_reuse(monkeypatch, tmp_path):
    import json, os, sys, yaml
    from rl_posttrain import run_sac, online_td3bc as o, paired_eval
    from rl_posttrain.collect_base import _episode_specs
    runtime=tmp_path/'runtime';(runtime/'locks').mkdir(parents=True)
    root=tmp_path/'run';root.mkdir();actor=tmp_path/'initial.pt';actor.write_bytes(b'test-only-checkpoint')
    cfg={'online':{'seed':0,'total_online_episodes':150,'init_checkpoint':str(actor),'model_path':'test-model','task':'Humanoid-Open-Laptop-v0'}}
    (root/'config.yaml').write_text(yaml.safe_dump(cfg))
    (root/'manifest.json').write_text(json.dumps(dict(gpu=0,runtime=str(runtime),code_assets=[],assets=[],initial_actor=str(actor))))
    monkeypatch.setenv('EGOVLA_GPU','0');monkeypatch.setenv('RL_SAC_MODE','training-parent-value')
    monkeypatch.setattr(sys,'argv',['run_sac','--run-root',str(root),'--eval-only'])
    calls=[]
    def fake_eval(mode,args,out,actor_checkpoint,run_name):
        pm=os.environ['RL_SAC_MODE'];seed=int(os.environ['RL_SAC_SEED']);calls.append((pm,args.room_idx,args.table_idx,seed))
        result=out/'results.txt';result.parent.mkdir(parents=True)
        values=[];lines=[]
        for i,spec in enumerate(_episode_specs(args.task,8)):
            for trial in range(2):
                ok=(i<4 if pm=='deterministic' else i<6);values.append(ok)
                lines.extend([f'Task: Open-Laptop, Room Idx: {args.room_idx}, Table Idx: {args.table_idx}, Episode Label: {spec[0]}, Trial Label: {trial}, Result: {ok}', 'eval_ablation: rl_actor_action_dim=38'])
        result.write_text('\n'.join(lines));return dict(result_path=str(result),results=values)
    monkeypatch.setattr(paired_eval,'_run_eval',fake_eval)
    # Restore module globals changed by the experiment wrapper after this test.
    for name in ['SafeWandbLogger','collect_online_episode','_run_updates','run_paired_eval']:
        monkeypatch.setattr(o,name,getattr(o,name))
    for name in ['save_actor_checkpoint','save_training_checkpoint']:
        monkeypatch.setattr(OnlineSACAgent,name,getattr(OnlineSACAgent,name))
    oldcwd=os.getcwd()
    try:
        run_sac.main()
        assert len(calls)==12 and os.environ['RL_SAC_MODE']=='training-parent-value'
        v=json.loads((root/'eval/episode_0000/verification.json').read_text())
        assert v['modes']['deterministic']['groups']['all']['successes']==48
        assert v['modes']['stochastic']['groups']['all']['successes']==72
        for mode in ['deterministic','stochastic']:
            assert v['modes'][mode]['groups']['room1_room2']['n']==64
            assert v['modes'][mode]['groups']['room3']['n']==32
        run_sac.main()
        assert len(calls)==12  # A verified scene is reused, not rerun.
    finally:os.chdir(oldcwd)
