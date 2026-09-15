"""Bounded SAC experiment with immutable inputs and two evaluation modes."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''): digest.update(block)
    return digest.hexdigest()


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


@contextlib.contextmanager
def policy_environment(mode, seed):
    values = {'RL_SAC_MODE': mode, 'RL_SAC_SEED': str(seed),
              'RL_EXPLORATION_NOISE_STD': '0', 'RL_EXPLORATION_NOISE_CLIP': '0'}
    before = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try: yield
    finally:
        for k, v in before.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def validate_scene(run, room, table):
    from rl_posttrain.collect_base import TASK_LINE_RE, _episode_specs
    from rl_posttrain.paired_eval import _parse_result_records
    p = Path(run['result_path']); matches = list(TASK_LINE_RE.finditer(p.read_text()))
    ids = [[m['task'], int(m['room_idx']), int(m['table_idx']), m['episode_label'], int(m['trial_idx'])] for m in matches]
    expected = [['Open-Laptop', room, table, spec[0], trial]
                for spec in _episode_specs('Humanoid-Open-Laptop-v0', 8) for trial in range(2)]
    assert ids == expected and len(set(map(tuple, ids))) == 16, 'Evaluation IDs differ'
    results = [m['result'] == 'True' for m in matches]
    assert results == run['results']
    records = _parse_result_records(p)
    assert len(records) == 16 and all(int(r['rl_actor_action_dim']) == 38 for r in records)
    return dict(run=run, result_path=str(p), result_sha256=sha(p), ids=ids,
                results=results, room=room, table=table)


def summarize(rows):
    groups = {}
    for key, subset in [('all', rows), ('room1_room2', [r for r in rows if r['room'] < 3]),
                        ('room3', [r for r in rows if r['room'] == 3])]:
        results = [x for r in subset for x in r['results']]
        groups[key] = dict(successes=sum(results), n=len(results), success_rate=sum(results)/len(results))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--mode', choices=['deterministic', 'stochastic'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; os.chdir(root)
    run = Path(args.run_root).resolve(); manifest = json.loads((run/'manifest.json').read_text())
    gpu = int(os.environ['EGOVLA_GPU']); assert gpu == manifest['gpu']
    import numpy as np
    import torch
    import yaml
    from rl_posttrain import online_td3bc as o
    from rl_posttrain.sac import OnlineSACAgent
    from rl_posttrain.paired_eval import _run_eval
    cfg = yaml.safe_load((run/'config.yaml').read_text()); current = 0
    if args.resume: current = int(torch.load(args.resume, map_location='cpu', weights_only=False)['online_episode'])
    def status(stage, **kw):
        atomic(run/'status.json', dict(stage=stage, pid=os.getpid(), gpu=gpu,
            updated=time.time(), online_episode=current, target_episode=cfg['online']['total_online_episodes'], **kw))
    def event(name, **kw):
        with (run/'events.jsonl').open('a') as f: f.write(json.dumps(dict(event=name,time=time.time(),**kw))+'\n')
    def verify(full=False):
        for item in manifest['code_assets'] + (manifest['assets'] if full else []):
            assert sha(item['path']) == item['sha256'], 'Input changed: '+item['path']
    def evaluate(cfg, output_root, init_checkpoint, actor, episode):
        nonlocal current
        current = episode; verify()
        periodic = run/"checkpoints"/f"episode_{episode:04d}_actor.pt"
        if episode > 0 and periodic.exists(): actor = periodic
        out = run/'eval'/f'episode_{episode:04d}'; out.mkdir(parents=True,exist_ok=True)
        fixed = out/'evaluated_actor.pt'
        if fixed.exists(): assert sha(fixed) == sha(actor), 'Checkpoint changed for a evaluated milestone'
        else: shutil.copyfile(actor,fixed)
        metrics = {"evaluation/episode": episode}; receipts = {}
        for mode in ([args.mode] if args.mode else ['deterministic','stochastic']):
            rows = []
            for room in [1,2,3]:
                for table in [1,2]:
                    # Common random numbers across budgets; independent of simulator/model RNG.
                    seed = 200000 + cfg['online']['seed']*10000 + room*100 + table
                    scene = out/mode/f'room{room}_table{table}'; rp = scene/'verified.json'
                    status('evaluating', evaluation_episode=episode, policy_mode=mode, room=room, table=table)
                    if rp.exists():
                        row = json.loads(rp.read_text())
                        assert row['actor_sha256'] == sha(fixed) and row['policy_mode'] == mode and row['sampling_seed'] == seed
                        assert sha(row['result_path']) == row['result_sha256']
                    else:
                        if scene.exists(): scene.rename(scene.with_name(scene.name+f'.incomplete-{int(time.time())}'))
                        eval_args = argparse.Namespace(room_idx=room, table_idx=table,
                            task=cfg['online']['task'],model_path=cfg['online']['model_path'],
                            num_episodes=8,num_trials=2,no_save_video=True,max_eval_steps=0)
                        with policy_environment(mode,seed):
                            result = _run_eval('actor',eval_args,scene,actor_checkpoint=fixed,run_name=mode)
                        row = validate_scene(result,room,table)
                        row.update(policy_mode=mode,sampling_seed=seed,actor_sha256=sha(fixed))
                        atomic(rp,row)
                    rows.append(row)
            groups = summarize(rows)
            receipts[mode] = dict(groups=groups,scenes=rows)
            atomic(out/mode/'verification.json', dict(episode=episode,policy_mode=mode,
                actor_checkpoint=str(fixed),actor_sha256=sha(fixed),**receipts[mode]))
            for key,values in groups.items(): metrics[f'eval_{mode}/{key}_success_rate'] = values['success_rate']
        receipt = dict(episode=episode,actor_checkpoint=str(fixed),actor_sha256=sha(fixed),modes=receipts)
        atomic(out/'verification.json',receipt)
        curve_path=run/'evaluation_curve.json';curve=json.loads(curve_path.read_text()) if curve_path.exists() else {}
        curve[str(episode)]=receipt;atomic(curve_path,curve)
        verify();event('evaluated',episode=episode,metrics=metrics)
        return out/'verification.json',metrics
    class SACLogger(o.SafeWandbLogger):
        def __init__(self, cfg, payload):
            super().__init__(cfg, payload)
            if self.run is not None:
                self.run.define_metric("*", step_metric="training/gradient_step")
                for prefix in ["eval_deterministic/*", "eval_stochastic/*"]:
                    self.run.define_metric(prefix, step_metric="evaluation/episode")
        def log(self, metrics, step=None):
            if self.run is not None:
                # Multiple events can share a gradient step (warm-up/evaluation).
                # W&B's history index is separate and strictly increases.
                self.run.log({"training/gradient_step": step, **dict(metrics)})
    o.SafeWandbLogger = SACLogger
    original_collect, original_updates = o.collect_online_episode, o._run_updates
    def collect(*a, **kw):
        verify();assert shutil.disk_usage(root).free > 20*1024**3
        status('collecting',next_episode=a[3]+1,scene=a[2]);t=time.time()
        result=original_collect(*a,**kw);event('collected',episode=a[3]+1,seconds=time.time()-t,info=result[1])
        return result
    def updates(*a,**kw):
        status('updating',next_episode=a[1].online_episode+1)
        result=original_updates(*a,**kw)
        assert all(np.isfinite(float(v)) for v in result[1].values())
        event('updated',episode=a[1].online_episode+1,updates=result[0],metrics=result[1])
        return result
    def save_wrapper(method):
        def save(self,path,cfg):
            nonlocal current
            p=Path(path);tmp=p.with_name(p.stem+'.writing.pt');method(self,tmp,cfg);tmp.replace(p)
            if p.name=='latest_online.pt':
                current=self.online_episode
                status('checkpoint_saved',env_steps=self.env_steps,critic_updates=self.critic_updates,actor_updates=self.actor_updates)
            return p
        return save
    o.collect_online_episode,o._run_updates,o.run_paired_eval=collect,updates,evaluate
    OnlineSACAgent.save_actor_checkpoint=save_wrapper(OnlineSACAgent.save_actor_checkpoint)
    OnlineSACAgent.save_training_checkpoint=save_wrapper(OnlineSACAgent.save_training_checkpoint)
    lock=(Path(manifest['runtime'])/'locks'/f'gpu{gpu}.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(full=True)
        event('started',runner_sha256=sha(__file__),resume=args.resume)
        if args.eval_only:
            actor=Path(manifest['initial_actor'])
            evaluate(cfg,run,Path(cfg['online']['init_checkpoint']),actor,0)
        else:
            assert Path(manifest['preflight_receipt']).is_file()
            if args.resume and current in [50,100,150] and not (run/f'eval/episode_{current:04d}/verification.json').exists():
                actor=run/'checkpoints'/f'episode_{current:04d}_actor.pt'
                evaluate(cfg,run,Path(cfg['online']['init_checkpoint']),actor,current)
            sys.argv=['online_td3bc','--config',str(run/'config.yaml'),'--strict_resume_manifest_match']
            if args.resume: sys.argv+=['--resume',args.resume]
            o.main()
            assert current==cfg['online']['total_online_episodes']
            for ep in [50,100,150]: assert (run/f'eval/episode_{ep:04d}/verification.json').is_file()
        verify(full=True);status('completed');event('completed')
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        # Explicitly close logging to avoid failed processes retaining GPU memory.
        try:
            import wandb
            if wandb.run: wandb.finish(exit_code=1)
        finally: raise


if __name__=='__main__': main()
