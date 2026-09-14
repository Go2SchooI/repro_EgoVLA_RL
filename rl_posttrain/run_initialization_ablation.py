"""Run B with immutable inputs, atomic saves and paired comparisons to archived A."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run_root).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    os.chdir(root)
    os.environ.update(manifest['environment'])
    os.environ['CONDA_PREFIX'] = str(Path(sys.executable).parent.parent)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    from rl_posttrain import online_td3bc as o
    from rl_posttrain.collect_base import _episode_specs, TASK_LINE_RE
    from rl_posttrain.paired_eval import _parse_result_records
    import torch
    import yaml

    current = 0
    if args.resume:
        current = int(torch.load(args.resume, map_location='cpu', weights_only=False)['online_episode'])
    start = time.time()

    def atomic(name, payload):
        p = run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2) + '\n')
        tmp.replace(p)

    def status(state, **kw):
        atomic('status.json', dict(state=state, updated=time.time(), started=start,
                                  pid=os.getpid(), online_episode=current, target_episode=300, **kw))

    def event(name, **kw):
        with (run / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), event=name, **kw)) + '\n')

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
                h.update(block)
        return h.hexdigest()

    def verify(full=False):
        for item in manifest['code_assets'] + manifest['static_assets'] + (manifest['assets'] if full else []):
            assert sha(item['path']) == item['sha256'], 'Input changed: ' + item['path']

    def compare(a, b):
        assert len(a) == len(b)
        return dict(recover=sum(not x and y for x, y in zip(a, b)),
                    regress=sum(x and not y for x, y in zip(a, b)), net=sum(b)-sum(a))

    def validate(summary_path, episode, actor):
        summary = json.loads(Path(summary_path).read_text())
        rows = []
        assert len(summary['scenes']) == 6
        for item in summary['scenes']:
            scene = json.loads(Path(item['summary_path']).read_text())
            runs = scene['runs']
            roles = dict(baseline=runs['baseline'], offline=runs['actors']['offline_td3bc'],
                         B=runs['actors'][o.paired_safe_label(Path(actor).stem)])
            room, table = int(roles['baseline']['room_idx']), int(roles['baseline']['table_idx'])
            expected = [['Open-Laptop', room, table, ep[0], trial]
                        for ep in _episode_specs('Humanoid-Open-Laptop-v0', 8) for trial in range(2)]
            outcomes = {}
            for name, outcome in roles.items():
                p = Path(outcome['result_path'])
                matches = list(TASK_LINE_RE.finditer(p.read_text()))
                ids = [[m['task'], int(m['room_idx']), int(m['table_idx']), m['episode_label'], int(m['trial_idx'])] for m in matches]
                values = [m['result'] == 'True' for m in matches]
                assert ids == expected and len(set(map(tuple, ids))) == 16, (episode, room, table, name, 'IDs')
                assert values == outcome['results']
                records = _parse_result_records(p)
                assert len(records) == 16
                if name == 'B':
                    assert all(int(v.get('rl_actor_action_dim', 0)) == 38 for v in records)
                outcomes[name] = values
            rows.append(dict(room=room, table=table, results=outcomes))
        assert {(s['room'], s['table']) for s in rows} == {(r, t) for r in (1, 2, 3) for t in (1, 2)}
        archived = json.loads(Path(manifest['A_curve']).read_text())
        a_entry = archived[str(episode)] if episode else archived['50']
        a_scenes = {(s['room'], s['table']): s for s in a_entry['scenes']}
        for s in rows:
            a_scene = a_scenes[(s['room'], s['table'])]
            assert s['results']['baseline'] == a_scene['results']['baseline']
            assert s['results']['offline'] == a_scene['results']['offline']
            s['results']['A'] = a_scene['results']['online' if episode else 'offline']
        groups = {}
        for name, scenes in [('all', rows), ('room1_room2', [s for s in rows if s['room'] < 3]),
                             ('room3', [s for s in rows if s['room'] == 3])]:
            values = {role: [v for s in scenes for v in s['results'][role]] for role in ['baseline', 'offline', 'A', 'B']}
            groups[name] = dict(policies={k: dict(successes=sum(v), n=len(v), success_rate=sum(v)/len(v)) for k, v in values.items()},
                                B_vs_A=compare(values['A'], values['B']),
                                B_vs_offline=compare(values['offline'], values['B']),
                                B_vs_baseline=compare(values['baseline'], values['B']))
        receipt = dict(episode=episode, actor_checkpoint=str(actor), actor_sha256=sha(actor),
                       summary_path=str(summary_path), groups=groups, scenes=rows, verified=time.time())
        atomic(f'eval/episode_{episode:04d}/verification.json', receipt)
        path = run / 'comparison_curve.json'
        curve = json.loads(path.read_text()) if path.exists() else {}
        curve[str(episode)] = receipt
        atomic('comparison_curve.json', curve)
        lines = ['# A/B initialization-route comparison', '',
                 'A: archived offline-initialized direct actor. B: reference-residual actor and random critic; same base replay. Single seed, different actor parameterization; not a pure offline-initialization causal ablation.', '',
                 'room3 is held out from RL replay but has been used for development/evaluation.', '',
                 '| Online episode | Split | A | B | B recover/regress vs A |', '| ---: | --- | ---: | ---: | ---: |']
        for ep in sorted(curve, key=int):
            for group, metrics in curve[ep]['groups'].items():
                a, b = metrics['policies']['A'], metrics['policies']['B']
                pair = metrics['B_vs_A']
                lines.append(f"| {ep} | {group} | {a['successes']}/{a['n']} | {b['successes']}/{b['n']} | {pair['recover']}/{pair['regress']} |")
        (run / 'comparison_curve.md').write_text('\n'.join(lines) + '\n')
        return groups

    original_collect, original_updates = o.collect_online_episode, o._run_updates

    def collect(*a, **kw):
        nonlocal current
        current = a[1].online_episode
        verify()
        assert shutil.disk_usage(run).free > 6 * 1024 ** 3, 'Disk space below 6 GiB'
        status('collecting', next_episode=a[3]+1, scene=a[2])
        t = time.time()
        result = original_collect(*a, **kw)
        event('collected', episode=a[3]+1, seconds=time.time()-t, info=result[1])
        return result

    def updates(*a, **kw):
        status('updating', next_episode=a[1].online_episode+1)
        result = original_updates(*a, **kw)
        assert all(math.isfinite(float(v)) for v in result[1].values()), 'Nonfinite training metrics'
        event('updated', episode=a[1].online_episode+1, updates=result[0], metrics=result[1],
              actor_updates=a[1].actor_updates, critic_updates=a[1].critic_updates)
        return result

    cache = json.loads(Path(manifest['static_cache']).read_text())

    def evaluate(cfg, output_root, init_checkpoint, actor, episode):
        verify()
        status('evaluating', evaluation_episode=episode)
        t = time.time()
        # Copy mutable latest exports before evaluation and record their exact hash.
        frozen = run / 'eval' / f'episode_{episode:04d}' / 'evaluated_actor.pt'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actor, frozen)
        summary = o._run_paired_eval_with_static_cache(cfg, frozen.parent, frozen, cache)
        groups = validate(summary, episode, frozen)
        verify()
        event('evaluated', episode=episode, seconds=time.time()-t, groups=groups)
        status('evaluation_completed', evaluation_episode=episode, groups=groups)
        metrics = {'eval_all/success_rate': groups['all']['policies']['B']['success_rate']}
        for label, key in [('all', 'all'), ('seen', 'room1_room2'), ('unseen', 'room3')]:
            metrics[f'eval_{label}/success_rate'] = groups[key]['policies']['B']['success_rate']
        return summary, metrics

    def wrap_save(method):
        def save(self, path, cfg):
            nonlocal current
            p = Path(path)
            tmp = p.with_name(p.stem + '.writing.pt')
            method(self, tmp, cfg)
            tmp.replace(p)
            if p.name == 'latest_online.pt':
                current = self.online_episode
                status('checkpoint_saved', checkpoint=str(p), actor_updates=self.actor_updates,
                       critic_updates=self.critic_updates, env_steps=self.env_steps)
            return p
        return save

    o.collect_online_episode, o._run_updates, o.run_paired_eval = collect, updates, evaluate
    o.OnlineTD3BCAgent.save_actor_checkpoint = wrap_save(o.OnlineTD3BCAgent.save_actor_checkpoint)
    o.OnlineTD3BCAgent.save_training_checkpoint = wrap_save(o.OnlineTD3BCAgent.save_training_checkpoint)
    lock = open('/tmp/egovla-lineage-eval-gpu0.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('validating')
        verify(full=True)
        assert (run / 'preflight/verification.json').is_file()
        event('start', resume=args.resume, runner_sha256=sha(__file__))
        cfg = yaml.safe_load((run / 'online_b.yaml').read_text())
        if args.resume and current > 0 and current % 50 == 0:
            receipt = run / f'eval/episode_{current:04d}/verification.json'
            if not receipt.exists():
                evaluate(cfg, run, Path(cfg['online']['init_checkpoint']), Path(args.resume), current)
        sys.argv = ['online_td3bc', '--config', str(run / 'online_b.yaml'), '--strict_resume_manifest_match']
        if args.resume:
            sys.argv += ['--resume', args.resume, '--allow_reuse_online_replay']
        o.main()
        assert current == 300
        # Full @0 evaluation can run after training: the immutable initial policy
        # is unchanged, and delaying it gives the first trained comparison sooner.
        cfg = yaml.safe_load((run / 'online_b.yaml').read_text())
        evaluate(cfg, run, Path(cfg['online']['init_checkpoint']), Path(cfg['online']['init_checkpoint']), 0)
        for ep in [0, 50, 100, 150, 200, 250, 300]:
            assert (run / f'eval/episode_{ep:04d}/verification.json').is_file()
        verify(full=True)
        status('completed', ended=time.time())
        event('completed')
    except BaseException as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        event('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
