"""Bound simulator failures and preserve failed attempts before retrying."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def _terminate_group(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=2)
    except subprocess.TimeoutExpired: pass
    # A shell may exit before a stuck simulator descendant does.
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=10)


def run_simulator(command, *, env, check=True, stdout=None, stderr=None,
                  max_attempts=3, idle_timeout=180, wall_timeout=None, poll_interval=1):
    run_dir=Path(env["RUN_DIR"]).resolve()
    result=Path(env["RESULT_PATH"]).resolve()
    if wall_timeout is None:
        trials=int(env.get("NUM_EPISODES",1))*int(env.get("NUM_TRIALS",1))
        wall_timeout=900 if trials==1 else 2400
    archive_root = run_dir.parent if result.is_relative_to(run_dir) else result.parent
    history=[]
    for attempt in range(1,max_attempts+1):
        run_dir.mkdir(parents=True,exist_ok=True);result.parent.mkdir(parents=True,exist_ok=True)
        owned=stdout is None
        output=(run_dir/"simulator.log").open("w") if owned else stdout
        log_path=Path(output.name)
        output.seek(0);output.truncate(0);output.flush()
        start=time.monotonic();last_change=start;last_size=0;reason=None
        process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                code=process.poll();now=time.monotonic()
                size=log_path.stat().st_size if log_path.exists() else 0
                if size!=last_size:last_change=now;last_size=size
                if size:
                    with log_path.open("rb") as f:
                        f.seek(max(0,size-65536));tail=f.read()
                    if b"ERROR_DEVICE_LOST" in tail:reason="Vulkan ERROR_DEVICE_LOST"
                if reason is None and now-start>wall_timeout:reason="simulator wall timeout"
                if reason is None and now-last_change>idle_timeout:reason="simulator log idle timeout"
                if reason is not None:
                    _terminate_group(process);break
                if code is not None:break
                time.sleep(poll_interval)
        except BaseException:
            _terminate_group(process)
            raise
        finally:
            output.flush()
            if owned:output.close()
        history.append(dict(attempt=attempt,returncode=process.returncode,reason=reason,seconds=time.monotonic()-start))
        (archive_root/"simulator_attempts.json").write_text(json.dumps(history,indent=2))
        if reason is None:
            code=process.returncode
            if check and code:raise subprocess.CalledProcessError(code,command)
            return subprocess.CompletedProcess(command,code)
        # Keep every interrupted result/replay outside the collector's active
        # file set; retry the identical scene, episode IDs and policy RNG seed.
        archive=archive_root/"simulator_retries"/f"attempt{attempt}-{time.time_ns()}"
        archive.mkdir(parents=True)
        shutil.copy2(log_path,archive/"simulator.log")
        (archive/"failure.json").write_text(json.dumps(history[-1],indent=2))
        replay=env.get("RL_COLLECT_REPLAY_PATH")
        for label,path in [("run",run_dir),("result.txt",result),("replay.npz",Path(replay).resolve() if replay else None)]:
            if path is not None and path.exists():path.rename(archive/label)
        print(f"[simulator-retry] attempt={attempt}/{max_attempts} reason={reason} archive={archive}",flush=True)
        if attempt==max_attempts:
            raise subprocess.CalledProcessError(process.returncode or 1,command,stderr=reason)
    raise AssertionError("Unreachable simulator retry state")
