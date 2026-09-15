"""Start a bounded SAC run after another run has fully completed evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rl_posttrain.run_sac import atomic, sha


def predecessor_complete(root):
    status_path = root / "status.json"
    if not status_path.exists():
        return False
    state = json.loads(status_path.read_text())
    if state.get("stage") != "completed" or state.get("online_episode") != state.get("target_episode"):
        return False
    target = state["target_episode"]
    for episode in range(50, target + 1, 50):
        path = root / "eval" / f"episode_{episode:04d}" / "verification.json"
        if not path.exists():
            return False
        receipt = json.loads(path.read_text())
        if receipt["episode"] != episode or set(receipt["modes"]) != {"deterministic", "stochastic"}:
            raise ValueError("Predecessor evaluation receipt is inconsistent")
        if sha(receipt["actor_checkpoint"]) != receipt["actor_sha256"]:
            raise ValueError("Predecessor evaluated checkpoint changed")
        for mode in receipt["modes"].values():
            if mode["groups"]["all"]["n"] != 96 or len(mode["scenes"]) != 6:
                raise ValueError("Predecessor evaluation incomplete")
            for scene in mode["scenes"]:
                if sha(scene["result_path"]) != scene["result_sha256"]:
                    raise ValueError("Predecessor evaluation result changed")
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-run",required=True)
    parser.add_argument("--run-root",required=True)
    args=parser.parse_args()
    root=Path(args.run_root).resolve(); previous=Path(args.after_run).resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    gpu=manifest["gpu"]
    assert int(os.environ["EGOVLA_GPU"]) == gpu
    lock=(root/"queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kw):
        atomic(root/"queue_status.json",dict(stage=stage,pid=os.getpid(),gpu=gpu,
            predecessor=str(previous),updated=time.time(),**kw))
    try:
        if (root/"status.json").exists():
            raise FileExistsError("Successor already started; explicit recovery is required")
        status("waiting_for_predecessor")
        while True:
            if predecessor_complete(previous):
                with (Path(manifest["runtime"])/"locks"/f"gpu{gpu}.lock").open("a") as gpu_lock:
                    try:
                        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(30)
                        continue
                    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits"],text=True).splitlines()
                    memory={int(r.split(",")[0]):int(r.split(",")[1]) for r in rows}
                    if memory[gpu] < 1000:
                        break
            time.sleep(30)
        # run_sac independently acquires the GPU lock before any model is loaded.
        status("starting")
        child=subprocess.Popen([sys.executable,"-m","rl_posttrain.run_sac","--run-root",str(root)])
        atomic(root/"launcher.json",dict(pid=child.pid,gpu=gpu,started=time.time(),queue_pid=os.getpid(),
            predecessor=str(previous),wandb_run_id=os.environ.get("WANDB_RUN_ID")))
        status("running",child_pid=child.pid)
        code=child.wait()
        status("completed" if code == 0 else "failed",child_pid=child.pid,exit_code=code)
        sys.exit(code)
    except Exception as exc:
        status("failed",error=str(exc))
        raise


if __name__ == "__main__":
    main()
