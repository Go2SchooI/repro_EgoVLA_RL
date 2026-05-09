from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


RESULT_RE = re.compile(r"Result:\s*(True|False)")


def _parse_results(path: Path) -> List[bool]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval result file: {path}")
    results: List[bool] = []
    for line in path.read_text().splitlines():
        match = RESULT_RE.search(line)
        if match:
            results.append(match.group(1) == "True")
    if not results:
        raise ValueError(f"No 'Result: True/False' lines found in {path}")
    return results


def _run_eval(mode: str, args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    run_dir = output_root / mode
    result_path = run_dir / "results_local_eval.txt"
    env = os.environ.copy()
    env["RL_MODE"] = mode
    env["RUN_DIR"] = str(run_dir)
    env["RESULT_PATH"] = str(result_path)
    env["NUM_EPISODES"] = str(args.num_episodes)
    env["NUM_TRIALS"] = str(args.num_trials)
    env["SAVE_VIDEO"] = "0" if args.no_save_video else "1"
    env.setdefault("SAVE_FRAMES", "0")
    env.setdefault("PROJECT_TRAJS", "0")
    if args.task:
        env["TASK"] = args.task
    if args.max_eval_steps is not None:
        env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    if mode == "actor":
        env["RL_ACTOR_CHECKPOINT"] = str(args.actor_checkpoint)
    else:
        env.pop("RL_ACTOR_CHECKPOINT", None)

    print(f"[paired-eval] running mode={mode} result_path={result_path}", flush=True)
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)
    results = _parse_results(result_path)
    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "results": results,
        "success_rate": sum(results) / len(results),
    }


def _compare(baseline: List[bool], identity: List[bool] | None, actor: List[bool]) -> Dict[str, int | None]:
    if len(baseline) != len(actor):
        raise ValueError(
            "Paired eval result count mismatch: "
            f"baseline={len(baseline)} actor={len(actor)}"
        )
    if identity is not None and len(identity) != len(baseline):
        raise ValueError(
            "Paired eval result count mismatch: "
            f"baseline={len(baseline)} identity={len(identity)} actor={len(actor)}"
        )
    return {
        "num_pairs": len(baseline),
        "identity_matches_baseline": (
            None if identity is None else sum(b == i for b, i in zip(baseline, identity))
        ),
        "baseline_success_actor_fail": sum(b and not a for b, a in zip(baseline, actor)),
        "baseline_fail_actor_success": sum((not b) and a for b, a in zip(baseline, actor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/identity/actor paired eval with matched seeds.")
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--no_save_video",
        action="store_true",
        help="Disable paired eval mp4 recording. By default videos are saved.",
    )
    parser.add_argument(
        "--skip_identity",
        action="store_true",
        help="Run only baseline/off and actor. The summary sets identity_matches_baseline to null.",
    )
    args = parser.parse_args()

    output_root = Path(
        args.output_root
        or Path("playground_eval") / "paired_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")

    runs = {"baseline": _run_eval("off", args, output_root)}
    if not args.skip_identity:
        runs["identity"] = _run_eval("identity", args, output_root)
    runs["actor"] = _run_eval("actor", args, output_root)
    comparison = _compare(
        runs["baseline"]["results"],
        None if args.skip_identity else runs["identity"]["results"],
        runs["actor"]["results"],
    )
    summary = {"runs": runs, "comparison": comparison}
    summary_path = output_root / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[paired-eval] summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
