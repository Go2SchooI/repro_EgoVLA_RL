from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


TASK_LINE_RE = re.compile(
    r"Task: (?P<task>.*?), Room Idx: (?P<room_idx>\d+), Table Idx: (?P<table_idx>\d+), "
    r"Episode Label: (?P<episode_label>[^,]+), Trial Label: (?P<trial_idx>\d+), "
    r"Result: (?P<result>True|False)"
)
EPISODE_LENGTH_RE = re.compile(r"\bepisode_length=(?P<episode_length>\d+)")
SUCCESS_RE = re.compile(r"(?P<key>[A-Za-z0-9_]*success): (?P<value>[-+]?\d+(?:\.\d+)?)")


def _task_key(task_name: str) -> str:
    if task_name.startswith("Humanoid-") and task_name.endswith("-v0"):
        return task_name[len("Humanoid-") : -len("-v0")]
    return task_name


def _episode_specs(task_name: str, requested: int):
    utils_path = Path(__file__).resolve().parents[1] / "human_plan" / "ego_bench_eval" / "utils.py"
    module = ast.parse(utils_path.read_text())
    task_init_episode = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "TASK_INIT_EPISODE" for target in node.targets):
            task_init_episode = ast.literal_eval(node.value)
            break
    if task_init_episode is None:
        raise KeyError(f"Could not find TASK_INIT_EPISODE in {utils_path}.")
    task_key = _task_key(task_name)
    if task_key not in task_init_episode:
        raise KeyError(f"Task {task_name!r} is not present in TASK_INIT_EPISODE.")
    episodes = task_init_episode[task_key]
    return episodes[: min(int(requested), len(episodes))]


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _npz_info(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        info = {
            "replay_path": str(path),
            "transitions": int(data["actor_obs"].shape[0]),
        }
        if "episode_length" in data.files:
            done = np.asarray(data["done"]).reshape(-1) > 0.5
            lengths = np.asarray(data["episode_length"]).reshape(-1)
            terminal_lengths = lengths[done]
            if terminal_lengths.size:
                info["episode_length"] = int(terminal_lengths[-1])
        if "episode_success" in data.files:
            successes = np.asarray(data["episode_success"]).reshape(-1)
            if successes.size:
                info["episode_success"] = bool(float(successes.max()) > 0.5)
        if "episode_timeout" in data.files:
            timeouts = np.asarray(data["episode_timeout"]).reshape(-1)
            if timeouts.size:
                info["episode_timeout"] = bool(float(timeouts.max()) > 0.5)
    return info


def _parse_result_file(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = path.read_text(errors="replace").splitlines()
    parsed: dict = {"result_path": str(path)}
    for line in lines:
        match = TASK_LINE_RE.search(line)
        if match:
            parsed.update(
                {
                    "task": match.group("task"),
                    "room_idx": int(match.group("room_idx")),
                    "table_idx": int(match.group("table_idx")),
                    "episode_label": match.group("episode_label").strip(),
                    "trial_idx": int(match.group("trial_idx")),
                    "success": match.group("result") == "True",
                }
            )
        length_match = EPISODE_LENGTH_RE.search(line)
        if length_match:
            parsed["episode_length"] = int(length_match.group("episode_length"))
        success_pairs = SUCCESS_RE.findall(line)
        if success_pairs:
            parsed.setdefault("subtask_success", {}).update(
                {key: float(value) for key, value in success_pairs}
            )
    return parsed


def _base_env(args, output: Path, num_episodes: int, num_trials: int) -> dict[str, str]:
    env = os.environ.copy()
    env["RL_MODE"] = "identity"
    env["RL_COLLECT_REPLAY_PATH"] = str(output)
    env["RL_COLLECT_SOURCE"] = args.source
    env["RL_COLLECT_SAVE_RAW"] = "1" if args.save_raw else "0"
    env["SAVE_VIDEO"] = "1" if args.save_video else "0"
    env.setdefault("SAVE_FRAMES", "0")
    env["NUM_EPISODES"] = str(num_episodes)
    env["NUM_TRIALS"] = str(num_trials)
    if args.max_eval_steps is not None:
        env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    if args.task:
        env["TASK"] = args.task
    if args.room_idx is not None:
        env["ROOM_IDX"] = str(args.room_idx)
    if args.table_idx is not None:
        env["TABLE_IDX"] = str(args.table_idx)
    return env


def _run_single(args, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = _base_env(args, output, args.num_episodes, args.num_trials)
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)


def _run_cached(args, output: Path) -> None:
    task_name = args.task or os.environ.get("TASK", "Humanoid-Push-Box-v0")
    room_idx = int(args.room_idx if args.room_idx is not None else os.environ.get("ROOM_IDX", 1))
    table_idx = int(args.table_idx if args.table_idx is not None else os.environ.get("TABLE_IDX", 1))
    episodes = _episode_specs(task_name, args.num_episodes)

    cache_root = output if output.suffix != ".npz" else output.with_suffix("")
    cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json) if args.summary_json else cache_root / "episodes.json"
    summary = _read_json(summary_path)
    summary.update(
        {
            "format": "rl_posttrain_collect_base_episode_summary_v1",
            "task": task_name,
            "room_idx": room_idx,
            "table_idx": table_idx,
            "source": args.source,
            "num_episodes_requested": int(args.num_episodes),
            "num_episodes_collected": len(episodes),
            "num_trials": int(args.num_trials),
            "cache_root": str(cache_root),
            "replay_glob": str(cache_root / "**" / "*.npz"),
        }
    )
    existing = {
        (
            int(item.get("episode_start_idx", -1)),
            int(item.get("trial_idx", -1)),
        ): item
        for item in summary.get("episodes", [])
    }

    records: list[dict] = []
    total_units = len(episodes) * int(args.num_trials)
    for episode_start_idx, episode_spec in enumerate(episodes):
        episode_label = str(episode_spec[0])
        label = _safe_label(Path(episode_label).stem)
        for trial_idx in range(int(args.num_trials)):
            unit_key = f"room{room_idx}_table{table_idx}_ep{episode_start_idx:03d}_{label}_trial{trial_idx:03d}"
            unit_dir = cache_root / unit_key
            replay_path = unit_dir / f"{unit_key}.npz"
            result_path = unit_dir / "results_local_eval.txt"
            run_dir = unit_dir / "run"
            record = {
                "status": "pending",
                "task": task_name,
                "room_idx": room_idx,
                "table_idx": table_idx,
                "episode_start_idx": episode_start_idx,
                "episode_label": episode_label,
                "trial_idx": trial_idx,
                "replay_path": str(replay_path),
                "result_path": str(result_path),
            }
            cached_record = existing.get((episode_start_idx, trial_idx))
            if replay_path.exists() and not args.force:
                try:
                    record.update(cached_record or {})
                    record.update(_npz_info(replay_path))
                    record.update(_parse_result_file(result_path))
                    record["status"] = "cached"
                    print(f"[collect-base-cache] skip existing {replay_path}")
                    records.append(record)
                    summary["episodes"] = records
                    summary["completed"] = len([item for item in records if item["status"] in ("cached", "completed")])
                    summary["total"] = total_units
                    _write_json(summary_path, summary)
                    continue
                except Exception as exc:
                    print(f"[collect-base-cache] existing replay is not usable, re-running: {replay_path} error={exc}")

            unit_dir.mkdir(parents=True, exist_ok=True)
            if result_path.exists():
                result_path.unlink()
            if replay_path.exists():
                replay_path.unlink()

            env = _base_env(args, replay_path, 1, 1)
            env["TASK"] = task_name
            env["ROOM_IDX"] = str(room_idx)
            env["TABLE_IDX"] = str(table_idx)
            env["EPISODE_START_IDX"] = str(episode_start_idx)
            env["TRIAL_START_IDX"] = str(trial_idx)
            env["RANDOMIZE_TOTAL_EPISODES"] = str(len(episodes))
            env["RANDOMIZE_TOTAL_TRIALS"] = str(args.num_trials)
            env["RESULT_PATH"] = str(result_path)
            env["RUN_DIR"] = str(run_dir)
            print(
                "[collect-base-cache] run "
                f"episode={episode_label} episode_start_idx={episode_start_idx} "
                f"trial={trial_idx} replay={replay_path}"
            )
            try:
                subprocess.run(["./run_local_eval.sh"], env=env, check=True)
                record.update(_npz_info(replay_path))
                record.update(_parse_result_file(result_path))
                record["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                record["status"] = "failed"
                record["returncode"] = int(exc.returncode)
                records.append(record)
                summary["episodes"] = records
                summary["completed"] = len([item for item in records if item["status"] in ("cached", "completed")])
                summary["total"] = total_units
                _write_json(summary_path, summary)
                raise

            records.append(record)
            summary["episodes"] = records
            summary["completed"] = len([item for item in records if item["status"] in ("cached", "completed")])
            summary["total"] = total_units
            _write_json(summary_path, summary)

    summary["episodes"] = records
    summary["completed"] = len([item for item in records if item["status"] in ("cached", "completed")])
    summary["total"] = total_units
    summary["success_count"] = len([item for item in records if item.get("success") is True])
    summary["failure_count"] = len([item for item in records if item.get("success") is False])
    summary["replay_files"] = [item["replay_path"] for item in records if item["status"] in ("cached", "completed")]
    _write_json(summary_path, summary)
    print(
        f"[collect-base-cache] summary={summary_path} "
        f"completed={summary['completed']}/{summary['total']} "
        f"success={summary['success_count']} failure={summary['failure_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect base/identity EgoVLA replay for TD3+BC post-training.")
    parser.add_argument(
        "--output",
        required=True,
        help="Replay .npz path for one-shot collection, or a directory for cached episode/trial shards.",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--source", choices=("base", "identity"), default="base")
    parser.add_argument("--room_idx", type=int, default=None)
    parser.add_argument("--table_idx", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--summary_json", default=None, help="Optional path for cached episode summary JSON.")
    parser.add_argument("--cache", action="store_true", help="Use cached episode/trial sharding for .npz outputs too.")
    parser.add_argument("--force", action="store_true", help="Re-run cached shards even if their replay npz exists.")
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cached sharding. Requires --output to be a .npz file.",
    )
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save eval mp4 videos during replay collection. Disabled by default.",
    )
    args = parser.parse_args()

    output = Path(args.output)

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")
    if args.no_cache:
        if output.suffix != ".npz":
            raise ValueError("--no_cache requires --output to be a .npz path.")
        _run_single(args, output)
    elif output.suffix == ".npz" and not args.cache:
        _run_single(args, output)
    else:
        _run_cached(args, output)


if __name__ == "__main__":
    main()
