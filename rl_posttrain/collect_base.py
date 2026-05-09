from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect base/identity EgoVLA replay for TD3+BC post-training.")
    parser.add_argument("--output", required=True, help="Replay .npz path to write.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--source", choices=("base", "identity"), default="base")
    parser.add_argument("--room_idx", type=int, default=None)
    parser.add_argument("--table_idx", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save eval mp4 videos during replay collection. Disabled by default.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RL_MODE"] = "identity"
    env["RL_COLLECT_REPLAY_PATH"] = str(output)
    env["RL_COLLECT_SOURCE"] = args.source
    env["RL_COLLECT_SAVE_RAW"] = "1" if args.save_raw else "0"
    env["SAVE_VIDEO"] = "1" if args.save_video else "0"
    env.setdefault("SAVE_FRAMES", "0")
    env["NUM_EPISODES"] = str(args.num_episodes)
    env["NUM_TRIALS"] = str(args.num_trials)
    if args.max_eval_steps is not None:
        env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    if args.task:
        env["TASK"] = args.task
    if args.room_idx is not None:
        env["ROOM_IDX"] = str(args.room_idx)
    if args.table_idx is not None:
        env["TABLE_IDX"] = str(args.table_idx)

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)


if __name__ == "__main__":
    main()
