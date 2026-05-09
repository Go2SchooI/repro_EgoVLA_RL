from __future__ import annotations

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline, identity, or TD3+BC actor eval.")
    parser.add_argument("--mode", choices=("off", "identity", "actor"), default="identity")
    parser.add_argument("--checkpoint", default=None, help="TD3+BC actor checkpoint for --mode=actor.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "actor" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for --mode=actor")

    env = os.environ.copy()
    env["RL_MODE"] = args.mode
    env["NUM_EPISODES"] = str(args.num_episodes)
    env["NUM_TRIALS"] = str(args.num_trials)
    if args.checkpoint:
        env["RL_ACTOR_CHECKPOINT"] = args.checkpoint
    if args.max_eval_steps is not None:
        env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    if args.task:
        env["TASK"] = args.task

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)


if __name__ == "__main__":
    main()
