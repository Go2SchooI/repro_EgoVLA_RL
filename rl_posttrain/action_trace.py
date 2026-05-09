from __future__ import annotations

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EgoVLA eval with RL action path tracing enabled.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--max_eval_steps", type=int, default=2)
    parser.add_argument("--trace_steps", type=int, default=2)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    args = parser.parse_args()

    env = os.environ.copy()
    env["RL_MODE"] = "identity"
    env["RL_ACTION_TRACE"] = "1"
    env["RL_ACTION_TRACE_STEPS"] = str(args.trace_steps)
    env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    env["NUM_EPISODES"] = str(args.num_episodes)
    env["NUM_TRIALS"] = str(args.num_trials)
    if args.task:
        env["TASK"] = args.task

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)


if __name__ == "__main__":
    main()

