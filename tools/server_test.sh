#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"
# Avoid loading unrelated plugins inherited from the environment bootstrap.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest rl_posttrain/test_action_utils.py rl_posttrain/test_obs_utils.py \
  rl_posttrain/test_replay_buffer.py rl_posttrain/test_td3bc_ref.py \
  rl_posttrain/test_online_td3bc.py "$@"
