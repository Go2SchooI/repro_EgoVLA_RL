#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-${REPO_ROOT:-<REPO_ROOT>}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$REPO" == "<REPO_ROOT>" ]]; then
  echo "Set REPO or REPO_ROOT to the cloned repository path before running." >&2
  exit 2
fi
cd "$REPO"
"$PYTHON_BIN" -m py_compile \
  openwam/train/vla/train_loop.py \
  openwam/train/vla/trainer.py \
  openwam/train/vla/wandb_protocol.py
