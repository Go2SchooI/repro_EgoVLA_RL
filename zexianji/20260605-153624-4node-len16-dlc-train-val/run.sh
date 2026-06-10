#!/usr/bin/env bash
set -euo pipefail

# Lightweight sandbox wrapper for SG DLC 4-node LEN16 training + EPIC-only validation.
# It records launch metadata here, while formal outputs stay under the LEN16 project output tree.
EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENTRY="${ENTRY:-<DLC_ENTRY_SCRIPT>}"
FORMAL_ROOT="${FORMAL_ROOT:-${EXP_DIR}/formal_outputs}"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/env" "${EXP_DIR}/artifacts"
mkdir -p "${FORMAL_ROOT}/runs_epic_val_epoch2" "${FORMAL_ROOT}/logs" "${FORMAL_ROOT}/cache/tmp"

# Keep formal output paths canonical; sandbox stores only records/manifests.
export SCALING_ROOT="${SCALING_ROOT:-${FORMAL_ROOT}}"
export RUN_BASE="${RUN_BASE:-${FORMAL_ROOT}/runs_epic_val_epoch2}"
export LOG_BASE="${LOG_BASE:-${FORMAL_ROOT}/logs}"
export CACHE_ROOT="${CACHE_ROOT:-${FORMAL_ROOT}/cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
mkdir -p "${RUN_BASE}" "${LOG_BASE}" "${CACHE_ROOT}" "${TMPDIR}"

# Keep data access consistent with recent full-read validation/training convention.
export VITRA_PARQUET_READ_MODE="${VITRA_PARQUET_READ_MODE:-full}"
export VITRA_PARQUET_WINDOW_READ="${VITRA_PARQUET_WINDOW_READ:-0}"

# SG/new-machine Python path.
# PATH is intentionally left unchanged in this sanitized record; set PYTHON_BIN if needed.
export PYTHON_BIN="${PYTHON_BIN:-python}"

# Defaults are repeated for auditability; upstream LEN16 entry has the same values.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000000000}"
export SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-1}"
export RUN_VAL="${RUN_VAL:-1}"
export VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS:-3000}"

ENV_FILE="${EXP_DIR}/env/launch_env_runtime.txt"
{
  echo "created=2026-06-05 15:36:24 Asia/Shanghai"
  echo "start_time=$(date '+%F %T %Z')"
  echo "host=<runtime-host>"
  echo "EXP_DIR=${EXP_DIR}"
  echo "ENTRY=${ENTRY}"
  echo "git_branch=<git-branch>"
  echo "git_head=6895add"
  echo "SCALING_ROOT=${SCALING_ROOT}"
  echo "RUN_BASE=${RUN_BASE}"
  echo "LOG_BASE=${LOG_BASE}"
  echo "CACHE_ROOT=${CACHE_ROOT}"
  echo "TMPDIR=${TMPDIR}"
  echo "VITRA_PARQUET_READ_MODE=${VITRA_PARQUET_READ_MODE}"
  echo "VITRA_PARQUET_WINDOW_READ=${VITRA_PARQUET_WINDOW_READ}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} BATCH_SIZE=${BATCH_SIZE} TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} MAX_EPOCHS=${MAX_EPOCHS} DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS}"
  echo "env_snapshot=omitted_for_privacy"
} > "${ENV_FILE}"

# Record formal output locations for later lookup without duplicating large files.
cat > "${EXP_DIR}/artifacts/output_paths.txt" <<EOF_PATHS
formal_root=${FORMAL_ROOT}
run_base=${RUN_BASE}
log_base=${LOG_BASE}
cache_root=${CACHE_ROOT}
entry=${ENTRY}
EOF_PATHS

if [[ "$ENTRY" == "<DLC_ENTRY_SCRIPT>" ]]; then
  echo "Set ENTRY to the DLC entry script before running." >&2
  exit 2
fi

bash "${ENTRY}"
rc=$?
echo "end_time=$(date '+%F %T %Z') rc=${rc}" >> "${ENV_FILE}"
exit "${rc}"
