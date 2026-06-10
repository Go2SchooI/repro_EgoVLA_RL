#!/usr/bin/env bash
set -euo pipefail

# Lightweight sandbox wrapper for SG DLC 2-node LEN16 per-episode-file training + EPIC-only validation.
# Formal outputs stay under FORMAL_ROOT; this sandbox records launch metadata and paths.
EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENTRY="${ENTRY:-<DLC_ENTRY_SCRIPT>}"
FORMAL_ROOT="${FORMAL_ROOT:-${EXP_DIR}/formal_outputs}"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/env" "${EXP_DIR}/artifacts"
mkdir -p "${FORMAL_ROOT}/runs_epic_val_epoch2" "${FORMAL_ROOT}/logs" "${FORMAL_ROOT}/cache/tmp"

export SCALING_ROOT="${SCALING_ROOT:-${FORMAL_ROOT}}"
export RUN_BASE="${RUN_BASE:-${FORMAL_ROOT}/runs_epic_val_epoch2}"
export LOG_BASE="${LOG_BASE:-${FORMAL_ROOT}/logs}"
export CACHE_ROOT="${CACHE_ROOT:-${FORMAL_ROOT}/cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
mkdir -p "${RUN_BASE}" "${LOG_BASE}" "${CACHE_ROOT}" "${TMPDIR}"

# Force 2-node topology; DLC/PET rank/address variables are still consumed by the upstream entry.
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TOTAL_GPUS_LABEL="${TOTAL_GPUS_LABEL:-16}"
export ALLOW_NON_32_GPUS="${ALLOW_NON_32_GPUS:-1}"

# Use the per-episode-file dataloader variants. Manifest/stats are unchanged.
export DATALOADER="${DATALOADER:-vitra_epic100_ego4d_len16_per_ep}"
export VAL_CONFIG="${VAL_CONFIG:-<VALIDATION_CONFIG>}"

# Requested scale: 2 nodes x 8 GPUs, per-GPU bs=32, workers=8.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-1000000000}"
export SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-1}"
export RUN_VAL="${RUN_VAL:-1}"
export VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS:-3000}"

# Keep data reads in full mode; the new root has smaller per-episode files.
export VITRA_PARQUET_READ_MODE="${VITRA_PARQUET_READ_MODE:-full}"
export VITRA_PARQUET_WINDOW_READ="${VITRA_PARQUET_WINDOW_READ:-0}"

# SG/new-machine Python path.
# PATH is intentionally left unchanged in this sanitized record; set PYTHON_BIN if needed.
export PYTHON_BIN="${PYTHON_BIN:-python}"

# Make run names and wandb labels distinguishable from previous LEN16 runs.
export RUN_NAME="${RUN_NAME:-epic100-ego4d-len16-per-ep-2node-bs32x16-w8-epoch${MAX_EPOCHS}-$(date -u +%Y%m%dT%H%M%SZ)}"
export WANDB_RUN_VARIANT="${WANDB_RUN_VARIANT:-epic100-ego4d-len16-per-ep-epoch2-2node-bs32x16-w8}"
export WANDB_TAGS="${WANDB_TAGS:-stage=scaling-law,backbone=paligemma2,dataset=epic100-plus-ego4d-len16,variant=ego4d-len16-per-ep-2node-bs32w8,scale=len16,bs=32,gpus=16,nodes=2,workers=8,read=full}"
export WANDB_NOTES="${WANDB_NOTES:-[epic100-ego4d-len16-per-ep-2node-bs32w8] EPIC100 fixed + ego4d screened-len16; data root=<DATA_ROOT>; 2 nodes x 8 GPUs; per_gpu_bs=32; total_batch=512; workers=8; max_epochs=2; full parquet read; validation=current EPIC seen-val max_windows=3000.}"

# Optional learning-rate overrides. If unset, keep the upstream defaults:
# training_strategy.learning_rate=1e-5 and action_model_learning_rate=1e-4.
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-training_strategy.max_steps=null}"
if [[ -n "${LEARNING_RATE:-}" ]]; then
  TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS} training_strategy.learning_rate=${LEARNING_RATE}"
fi
if [[ -n "${ACTION_MODEL_LEARNING_RATE:-}" ]]; then
  TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS} training_strategy.action_model_learning_rate=${ACTION_MODEL_LEARNING_RATE}"
fi
export TRAIN_EXTRA_ARGS

ENV_FILE="${EXP_DIR}/env/launch_env_runtime.txt"
{
  echo "created=2026-06-08 12:00:48 Asia/Shanghai"
  echo "start_time=$(date '+%F %T %Z')"
  echo "host=<runtime-host>"
  echo "EXP_DIR=${EXP_DIR}"
  echo "ENTRY=${ENTRY}"
  echo "git_branch=<git-branch>"
  echo "git_head=6895add"
  echo "FORMAL_ROOT=${FORMAL_ROOT}"
  echo "SCALING_ROOT=${SCALING_ROOT}"
  echo "RUN_BASE=${RUN_BASE}"
  echo "LOG_BASE=${LOG_BASE}"
  echo "CACHE_ROOT=${CACHE_ROOT}"
  echo "TMPDIR=${TMPDIR}"
  echo "DATALOADER=${DATALOADER}"
  echo "VAL_CONFIG=${VAL_CONFIG}"
  echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} TOTAL_GPUS_LABEL=${TOTAL_GPUS_LABEL}"
  echo "BATCH_SIZE=${BATCH_SIZE} TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS} MAX_EPOCHS=${MAX_EPOCHS}"
  echo "LEARNING_RATE=${LEARNING_RATE:-<default 1e-5>} ACTION_MODEL_LEARNING_RATE=${ACTION_MODEL_LEARNING_RATE:-<default 1e-4>}"
  echo "TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS}"
  echo "VITRA_PARQUET_READ_MODE=${VITRA_PARQUET_READ_MODE}"
  echo "VITRA_PARQUET_WINDOW_READ=${VITRA_PARQUET_WINDOW_READ}"
  echo "env_snapshot=omitted_for_privacy"
} > "${ENV_FILE}"

cat > "${EXP_DIR}/artifacts/output_paths.txt" <<EOF_PATHS
formal_root=${FORMAL_ROOT}
run_base=${RUN_BASE}
log_base=${LOG_BASE}
cache_root=${CACHE_ROOT}
entry=${ENTRY}
dataloader=${DATALOADER}
val_config=${VAL_CONFIG}
train_dataloader_yaml=<TRAIN_DATALOADER_YAML>
val_dataloader_yaml=<VAL_DATALOADER_YAML>
data_root=<DATA_ROOT>
EOF_PATHS

if [[ "$ENTRY" == "<DLC_ENTRY_SCRIPT>" ]]; then
  echo "Set ENTRY to the DLC entry script before running." >&2
  exit 2
fi

bash "${ENTRY}"
rc=$?
echo "end_time=$(date '+%F %T %Z') rc=${rc}" >> "${ENV_FILE}"
exit "${rc}"
