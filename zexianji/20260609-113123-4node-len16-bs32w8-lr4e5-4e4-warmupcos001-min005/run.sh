#!/usr/bin/env bash
set -euo pipefail

# Lightweight sandbox wrapper for SG/new DLC 4-node LEN16 per-episode-file training + EPIC-only validation.
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

# 4 nodes x 8 GPUs, per-GPU bs=32.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1024}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-1000000000}"
export SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-1}"
export RUN_VAL="${RUN_VAL:-1}"
export VAL_MAX_WINDOWS="${VAL_MAX_WINDOWS:-3000}"

# Use the per-episode-file dataloader variants. Manifest/stats are unchanged.
export DATALOADER="${DATALOADER:-vitra_epic100_ego4d_len16_per_ep}"
export VAL_CONFIG="${VAL_CONFIG:-<VALIDATION_CONFIG>}"

# Keep data reads in full mode; the per-episode root has smaller files.
export VITRA_PARQUET_READ_MODE="${VITRA_PARQUET_READ_MODE:-full}"
export VITRA_PARQUET_WINDOW_READ="${VITRA_PARQUET_WINDOW_READ:-0}"

# SG/new-machine Python path.
# PATH is intentionally left unchanged in this sanitized record; set PYTHON_BIN if needed.
export PYTHON_BIN="${PYTHON_BIN:-python}"

# Linear warmup + cosine decay. Warmup is 1%, cosine min LR is 1/20 peak.
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
export LLM_FREEZE_STEP="${LLM_FREEZE_STEP:-0}"

# Backbone/action LR plus explicit scheduler/min LR. Keep max_steps=null so MAX_EPOCHS controls training length.
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-training_strategy.max_steps=null training_strategy.lr_scheduler_type=warmup_cosine training_strategy.learning_rate=4e-5 training_strategy.action_model_learning_rate=4e-4 training_strategy.cosine_min_lr_rate=0.05}"
export TRAIN_EXTRA_ARGS

# Make filesystem run name and wandb labels distinguishable. Keep tags short enough for wandb's 64-char limit.
export RUN_NAME="${RUN_NAME:-len16_4node_lr44_wcos001_min005_sg}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_RUN_VARIANT="${WANDB_RUN_VARIANT:-len16-4node-bs32-lr44-wcos001-min005-sg}"
export WANDB_TAGS="${WANDB_TAGS:-stage=scaling-law,backbone=pg2,dataset=len16,variant=wcos001-min005,scale=len16,bs=32,gpus=32,nodes=4,workers=8,read=full,scheduler=warmup_cosine,warmup=0.01,minlr=0.05,host=<host>}"
export WANDB_NOTES="${WANDB_NOTES:-[SG new: len16 4node bs32 lr44 warmup-cosine001 min005] EPIC100 fixed + ego4d screened-len16; data root=<DATA_ROOT>; 4 nodes x 8 GPUs; per_gpu_bs=32; total_batch=1024; workers=8; max_epochs=2; backbone_lr=4e-5; action_model_lr=4e-4; scheduler=linear-warmup+cosine-decay; warmup_ratio=0.01; cosine_min_lr_rate=0.05 (1/20 peak); LLM_FREEZE_STEP=0; full parquet read; validation=current EPIC seen-val max_windows=3000.}"

ENV_FILE="${EXP_DIR}/env/launch_env_runtime.txt"
{
  echo "created=$(date '+%F %T %Z')"
  echo "host=<runtime-host>"
  echo "EXP_DIR=${EXP_DIR}"
  echo "ENTRY=${ENTRY}"
  echo "FORMAL_ROOT=${FORMAL_ROOT}"
  echo "SCALING_ROOT=${SCALING_ROOT}"
  echo "RUN_BASE=${RUN_BASE}"
  echo "LOG_BASE=${LOG_BASE}"
  echo "CACHE_ROOT=${CACHE_ROOT}"
  echo "TMPDIR=${TMPDIR}"
  echo "DATALOADER=${DATALOADER}"
  echo "VAL_CONFIG=${VAL_CONFIG}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "BATCH_SIZE=${BATCH_SIZE} TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS} MAX_EPOCHS=${MAX_EPOCHS}"
  echo "LLM_FREEZE_STEP=${LLM_FREEZE_STEP} WARMUP_RATIO=${WARMUP_RATIO}"
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
train_windows=6143126
steps_per_epoch_at_global_bs_1024=6000
total_steps_epoch2=12000
warmup_ratio=0.01
warmup_steps=120
cosine_min_lr_rate=0.05
scheduler=warmup_cosine
EOF_PATHS

if [[ "$ENTRY" == "<DLC_ENTRY_SCRIPT>" ]]; then
  echo "Set ENTRY to the DLC entry script before running." >&2
  exit 2
fi

bash "${ENTRY}"
rc=$?
echo "end_time=$(date '+%F %T %Z') rc=${rc}" >> "${ENV_FILE}"
exit "${rc}"
