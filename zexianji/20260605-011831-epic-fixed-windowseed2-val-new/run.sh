#!/usr/bin/env bash
set -Eeuo pipefail
EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="${REPO:-${REPO_ROOT:-<REPO_ROOT>}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VAL_CONFIG=$EXP_DIR/config/validation_epic100_ego4d100_fixed_windows.yaml
INDICES_JSON=$EXP_DIR/config/window_indices_seed20260606_n3000.json
WINDOW_SEED=${WINDOW_SEED:-20260606}
DATASET_SIZE=${DATASET_SIZE:-69606}
NUM_WINDOWS=${NUM_WINDOWS:-3000}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/artifacts/shards" "$EXP_DIR/artifacts/merged" "$EXP_DIR/cache/tmp" "$EXP_DIR/cache/hf" "$EXP_DIR/cache/xdg" "$EXP_DIR/env"
export TMPDIR=$EXP_DIR/cache/tmp
export HF_HOME=$EXP_DIR/cache/hf
export XDG_CACHE_HOME=$EXP_DIR/cache/xdg
export WANDB_DIR=$EXP_DIR/logs/wandb
export WANDB_MODE=${WANDB_MODE:-offline}
export VITRA_PARQUET_READ_MODE=full
export VITRA_PARQUET_WINDOW_READ=0
export WUJI_REPO_ROOT=$REPO
if [[ "$REPO" == "<REPO_ROOT>" ]]; then
  echo "Set REPO or REPO_ROOT to the cloned repository path before running." >&2
  exit 2
fi
export WUJI_VAL_WINDOW_INDICES=$INDICES_JSON
"$PYTHON_BIN" "$EXP_DIR/scripts/make_window_indices.py" --output "$INDICES_JSON" --dataset-size "$DATASET_SIZE" --num-windows "$NUM_WINDOWS" --seed "$WINDOW_SEED" | tee "$EXP_DIR/logs/window_indices.log"
(cd "$REPO" && git rev-parse HEAD > "$EXP_DIR/env/git_head.txt" 2>/dev/null || true)
"$PYTHON_BIN" - <<'PY' > "$EXP_DIR/env/python.txt" 2>&1 || true
import sys, torch
print(sys.executable)
print(sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
PY
if [ -n "${GPU_LIST:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 1 ]; then GPU_COUNT=1; fi
  GPUS=()
  for ((i=0; i<GPU_COUNT; i++)); do GPUS+=("$i"); done
fi
NUM_SHARDS=${NUM_SHARDS:-${#GPUS[@]}}
if [ "$NUM_SHARDS" -gt "${#GPUS[@]}" ]; then
  echo "NUM_SHARDS=$NUM_SHARDS exceeds GPU count/list ${#GPUS[@]}" >&2
  exit 2
fi
declare -a LABELS=("3-node-bs4x24-h200" "4-node-bs2x32-h200")
declare -a CKPTS=(
  "${CKPT_0:-<CHECKPOINT_0>}"
  "${CKPT_1:-<CHECKPOINT_1>}"
)
echo "start_time=$(date -Is)" | tee "$EXP_DIR/logs/run.log"
echo "num_shards=$NUM_SHARDS gpus=${GPUS[*]} batch_size=$BATCH_SIZE" | tee -a "$EXP_DIR/logs/run.log"
for idx in "${!LABELS[@]}"; do
  RUN_LABEL=${LABELS[$idx]}
  CKPT=${CKPTS[$idx]}
  echo "=== checkpoint $RUN_LABEL ===" | tee -a "$EXP_DIR/logs/run.log"
  PIDS=()
  SHARD_DIRS=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu=${GPUS[$shard]}
    run_name=${RUN_LABEL}-fixedwin-seed${WINDOW_SEED}-shard${shard}of${NUM_SHARDS}
    out_root=$EXP_DIR/artifacts/shards/$RUN_LABEL
    log=$EXP_DIR/logs/${run_name}.log
    mkdir -p "$out_root"
    SHARD_DIRS+=("$out_root/$run_name")
    echo "launch label=$RUN_LABEL shard=$shard gpu=$gpu log=$log" | tee -a "$EXP_DIR/logs/run.log"
    (
      cd "$REPO"
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$EXP_DIR/scripts/fixed_index_run_vitra.py" \
        --config "$VAL_CONFIG" \
        --checkpoint "$CKPT" \
        --output-root "$out_root" \
        --run-name "$run_name" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --max-windows "$NUM_WINDOWS"
    ) > "$log" 2>&1 &
    PIDS+=("$!")
  done
  rc=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then rc=1; fi
  done
  if [ "$rc" -ne 0 ]; then
    echo "one or more shards failed for $RUN_LABEL; see $EXP_DIR/logs" | tee -a "$EXP_DIR/logs/run.log"
    exit "$rc"
  fi
  MERGED=$EXP_DIR/artifacts/merged/$RUN_LABEL
  (
    cd "$REPO"
    "$PYTHON_BIN" validation/scripts/merge_vitra_shards.py "${SHARD_DIRS[@]}" --output-dir "$MERGED" --top-k 20
  ) > "$EXP_DIR/logs/${RUN_LABEL}-merge.log" 2>&1
  echo "merged $RUN_LABEL -> $MERGED" | tee -a "$EXP_DIR/logs/run.log"
done
"$PYTHON_BIN" - "$EXP_DIR" <<'PY'
import json, pathlib, sys, time
exp=pathlib.Path(sys.argv[1])
labels=['3-node-bs4x24-h200','4-node-bs2x32-h200']
lines=['# Results','', '## Status','success','', '## Timing', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %z")}', '', '## Metrics']
for label in labels:
    p=exp/'artifacts'/'merged'/label/'metrics.json'
    m=json.load(open(p)); raw=m.get('raw_action_mse') or {}
    lines += ['', f'### {label}', f'- n_samples: {m.get("n_samples")}', f'- normalized_mse_mean: {m.get("mse_mean")}', f'- normalized_sample_mse_mean: {m.get("sample_mse_mean")}', f'- raw_action_mse_mean: {raw.get("mse_mean")}', f'- raw_action_sample_mse_mean: {raw.get("sample_mse_mean")}', f'- merged: {p.parent}']
(exp/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "done" | tee -a "$EXP_DIR/logs/run.log"
