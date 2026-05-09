#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ISAACLAB_PATH="${IsaacLab_PATH:-/home/jizexian/IsaacLab}"
CUDA_LIB_PATH="${CUDA_LIB_PATH:-/usr/local/cuda-12.8/lib64}"
SYSTEM_LIBSTDCPP_PATH="${SYSTEM_LIBSTDCPP_PATH:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
VILA_PURELIB="${VILA_PURELIB:-/home/jizexian/anaconda3/envs/vila/lib/python3.10/site-packages}"
VENDOR_PY="${VENDOR_PY:-$REPO_ROOT/.vendor_py}"

TASK="${TASK:-Humanoid-Push-Box-v0}"
ROOM_IDX="${ROOM_IDX:-1}"
TABLE_IDX="${TABLE_IDX:-1}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
EPISODE_START_IDX="${EPISODE_START_IDX:-0}"
TRIAL_START_IDX="${TRIAL_START_IDX:-0}"
RANDOMIZE_TOTAL_EPISODES="${RANDOMIZE_TOTAL_EPISODES:-$NUM_EPISODES}"
RANDOMIZE_TOTAL_TRIALS="${RANDOMIZE_TOTAL_TRIALS:-$NUM_TRIALS}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.2}"
HAND_SMOOTH_WEIGHT="${HAND_SMOOTH_WEIGHT:-0.8}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_FRAMES="${SAVE_FRAMES:-0}"
PROJECT_TRAJS="${PROJECT_TRAJS:-0}"
VISION_INPUT_MODE="${VISION_INPUT_MODE:-real}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-0}"
CHUNK_EXEC_LEN="${CHUNK_EXEC_LEN:-none}"
IMAGE_UPDATE_INTERVAL="${IMAGE_UPDATE_INTERVAL:-1}"
IMAGE_DELAY_STEPS="${IMAGE_DELAY_STEPS:-0}"
PROPRIO_ABLATION_MODE="${PROPRIO_ABLATION_MODE:-none}"
PROPRIO_DELAY_STEPS="${PROPRIO_DELAY_STEPS:-0}"
EVAL_ABLATION_DEBUG="${EVAL_ABLATION_DEBUG:-0}"
RL_MODE="${RL_MODE:-off}"
RL_ACTION_TRACE="${RL_ACTION_TRACE:-0}"
RL_ACTION_TRACE_STEPS="${RL_ACTION_TRACE_STEPS:-2}"
RL_IDENTITY_TOLERANCE="${RL_IDENTITY_TOLERANCE:-1e-5}"
RL_ACTOR_CHECKPOINT="${RL_ACTOR_CHECKPOINT:-}"
RL_COLLECT_REPLAY_PATH="${RL_COLLECT_REPLAY_PATH:-}"
RL_COLLECT_SOURCE="${RL_COLLECT_SOURCE:-base}"
RL_COLLECT_SAVE_RAW="${RL_COLLECT_SAVE_RAW:-0}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/ego_vla_checkpoint/ckpt-human-video-pretrained}"
DEFAULT_ADDITIONAL_LABEL="$(basename "${MODEL_PATH%/}")"
if [[ "$VISION_INPUT_MODE" != "real" ]]; then
  DEFAULT_ADDITIONAL_LABEL="${DEFAULT_ADDITIONAL_LABEL}_${VISION_INPUT_MODE}"
fi
DEFAULT_ADDITIONAL_LABEL="${DEFAULT_ADDITIONAL_LABEL//[^[:alnum:]_.-]/_}"
ADDITIONAL_LABEL="${ADDITIONAL_LABEL:-$DEFAULT_ADDITIONAL_LABEL}"
RUN_DIR="${RUN_DIR:-}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="$REPO_ROOT/playground_eval/runs/$RUN_STAMP"
  while [[ -e "$RUN_DIR" ]]; do
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$RANDOM"
    RUN_DIR="$REPO_ROOT/playground_eval/runs/$RUN_STAMP"
  done
fi

VIDEO_ROOT="${VIDEO_ROOT:-$RUN_DIR/videos}"
RESULT_PATH="${RESULT_PATH:-$RUN_DIR/results_local_eval.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/logs/local_headless_eval}"

# Optional one-time helper: if the target model path is missing, create it from a
# separately downloaded checkpoint directory.
SRC_CHECKPOINT="${SRC_CHECKPOINT:-}"
if [[ ! -e "$MODEL_PATH" && -n "$SRC_CHECKPOINT" ]]; then
  mkdir -p "$(dirname "$MODEL_PATH")"
  ln -sfn "$SRC_CHECKPOINT" "$MODEL_PATH"
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Missing checkpoint directory: $MODEL_PATH" >&2
  echo "Set MODEL_PATH or SRC_CHECKPOINT before running." >&2
  exit 1
fi

mkdir -p "$VIDEO_ROOT" "$(dirname "$RESULT_PATH")" "$OUTPUT_DIR"
mkdir -p "$VENDOR_PY"

echo "Run directory: $RUN_DIR"
echo "Video root: $VIDEO_ROOT"
echo "Result file: $RESULT_PATH"
echo "Additional label: $ADDITIONAL_LABEL"
echo "Vision input mode: $VISION_INPUT_MODE"
echo "Save video: $SAVE_VIDEO"
echo "Save frames: $SAVE_FRAMES"
echo "Project trajs: $PROJECT_TRAJS"
echo "Max eval steps: $MAX_EVAL_STEPS"
echo "Chunk exec len: $CHUNK_EXEC_LEN"
echo "Image update interval: $IMAGE_UPDATE_INTERVAL"
echo "Image delay steps: $IMAGE_DELAY_STEPS"
echo "Proprio ablation mode: $PROPRIO_ABLATION_MODE"
echo "Proprio delay steps: $PROPRIO_DELAY_STEPS"
echo "RL mode: $RL_MODE"
echo "RL action trace: $RL_ACTION_TRACE"
echo "RL action trace steps: $RL_ACTION_TRACE_STEPS"
echo "RL actor checkpoint: $RL_ACTOR_CHECKPOINT"
echo "RL collect replay path: $RL_COLLECT_REPLAY_PATH"
echo "RL collect source: $RL_COLLECT_SOURCE"
echo "Episode start idx: $EPISODE_START_IDX"
echo "Trial start idx: $TRIAL_START_IDX"
echo "Randomize total episodes: $RANDOMIZE_TOTAL_EPISODES"
echo "Randomize total trials: $RANDOMIZE_TOTAL_TRIALS"
if [[ "${TERM:-dumb}" == "dumb" ]]; then
  export TERM=xterm
fi

for pkg in smplx chumpy accelerate loguru s2wrapper; do
  if [[ -e "$VILA_PURELIB/$pkg" ]]; then
    ln -sfn "$VILA_PURELIB/$pkg" "$VENDOR_PY/$pkg"
  fi
done

for meta_pattern in accelerate-*.dist-info; do
  if compgen -G "$VILA_PURELIB/$meta_pattern" > /dev/null; then
    for meta_path in "$VILA_PURELIB"/$meta_pattern; do
      ln -sfn "$meta_path" "$VENDOR_PY/$(basename "$meta_path")"
    done
  fi
done

export LD_LIBRARY_PATH="$CUDA_LIB_PATH"
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SYSTEM_LIBSTDCPP_PATH"
export PYTHONPATH="$VENDOR_PY:$REPO_ROOT:$REPO_ROOT/VILA:$REPO_ROOT/manopth${PYTHONPATH:+:$PYTHONPATH}"

"$ISAACLAB_PATH/isaaclab.sh" -p human_plan/ego_bench_eval/ik_agent_30hz.py \
  --model_name_or_path "$MODEL_PATH" \
  --version qwen2 \
  --vision_tower google/siglip-so400m-patch14-384 \
  --data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train \
  --mm_vision_select_feature cls_patch \
  --mm_projector mlp_downsample \
  --tune_vision_tower True \
  --tune_mm_projector True \
  --tune_language_model True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio resize \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --report_to none \
  --run_name local_headless_eval \
  --future_index 1 \
  --predict_future_step 30 \
  --max_action 1 \
  --min_action 0 \
  --add_his_obs_step 5 \
  --add_his_imgs True \
  --add_his_img_skip 6 \
  --num_action_bins 256 \
  --action_tokenizer uniform \
  --invalid_token_weight 0.1 \
  --mask_input True \
  --add_current_language_description False \
  --traj_decoder_type transformer_split_action_v2 \
  --raw_action_label True \
  --traj_action_output_dim 48 \
  --input_placeholder_diff_index True \
  --ee_loss_coeff 20.0 \
  --hand_loss_coeff 5.0 \
  --hand_loss_dim 6 \
  --ee_2d_loss_coeff 0.0 \
  --ee_rot_loss_coeff 5.0 \
  --hand_kp_loss_coeff 0.0 \
  --next_token_loss_coeff 0.0 \
  --traj_action_output_ee_2d_dim 0 \
  --traj_action_output_ee_dim 6 \
  --traj_action_output_hand_dim 30 \
  --traj_action_output_ee_rot_dim 12 \
  --ee_rot_representation rot6d \
  --correct_transformation True \
  --include_2d_label True \
  --include_rot_label True \
  --use_short_language_label True \
  --no_norm_ee_label True \
  --tf32 True \
  --merge_hand True \
  --use_mano True \
  --sep_proprio True \
  --sep_query_token True \
  --loss_use_l1 True \
  --task "$TASK" \
  --room_idx "$ROOM_IDX" \
  --table_idx "$TABLE_IDX" \
  --smooth_weight "$SMOOTH_WEIGHT" \
  --num_episodes "$NUM_EPISODES" \
  --num_trials "$NUM_TRIALS" \
  --episode_start_idx "$EPISODE_START_IDX" \
  --trial_start_idx "$TRIAL_START_IDX" \
  --randomize_total_episodes "$RANDOMIZE_TOTAL_EPISODES" \
  --randomize_total_trials "$RANDOMIZE_TOTAL_TRIALS" \
  --result_saving_path "$RESULT_PATH" \
  --save_video "$SAVE_VIDEO" \
  --save_frames "$SAVE_FRAMES" \
  --project_trajs "$PROJECT_TRAJS" \
  --vision_input_mode "$VISION_INPUT_MODE" \
  --max_eval_steps "$MAX_EVAL_STEPS" \
  --chunk_exec_len "$CHUNK_EXEC_LEN" \
  --image_update_interval "$IMAGE_UPDATE_INTERVAL" \
  --image_delay_steps "$IMAGE_DELAY_STEPS" \
  --proprio_ablation_mode "$PROPRIO_ABLATION_MODE" \
  --proprio_delay_steps "$PROPRIO_DELAY_STEPS" \
  --rl_mode "$RL_MODE" \
  --rl_action_trace_steps "$RL_ACTION_TRACE_STEPS" \
  --rl_identity_tolerance "$RL_IDENTITY_TOLERANCE" \
  --rl_collect_source "$RL_COLLECT_SOURCE" \
  $([[ "$RL_ACTION_TRACE" == "1" ]] && printf '%s' "--rl_action_trace") \
  $([[ -n "$RL_ACTOR_CHECKPOINT" ]] && printf '%s %q' "--rl_actor_checkpoint" "$RL_ACTOR_CHECKPOINT") \
  $([[ -n "$RL_COLLECT_REPLAY_PATH" ]] && printf '%s %q' "--rl_collect_replay_path" "$RL_COLLECT_REPLAY_PATH") \
  $([[ "$RL_COLLECT_SAVE_RAW" == "1" ]] && printf '%s' "--rl_collect_save_raw") \
  $([[ "$EVAL_ABLATION_DEBUG" == "1" ]] && printf '%s' "--eval_ablation_debug") \
  --hand_smooth_weight "$HAND_SMOOTH_WEIGHT" \
  --video_saving_path "$VIDEO_ROOT" \
  --additional_label "$ADDITIONAL_LABEL"
