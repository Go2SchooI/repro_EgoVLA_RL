# EgoVLA Commands

## Remote / Checkpoint

```bash
scp -3 -r \
  183.147.142.40:/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v5-from14000/checkpoint-3000 \
  100.124.11.120:/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/

ssh.exe -p 31708 -N -R 17890:127.0.0.1:7890 root@183.147.142.40
```

## Common Setup

```bash
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000
export TASK=Humanoid-Open-Laptop-v0
```

Useful task names:

```text
Humanoid-Push-Box-v0
Humanoid-Open-Drawer-v0
Humanoid-Close-Drawer-v0
Humanoid-Pour-Balls-v0
Humanoid-Flip-Mug-v0
Humanoid-Open-Laptop-v0
Humanoid-Stack-Can-v0
Humanoid-Unload-Cans-v0
Humanoid-Insert-Cans-v0
Humanoid-Stack-Can-Into-Drawer-v0
Humanoid-Sort-Cans-v0
Humanoid-Insert-And-Unload-Cans-v0
```

## Local Eval / Ablation

```bash
# Default Open-Laptop eval with checkpoint-3000.
./run_local_eval.sh

# Vision input ablations.
TASK=Humanoid-Open-Laptop-v0 VISION_INPUT_MODE=noise ./run_local_eval.sh
TASK=Humanoid-Open-Laptop-v0 VISION_INPUT_MODE=initial ./run_local_eval.sh
TASK=Humanoid-Pour-Balls-v0 VISION_INPUT_MODE=real ./run_local_eval.sh

# Image update every K steps.
IMAGE_UPDATE_INTERVAL=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# Fixed initial image.
IMAGE_UPDATE_INTERVAL=inf ./run_local_eval.sh

# Image delay.
IMAGE_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
IMAGE_DELAY_STEPS=10 ./run_local_eval.sh

# Proprio ablations.
TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
PROPRIO_ABLATION_MODE=freeze TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=10 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# Combined image + proprio delay.
IMAGE_DELAY_STEPS=5 PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
```

Old pretrained model path, only use intentionally:

```bash
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/ckpt-human-video-pretrained
```

## Offline RL: Open-Laptop

### 1. Collect Base Replay

```bash
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000

TASK=Humanoid-Open-Laptop-v0
TAG=open_laptop_v4_checkpoint3000_$(date +%Y%m%d_%H%M%S)
REPLAY_DIR=playground_eval/replays/${TAG}
mkdir -p "${REPLAY_DIR}"

# Offline replay can include all six scenes.
ROOM_TABLES=(
  "1 1"
  "1 2"
  "2 1"
  "2 2"
  "3 1"
  "3 2"
)

for RT in "${ROOM_TABLES[@]}"; do
  read -r ROOM_IDX TABLE_IDX <<< "${RT}"
  python -m rl_posttrain.collect_base \
    --output "${REPLAY_DIR}/room${ROOM_IDX}_table${TABLE_IDX}_base" \
    --task "${TASK}" \
    --room_idx "${ROOM_IDX}" \
    --table_idx "${TABLE_IDX}" \
    --source base \
    --num_episodes 10 \
    --num_trials 2
done
```

### 2A. Train Pure BC Actor

```bash
PURE_BC_CKPT=playground_eval/rl_checkpoints/${TAG}_pure_bc

python -m rl_posttrain.td3bc_ref \
  --replay "${REPLAY_DIR}" \
  --output "${PURE_BC_CKPT}" \
  --steps 75000 \
  --batch_size 256 \
  --td3bc_alpha 0.0 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda
```

W&B version:

```bash
python -m rl_posttrain.td3bc_ref \
  --replay "${REPLAY_DIR}" \
  --output "${PURE_BC_CKPT}" \
  --steps 75000 \
  --batch_size 256 \
  --td3bc_alpha 0.0 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda \
  --wandb_project egovla-td3bc \
  --wandb_run_name "${TAG}_pure_bc" \
  --wandb_tags open_laptop,pure_bc
```

### 2B. Train Weak-Q TD3+BC Actor

```bash
ALPHA_TAG=alpha0015
TD3BC_CKPT=playground_eval/rl_checkpoints/${TAG}_td3bc_${ALPHA_TAG}

python -m rl_posttrain.td3bc_ref \
  --replay "${REPLAY_DIR}" \
  --output "${TD3BC_CKPT}" \
  --steps 150000 \
  --batch_size 256 \
  --td3bc_alpha 0.015 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda
```

Known replay retrain example:

```bash
python -m rl_posttrain.td3bc_ref \
  --replay /home/jizexian/dexhand/EgoVLA_Release/playground_eval/replays/open_laptop_v4_checkpoint3000_20260509_174957 \
  --output playground_eval/rl_checkpoints/open_laptop_v4_retrain_checkpoint3000_alpha0005 \
  --steps 150000 \
  --batch_size 256 \
  --td3bc_alpha 0.005 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda \
  --wandb_project egovla-td3bc \
  --wandb_run_name open_laptop_v4_retrain_checkpoint3000_alpha0005
```

### 3. Paired Eval Offline Actors

```bash
OUTPUT_ROOT=playground_eval/paired_eval/${TAG}_multi_scene_multi_actor

python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${PURE_BC_CKPT}" "${TD3BC_CKPT}" \
  --task "${TASK}" \
  --model_path "${MODEL_PATH}" \
  --scene 1 1 \
  --scene 1 2 \
  --scene 2 1 \
  --scene 2 2 \
  --scene 3 1 \
  --scene 3 2 \
  --num_episodes 5 \
  --num_trials 3 \
  --skip_identity \
  --no_save_video \
  --output_root "${OUTPUT_ROOT}"
```

Existing h_proj64 alpha0001 vs alpha0003 all-six-scene eval:

```bash
TASK=Humanoid-Open-Laptop-v0
MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000

CKPT_ALPHA0001=/home/jizexian/dexhand/EgoVLA_Release/playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj64_alpha0001
CKPT_ALPHA0003=/home/jizexian/dexhand/EgoVLA_Release/playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj64_alpha0003

OUTPUT_ROOT=playground_eval/paired_eval/open_laptop_hsummary_h_proj64_alpha0001_alpha0003_all6_$(date +%Y%m%d_%H%M%S)

python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${CKPT_ALPHA0001}" "${CKPT_ALPHA0003}" \
  --task "${TASK}" \
  --model_path "${MODEL_PATH}" \
  --scene 1 1 \
  --scene 1 2 \
  --scene 2 1 \
  --scene 2 2 \
  --scene 3 1 \
  --scene 3 2 \
  --num_episodes 8 \
  --num_trials 2 \
  --skip_identity \
  --no_save_video \
  --output_root "${OUTPUT_ROOT}"
```

## Online RL: Open-Laptop TD3+BC

Important:

- Do not resume the old contaminated run `playground_eval/online_td3bc/h_proj128_alpha0001_online_v1`.
- Use `checkpoint-3000` as `online.model_path`.
- Online training scenes are room1/room2 only; room3 is eval-only.
- Periodic eval uses `eval.cache_static: true`: first eval runs baseline/offline_init/current, later evals rerun only current actor and reuse static baseline/offline_init results.

### 1. Collector Smoke

```bash
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root playground_eval/online_td3bc/smoke_rollout_$(date +%Y%m%d_%H%M%S) \
  --total_online_episodes 1 \
  --rollout_only \
  --no_eval \
  --no_wandb
```

### 2. Critic-Only Smoke

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root playground_eval/online_td3bc/smoke_critic_only_$(date +%Y%m%d_%H%M%S) \
  --total_online_episodes 2 \
  --min_online_transitions_for_training 1 \
  --no_eval \
  --no_wandb
```

### 3. Joint Update Smoke

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root playground_eval/online_td3bc/smoke_joint_$(date +%Y%m%d_%H%M%S) \
  --total_online_episodes 2 \
  --critic_only_episodes 0 \
  --min_online_transitions_for_training 1 \
  --no_eval \
  --no_wandb
```

### 4. Full Online Train: h_proj128 alpha0001, bc_weight=0

This follows `rl_posttrain/configs/online_td3bc_v1.yaml`.

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --auto_name \
  --output_root_base playground_eval/online_td3bc \
  --wandb_mode online
```

Equivalent fixed output root:

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root playground_eval/online_td3bc/h_proj128_alpha0001_online_v1_ckpt3000_td3alpha0001_bc0 \
  --auto_name \
  --wandb_mode online
```

### 5. Full Online Train: h_proj128 alpha0 init, td3alpha0003, bc_weight=0.1

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --init_checkpoint playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj128_alpha0 \
  --td3bc_alpha 0.003 \
  --bc_weight 0.1 \
  --auto_name \
  --output_root_base playground_eval/online_td3bc \
  --wandb_mode online
```

This auto-generates a run name like:

```text
h_proj128_alpha0_online_v1_ckpt3000_td3alpha0003_bc01
```

### 6. Resume Online Train

Use the same config and command-line overrides as the original run. Replace `RUN_DIR`.

```bash
RUN_DIR=playground_eval/online_td3bc/h_proj128_alpha0001_online_v1_ckpt3000_td3alpha0001_bc0

python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root "${RUN_DIR}" \
  --resume "${RUN_DIR}/checkpoints/latest_online.pt" \
  --allow_reuse_online_replay \
  --strict_resume_manifest_match \
  --auto_name \
  --wandb_mode online
```

Resume the h_proj128 alpha0 / td3alpha0003 / bc01 run:

```bash
RUN_DIR=playground_eval/online_td3bc/h_proj128_alpha0_online_v1_ckpt3000_td3alpha0003_bc01

python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --init_checkpoint playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj128_alpha0 \
  --td3bc_alpha 0.003 \
  --bc_weight 0.1 \
  --output_root "${RUN_DIR}" \
  --resume "${RUN_DIR}/checkpoints/latest_online.pt" \
  --allow_reuse_online_replay \
  --strict_resume_manifest_match \
  --auto_name \
  --wandb_mode online
```

### 7. Eval-Only From Online Checkpoint

```bash
RUN_DIR=playground_eval/online_td3bc/h_proj128_alpha0001_online_v1_ckpt3000_td3alpha0001_bc0

python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --output_root "${RUN_DIR}" \
  --resume "${RUN_DIR}/checkpoints/latest_online.pt" \
  --allow_reuse_online_replay \
  --strict_resume_manifest_match \
  --eval_only \
  --auto_name \
  --wandb_mode online
```

Direct paired eval for a latest actor:

```bash
RUN_DIR=playground_eval/online_td3bc/h_proj128_alpha0001_online_v1_ckpt3000_td3alpha0001_bc0

python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${RUN_DIR}/checkpoints/latest_actor.pt" \
  --task Humanoid-Open-Laptop-v0 \
  --model_path /home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000 \
  --scene 1 1 \
  --scene 1 2 \
  --scene 2 1 \
  --scene 2 2 \
  --scene 3 1 \
  --scene 3 2 \
  --num_episodes 6 \
  --num_trials 1 \
  --skip_identity \
  --no_save_video \
  --output_root "${RUN_DIR}/manual_paired_eval_$(date +%Y%m%d_%H%M%S)"
```
