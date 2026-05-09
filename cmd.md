scp -3 -r 183.147.142.40:/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v5-from14000/checkpoint-3000 100.124.11.120:/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/

ssh.exe -p 31708 -N -R 17890:127.0.0.1:7890 root@183.147.142.40


conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000

./run_local_eval.sh
TASK=Humanoid-Open-Laptop-v0 VISION_INPUT_MODE=noise ./run_local_eval.sh
TASK=Humanoid-Open-Laptop-v0 VISION_INPUT_MODE=initial ./run_local_eval.sh
TASK=Humanoid-Pour-Balls-v0 VISION_INPUT_MODE=real ./run_local_eval.sh

# image every K steps
IMAGE_UPDATE_INTERVAL=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# fixed initial image
IMAGE_UPDATE_INTERVAL=inf ./run_local_eval.sh

# image delay
IMAGE_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh
IMAGE_DELAY_STEPS=10 ./run_local_eval.sh

# 默认，不启用 proprio ablation
TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# freeze proprio: image 正常更新，proprio 固定为 episode 初始 proprio
PROPRIO_ABLATION_MODE=freeze TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# delay proprio: image 当前，proprio 使用 5 step 前
PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# delay 10 steps
PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=10 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh

# 可和 image ablation 组合
IMAGE_DELAY_STEPS=5 PROPRIO_ABLATION_MODE=delay PROPRIO_DELAY_STEPS=5 TASK=Humanoid-Pour-Balls-v0 ./run_local_eval.sh


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


export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/ckpt-human-video-pretrained


# ================================
# RL posttrain: Open-Laptop
# ================================

# 进入 IsaacLab 环境，并固定使用 checkpoint-3000
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000

# 通用实验变量：改 TAG 就能避免覆盖旧结果
TASK=Humanoid-Open-Laptop-v0
TAG=open_laptop_v4_checkpoint3000_$(date +%Y%m%d_%H%M%S)
REPLAY_DIR=playground_eval/replays/${TAG}
mkdir -p ${REPLAY_DIR}

# 按需增删 room/table 组合。每个组合会生成一个独立 npz，
# 训练时 --replay 直接传文件夹，代码会混合文件夹内所有 .npz。
ROOM_TABLES=(
  "1 1"
  "1 2"
  "2 1"
  "2 2"
  "3 1"
  "3 2"
)

# 1. 采 Base Replay
# collect_base 默认不保存视频；如需保存视频再额外加 --save_video
for RT in "${ROOM_TABLES[@]}"; do
  read -r ROOM_IDX TABLE_IDX <<< "${RT}"
  python -m rl_posttrain.collect_base \
    --output ${REPLAY_DIR}/room${ROOM_IDX}_table${TABLE_IDX}_base.npz \
    --task ${TASK} \
    --room_idx ${ROOM_IDX} \
    --table_idx ${TABLE_IDX} \
    --source base \
    --num_episodes 10 \
    --num_trials 2
done

# 2A. 训练 pure BC actor：td3bc_alpha=0.0
PURE_BC_CKPT=playground_eval/rl_checkpoints/${TAG}_pure_bc.pt
python -m rl_posttrain.td3bc_ref \
  --replay ${REPLAY_DIR} \
  --output ${PURE_BC_CKPT} \
  --steps 75000 \
  --batch_size 256 \
  --td3bc_alpha 0.0 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda

# 2B. 训练 weak-Q TD3+BC actor：td3bc_alpha=0.03
# pure BC paired eval 不明显破坏 baseline 后，再跑这个版本
ALPHA_TAG=alpha003
TD3BC_CKPT=playground_eval/rl_checkpoints/${TAG}_td3bc_${ALPHA_TAG}.pt
python -m rl_posttrain.td3bc_ref \
  --replay ${REPLAY_DIR} \
  --output ${TD3BC_CKPT} \
  --steps 75000 \
  --batch_size 256 \
  --td3bc_alpha 0.03 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda

# 3A. Paired Eval: pure BC，对比 baseline/off 和 actor，跳过 identity
# paired_eval 默认保存视频；如不想保存视频，加 --no_save_video
python -m rl_posttrain.paired_eval \
  --actor_checkpoint ${PURE_BC_CKPT} \
  --task ${TASK} \
  --num_episodes 5 \
  --num_trials 3 \
  --skip_identity \
  --output_root playground_eval/paired_eval/${TAG}_pure_bc

# 3B. Paired Eval: weak-Q TD3+BC alpha=0.03，对比 baseline/off 和 actor，跳过 identity
python -m rl_posttrain.paired_eval \
  --actor_checkpoint ${TD3BC_CKPT} \
  --task ${TASK} \
  --num_episodes 5 \
  --num_trials 3 \
  --skip_identity \
  --output_root playground_eval/paired_eval/${TAG}_td3bc_${ALPHA_TAG}
