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

# 按需增删 room/table 组合。每个组合会生成一个缓存子目录，
# 里面每个 episode/trial 一个 npz；训练时 --replay 直接传总文件夹。
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
# --output 给目录时会启用缓存：每个 episode/trial 单独保存 npz，并写 episodes.json
for RT in "${ROOM_TABLES[@]}"; do
  read -r ROOM_IDX TABLE_IDX <<< "${RT}"
  python -m rl_posttrain.collect_base \
    --output ${REPLAY_DIR}/room${ROOM_IDX}_table${TABLE_IDX}_base \
    --task ${TASK} \
    --room_idx ${ROOM_IDX} \
    --table_idx ${TABLE_IDX} \
    --source base \
    --num_episodes 10 \
    --num_trials 2
done

# 2A. 训练 pure BC actor：td3bc_alpha=0.0
# 训练会输出一个目录，里面包含 actor.pt 和 config.yaml。
# 如需推送 wandb，在命令末尾追加：
#   --wandb_project egovla-td3bc --wandb_run_name ${TAG}_pure_bc --wandb_tags open_laptop,pure_bc
PURE_BC_CKPT=playground_eval/rl_checkpoints/${TAG}_pure_bc
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
# 训练会输出一个目录，里面包含 actor.pt 和 config.yaml。
# 如需推送 wandb，在命令末尾追加：
#   --wandb_project egovla-td3bc --wandb_run_name ${TAG}_${ALPHA_TAG} --wandb_tags open_laptop,td3bc,${ALPHA_TAG}
ALPHA_TAG=alpha0015
TD3BC_CKPT=playground_eval/rl_checkpoints/${TAG}_td3bc_${ALPHA_TAG}
python -m rl_posttrain.td3bc_ref \
  --replay ${REPLAY_DIR} \
  --output ${TD3BC_CKPT} \
  --steps 150000 \
  --batch_size 256 \
  --td3bc_alpha 0.015 \
  --td3bc_bc_weight 1.0 \
  --policy_delay 2 \
  --log_every 100 \
  --device cuda 

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

# 3. Paired Eval: 一次对比多个 actor checkpoint，跳过 identity
# paired_eval 默认保存视频；如不想保存视频，加 --no_save_video
# 可以重复传 --scene，一次生成总目录和总 paired_summary.json。
python -m rl_posttrain.paired_eval \
  --actor_checkpoint ${PURE_BC_CKPT} ${TD3BC_CKPT} \
  --task ${TASK} \
  --model_path ${MODEL_PATH} \
  --scene 1 1 \
  --scene 1 2 \
  --scene 2 1 \
  --scene 2 2 \
  --num_episodes 5 \
  --num_trials 3 \
  --skip_identity \
  --output_root playground_eval/paired_eval/${TAG}_multi_scene_multi_actor



python -m rl_posttrain.online_td3bc   --config rl_posttrain/configs/online_td3bc_v1.yaml   --output_root playground_eval/online_td3bc/h_proj128_alpha0001_online_v1_ckpt3000_bc0   --wandb_mode online


python -m rl_posttrain.online_td3bc   --config rl_posttrain/configs/online_td3bc_h_proj128_alpha0_init.yaml   --wandb_mode online


python -m rl_posttrain.online_td3bc   --config rl_posttrain/configs/online_td3bc_v1.yaml   --output_root playground_eval/online_td3bc/h_proj128_alpha0001_online_v1   --resume playground_eval/online_td3bc/h_proj128_alpha0001_online_v1/checkpoints/latest_online.pt   --allow_reuse_online_replay   --strict_resume_manifest_match   --wandb_mode online
