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