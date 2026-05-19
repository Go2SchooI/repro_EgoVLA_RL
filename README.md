# EgoVLA RL Post-Training

Original EgoVLA documentation: [README_EGOVLA_ORIGINAL.md](README_EGOVLA_ORIGINAL.md)

This repository extends the original EgoVLA simulation pipeline with an RL post-training stack for improving humanoid manipulation policies after VLA pretraining. The current focus is Open-Laptop in IsaacLab, with both offline TD3+BC post-training and online RL fine-tuning.

The RL code is intentionally layered on top of the existing EgoVLA inference and evaluation path: EgoVLA still produces the reference action, the original temporal smoothing / IK / retargeting / environment stepping path is reused, and RL actors are inserted as a post-training control module rather than replacing the VLA model.

## Highlights

- Offline TD3+BC post-training from EgoVLA-collected replay.
- Online TD3+BC fine-tuning with mixed base replay and fresh online replay.
- Frozen EgoVLA backbone during online RL.
- Seen/unseen scene split for Open-Laptop:
  - train: `room1_table1`, `room1_table2`, `room2_table1`, `room2_table2`
  - unseen eval: `room3_table1`, `room3_table2`
- Paired evaluation against baseline EgoVLA, identity insertion path, offline init actor, and current online actor.
- W&B logging for training, Q values, action deltas, replay mix ratios, and seen/unseen evaluation metrics.
- Checkpoint / resume support for actor, critic, targets, optimizers, normalizers, replay manifest, and frozen EgoVLA model path.

## RL Control Path

```text
Frozen EgoVLA
  -> temporal smoothing
  -> reference action a_ref
  -> RL actor
  -> executed action a_exec
  -> denormalize / unpack / IK / retarget
  -> IsaacLab env.step
```

The RL actor observes the same post-processed state used by the post-training pipeline, including proprioception, history summary features, and normalized reference action. The critic is trained on the true executed action `a_exec_norm`, while the TD3+BC actor regularizer uses the EgoVLA reference `a_ref_norm` as the behavior-cloning target.

## Repository Entry Points

| File | Purpose |
| --- | --- |
| `rl_posttrain/collect_base.py` | Collect base replay from the frozen EgoVLA policy. |
| `rl_posttrain/td3bc_ref.py` | Offline TD3+BC / BC-style actor post-training. |
| `rl_posttrain/paired_eval.py` | Paired baseline / identity / actor evaluation. |
| `rl_posttrain/online_td3bc.py` | Online TD3+BC fine-tuning loop. |
| `rl_posttrain/replay_buffer.py` | Offline and online replay schemas and loading utilities. |
| `rl_posttrain/configs/online_td3bc_v1.yaml` | Default online RL configuration. |
| `cmd.md` | Working command cookbook for experiments. |

## Setup

Activate the IsaacLab environment before running collection, training, or evaluation:

```bash
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH=/home/jizexian/dexhand/EgoVLA_Release/checkpoints/ego_vla_checkpoint/checkpoint-3000
export TASK=Humanoid-Open-Laptop-v0
```

The online configuration also stores the frozen EgoVLA checkpoint path:

```yaml
online:
  model_path: checkpoints/ego_vla_checkpoint/checkpoint-3000
```

This path is part of the resume manifest. Runs collected with a different frozen EgoVLA checkpoint should not be resumed into a new run.

## Offline RL Post-Training

Offline post-training first collects replay with the frozen EgoVLA policy, then trains a compact actor with TD3+BC losses.

### 1. Collect Base Replay

```bash
TAG=open_laptop_v4_checkpoint3000_$(date +%Y%m%d_%H%M%S)
REPLAY_DIR=playground_eval/replays/${TAG}
mkdir -p "${REPLAY_DIR}"

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

### 2. Train Offline TD3+BC Actor

```bash
TD3BC_CKPT=playground_eval/rl_checkpoints/${TAG}_td3bc_alpha0015

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

For pure behavior cloning, use `--td3bc_alpha 0.0`.

### 3. Paired Offline Evaluation

```bash
python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${TD3BC_CKPT}" \
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
  --output_root "playground_eval/paired_eval/${TAG}_td3bc_all6"
```

The paired evaluator reports baseline success, actor success, regression cases, recovery cases, and net improvement.

## Online RL Fine-Tuning

Online training alternates data collection and gradient updates:

```text
for each online episode:
  1. collect one episode with the current actor on a balanced train scene
  2. append the episode to online replay
  3. sample mixed batches from base replay and online replay
  4. update critics, and update actor after the critic-only warmup phase
  5. periodically run seen/unseen paired eval
```

Default online settings:

| Setting | Value |
| --- | --- |
| task | `Humanoid-Open-Laptop-v0` |
| init checkpoint | `h_proj128_alpha0001` |
| total online episodes | `300` |
| critic-only episodes | `30` |
| UTD ratio | `4` |
| max updates per episode | `800` |
| reward | sparse final success |
| exploration noise | normalized Gaussian noise after episode 10 |
| eval interval | every 50 online episodes |

### Full Online Training

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --auto_name \
  --output_root_base playground_eval/online_td3bc \
  --wandb_mode online
```

Example for an offline `h_proj128_alpha0` initialization with TD3 alpha `0.003` and BC weight `0.1`:

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

This creates a run name like:

```text
h_proj128_alpha0_online_v1_ckpt3000_td3alpha0003_bc01
```

### Resume Online Training

Use the same config and overrides as the original run:

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

## Evaluation Metrics

The main W&B metrics are grouped by training progress, replay composition, losses, action deltas, and evaluation.

| Metric group | Examples | How to read it |
| --- | --- | --- |
| progress | `online/episode`, `online/env_steps`, `online/online_buffer_size` | Confirms online collection and training are advancing. |
| updates | `online/critic_updates`, `online/actor_updates`, `online/actual_utd_ratio` | Actor updates should stay at zero during critic-only warmup. |
| replay mix | `online/base_sample_ratio`, `online/online_sample_ratio` | Should match the configured base/online replay ratios after enough online data exists. |
| losses | `loss/critic_loss`, `loss/actor_loss`, `loss/bc_loss` | Useful for debugging, but eval success is the primary metric. |
| Q values | `q/q_ref`, `q/q_exec`, `q/q_adv` | `q_adv = q_exec - q_ref`; early Q estimates can be noisy. |
| action deltas | `action/mean_abs_actor_minus_ref_norm`, `action/num_clipped_dims` | Large deltas or many clipped dims may indicate unstable actor behavior. |
| eval seen | `eval_seen/success_rate`, `eval_seen/regress`, `eval_seen/recover`, `eval_seen/net` | Performance on online training scenes. |
| eval unseen | `eval_unseen/success_rate`, `eval_unseen/net` | Generalization to held-out room3 scenes. |
| eval all | `eval_all/success_rate`, `eval_all/net` | Aggregate across all six scenes. |
| eval cache | `eval/static_cache_used` | `0` means full eval; `1` means static baseline/offline-init results were reused. |

The most important online result is the paired eval tradeoff:

```text
net = recover - regress
```

where `recover` means baseline failed but the actor succeeded, and `regress` means baseline succeeded but the actor failed.

## Output Layout

Typical experiment outputs:

```text
playground_eval/
  replays/
    <offline_replay_name>/
  rl_checkpoints/
    <offline_actor_name>/
      actor.pt
      config.yaml
  paired_eval/
    <paired_eval_name>/
      paired_summary.json
  online_td3bc/
    <online_run_name>/
      online_config.yaml
      online_replay/
      checkpoints/
        latest_online.pt
        latest_actor.pt
      eval/
        episode_0050/
        episode_0100/
```

## Results

### Training Curves

The following figure slots are intentionally left as placeholders. Fill them after selecting the final W&B screenshots or exported plots.

| Figure | Path / Link | Notes |
| --- | --- | --- |
| Offline TD3+BC training curve | `TODO: media/rl_posttrain/offline_training_curve.png` | Loss / Q / BC curve. |
| Online success rate curve | `TODO: media/rl_posttrain/online_eval_success_rate.png` | Seen vs unseen success rate. |
| Online net improvement curve | `TODO: media/rl_posttrain/online_eval_net.png` | Recover minus regress. |
| Action delta curve | `TODO: media/rl_posttrain/action_delta.png` | Actor deviation from EgoVLA reference. |

### Evaluation Videos

The following MP4 links are placeholders. Replace them with final selected rollouts.

| Demo | Path / Link | Description |
| --- | --- | --- |
| Baseline EgoVLA success | `TODO: media/rl_posttrain/baseline_success.mp4` | Frozen EgoVLA baseline rollout. |
| Offline actor rollout | `TODO: media/rl_posttrain/offline_actor.mp4` | Offline TD3+BC actor inserted into the action path. |
| Online actor seen-scene rollout | `TODO: media/rl_posttrain/online_seen_success.mp4` | Online actor on room1/room2. |
| Online actor unseen-scene rollout | `TODO: media/rl_posttrain/online_unseen_success.mp4` | Online actor on room3. |
| Recovery example | `TODO: media/rl_posttrain/recovery_case.mp4` | Baseline fails, RL actor succeeds. |
| Regression example | `TODO: media/rl_posttrain/regression_case.mp4` | Baseline succeeds, RL actor fails. |

## Notes

- Room3 is reserved for unseen evaluation in online RL. Online training rollout should not collect room3 data.
- Eval transitions are not written into online replay.
- The online actor action dimension is loaded from the checkpoint / action spec; it is not hard-coded to the raw EgoVLA action size.
- `eval.cache_static: true` avoids repeatedly rerunning static baseline and offline-init evaluations during periodic online eval.
- The old contaminated output root `playground_eval/online_td3bc/h_proj128_alpha0001_online_v1` should not be resumed.

