# EgoVLA-RL: RLT-Inspired RL Post-Training

An independently developed RL post-training extension to EgoVLA for **Open-Laptop in IsaacLab simulation**. Both offline and online pipelines have run end to end. A completed single-seed [online interaction-budget experiment](docs/experiments/online-budget-20260915.md) reports matched evaluation through 300 online episodes, including separate RL-ID and RL-OOD results.

## Scope and Attribution

- **Upstream:** the EgoVLA baseline, original inference pipeline, and Open-Laptop task/benchmark come from the original open-source projects. Their documentation and credits are preserved in [README_EGOVLA_ORIGINAL.md](README_EGOVLA_ORIGINAL.md).
- **This extension:** the repository maintainer independently developed the post-training code, including replay collection/integration, offline and online actor-critic training, observation/normalization handling, and paired evaluation.
- **Method:** RLT-inspired post-training with deterministic TD3+BC-style actors and asymmetric critics. The EgoVLA backbone stays frozen during RL. This is not end-to-end RL optimization of EgoVLA weights or a claim of an exact RLT reproduction.
- **Validation scope:** simulation. The reported single-seed results do not establish consistent improvements across seeds; real-robot validation is not reported here.

## Control Path

```text
Frozen EgoVLA -> temporal smoothing -> reference action a_ref
                                             |
        EgoVLA latent + proprio + chunk summary + a_ref_norm
                                             |
                                     deterministic actor
                                             |
                               executed action a_exec_norm
                                             |
                         denormalize / unpack / IK / retarget
                                             |
                                    IsaacLab env.step
```

The actor outputs the normalized execution command. The critic sees actor observations plus privileged simulator state and learns from executed actions. When enabled, the behavior-cloning term targets the frozen policy's reference action. Latent-feature ablations include full, zeroed, and projected `h_summary` inputs. See the [observation and action specification](rl_posttrain/actor_critic_observation_spec.md); dimensions depend on the action specification and checkpoint.

## Repository Map and Entry Points

| Path | Role |
| --- | --- |
| [rl_posttrain/](rl_posttrain/) | Post-training modules and existing unit tests. |
| [collect_base.py](rl_posttrain/collect_base.py) | Base/identity replay collection with cached episode/trial shards. |
| [td3bc_ref.py](rl_posttrain/td3bc_ref.py) | Offline TD3+BC or pure-BC training. |
| [online_td3bc.py](rl_posttrain/online_td3bc.py) | Online rollout/update loop, mixed replay, periodic evaluation, resume. |
| [paired_eval.py](rl_posttrain/paired_eval.py) | Baseline/identity/actor evaluation with matching episode/trial settings. |
| [replay_buffer.py](rl_posttrain/replay_buffer.py) | Replay schemas, loading, and normalization metadata. |
| [actors.py](rl_posttrain/actors.py), [critics.py](rl_posttrain/critics.py), [h_summary.py](rl_posttrain/h_summary.py) | Networks and latent-feature ablations. |
| [online_td3bc_v1.yaml](rl_posttrain/configs/online_td3bc_v1.yaml) | Online configuration, including local asset paths. |
| [run_local_eval.sh](run_local_eval.sh) | Simulator launcher used by collection and evaluation. |
| [ik_agent_30hz.py](human_plan/ego_bench_eval/ik_agent_30hz.py) | EgoVLA inference and simulation integration. |
| [human_plan/](human_plan/), [VILA/](VILA/), [training_scripts/](training_scripts/) | EgoVLA/VILA model, data, and original training code. |
| `checkpoints/`, `playground_eval/`, `wandb/` | Local models, experiment outputs, and logs; not a downloadable results bundle. |
| [cmd.md](cmd.md) | Historical experiment cookbook with machine-specific paths and older protocols. Use the workflow below for a consistent split. |

## Environment and Assets

On a fresh machine, start with the upstream [installation instructions](README_EGOVLA_ORIGINAL.md#installation) and obtain the benchmark assets, MANO models, VILA dependencies, and an appropriate EgoVLA checkpoint. [build_env.sh](build_env.sh) installs additional dependencies; it does not provision the simulator or download all assets. A Git clone alone is insufficient for simulation.

The setup below describes the existing `polyu_pc` installation. Adapt paths on another host:

```bash
source /home/jizexian/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jizexian/dexhand/EgoVLA_Release

export IsaacLab_PATH=/home/jizexian/IsaacLab
export MODEL_PATH="$PWD/checkpoints/ego_vla_checkpoint/checkpoint-3000"
export TASK=Humanoid-Open-Laptop-v0

test -x "$IsaacLab_PATH/isaaclab.sh"
test -d "$MODEL_PATH"
```

The launcher also uses `VILA_PURELIB`, `CUDA_LIB_PATH`, and `SYSTEM_LIBSTDCPP_PATH`, with local defaults in `run_local_eval.sh`. Check those paths when moving environments. The upstream installation document and the current local environment may differ; there is no fully pinned environment lockfile. `checkpoint-3000` is a local experiment checkpoint, not a promised filename in an upstream download.

Run the remaining commands from the repository root in the same activated Bash session. Explicitly set `TASK` and `MODEL_PATH`: the bare launcher's defaults are Push-Box and `ckpt-human-video-pretrained`.

Inspect options without launching simulation or training:

```bash
python -m rl_posttrain.collect_base --help
python -m rl_posttrain.td3bc_ref --help
python -m rl_posttrain.paired_eval --help
python -m rl_posttrain.online_td3bc --help
```

## Reproduction Workflow

### 1. Collect Training Replay

Use room1/room2 for post-training and reserve room3 for evaluation. Keeping room3 held out from **both offline initialization and online replay** is required to call it unseen to the entire post-training process. This does not establish that room3 was unseen during upstream EgoVLA pretraining/fine-tuning.

```bash
TAG=open_laptop_train4_checkpoint3000_$(date +%Y%m%d_%H%M%S)
REPLAY_DIR=playground_eval/replays/${TAG}
mkdir -p "${REPLAY_DIR}"

ROOM_TABLES=("1 1" "1 2" "2 1" "2 2")
for RT in "${ROOM_TABLES[@]}"; do
  read -r ROOM_IDX TABLE_IDX <<< "${RT}"
  python -m rl_posttrain.collect_base \
    --output "${REPLAY_DIR}/room${ROOM_IDX}_table${TABLE_IDX}_base" \
    --task "${TASK}" \
    --room_idx "${ROOM_IDX}" --table_idx "${TABLE_IDX}" \
    --source base --num_episodes 10 --num_trials 2
done
```

Collection inherits `MODEL_PATH` and disables video by default; add `--save_video` to record it. Directory outputs use cached shards. Review collection summaries and failed shards before training; requested episode counts are not proof that all episodes completed.

### 2. Train an Offline Actor

```bash
TD3BC_CKPT=playground_eval/rl_checkpoints/${TAG}_h_proj128_alpha0001

python -m rl_posttrain.td3bc_ref \
  --replay "${REPLAY_DIR}" --output "${TD3BC_CKPT}" \
  --steps 150000 --batch_size 256 \
  --td3bc_alpha 0.001 --td3bc_bc_weight 1.0 \
  --h_summary_mode h_proj128 --h_summary_h_dim 1536 \
  --policy_delay 2 --log_every 100 --device cuda --seed 0
```

This writes `actor.pt` and `config.yaml`. For pure BC, use `--td3bc_alpha 0.0` with a positive BC weight and a separate output directory. These hyperparameters are an experiment recipe, not a claim of the best final setting. Replay and checkpoint must agree on observation layout, latent dimension, action specification, and normalizer.

### 3. Evaluate the Offline Actor

```bash
python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${TD3BC_CKPT}" \
  --task "${TASK}" --model_path "${MODEL_PATH}" \
  --scene 1 1 --scene 1 2 --scene 2 1 --scene 2 2 \
  --scene 3 1 --scene 3 2 \
  --num_episodes 8 --num_trials 2 --no_save_video \
  --output_root "playground_eval/paired_eval/${TAG}_all6"
```

Baseline, identity insertion, and actor evaluations are included. `--skip_identity` is available after validating the insertion path. `--actor_checkpoint` accepts one or more actor files/directories. A checkpoint directory should contain `actor.pt`; use the specific `.pt` path for online checkpoints.

### 4. Run Online Post-Training

The loop collects episodes on balanced train scenes, appends fresh online replay, samples mixed base/online batches, warms up the critic, then updates the actor and runs periodic paired evaluation. Explicitly use the replay and actor created above:

```bash
RUN_DIR=playground_eval/online_td3bc/${TAG}_online

python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --model_path "${MODEL_PATH}" \
  --base_replay "${REPLAY_DIR}" \
  --init_checkpoint "${TD3BC_CKPT}" \
  --output_root "${RUN_DIR}" \
  --wandb_mode disabled
```

Use `--wandb_mode online` with configured credentials for remote logging. `--auto_name --output_root_base playground_eval/online_td3bc` can derive a name when an explicit output root is omitted; use `--name_suffix` to distinguish repeats.

The checked-in YAML currently specifies:

| Setting | Value |
| --- | --- |
| Online episodes / critic-only episodes | `300` / `30` |
| Minimum online transitions before updates | `1024` |
| UTD ratio / maximum updates per episode | `4` / `800` |
| Base/online replay mix during critic warmup | `0.3` / `0.7` |
| Base/online replay mix during joint updates | `0.1` / `0.9` |
| TD3+BC alpha / BC weight | `0.001` / `0.0` |
| Reward | Sparse final success |
| Exploration | Normalized Gaussian noise after episode 10 |
| Periodic evaluation | Every 50 online episodes; no videos by default |

**The YAML's BC weight is zero**, so its online actor objective has no BC penalty. Add `--bc_weight 0.1`, for example, to run a BC-regularized variant and record the override. Always pass `--config`: some in-code defaults differ from the YAML.

The YAML's local defaults use `playground_eval/replays/open_laptop_v4_checkpoint3000_20260509_174957` and alias `h_proj128_alpha0001`, resolved to `playground_eval/rl_checkpoints/open_laptop_hsummary_h_proj128_alpha0001/actor.pt`. These are existing experiment artifacts, not files supplied by a fresh clone. The explicit overrides above avoid depending on them.

### 5. Resume the Same Run

Keep the original replay, offline initialization, model path, configuration, and any overrides:

```bash
python -m rl_posttrain.online_td3bc \
  --config rl_posttrain/configs/online_td3bc_v1.yaml \
  --model_path "${MODEL_PATH}" \
  --base_replay "${REPLAY_DIR}" \
  --init_checkpoint "${TD3BC_CKPT}" \
  --output_root "${RUN_DIR}" \
  --resume "${RUN_DIR}/checkpoints/latest_online.pt" \
  --allow_reuse_online_replay --strict_resume_manifest_match \
  --wandb_mode disabled
```

The full online checkpoint preserves actor/critic networks, targets, optimizers, normalizers, progress, and replay/model-path metadata. Historical checkpoints do not serialize RNG states, so resume is not guaranteed to reproduce uninterrupted training bit for bit. `latest_actor.pt` is for actor evaluation, not full training resume. The manifest guards paths and replay metadata; keep the frozen checkpoint's contents unchanged as well. Use a new output directory for a new experiment. The historical root `playground_eval/online_td3bc/h_proj128_alpha0001_online_v1` was previously marked unsuitable for resume in this project's notes.

## Evaluation and Outputs

For paired outcomes, `recover` counts baseline failures changed to actor successes, and `regress` counts baseline successes changed to actor failures:

```text
net = recover - regress
```

Report success rates with episode/trial counts, scene splits, checkpoint identity, and paired recovery/regression counts. Training losses and Q values are diagnostics, not substitutes for evaluation. Check offline replay provenance before calling an old experiment held out.

W&B groups include `online/*` for progress and replay mix, `loss/*`, `q/*`, `action/*`, and `eval_seen/*`, `eval_unseen/*`, `eval_all/*`. `eval/static_cache_used=1` indicates reuse of static baseline/offline-init evaluation; current actor evaluation still runs.

```text
playground_eval/
  replays/<tag>/                         # episode/trial replay shards
  rl_checkpoints/<actor>/
    actor.pt
    config.yaml
  paired_eval/<evaluation>/
    paired_summary.json                 # aggregate for a multi-scene run
    room1_table1/
      paired_summary.json
      off/results_local_eval.txt        # baseline directory is named off
      identity/results_local_eval.txt
      actor/results_local_eval.txt      # actor_<name> for multiple actors
      actor/videos/...                  # when recording is enabled
  online_td3bc/<run>/
    online_config.yaml
    online_replay/
    checkpoints/
      latest_online.pt
      latest_actor.pt
    eval/
      episode_0050/
      episode_0100/
```

`playground_eval/` is ignored by Git. Experiment outputs and local absolute paths do not become publicly accessible by linking to them in this README.

## Results and Demos

The completed [300-episode budget report](docs/experiments/online-budget-20260915.md) includes the full milestone table, actual training settings, paired outcomes, and limitations. Online@300 reached 75/96 successes versus 74/96 for offline initialization; room1/room2 gained two successes and room3 lost one. Additional budget did not yield monotonic improvement.

Selected public demos are pending consolidation. Unpopulated figure/video placeholders have been removed. The existing `media/EgoVLA-Teaser.jpg` belongs to upstream EgoVLA and is not evidence of this extension's results.

Local videos exist under `playground_eval/paired_eval/` and `playground_eval/runs/`. No MP4 files were found under `playground_eval/online_td3bc/` in the 2026-09-14 audit, consistent with the configuration disabling evaluation videos. Do not label an offline rollout as an online result.

To record a paired demo **after the online actor checkpoint exists**, omit `--no_save_video` and choose a new output root:

```bash
DEMO_ROOT=playground_eval/paired_eval/demo_$(date +%Y%m%d_%H%M%S)
python -m rl_posttrain.paired_eval \
  --actor_checkpoint "${TD3BC_CKPT}" "${RUN_DIR}/checkpoints/latest_actor.pt" \
  --task "${TASK}" --model_path "${MODEL_PATH}" \
  --scene 1 1 --scene 3 1 \
  --num_episodes 2 --num_trials 1 \
  --output_root "${DEMO_ROOT}"

find "${DEMO_ROOT}" -type f -name '*.mp4'
```

Videos are nested below each mode's `videos/` directory. Use matching `paired_summary.json` and `results_local_eval.txt` files to identify outcomes. A recovery/regression example must match scene, episode, and trial between baseline and actor. Two episodes are a recording example, not the final evaluation protocol.

Before adding public media links, select and inspect clips, record model/actor checkpoint and scene/episode/trial provenance, and include the actual assets in Git or upload them to a stable release/attachment location. Verify published links from a fresh clone or browser. Curated videos illustrate behavior; final result tables should cover the complete evaluation protocol rather than selected clips.

