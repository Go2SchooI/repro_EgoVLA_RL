# Experiment: 2node-len16-per-ep-bs32w8

## Created
2026-06-08 12:00:48 Asia/Shanghai

## Goal
Run SG DLC 2-node VITRA EPIC100 + Ego4D LEN16 training using the per-episode-file data root, with per-GPU batch size 32 and DataLoader workers 8, then EPIC-only validation.

## Reproduce
Use this DLC UserCommand:

```bash
bash 
```

DLC resource shape: 2 Worker pods, 8 GPU per pod, CPFS mounted at ``.

Optional learning-rate environment variables:

```text
LEARNING_RATE=<backbone_lr>                 # overrides training_strategy.learning_rate, default 1e-5
ACTION_MODEL_LEARNING_RATE=<action_head_lr> # overrides training_strategy.action_model_learning_rate, default 1e-4
```

## Config
See `config.yaml`.

## Environment
- Host: SG DLC
- Root: ``
- Repo: ``
- Git: `<git-branch>` / `6895add`
- Python: ``
- Data root: ``
- LEN16 handoff: ``

## Inputs
- Train dataloader: ``
- Val dataloader: ``
- Train/val manifest: ``
- Statistics: ``
- Validation config: ``

## Outputs
Formal outputs are not duplicated into this sandbox:

- Runs/checkpoints/validation: ``
- Logs: ``
- Cache/temp: ``
- Sandbox launch env snapshots: `env/`
- Output path manifest: `artifacts/output_paths.txt`

## Notes
Manifest and statistics are reused because the per-episode-file root has the same dataset ids, episode indices, and episode lengths as the original root. This run intentionally uses 2 nodes x 8 GPUs, per-GPU bs=32, total batch 512.
