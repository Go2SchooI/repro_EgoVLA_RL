# Experiment: 4node-len16-per-ep-bs32w8

## Created
2026-06-08 11:53:59 Asia/Shanghai

## Goal
Run SG DLC 4-node VITRA EPIC100 + Ego4D LEN16 training using the per-episode-file data root, with per-GPU batch size 32 and DataLoader workers 8, then EPIC-only validation.

## Reproduce
Use this DLC UserCommand:

```bash
bash 
```

DLC resource shape: 4 Worker pods, 8 GPU per pod, CPFS mounted at ``.

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
Manifest and statistics are reused because the new per-episode-file root has the same dataset ids, episode indices, and episode lengths as the original root.
