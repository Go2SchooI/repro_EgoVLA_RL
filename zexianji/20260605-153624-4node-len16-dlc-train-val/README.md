# Experiment: 4node-len16-dlc-train-val

## Created
2026-06-05 15:36:24 Asia/Shanghai

## Goal
Run SG DLC 4-node VITRA EPIC100 + Ego4D LEN16 training and then EPIC-only validation.

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
- LEN16 handoff: ``

## Inputs
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
The wrapper sets `VITRA_PARQUET_READ_MODE=full` and `VITRA_PARQUET_WINDOW_READ=0` by default.
