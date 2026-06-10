# Experiment: EPIC full-2048 all-window validation

## Created
2026-06-05 02:23:04 Asia/Shanghai

## Goal
Validate checkpoint(s) on the fixed EPIC seen-val split of 2048 episodes without the 3000-window cap. In this codebase validation samples are windows, so this run expands the fixed 2048 episodes to all available windows (dataset_size=69606) and validates every window.

## Reproduce
```bash
cd <EXPERIMENT_RECORD_DIR>
nohup bash run.sh > logs/driver.nohup.log 2>&1 & echo pid=$!
```

Progress is printed to logs/driver.nohup.log and logs/progress.log. You can also run:

```bash
cd <EXPERIMENT_RECORD_DIR>
./show_progress.sh
```

## Config
See config.yaml.

## Environment
- Host: <host-redacted>
- Root: <WORKSPACE_ROOT>
- Repo: <REPO_ROOT>
- Python: <PYTHON_BIN>
- Read mode: full (VITRA_PARQUET_READ_MODE=full, VITRA_PARQUET_WINDOW_READ=0)
- Seed: 42
- Per-GPU validation batch size: 8
- Dataset windows: 69606
- Max windows: 0 (no cap)

## Inputs
- Manifest: <MANIFEST>
- Checkpoint label: 3-node-bs4x24-h200
- Checkpoint label: 4-node-bs2x32-h200
- Checkpoint path: <CHECKPOINT_PATH>
- Checkpoint path: <CHECKPOINT_PATH>

## Outputs
- Logs: logs/
- Shards: artifacts/shards/
- Merged metrics: artifacts/merged/ or <MERGED_OUTPUT_DIR>
- Results: results.md

## Notes
All writes are kept under this experiment directory. Large datasets/checkpoints are referenced in place rather than copied.
