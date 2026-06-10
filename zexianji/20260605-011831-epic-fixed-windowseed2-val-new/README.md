# Experiment: epic-fixed-windowseed2-val-new

## Created
2026-06-05 01:18:31 Asia/Shanghai

## Goal
Revalidate the new-machine `3-node bs4x24 H200` and `4-node bs2x32 H200` EPIC100+Ego4D100 checkpoints on the same 2048 EPIC seen-val episodes, but with the same third deterministic sample of 3000 global validation windows used by the old-machine script.

## Reproduce
```bash
bash 
```

## Notes
- All writes stay under this experiment directory.
- Window indices are generated deterministically from `window_seed=20260606`, `dataset_size=69606`, `num_windows=3000`.
- Validation uses `VITRA_PARQUET_READ_MODE=full`, validation seed `42`, and per-GPU batch size `8`.
- Each checkpoint is sharded over all visible GPUs by default; set `GPU_LIST=0,1,...` or `NUM_SHARDS=N` to override.
