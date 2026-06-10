# Experiment: 4node-len16-bs32w8-lr4e5-4e4-warmupcos001-min005

## Created
2026-06-09 11:31:23 CST

## Goal
Launch SG/new 4-node LEN16 per-episode-file training with per-GPU batch 32, workers 8, backbone lr 4e-5, action lr 4e-4, linear-warmup+cosine-decay, WARMUP_RATIO=0.01, and cosine_min_lr_rate=0.05.

## Reproduce
```bash
bash 
```

## Config
See `run.sh` and `artifacts/output_paths.txt`.

## Notes
This requires the local repo patch that promotes training_strategy.cosine_min_lr_rate into the warmup_cosine scheduler. For global batch 1024 and 2 epochs, total optimizer steps are about 12000, so warmup is about 120 steps.
