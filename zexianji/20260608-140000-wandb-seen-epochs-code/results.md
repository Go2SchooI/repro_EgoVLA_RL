# Results

## Status
success

## Command
```bash
<PYTHON_BIN> -m py_compile openwam/train/vla/train_loop.py openwam/train/vla/trainer.py openwam/train/vla/wandb_protocol.py
```

## Timing
- Start: 2026-06-08 14:00:00 Asia/Shanghai
- End: 2026-06-08 14:00:00 Asia/Shanghai

## Metrics
- train/effective_batch_size: logged to wandb
- train/samples_seen: logged to wandb
- train/train_windows: logged to wandb when dataset length is available
- train/seen_epochs: logged to wandb when dataset length is available
- train/loss_vs_seen_epochs: logged to wandb using train/seen_epochs as step metric

## Summary
Code compiles and now supports loss comparison by seen epochs without hard-coding the number of train windows.

## Notes
No training job was started by this record.
