# Experiment: wandb-seen-epochs-code

## Created
2026-06-08 14:00:00 Asia/Shanghai

## Goal
Record the code instrumentation that logs train/samples_seen and train/seen_epochs to wandb for fair loss comparison across different effective batch sizes.

## Reproduce
Use the normal LEN16 DLC entry/run scripts; no separate training command is launched by this record.

## Config
See config.yaml.

## Environment
- Host: <host-ip-redacted> (new-sg)
- Root: <WORKSPACE_ROOT>
- Repo: <REPO_ROOT>
- Git commit: 6895add
- Dirty entries: 39
- Python compile: <PYTHON_BIN> -m py_compile openwam/train/vla/train_loop.py openwam/train/vla/trainer.py openwam/train/vla/wandb_protocol.py

## Inputs
- Existing training code and current LEN16 run wrappers under 

## Outputs
- Code changes in openwam/train/vla/train_loop.py, openwam/train/vla/trainer.py, openwam/train/vla/wandb_protocol.py.
- Diff snapshot: artifacts/seen_epochs.diff
- Validation log: logs/py_compile.log

## Notes
- Metrics use len(train dataset) dynamically; train_windows is not hard-coded.
- Formal checkpoints/logs remain in normal training output paths; this sandbox record only documents the code instrumentation.
