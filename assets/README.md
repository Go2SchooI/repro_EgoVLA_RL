# Bundled simulation assets

These files were already bundled with the upstream-derived repository. They are support assets for the EgoVLA inference and simulator reset path, separate from the EgoVLA backbone and offline/online RL checkpoints. This reorganization preserves their bytes; it does not retrain models or regenerate evaluation initializations.

| Path | Purpose | Current use |
| --- | --- | --- |
| [retargeting/hand_actuation_net.pth](retargeting/hand_actuation_net.pth) | Small MLP mapping 30 hand-keypoint coordinates (15 per hand) to 24 robot hand joint targets (12 per hand). | Loaded by `human_plan/ego_bench_eval/utils.py` for action retargeting. Required by the current inference path. |
| [retargeting/hand_mano_retarget_net.pth](retargeting/hand_mano_retarget_net.pth) | Companion MLP mapping 24 robot hand joint positions to 30 MANO pose parameters. | Also loaded by the evaluation utilities for the robot-to-MANO representation conversion. |
| [initial_poses/init_poses_fixed_set_100traj.pkl](initial_poses/init_poses_fixed_set_100traj.pkl) | Per-task, per-episode initialization table containing left/right hand joint targets and end-effector poses, indexed by starting-frame offset. | The table actually loaded by `human_plan/ego_bench_eval/ik_agent_30hz.py` to initialize the arms and hands after reset. |
| [initial_poses/legacy/init_poses.pkl](initial_poses/legacy/init_poses.pkl) | Historical initial-pose table with the same kinds of hand/end-effector fields. | No reader found in the current tracked code, and this table contains no Open-Laptop entry. Retained for historical compatibility; not used by the current evaluator. |
| [initial_poses/legacy/init_poses_fixed_set.pkl](initial_poses/legacy/init_poses_fixed_set.pkl) | Earlier fixed-set initial-pose table. | Written by the older `tools/parse_initial_poses.py`; not loaded by the current evaluator. |

## Paths and regeneration

[human_plan/asset_paths.py](../human_plan/asset_paths.py) centralizes paths relative to this source checkout, so asset lookup does not depend on the shell's current directory. Other model, dataset and simulator paths still follow the main setup instructions; this is not a standalone wheel containing all runtime assets.

The root-level copies have moved to the locations above. The original upstream README retains its historical filenames; use this table to locate them. Existing experiment worktrees and their frozen inputs remain separate.

- `human_plan/utils/nn_retarget.py` trains the hand-actuation network; `nn_retarget_tomano.py` trains the companion network. Their save/load locations now point to `retargeting/`. These scripts require their original datasets and are not needed to use the supplied weights.
- `tools/parse_initial_poses_fixed_set.py` generates the active initialization table from `data/EgoVLA_SIM`. The older generator writes into `initial_poses/legacy/`.
- Generation scripts overwrite their corresponding bundled asset. Do not regenerate these files merely to run evaluation: changing them changes the execution or initialization protocol.

The filenames alone do not establish how many usable trajectories each task contains. The evaluator chooses its episode list and starting offset separately; the reported Open-Laptop protocol uses eight fixed initializations and two trials per scene.
