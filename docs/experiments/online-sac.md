# SAC from the offline EgoVLA actor

The TD3+BC budget experiments showed limited additional gains after offline
training. This experiment changes the online learner to SAC while retaining
the frozen EgoVLA features, action interface, prior replay, reward, and scene
protocol. The implementation starts from the original offline direct-tanh
actor and critic, not from B's reference-residual actor or online@50.

## Policy and updates

The Gaussian mean reuses the original actor's feature processor and MLP.
An additional state-dependent log-standard-deviation head starts at -5.8
(pre-tanh std approximately 0.0030), with zero weights and bounds [-10, 0].
Deterministic inference is exactly the source actor at initialization for
the same normalized observation. Stochastic actions use tanh(mu + std * eps),
with a stable tanh Jacobian correction in the log density.

SAC uses two critics, their minimum for the target and actor objective,
entropy in the Bellman backup, and an automatically learned temperature.
There is no BC term, Q-magnitude normalization, TD3 target-action noise, or
delayed actor update. Only target critics are used by SAC. The inherited
target actor exists for export compatibility and is not used in its updates.

The first preset is deliberately bounded rather than a tuned optimum:

- 150 online trajectories each, training seeds 0 and 1; UTD 1, batch 256,
  maximum 800 critic updates per trajectory.
- First 30 trajectories: critic-only warm-up, mean/std/temperature frozen;
  updates begin after at least 1,024 online transitions.
- Thereafter, one actor and temperature update per critic update.
- Actor LR 1e-4, critic LR 3e-4, temperature LR 3e-4; initial temperature
  1e-4, target entropy -38; gamma 0.99, target-critic tau 0.005.
- Prior/online replay mix: 50/50 during warm-up, 25/75 during joint updates.
- Sparse terminal success reward 1, failure 0; preserve the existing
  termination and timeout semantics. No added shaping reward.
- Train on room1/room2, evaluate room1/room2 and room3 separately.
- Every training rollout samples from SAC from the first episode and uses
  no additional TD3 exploration noise.

The initial critic weights come from TD3+BC; warm-up adapts them to the new
entropy-regularized target. Actor/critic optimizer states are fresh at this
algorithm transition. This is warm-started SAC with prior replay, not RLPD.

## Evaluation and reproducibility

At online@50/100/150 evaluate both deterministic and stochastic execution:
six scenes x eight initialization labels x two trials = 96 per mode.
Two independent initial-evaluation jobs report the same two modes at @0.
Their shared actor is numerically identical at initialization across the
training seeds, because the only newly initialized policy head has zero
weights and a fixed bias.

Both modes use the same frozen checkpoint. Stochastic inference has a
dedicated CUDA generator, independent of simulator/model global RNG.
The generator seed is 200000 + training_seed*10000 + room*100 + table,
reset per scene and consumed in the fixed episode/trial order. Seeds are
shared across milestones. Training rollout seeds are
100000 + training_seed*10000 + zero_based_episode.

Each scene receipt records mode, seed, actor SHA256, ordered episode IDs,
outcomes, and original result SHA256. Incomplete scenes are preserved and
retried; completed scenes are reused only if their hashes and metadata
match. Full checkpoints include actor, critics, target critics, temperature,
optimizers, counters, replay manifest, and Python/NumPy/Torch/sampler RNG.
Milestone recovery evaluates the fixed periodic export before continuing.
No best-checkpoint selection on room3 is performed.

W&B training runs belong to one SAC group, with `sac/*` diagnostics and
separate `eval_deterministic/*` and `eval_stochastic/*` results. Evaluation
plots use `evaluation/episode`; training plots use `training/gradient_step`.
The W&B history index is independent of gradient-update counts so that
events at the same update do not overwrite or drop evaluation records.
Negative differential entropy or positive log probability densities are
valid for concentrated continuous distributions.

## Entry points

`rl_posttrain/configs/online_sac_v1.yaml` provides the learner settings.
Configure paths and GPU mapping for the local runtime. A bounded run uses
`python -m rl_posttrain.run_sac --run-root <run-directory>` and a frozen
`manifest.json` containing code/assets SHA256 entries, GPU, runtime,
initial actor, and preflight receipt. Its `config.yaml` supplies training
settings. Resume uses `--resume <full-checkpoint>`. Initial evaluation uses
`--eval-only --mode deterministic` or `--mode stochastic` in separate run
directories. All output directories must be separate from TD3+BC and B.

The basic learner is also reachable through
`python -m rl_posttrain.online_td3bc --config <sac-config> --no_eval`;
the bounded runner supplies the required dual-mode evaluation protocol.

## Validation before launch

- Unit/regression checks cover exact mean transfer, checkpoint loading,
  independent sampling RNG, density against PyTorch's transformed Normal,
  stable saturation gradients, terminal masks/entropy backup, warm-up,
  mean/std/temperature updates, and exact CPU training resume.
- All 14,304 prior replay observations: deterministic conversion error 0.
- Real-replay diagnostic: 40 critic updates and 20 actor updates with finite
  metrics; warm-up leaves the actor unchanged. This is not a scientific run.
- Five-step room3 simulator probes for the original actor and both SAC
  modes reconstruct executed normalized actions with error 0; stochastic
  execution has nonzero sampling deviation. These are interface checks,
  not full success-rate evaluations.

This evaluates the RL action head on frozen EgoVLA features, not end-to-end
VLA weight fine-tuning. Two conditional training seeds and a single task
do not establish a general algorithm ranking. Room3 is held out from RL
replay but has previously been used for development and evaluation.
