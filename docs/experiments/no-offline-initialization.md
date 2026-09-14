# B: online training without offline parameter initialization

The first B run compares a zero-initialized reference-residual actor and random critic with the archived A lineage in [the online budget report](online-budget-20260915.md). Both retain the same frozen EgoVLA, base replay and observation/action normalizers. B removes offline parameter pretraining, not offline data access.

## Actor and initialization

The actor computes `clamp(a_ref_norm + tanh(MLP(obs)), -1, 1)`. The reference action is recovered from the normalized observation tail using fixed observation-normalizer statistics. The final MLP layer starts at zero. The learned MLP and h-summary projector have the same sizes as A, but the output parameterization differs. A uses a direct tanh actor.

`python -m rl_posttrain.initialize_reference_actor --base_replay <base-replay> --output <initial.pt> --seed 0` creates an initialization artifact from base replay only. The initializer accepts no offline checkpoint, creates fresh actor/critic parameters, checks replay scenes, fits normalizers, and validates initial reference-action error. Targets and empty Adam optimizers are constructed by the online agent; online and update counters start at zero.

The initial artifact is an initialization checkpoint, not an offline-trained model. The `initialization` and `actor_parameterization` fields persist in actor and training exports. Legacy checkpoints without these fields retain direct-tanh behavior.

## Fixed comparison protocol

- Training seed 0, 300 total online episodes, balanced room1/room2 training scenes.
- Same 14,304-transition base replay as A. Actual real-replay comparison found identical means/scales for action, actor-observation and critic-observation normalization.
- 30 critic-only episodes; minimum 1,024 online transitions before updates; UTD 4 capped at 800 critic updates per episode; actor updates every 2 critic updates.
- Base/online replay ratios: 0.5/0.5 during warm-up, 0.25/0.75 during joint training. TD3+BC alpha 0.001 and BC weight 0.1.
- Same reward, learning rates, action noise, simulator settings and 300-episode collection initialization schedule as the archived A configuration.
- Evaluate B at 50, 100, 150, 200, 250 and 300 with six scenes, eight fixed initializations per scene, two trials each. Static baseline/offline references are reused from A's verified cache. Report A/B success counts and paired recoveries/regressions separately for all scenes, RL-ID room1/room2 and RL-OOD room3.
- Preserve B's initial checkpoint and evaluate it on the full 96-rollout protocol after training, so the first trained comparison is not delayed by a full initial-policy evaluation. The earlier 5-step simulator preflight is technical validation, not a success-rate result.
- Do not select a best checkpoint using room3. `eval.save_best` is disabled for this run. Local W&B logging uses offline mode.

## Validation and orchestration

The 2026-09-15 preflight passed all 35 post-training tests, including identity initialization without weight loading, legacy checkpoint compatibility, warm-up/actor gradients, held-out replay rejection and exact CPU continuation after RNG restoration. On the real base replay, maximum initial reference-action error was `1.7881393432617188e-7`. Short room1 and room3 simulator traces passed the `1e-6` reference-action tolerance. These checks do not establish equality of complete stochastic trajectories.

`python -m rl_posttrain.run_initialization_ablation --run_root <run>` runs the prepared experiment. Its run directory contains `online_b.yaml`, an immutable-input `manifest.json`, the initial artifact, preflight verification, and references to A's archived evaluation curve/static cache. It uses a GPU lock, checks input hashes and available disk space, saves checkpoints atomically, validates evaluation IDs and outcomes, and emits `comparison_curve.json` / `comparison_curve.md` after each evaluation. Full training checkpoints preserve replay-generator, Python, NumPy, CPU Torch and CUDA RNG states. Use `--resume <full-online-checkpoint>` with the same run and manifest after inspecting a failure; interrupted milestone evaluation is completed before training proceeds.

## Interpretation

This first run is a comparison of complete training routes. It is not a pure causal ablation of offline initialization because the actor output parameterization differs. A's historical interruptions also lack serialized RNG state; equal configured seeds do not imply identical random draws. Both use a single training seed. A parameterization-matched control and independent seeds are required before attributing a difference specifically to offline initialization.

room3 is excluded from RL replay but has historically been used for development/evaluation. It is not an untouched upstream test set. This document describes implementation and launch validation; trained B performance is reported only after verified evaluations finish.
