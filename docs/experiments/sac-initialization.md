# SAC actor/critic initialization comparison

Reuse the original A seeds 0/1 without retraining. Add B and C, each with
150 online trajectories, seeds 0/1, 30 critic-only trajectories, UTD 1,
batch 256, and the same replay ratios, reward, learning rates, entropy
settings, simulator settings and deterministic/stochastic evaluation at
50/100/150 as A. Each mode evaluates 96 episodes and reports room1/room2
separately from room3. No actor anchoring or entropy stabilization is added.

| Condition | Actor mean initialization | Critic initialization |
| --- | --- | --- |
| A (existing) | Offline TD3+BC actor | Offline TD3+BC critics |
| B | Identical offline TD3+BC actor | Fresh twin critics and matching targets |
| C | EgoVLA reference actions, zero logit residual | Fresh twin critics and matching targets |

B/C use identical fresh critic tensors for the same seed (CPU RNG seed
400000 + training seed). C's feature projection/MLP is fresh (seed 300000 +
training seed); its final mean layer is zero. All optimizer states and
training counters start fresh. Normalization metadata and the same prior
replay remain in use, so C is not an experiment without offline data.

C's mean is atanh(clamp(reference, -1+1e-6, 1-1e-6)) + learned_residual.
The actor observes the same frozen EgoVLA features and normalized reference
actions. Actions are tanh(mean + std * noise), with the same exact squashed
Gaussian density as A/B. There is no clipping after Gaussian sampling.
Deterministic actions initially match the clipped reference to float32
precision. The reference skip remains during training; therefore A/B is a
strict critic-initialization comparison, while B/C also differs in actor
parameterization. C must not be described as a pure weight-only ablation.

Initial pre-tanh std is exp(-5.8), alpha 1e-4, target entropy -38, learned
after warm-up. Known instability in A is deliberately retained in this
initialization round; changing it would require a separate matched control.

GPU2 runs B seed0; GPU3 runs C seed0. B seed1 waits for A seed0 on GPU0;
C seed1 waits for A seed1 on GPU1. The successor starts only after the
predecessor is marked completed with all milestone dual-mode receipts,
its cooperative GPU lock is released and GPU memory is below 1000 MiB.
A failed predecessor is not considered completed: its successor remains
queued for recovery. Queue processes use no CUDA context and survive SSH
disconnection; machine reboot requires explicit relaunch.

Entry points: initialize_sac_ablation.initialize creates the pre-SAC
checkpoint; run_sac provides bounded training/evaluation; queue_sac provides
the completion dependency. Per-run manifests freeze code, configs, assets
and preflight evidence. Diagnostic updates/rollouts are excluded from all
training and evaluation results. W&B group: sac_init_ablation_20260916.
