# SAC behavior anchoring and exploration control

Original SAC (A) lost most initial behavior after actor updates began. Both
A seeds reached 0/96 in both evaluation modes at episode 100. Resetting only
the critic (B seed0) did not prevent this. These runs reuse A's original
offline actor AND critic initialization, not a later online checkpoint.
C continues independently with its original configuration.

## Three interventions, paired seeds 0/1

| Variant | Initial-actor mean anchor | Exploration settings |
| --- | --- | --- |
| Existing A | none | log std max 0, target entropy -38 |
| anchor | mean MSE weight 0.1 | unchanged from A |
| lowexplore | none | std max 0.03, target entropy -140 |
| combined | mean MSE weight 0.1 | std max 0.03, target entropy -140 |

Anchor loss compares tanh(mu) to a frozen copy of the original actor on the
same observations sampled for the actor update. It is not a loss against
replay actions, and it does not directly constrain policy std. The initial
teacher is excluded from optimization and is saved/restored in checkpoints;
resuming cannot silently replace it with the current policy. Changing anchor
weight, sampling bounds or target entropy during resume is rejected.

The exploration intervention changes two coupled settings together; this
round does not separately identify the std cap and entropy-target effects.
Using only a 0.03 cap with target -38 would make that target unattainable.
A base-replay calibration uses 1024 fixed states, eight common Gaussian noise
samples per state, and the full tanh Jacobian. At frozen initial means,
entropy is -182.838 for std exp(-5.8), -137.437 for std 0.01 and -95.711 for
std 0.03. Target -140 corresponds to std 0.00935 at those means. This checks
initial feasibility; changing means or state distribution changes achievable
entropy. Alpha remains automatically learned; there is no new alpha clamp.

Initial std exp(-5.8), temperature 1e-4, all actor/critic weights and all
normalizers are identical to A. Initial deterministic and sampled actions
match A exactly, so the original initial-policy evaluation is reusable.
No diagnostic updates enter the scientific initial checkpoints.

## Matched protocol and limits

150 trajectories per seed, first 30 critic-only, minimum 1024 online
transitions, UTD 1, batch 256, max 800 updates per trajectory; replay base/
online 50/50 then 25/75; actor LR 1e-4, critic LR 3e-4, temperature LR 3e-4,
gamma .99, tau .005, sparse terminal-success reward and inherited timeout
semantics. Train room1/2; evaluate all six scenes at 50/100/150 with 96
deterministic and 96 stochastic trials and separate room1/2 and room3 results.
Matched trajectories need not imply equal steps or optimizer updates.

A seed0 has completed all evaluations. A seed1 was stopped by the user after
133 trajectories, so its @50/@100 results remain available but @150 does not.
B seed0 was stopped after121, B seed1 after50 before complete @50 evaluation.
Do not invent missing baseline results or select a best checkpoint on room3.
The controls reuse A from the earlier code version; numerical compatibility
checks verify that disabling the new anchor preserves the original updates.

GPU mapping: anchor seeds0/1 -> 0/4; lowexplore seeds0/1 -> 2/5;
combined seeds0/1 -> 6/7. combined seed1 waits for the previously running
alpha10 seed1 to complete its final evaluation on GPU7. C seeds0/1 use GPU3/1.
Each group has independent outputs, frozen manifests and W&B IDs.

## Simulator failure handling

New runs bound each simulator invocation: 180 seconds without log output,
900 seconds total for a training rollout, 2400 seconds for an evaluation
scene. Vulkan ERROR_DEVICE_LOST or timeout terminates the entire simulator
process group and archives partial run/results/replay before retrying the
same IDs and policy seed, at most three total attempts. Other errors fail
immediately. Retries and interrupted outputs are retained in per-scene
simulator_attempts.json and simulator_retries directories; they are not
scientific episodes. No recurring app monitor is created, and existing C
runs keep their frozen code. A bounded retry is recovery, not guaranteed
resolution of the underlying Isaac Sim/driver fault.
