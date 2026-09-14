# Online interaction-budget experiment

Verified on 2026-09-15. This is one completed training lineage, using training seed 0, from an offline TD3+BC initialization to 300 total online episodes. It is a preliminary budget comparison, not a multi-seed claim of improvement.

## Evaluation protocol

- Task: `Humanoid-Open-Laptop-v0`, frozen EgoVLA `checkpoint-3000`.
- Six scenes: two table settings in each of room1, room2, and room3.
- Eight fixed episode initializations per scene, two trials each: 96 rollouts per policy and evaluation checkpoint. Evaluation task seed: 8. Exploration is disabled for evaluation.
- room1/room2 are RL training scenes (64 evaluation rollouts, RL-ID). room3 is excluded from offline and online training replay (32 rollouts, RL-OOD relative to this RL split).
- room3 has historically been used for development/evaluation. It is not an untouched test set, and its exclusion from upstream EgoVLA training is not established.
- Baseline/offline results are reused from the matched static evaluation cache. Each online milestone is evaluated afresh with the same scene/episode/trial protocol.

## Results

| Policy | All six scenes | room1/room2 (RL-ID) | room3 (RL-OOD) |
| --- | ---: | ---: | ---: |
| EgoVLA baseline | 70/96 (72.92%) | 48/64 (75.00%) | 22/32 (68.75%) |
| Offline TD3+BC | 74/96 (77.08%) | 49/64 (76.56%) | 25/32 (78.13%) |
| Online@50 | 73/96 (76.04%) | 47/64 (73.44%) | 26/32 (81.25%) |
| Online@100 | 71/96 (73.96%) | 49/64 (76.56%) | 22/32 (68.75%) |
| Online@150 | 74/96 (77.08%) | 51/64 (79.69%) | 23/32 (71.88%) |
| Online@200 | 75/96 (78.13%) | 53/64 (82.81%) | 22/32 (68.75%) |
| Online@250 | 74/96 (77.08%) | 50/64 (78.13%) | 24/32 (75.00%) |
| Online@300 | 75/96 (78.13%) | 51/64 (79.69%) | 24/32 (75.00%) |

The budget curve is not monotonic. Online@300 has one additional success over offline TD3+BC overall: six recoveries and five regressions. In room1/room2 it has four recoveries and two regressions; in room3 it has two recoveries and three regressions. The aggregate therefore hides different ID/OOD behavior.

Relative to Online@50, Online@300 gains four successes on room1/room2 and loses two on room3. The highest observed room3 success rate in this lineage is at episode 50; this is a descriptive observation, not justification for selecting checkpoints using a held-out test set. These counts do not establish statistical significance or consistent gains across training seeds.

## Actual training configuration

These settings come from the completed run's configuration. They differ from the checked-in default YAML, especially in BC weight and replay mixture. The README's fresh collection/training recipe is not an exact reconstruction of this historical run.

| Setting | Value |
| --- | --- |
| Initialization | Offline `h_proj128_alpha0001` actor and critic |
| Online training scenes | room1_table1, room1_table2, room2_table1, room2_table2; balanced sampling |
| Total online budget / critic-only episodes | 300 / 30 |
| Minimum online transitions for updates | 1024 |
| UTD / maximum critic updates per episode | 4 / 800 |
| Actor update interval | Every 2 critic updates during joint training |
| Base/online replay mix, critic warmup | 0.5 / 0.5 |
| Base/online replay mix, joint training | 0.25 / 0.75 |
| TD3+BC alpha / BC weight | 0.001 / 0.1 |
| Actor LR / critic LR / batch size | 0.0001 / 0.0003 / 256 |
| Gamma / target-update tau | 0.99 / 0.005 |
| Reward | Sparse final success: success 1, failure 0; terminate on success or timeout |
| Exploration | After 10 episodes; normalized noise std 0.003, clip 0.01 |
| Budget-continuation W&B logging | Local offline logs |

The continuation resumed at episode 55 and added 245 episodes; it did not collect 300 new episodes after the original 50. At completion the checkpoint records 37,198 environment steps, 144,780 critic updates, and 66,980 actor updates. Training was interrupted and resumed. Network, optimizer, target, normalizer and replay states were checked, but the historical checkpoints do not serialize RNG states, so bitwise equivalence to an uninterrupted run is not claimed.

## Evidence and scope

The [machine-readable results](online-budget-20260915.json) contain aggregate counts, paired recovery/regression counts, and per-scene ordered outcomes for all six milestones. Counts and paired outcomes were recomputed from these outcome arrays before committing this report. The arrays retain the source evaluation order within each scene; they are not full simulator trajectories.

The local completion audit verified 300 replay shards, strict replay-manifest matching, 49 registered input hashes, and all five continuation milestone evaluations (100 through 300). The final full training checkpoint SHA-256 is:

```text
2ab08e1f7304314f776874900ad4fed2d46cc249d8c1995f21b698a13d06e41f
```

Raw logs, replay, model weights and run-specific orchestration scripts remain local experiment artifacts. The committed summary supports inspection of the reported counts; it is not a public checkpoint or complete reproducibility bundle. No real-robot results are claimed. The separate server hyperparameter grid and the planned no-offline-initialization experiment are outside this report.
