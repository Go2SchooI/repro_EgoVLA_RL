# Actor/Critic Observation Spec

生成时间：2026-05-12  
适用代码：当前 `rl_posttrain` TD3+BC 后训练实现  
主要任务例子：`Humanoid-Open-Laptop-v0`

这份文档记录当前代码里 actor 和 critic 的真实输入，而不是最初设计草案。核心代码位置：

- Actor obs 构造：`rl_posttrain/obs_utils.py::build_actor_obs`
- Critic obs 构造：`rl_posttrain/obs_utils.py::build_critic_obs`
- Actor 网络：`rl_posttrain/actors.py::DeterministicActor`
- Critic 网络：`rl_posttrain/critics.py::DoubleQCritic`
- Replay/normalization/training：`rl_posttrain/td3bc_ref.py::PreparedReplay` 和 `TD3BCTrainer`

## 1. Action Space

当前 actor 插入点在 temporal smoothing 之后。actor 的 action 不是 EgoVLA raw action head 的 48 维，而是 smoothing 后即将送入 IK/retarget/env.step 的 packed execution command。

当前 Open-Laptop action spec：

| field | dim |
|---|---:|
| `left_ee_pose` | 7 |
| `right_ee_pose` | 7 |
| `left_qpos` | 12 |
| `right_qpos` | 12 |
| total | 38 |

所以：

```text
action_dim = 38
```

actor 输出的是 canonical normalized action：

```text
a_exec_norm = actor(actor_obs)
a_exec_norm in [-1, 1]
```

然后代码通过 replay/checkpoint 中保存的 action normalizer 做 denormalize，得到实际执行动作 `a_exec`。

## 2. Actor Input

actor obs 在 `build_actor_obs()` 中构造：

```text
actor_obs = [
  h_summary,
  proprio,
  base_chunk_summary,
  a_ref_norm
]
```

### 2.1 `h_summary`

来源：EgoVLA latent hook 的 `h_in`。

当前实现：

- 如果 `h_in` 是 1D，直接 flatten。
- 如果 `h_in` 是多维，例如 Open-Laptop 中常见的 `(60, 1536)`，则 reshape 成 `(-1, hidden_dim)` 后对 token/time 维做 mean pooling。

因此 Open-Laptop 中：

```text
h_in shape = (60, 1536)
h_summary dim = 1536
```

这部分包含 EgoVLA 的视觉、语言、历史和当前策略意图表征。actor 可以使用它，但 EgoVLA 本体保持 frozen。

### 2.2 `proprio`

actor 只使用 deployable proprio，不直接使用 privileged object pose。

当前 actor proprio keys：

```text
proprio_input_3d
proprio_input_rot
proprio_input_handdof
proprio_input_hand_finger_tip
```

Open-Laptop 当前实际 shape：

| key | shape | dim |
|---|---:|---:|
| `proprio_input_3d` | `(1, 6)` | 6 |
| `proprio_input_rot` | `(1, 6)` | 6 |
| `proprio_input_handdof` | `(1, 30)` | 30 |
| `proprio_input_hand_finger_tip` | `(1, 30)` | 30 |
| total | - | 72 |

所以：

```text
proprio dim = 72
```

### 2.3 `base_chunk_summary`

EgoVLA 输出 action chunk，但 TD3+BC actor 第一版输出 single-step action。为了让 actor 看到短期 base policy 意图，代码把当前 action chunk 压成一个 summary。

当前 chunk summary：

```text
base_chunk_summary = [
  a_0,
  a_last,
  a_last - a_0,
  mean(A_chunk)
]
```

其中每个 `a_*` 都是按当前 action spec pack 后的 38 维动作。

所以：

```text
base_chunk_summary dim = 4 * action_dim = 4 * 38 = 152
```

注意：这里用的是 EgoVLA chunk 的 postprocess 结果，字段包括：

```text
left_ee_pose
right_ee_pose
left_qpos_multi_step
right_qpos_multi_step
```

### 2.4 `a_ref_norm`

`a_ref` 是 temporal smoothing 之后，原 pipeline 在没有 RL actor 时本来要执行的动作。

代码将它 pack 成 38 维，并通过 replay 中的 canonical action normalizer 归一化：

```text
a_ref_norm dim = 38
```

它被拼在 actor_obs 的最后 38 维。训练时 `PreparedReplay` 会强制检查：

```text
actor_obs[:, -action_dim:] == bc_target_norm
next_actor_obs[:, -action_dim:] == next_bc_target_norm
```

这个检查保证 actor 输入里的 reference action tail 和 BC target 处在同一个 canonical normalized action space。

### 2.5 Open-Laptop Actor Obs 总维度

Open-Laptop 当前 actor obs：

| component | dim |
|---|---:|
| `h_summary` | 1536 |
| `proprio` | 72 |
| `base_chunk_summary` | 152 |
| `a_ref_norm` | 38 |
| total | 1798 |

因此：

```text
actor_obs_dim = 1798
```

## 3. Actor Network

当前 actor 是 deterministic actor，不是 stochastic actor。

网络形式：

```text
action_norm = tanh(MLP(actor_obs))
```

默认 hidden dims：

```text
[1024, 1024, 512]
```

输出：

```text
action_norm dim = 38
range = [-1, 1]
```

没有：

- `log_std`
- `log_prob`
- Gaussian sampling
- entropy loss
- SAC actor

## 4. Critic Input

critic obs 在 `build_critic_obs()` 中构造：

```text
critic_obs = [
  actor_obs,
  required_env_state,
  optional_env_state_if_present
]
```

也就是说，critic_obs 的前缀就是完整 actor_obs。训练时 `PreparedReplay` 会检查：

```text
critic_obs[:, :actor_obs_dim] == actor_obs
next_critic_obs[:, :actor_obs_dim] == next_actor_obs
```

这个检查保证 actor/critic 的上下文对齐。

## 5. Required Critic Env Fields

当前 required critic env keys：

```text
qpos
qvel
left_ee_pose
right_ee_pose
left_target_ee_pose
right_target_ee_pose
left_finger_tip_pos
right_finger_tip_pos
left_hand_contact_force
right_hand_contact_force
action
success
```

这些字段来自当前 benchmark/env observation 字典。代码没有用新版 IsaacLab 2.x/3.x API 去额外查询状态。

Open-Laptop 当前实际 shape：

| key | shape | flattened dim |
|---|---:|---:|
| `actor_obs` | `(1798,)` | 1798 |
| `qpos` | `(1, 50)` | 50 |
| `qvel` | `(1, 50)` | 50 |
| `left_ee_pose` | `(1, 7)` | 7 |
| `right_ee_pose` | `(1, 7)` | 7 |
| `left_target_ee_pose` | `(1, 7)` | 7 |
| `right_target_ee_pose` | `(1, 7)` | 7 |
| `left_finger_tip_pos` | `(1, 5, 3)` | 15 |
| `right_finger_tip_pos` | `(1, 5, 3)` | 15 |
| `left_hand_contact_force` | `(1, 1)` | 1 |
| `right_hand_contact_force` | `(1, 1)` | 1 |
| `action` | `(1, 50)` | 50 |
| `success` | `(1,)` | 1 |

Required env state 部分合计：

```text
50 + 50 + 7 + 7 + 7 + 7 + 15 + 15 + 1 + 1 + 50 + 1 = 211
```

## 6. Optional Critic Env Fields

当前 optional critic env keys：

```text
object_pose
reach_success
lift_success
insert_success
unload_success
sort_success
move_lid_success
flip_mug_pose_success
```

如果 env_obs 中存在这些字段，critic 会拼进去；如果不存在，就记录 missing，不强行 crash。

Open-Laptop 当前存在：

```text
move_lid_success shape = (1,)
```

因此 optional 部分通常额外增加 1 维。

Open-Laptop 当前缺失的 optional 字段通常包括：

```text
object_pose
reach_success
lift_success
insert_success
unload_success
sort_success
flip_mug_pose_success
```

## 7. Open-Laptop Critic Obs 总维度

Open-Laptop 当前 critic obs：

| component | dim |
|---|---:|
| `actor_obs` prefix | 1798 |
| required env state | 211 |
| optional `move_lid_success` | 1 |
| total | 2010 |

因此：

```text
critic_obs_dim = 2010
```

## 8. Critic Network

critic 是 double-Q critic：

```text
Q1(critic_obs, action_norm)
Q2(critic_obs, action_norm)
```

内部实际输入是：

```text
concat([critic_obs, action_norm])
```

Open-Laptop 当前：

```text
critic_obs_dim = 2010
action_dim = 38
critic MLP input dim = 2010 + 38 = 2048
```

默认 hidden dims：

```text
[1024, 1024, 512]
```

输出：

```text
Q1 scalar
Q2 scalar
```

## 9. Training-Time Usage

### 9.1 Critic update

critic 使用 replay 里的 normalized action：

```text
current_q1, current_q2 = critic(critic_obs, action_norm)
```

target action：

```text
next_action = target_actor(next_actor_obs)
noise = clipped Gaussian noise
next_action = clip(next_action + noise, -1, 1)
target_q = min(target_q1, target_q2)
y = reward + gamma * (1 - done) * target_q
```

critic loss：

```text
MSE(Q1(critic_obs, action_norm), y)
+ MSE(Q2(critic_obs, action_norm), y)
```

### 9.2 Actor update

actor 输出：

```text
action_pi = actor(actor_obs)
```

actor 的 Q term：

```text
q_pi = critic.q1_value(critic_obs, action_pi)
```

BC target：

```text
bc_target_norm = a_ref_norm
```

BC loss：

```text
bc_loss = mean((action_pi - bc_target_norm)^2)
```

TD3+BC actor loss：

```text
lambda_q = td3bc_alpha / mean(abs(q_pi)).detach()
actor_loss = -lambda_q * mean(q_pi) + td3bc_bc_weight * bc_loss
```

如果：

```text
td3bc_alpha = 0
```

则：

```text
lambda_q = 0
actor_loss = td3bc_bc_weight * bc_loss
```

即 pure BC。

## 10. Normalization

### 10.1 Observation normalization

如果 `obs_normalize=True`，训练前会分别 fit：

```text
actor_obs_normalizer = standard normalizer fitted on actor_obs
critic_obs_normalizer = standard normalizer fitted on critic_obs
```

然后训练使用 normalized actor_obs / critic_obs。

checkpoint 中会保存：

```text
actor_obs_normalizer
critic_obs_normalizer
```

eval 时加载 actor checkpoint，会用保存的 actor obs normalizer normalize 当前 actor_obs。

### 10.2 Action normalization

replay 文件中必须已经包含 canonical normalized action fields：

```text
action_norm
bc_target_norm
next_bc_target_norm
```

并且 replay metadata/checkpoint 里保存 action normalizer。

训练阶段不会重新 fit `bc_target_norm`，而是直接使用 replay 保存的 canonical normalized action space。

checkpoint 中会保存：

```text
action_normalizer
```

eval 时 actor 输出 `a_exec_norm` 后，用这个 normalizer denormalize 成实际执行 command。

## 11. Privileged vs Deployable 信息边界

actor 输入：

- 使用 EgoVLA latent。
- 使用 deployable proprio。
- 使用 base action chunk summary。
- 使用 `a_ref_norm`。
- 不直接使用 object pose / success / contact force 等 privileged env state。

critic 输入：

- 以完整 actor_obs 为前缀。
- 额外使用 env observation 中已有的 privileged state，例如 qpos/qvel/contact/success 等。
- 如果当前 task/env 提供 object_pose，也可以使用。
- 不通过新版 IsaacLab API 额外抓状态；只复用当前 env observation 字典已有字段。

这个设计是 asymmetric actor-critic：

```text
actor: deployable obs + EgoVLA latent + base intent
critic: actor obs + privileged simulator/env state
```

## 12. Open-Laptop 当前维度总览

| item | dim |
|---|---:|
| action_dim | 38 |
| h_summary | 1536 |
| proprio | 72 |
| base_chunk_summary | 152 |
| a_ref_norm | 38 |
| actor_obs_dim | 1798 |
| required critic env state | 211 |
| optional move_lid_success | 1 |
| critic_obs_dim | 2010 |
| critic network input dim | 2048 |

一句话总结：

```text
actor 输入 raw 1798 维，输出 38 维 normalized correction action；
critic 输入 raw 2010 维 critic_obs 加 38 维 action_norm，输出双 Q。
```

## 13. H Summary Ablation Extension

当前代码额外支持一个可插拔的 h_summary ablation。这个 ablation 发生在网络内部，不改变 replay schema。

顺序是：

```text
raw actor_obs / critic_obs
-> existing obs normalizer
-> split h_summary prefix and rest
-> h_summary processor
-> actor / critic MLP
```

支持的 mode：

| mode | 行为 | processed h dim |
|---|---|---:|
| `full_h` | 原始 h_summary 直接进入 MLP，等价旧路径 | 1536 |
| `h_zero` | h_summary 置零，但保留原始维度 | 1536 |
| `h_proj256` | `LayerNorm -> Linear(1536,512) -> GELU -> Linear(512,256) -> LayerNorm` | 256 |
| `h_proj128` | `LayerNorm -> Linear(1536,512) -> GELU -> Linear(512,128) -> LayerNorm` | 128 |

`full_h` 是默认值；如果不显式传 `--h_summary_mode`，训练和 eval 的主路径保持等价。`h_proj256/h_proj128` 的 projection 参数属于 actor/critic 网络参数，会随 checkpoint 保存和加载。eval 加载 checkpoint 时会自动恢复 h mode 和 projector 结构。

Open-Laptop 下不同 mode 的网络输入维度：

| mode | actor MLP input dim | critic processed obs dim | critic MLP input dim |
|---|---:|---:|---:|
| `full_h` | 1798 | 2010 | 2048 |
| `h_zero` | 1798 | 2010 | 2048 |
| `h_proj256` | 518 | 730 | 768 |
| `h_proj128` | 390 | 602 | 640 |

这里 critic MLP input dim = processed critic obs dim + action dim 38。
