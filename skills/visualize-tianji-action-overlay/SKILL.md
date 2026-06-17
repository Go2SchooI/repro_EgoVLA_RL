---
name: visualize-tianji-action-overlay
description: 在 VITRA / wuji-vla-post-train 的 action chunk 评测结果上使用 wuji-hand-teleop-private/test/visualize_tianji 做天机臂真实头相机 overlay 可视化。用于把 predictions.npz 里的 pred_action/target_action 或 lerobot v3 数据集里的 54D action 通过 URDF/FK、camera.yaml/info.yaml 标定投影到真实视频上，渲染骨架、手部 mesh、EEF 前瞻轨迹，检查模型输出 action chunk 与真实画面的几何一致性。用户提到 visualize_tianji、天机臂 overlay、URDF/FK 可视化、camera.yaml/info.yaml、predictions.npz、action chunk 视频叠加、skeleton/hand mesh/trajectory 时使用。
---

# visualize_tianji 天机臂 action overlay 可视化

> 这个 skill 记录如何在 `wuji-vla-post-train` 的 SFT / val 结果上调用
> `visualize_tianji` 工具，把 54D action chunk 通过 URDF/FK 投影到真实头相机视频上。
>
> - 可视化工具目录：`/cpfs/username/wuji-hand-teleop-private/test/visualize_tianji`
> - SFT / val 结果通常来自：`/cpfs/username/wuji-vla-post-train` 或用户指定的评测输出目录
> - 大输出 mp4 / npz 放 `/cpfs/username/...`，不要提交到 git
> - 示例里统一用 `/cpfs/username`、`<run>`、`<dataset>` 这类占位符；实际使用时替换成自己的路径

## 适用场景

当用户想看 VLA / VITRA 输出的 action chunk 是否“真的对”，优先使用这个工具，而不是只看折线图。

它和 `scripts/benchmark/render_chunk_overlay.py` 不一样：

- `render_chunk_overlay.py` 主要画 pred-vs-GT 的 54D 曲线和简单视频背景；
- `visualize_tianji` 会把 action 通过天机臂/灵巧手 URDF 和 FK 转成 3D 关节/末端轨迹，再用相机标定投影回真实头相机画面；
- 默认 `skeleton` 模式先跑通，`--hand-mesh` 可切到手部 mesh，`--trajectory` 可叠 EEF 前瞻轨迹。

## 输入文件

### 1. action chunk 评测输出

`run_action_chunk_eval.py` 产出的 `predictions.npz` 可以直接用于批量渲染。常见 key：

```text
pred_action          (S, N, 54)  模型预测 action chunk
target_action        (S, N, 54)  GT action chunk；batch_render 默认渲染 pred_action
frame0_global_index  (S,)        每段 action 的全局起始帧
episode_index        (S,)        每段 action 对应的 episode
frame0_image         (S,H,W,3)   评测时的首帧截图
```

`batch_render.py` 会根据 `episode_index` 和 `frame0_global_index` 找到 lerobot 数据集里的真实视频窗口：

```text
<dataset>/videos/observation.images.stereo/chunk-XXX/file-YYY.mp4
```

注意不要假设只有 `chunk-000`；LeRobot v3 数据集可能有多个 `chunk-*`。

### 2. lerobot v3 数据集

数据集需要有：

```text
data/chunk-*/file-*.parquet
videos/observation.images.stereo/chunk-*/file-*.mp4
meta/
```

示例占位路径：

```text
/cpfs/username/wuji-hand-teleop-data/<task>/<version-or-date>/lerobotv3/<robot-or-host>
```

### 3. 相机标定

真实画面 overlay 必须有 `camera.yaml` 或原始 `info.yaml`，否则只能猜相机外参，投影会偏。

`visualize_tianji` 仓库内的标定示例/默认配置目录：

```text
/cpfs/username/wuji-hand-teleop-private/test/visualize_tianji/config
```

使用时优先选择与采集机器、相机、分辨率匹配的 `camera*.yaml`；如果只有原始 `info.yaml`，也可以把它传给 `--calib`，工具会按自己的逻辑读取/转换。不要把某台机器的 `camera.yaml` 当成通用外参；借用其它机器外参只能用于跑通流程，不能据此严格判断模型动作对错。

## 常用命令

### 批量渲染 val 的模型预测 action

先把路径写成变量，避免把某个人的实验目录硬编码进命令：

```bash
USER_ROOT=/cpfs/username
VIS_DIR=$USER_ROOT/wuji-hand-teleop-private/test/visualize_tianji

# run_action_chunk_eval.py 输出的 predictions.npz
PRED_NPZ=$USER_ROOT/<path-to-eval-output>/<dataset>/predictions.npz

# 与 predictions.npz 对应的 LeRobot v3 数据集根目录
DATASET=$USER_ROOT/wuji-hand-teleop-data/<task>/<version-or-date>/lerobotv3/<dataset>

# 与采集机器/相机匹配的标定文件；也可以换成某个原始 info.yaml
CALIB=$VIS_DIR/config/camera_<host-or-robot>.yaml

# 输出目录，建议和 eval run 一一对应
OUT_DIR=$USER_ROOT/<path-to-eval-overlay>/<run>/<dataset>_pred
```

然后运行：

```bash
cd "$VIS_DIR"

uv run python -m tianji_vis.batch_render "$PRED_NPZ" \
  --dataset "$DATASET" \
  --calib "$CALIB" \
  --out-dir "$OUT_DIR" \
  --fps 6 \
  --trajectory
```

输出类似：

```text
pred_anim_s0_ep0.mp4
pred_anim_s1_ep1.mp4
...
```

### 快速单段 video + action 预览

如果已经有单段 `(N,54)` 的 `.npy/.npz/.parquet`：

```bash
USER_ROOT=/cpfs/username
VIS_DIR=$USER_ROOT/wuji-hand-teleop-private/test/visualize_tianji
VIDEO=/path/to/file-000.mp4
ACTION=/path/to/action.npy
CALIB=$VIS_DIR/config/camera_<host-or-robot>.yaml
OUT=$USER_ROOT/tmp/overlay_preview.mp4

cd "$VIS_DIR"

uv run python visualize_tianji.py \
  --video "$VIDEO" \
  --action "$ACTION" \
  --calib "$CALIB" \
  --out "$OUT" \
  --max-frames 80 \
  --fps 6 \
  --chunk 16 \
  --chunk-stride 16
```

### rosbag / mcap 原始数据可视化

如果原始数据同目录有 `info.yaml`：

```bash
USER_ROOT=/cpfs/username
VIS_DIR=$USER_ROOT/wuji-hand-teleop-private/test/visualize_tianji
MCAP=/path/to/rosbag_xxx_0.mcap

cd "$VIS_DIR"

uv run python visualize_tianji.py \
  --mcap "$MCAP" \
  --max-frames 300
```

工具会尝试从 `info.yaml` 生成/读取 `camera.yaml`。

## 参数选择

- 先用默认 skeleton 模式，不加 `--hand-mesh`，用于验证整体投影链路；
- 需要看手部包络时再加 `--hand-mesh`；
- 需要看模型接下来要往哪里走时加 `--trajectory`，批量模式会把整段 chunk 当作前瞻轨迹；
- `--fps 6` 适合快速看 action chunk，输出较小；
- `--scale-calib` 在单段 CLI 里用于视频被纯 resize 后缩放内参；批量模式内部已经用 `scale_calib=True`；
- `--max-frames` 适合 smoke 预览，避免一次渲染很久。

## 54D action 约定

`visualize_tianji` 期望 action 每帧是 54D：

```text
[左臂 7, 左手 20, 右臂 7, 右手 20]
```

单位跟数据集保持一致：臂关节通常按代码/数据约定使用角度或等价 motor command，手部为灵巧手关节量。不要在渲染前自行归一化；传给工具的应是反归一化后的 raw action。

## 结果判断

- 骨架 / 手 mesh 应大体贴住真实画面里的机械臂和灵巧手；
- 单侧整体偏移通常优先怀疑该侧手眼外参或相机标定；
- 两侧都整体偏移，优先检查用错 `camera.yaml`、视频分辨率与标定不一致、或数据集和 predictions.npz 不匹配；
- 若轨迹方向明显不合理，再结合 MSE、`pred_action` / `target_action` 曲线和 episode 编号定位模型问题。

## 常见问题

- **只有折线图，没有 3D overlay**：你看的可能是 `render_chunk_overlay.py` 输出，不是 `visualize_tianji`。
- **视频找不到**：确认 `--dataset` 指向 lerobot 数据集根目录，而不是 `videos/` 子目录。
- **投影偏很多**：确认 `--calib` 与机器/相机一致；不要拿不匹配机器的标定严格评判模型动作。
- **渲染很慢**：先去掉 `--hand-mesh`，保留 skeleton；减少 `--max-frames` 或只渲几个 sample。
- **工具依赖问题**：在 `visualize_tianji` 目录用 `uv run ...`，不要在 `wuji-vla-post-train` 根目录直接跑模块。

## 与 SFT val 的推荐工作流

1. 在 `wuji-vla-post-train` 里跑 `scripts/benchmark/run_action_chunk_eval.py`，得到 `predictions.npz` 和 `metrics.json`；
2. 在 `wuji-hand-teleop-private/test/visualize_tianji` 里用 `tianji_vis.batch_render` 渲染同一个 `predictions.npz`；
3. 先看 metrics 里 raw / normalized MSE 和 arm/hand segment，再看对应 `pred_anim_s*_ep*.mp4`；
4. 输出路径放到 `/cpfs/username/.../eval_overlay/<run>/<dataset>_pred`，保持和 eval 目录一一对应，方便复现。