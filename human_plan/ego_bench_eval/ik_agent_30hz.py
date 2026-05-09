import argparse
import builtins
import contextlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tqdm

from collections import deque

import numpy as np

if not hasattr(builtins, "ParallelismConfig"):
    class ParallelismConfig:  # pragma: no cover - compatibility shim
        pass

    builtins.ParallelismConfig = ParallelismConfig

# IsaacLab 5.x exposes the package as `isaaclab`/`isaaclab_tasks` instead of
# `omni.isaac.lab`/`omni.isaac.lab_tasks`. We alias the commonly used modules so
# the benchmark code can keep its older import paths.
def _alias_module(old_name: str, new_name: str):
    with contextlib.suppress(ModuleNotFoundError, AttributeError):
        module = importlib.import_module(new_name)
        sys.modules[old_name] = module

        if "." in old_name:
            parent_name, child_name = old_name.rsplit(".", 1)
            parent_module = sys.modules.get(parent_name)
            if parent_module is None:
                with contextlib.suppress(ModuleNotFoundError):
                    parent_module = importlib.import_module(parent_name)
            if parent_module is not None:
                setattr(parent_module, child_name, module)


def _ensure_isaaclab_aliases():
    aliases = (
        ("omni.isaac.lab", "isaaclab"),
        ("omni.isaac.lab.utils", "isaaclab.utils"),
        ("omni.isaac.lab.utils.math", "isaaclab.utils.math"),
        ("omni.isaac.lab.utils.sensors", "isaaclab.utils.sensors"),
        ("omni.isaac.lab.sim", "isaaclab.sim"),
        ("omni.isaac.lab.controllers", "isaaclab.controllers"),
        ("omni.isaac.lab.managers", "isaaclab.managers"),
        ("omni.isaac.lab.managers.scene_entity_cfg", "isaaclab.managers.scene_entity_cfg"),
        ("omni.isaac.lab.assets", "isaaclab.assets"),
        ("omni.isaac.lab.assets.articulation", "isaaclab.assets.articulation"),
        (
            "omni.isaac.lab.assets.articulation.articulation_cfg",
            "isaaclab.assets.articulation.articulation_cfg",
        ),
        ("omni.isaac.lab.actuators", "isaaclab.actuators"),
        ("omni.isaac.lab_tasks", "isaaclab_tasks"),
        ("omni.isaac.lab_tasks.utils", "isaaclab_tasks.utils"),
    )
    for old_name, new_name in aliases:
        _alias_module(old_name, new_name)


_ensure_isaaclab_aliases()

from omni.isaac.lab.app import AppLauncher

# We fix the seed for tasks to make sure the object position during evaluation
# are never seen during training.
seed_map = {
    "Humanoid-Pour-Balls-v0": 0,
    "Humanoid-Sort-Cans-v0": 1,
    "Humanoid-Insert-Cans-v0": 2,
    "Humanoid-Unload-Cans-v0": 3,
    "Humanoid-Insert-And-Unload-Cans-v0": 4,
    "Humanoid-Push-Box-v0": 5,
    "Humanoid-Open-Drawer-v0": 6,
    "Humanoid-Close-Drawer-v0": 7,
    "Humanoid-Open-Laptop-v0": 8,
    "Humanoid-Flip-Mug-v0": 9,
    "Humanoid-Stack-Can-v0": 10,
    "Humanoid-Stack-Can-Into-Drawer-v0": 11,
}

# Launch the simulator before importing heavyweight ML deps such as transformers.
app_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(app_parser)
app_parser.set_defaults(headless=True, enable_cameras=True, device="cuda")
app_args, _ = app_parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(app_args)
simulation_app = app_launcher.app
_ensure_isaaclab_aliases()

from transformers import HfArgumentParser
from human_plan.vila_train.args import (
  VLATrainingArguments, VLAModelArguments, VLADataArguments
)

parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
# add argparse arguments
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--room_idx", type=int, default=None, help="Room Idx")
parser.add_argument("--table_idx", type=int, default=None, help="Table Idx")
parser.add_argument("--smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--hand_smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--num_episodes", type=int, default=None, help="episode_label")
parser.add_argument("--num_trials", type=int, default=None, help="trial label")
parser.add_argument("--episode_start_idx", type=int, default=0, help="start index into TASK_INIT_EPISODE")
parser.add_argument("--trial_start_idx", type=int, default=0, help="first trial label to execute")
parser.add_argument("--randomize_total_episodes", type=int, default=None, help="total episode count used for stable randomize_idx slicing")
parser.add_argument("--randomize_total_trials", type=int, default=None, help="total trial count used for stable randomize_idx slicing")
parser.add_argument("--result_saving_path", type=str, default=None, help="result saving path")
parser.add_argument("--video_saving_path", type=str, default=None, help="video saving path")
parser.add_argument("--save_video", type=int, default=1, help="save episode mp4 video")
parser.add_argument("--save_frames", type=int, default=0, help="result saving path")
parser.add_argument("--project_trajs", type=int, default=0, help="result saving path")
parser.add_argument("--additional_label", type=str, default=None, help="additional_label")
parser.add_argument(
    "--max_eval_steps",
    type=int,
    default=None,
    help="Optional eval-only smoke-test cap. None or <=0 keeps the task horizon unchanged.",
)
parser.add_argument(
    "--chunk_exec_len",
    type=str,
    default=None,
    help="Eval-only ablation. Kept for logging; current eval executes one action per model query.",
)
parser.add_argument(
    "--image_update_interval",
    type=str,
    default=None,
    help="Eval-only image ablation. 1/none=normal, K>1=refresh cached image every K env steps, <=0/inf=fixed reset image.",
)
parser.add_argument(
    "--image_delay_steps",
    type=int,
    default=0,
    help="Eval-only image ablation. 0=normal, k>0 feeds the image from k env steps earlier.",
)
parser.add_argument(
    "--proprio_ablation_mode",
    type=str,
    default="none",
    choices=("none", "freeze", "delay"),
    help="Eval-only proprio ablation. none=normal, freeze=reset proprio, delay=k-step old proprio.",
)
parser.add_argument(
    "--proprio_delay_steps",
    type=int,
    default=0,
    help="Delay in eval env steps when --proprio_ablation_mode=delay.",
)
parser.add_argument(
    "--eval_ablation_debug",
    action="store_true",
    help="Print per-step debug information for eval-only ablations.",
)
parser.add_argument(
    "--vision_input_mode",
    type=str,
    default="real",
    choices=("real", "noise", "initial"),
    help="Image source used by the model during evaluation.",
)
parser.add_argument(
    "--rl_mode",
    type=str,
    default="off",
    choices=("off", "identity", "actor"),
    help=(
        "Offline-RL correction mode. 'identity' routes through the RL insertion path "
        "with a_exec=a_ref; 'actor' applies a TD3+BC checkpoint after smoothing."
    ),
)
parser.add_argument(
    "--rl_actor_checkpoint",
    type=str,
    default=None,
    help="TD3+BC actor checkpoint used when --rl_mode=actor.",
)
parser.add_argument(
    "--rl_action_trace",
    action="store_true",
    help="Print Stage-0 action path tracing logs at the post-smoothing RL insertion point.",
)
parser.add_argument(
    "--rl_action_trace_steps",
    type=int,
    default=2,
    help="Number of eval steps to print Stage-0 action trace logs for.",
)
parser.add_argument(
    "--rl_identity_tolerance",
    type=float,
    default=1.0e-5,
    help="Max allowed pack/unpack identity error before raising in identity mode.",
)
parser.add_argument(
    "--rl_collect_replay_path",
    type=str,
    default=None,
    help="Optional Stage-2 replay path. When set, saves base/identity transitions as .npz.",
)
parser.add_argument(
    "--rl_collect_source",
    type=str,
    default="base",
    choices=("base", "identity"),
    help="Replay source label for Stage-2 base collection.",
)
parser.add_argument(
    "--rl_collect_save_raw",
    action="store_true",
    help="Also save a small raw/debug sidecar next to the fast replay .npz.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

from human_plan.ego_bench_eval.utils import (
    process_input,
    ik_step,
    ik_eval_single_step,
    get_language_instruction,
    sanitize_task_success,
    update_eval_success,
)
import gymnasium as gym
import torch

try:
    from omni.isaac.lab_tasks.utils import parse_env_cfg
except ModuleNotFoundError:
    from isaaclab_tasks.utils import parse_env_cfg
import torch

from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.markers import VisualizationMarkers
# from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.lab.utils.math import subtract_frame_transforms
from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

import cv2
from human_plan.vila_eval.utils.load_model import load_model_eval
from rl_posttrain.action_utils import (
    ActionSpec,
    array_stats,
    format_action_spec,
    format_stats,
    make_exec_action_dict,
    max_abs_action_diff,
    mean_abs_action_diff,
    pack_action,
    shape_summary,
    unpack_action,
)
from rl_posttrain.normalizer import AffineNormalizer
from rl_posttrain.feature_hooks import register_traj_decoder_input_hook
from rl_posttrain.obs_utils import (
    build_actor_obs,
    build_critic_obs,
    subtask_success_metrics,
)
from rl_posttrain.replay_buffer import ReplayBufferWriter
from rl_posttrain.td3bc_ref import load_actor_policy


def _get_action_dim(env: BaseEnv) -> int:
    """Compat helper for IsaacLab API changes around action dimensions."""
    if hasattr(env, "num_actions"):
        return int(env.num_actions)

    action_space = getattr(env, "action_space", None)
    if action_space is not None and getattr(action_space, "shape", None):
        return int(math.prod(action_space.shape))

    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        if getattr(cfg, "action_space", None) is not None:
            action_space = cfg.action_space
            if isinstance(action_space, int):
                return int(action_space)
            if isinstance(action_space, (tuple, list)):
                return int(math.prod(action_space))
        if getattr(cfg, "num_actions", None) is not None:
            return int(cfg.num_actions)

    raise AttributeError("Unable to infer action dimension from environment.")


def _make_non_overwriting_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    stem, suffix = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _convert_video_to_h264(output_path: str) -> None:
    """Convert OpenCV-written MP4 to broadly compatible H.264/AVC."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[video] ffmpeg not found; keeping original video: {output_path}")
        return

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"[video] skip H.264 conversion because video is missing or empty: {output_path}")
        return

    stem, suffix = os.path.splitext(output_path)
    temp_output_path = _make_non_overwriting_path(f"{stem}.h264_tmp{suffix or '.mp4'}")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        output_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        temp_output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        with contextlib.suppress(OSError):
            os.remove(temp_output_path)
        stderr = getattr(exc, "stderr", "") or str(exc)
        print(f"[video] H.264 conversion failed; keeping original video: {output_path}\n{stderr}")
        return

    os.replace(temp_output_path, output_path)
    print(f"[video] converted to H.264/AVC libx264 yuv420p without audio: {output_path}")


def _apply_task_cfg_overrides(task_name: str, env_cfg: BaseEnvCfg) -> None:
    # IsaacLab 5 validates articulation defaults during environment creation.
    # The benchmark's laptop asset uses a legacy default joint position (0.088)
    # that falls below the current lower joint limit (0.175), so we bump it into
    # the valid range before gym.make() instantiates the task.
    if task_name == "Humanoid-Open-Laptop-v0":
        env_cfg.laptop.init_state.joint_pos = {".*joint": 0.18}
    elif task_name == "Humanoid-Stack-Can-Into-Drawer-v0":
        # This task widens both shoulder yaw joints to 1.5 to avoid drawer-door
        # collisions, but IsaacLab 5 enforces an upper limit of 1.3 during
        # articulation initialization. Keep the shoulders opened up while
        # staying safely inside the current joint limits.
        env_cfg.robot.init_state.joint_pos[".*_shoulder_yaw_joint"] = 1.25


def _get_model_rgb_obs(rgb_obs, vision_input_mode: str, initial_rgb_obs, rng):
    if vision_input_mode == "real":
        return rgb_obs.copy()
    if vision_input_mode == "initial":
        return initial_rgb_obs.copy()
    if vision_input_mode == "noise":
        return rng.integers(0, 256, size=rgb_obs.shape, dtype=rgb_obs.dtype)
    raise ValueError(f"Unsupported vision_input_mode: {vision_input_mode}")


def _parse_optional_positive_int(value, name: str):
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    value_str = str(value).strip().lower()
    if value_str in ("", "none", "null"):
        return None
    parsed = int(value_str)
    return parsed if parsed > 0 else None


def _parse_image_update_interval(value):
    if value is None:
        return 1
    value_str = str(value).strip().lower()
    if value_str in ("", "none", "null"):
        return 1
    if value_str in ("inf", "infinite", "infinity"):
        return 0
    parsed = int(value_str)
    return parsed


def _copy_image_like(image):
    if hasattr(image, "detach"):
        return image.detach().clone()
    return image.copy()


def _copy_value(value):
    if hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def _copy_proprio_pack(proprio_pack):
    return {key: _copy_value(value) for key, value in proprio_pack.items()}


def _image_mean_abs_diff(a, b):
    if hasattr(a, "detach"):
        return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _proprio_mean_abs_diff(current_pack, used_pack):
    current = current_pack["proprio_input"]
    used = used_pack["proprio_input"]
    if hasattr(current, "detach"):
        return (current.to(torch.float32) - used.to(torch.float32)).abs().mean().item()
    return float(np.mean(np.abs(current.astype(np.float32) - used.astype(np.float32))))


class ImageAblationBuffer:
    """Eval-only image freeze/update/delay state for the model RGB input."""

    def __init__(self, image_update_interval=1, image_delay_steps=0, debug=False):
        self.image_update_interval = _parse_image_update_interval(image_update_interval)
        self.image_delay_steps = int(image_delay_steps or 0)
        self.debug = debug
        self.cached_image = None
        self.history = []
        self.step = 0
        self.diff_sum = 0.0
        self.diff_count = 0

        update_active = self.image_update_interval <= 0 or self.image_update_interval > 1
        if update_active and self.image_delay_steps > 0:
            raise ValueError(
                "image_update_interval and image_delay_steps are mutually exclusive for eval ablations. "
                "Please enable only one so the experiment remains interpretable."
            )
        if self.image_delay_steps < 0:
            raise ValueError("image_delay_steps must be >= 0.")

    @property
    def enabled(self):
        return (
            self.image_delay_steps > 0
            or self.image_update_interval <= 0
            or self.image_update_interval > 1
        )

    @property
    def mode(self):
        if self.image_delay_steps > 0:
            return f"delay_{self.image_delay_steps}"
        if self.image_update_interval <= 0:
            return "fixed_initial"
        if self.image_update_interval > 1:
            return f"update_every_{self.image_update_interval}"
        return "normal"

    def reset(self, image):
        image_copy = _copy_image_like(image)
        self.cached_image = image_copy
        self.history = [image_copy]
        self.step = 0
        self.diff_sum = 0.0
        self.diff_count = 0

    def apply(self, image):
        if self.cached_image is None:
            self.reset(image)

        current = _copy_image_like(image)
        used = current
        source = "current"

        if self.image_delay_steps > 0:
            self.history.append(current)
            used_index = max(0, len(self.history) - 1 - self.image_delay_steps)
            used = self.history[used_index]
            source = f"history[{used_index}]"
        elif self.image_update_interval <= 0:
            used = self.cached_image
            source = "reset"
        elif self.image_update_interval > 1:
            if self.step % self.image_update_interval == 0:
                self.cached_image = current
                source = "updated"
            else:
                source = "cached"
            used = self.cached_image

        used = _copy_image_like(used)
        diff = _image_mean_abs_diff(current, used)
        self.diff_sum += diff
        self.diff_count += 1
        if self.debug:
            print(
                f"[eval-ablation:image] step={self.step} mode={self.mode} source={source} "
                f"mean_abs_diff={diff:.6f}"
            )
        self.step += 1
        return used

    def mean_abs_diff(self):
        if self.diff_count == 0:
            return 0.0
        return self.diff_sum / self.diff_count


class ProprioAblationBuffer:
    """Eval-only proprio freeze/delay state for the model proprio input."""

    def __init__(self, mode="none", delay_steps=0, debug=False):
        self.mode = mode
        self.delay_steps = int(delay_steps or 0)
        self.debug = debug
        self.initial_proprio = None
        self.history = []
        self.step = 0
        self.diff_sum = 0.0
        self.diff_count = 0

        if self.mode not in ("none", "freeze", "delay"):
            raise ValueError(f"Unsupported proprio_ablation_mode: {self.mode}")
        if self.delay_steps < 0:
            raise ValueError("proprio_delay_steps must be >= 0.")
        if self.mode == "delay" and self.delay_steps <= 0:
            raise ValueError("proprio_ablation_mode='delay' requires proprio_delay_steps > 0.")

    @property
    def enabled(self):
        return self.mode != "none"

    def reset(self, proprio_pack):
        pack_copy = _copy_proprio_pack(proprio_pack)
        self.initial_proprio = pack_copy
        self.history = [pack_copy]
        self.step = 0
        self.diff_sum = 0.0
        self.diff_count = 0

    def apply(self, proprio_pack):
        if "proprio_input" not in proprio_pack:
            raise KeyError(
                "Eval proprio ablation expected a 'proprio_input' key in the processed proprio pack, "
                f"but available keys are: {sorted(proprio_pack.keys())}"
            )
        if self.mode == "none":
            return proprio_pack
        if self.initial_proprio is None:
            self.reset(proprio_pack)

        current = _copy_proprio_pack(proprio_pack)
        used = current
        source = "current"

        if self.mode == "freeze":
            used = self.initial_proprio
            source = "reset"
        elif self.mode == "delay":
            self.history.append(current)
            used_index = max(0, len(self.history) - 1 - self.delay_steps)
            used = self.history[used_index]
            source = f"history[{used_index}]"

        used = _copy_proprio_pack(used)
        diff = _proprio_mean_abs_diff(current, used)
        self.diff_sum += diff
        self.diff_count += 1
        if self.debug:
            print(
                f"[eval-ablation:proprio] step={self.step} mode={self.mode} source={source} "
                f"mean_abs_diff={diff:.6f}"
            )
        self.step += 1
        return used

    def mean_abs_diff(self):
        if self.diff_count == 0:
            return 0.0
        return self.diff_sum / self.diff_count


PROPRIO_ENV_KEYS = (
    "left_finger_tip_pos",
    "right_finger_tip_pos",
    "left_ee_pose",
    "right_ee_pose",
    "qpos",
)


PROPRIO_MODEL_KEYS = (
    "proprio_input",
    "proprio_input_2d",
    "proprio_input_3d",
    "proprio_input_rot",
    "proprio_input_handdof",
    "proprio_input_hand_finger_tip",
)


def _assert_proprio_env_keys(env_obs):
    missing_keys = [key for key in PROPRIO_ENV_KEYS if key not in env_obs]
    if missing_keys:
        raise KeyError(
            "Eval proprio ablation could not find required proprio env observation keys. "
            f"missing={missing_keys}, available={sorted(env_obs.keys())}"
        )


def _make_proprio_pack(proprio_input, raw_proprio_inputs):
    proprio_pack = {"proprio_input": proprio_input}
    proprio_pack.update(raw_proprio_inputs)
    missing_model_keys = [key for key in PROPRIO_MODEL_KEYS if key not in proprio_pack]
    if missing_model_keys:
        raise KeyError(
            "Eval proprio ablation could not find required processed proprio keys. "
            f"missing={missing_model_keys}, available={sorted(proprio_pack.keys())}"
        )
    return proprio_pack


def _raw_proprio_from_pack(proprio_pack):
    return {key: value for key, value in proprio_pack.items() if key != "proprio_input"}


def _format_action_shape(action_dict):
    return {
        key: tuple(value.shape)
        for key, value in action_dict.items()
        if key in ("left_ee_pose", "right_ee_pose", "left_qpos_multi_step", "right_qpos_multi_step")
        and hasattr(value, "shape")
    }


def _record_trace_line(trace_lines, line: str) -> None:
    print(line, flush=True)
    trace_lines.append(line)


def _finalize_replay_transition(replay_writer, pending, next_actor_obs, next_critic_obs, next_bc_target_raw):
    transition = {
        "actor_obs": pending["actor_obs"],
        "critic_obs": pending["critic_obs"],
        "action_raw": pending["action_raw"],
        "bc_target_raw": pending["bc_target_raw"],
        "reward": np.asarray([pending["reward"]], dtype=np.float32),
        "done": np.asarray([pending["done"]], dtype=np.float32),
        "next_actor_obs": next_actor_obs,
        "next_critic_obs": next_critic_obs,
        "next_bc_target_raw": next_bc_target_raw,
        "source": pending["source"],
        "success": np.asarray([pending["success"]], dtype=np.float32),
        "timeout": np.asarray([pending["timeout"]], dtype=np.float32),
    }
    for optional in ("episode_id", "episode_step", "episode_length", "episode_success", "episode_timeout"):
      if optional in pending:
        transition[optional] = np.asarray([pending[optional]], dtype=np.float32)
    raw = pending.get("raw")
    replay_writer.add(transition, raw=raw)

def main():

    model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()
    chunk_exec_len = _parse_optional_positive_int(task_args.chunk_exec_len, "chunk_exec_len")
    image_update_interval = _parse_image_update_interval(task_args.image_update_interval)
    image_delay_steps = int(task_args.image_delay_steps or 0)
    proprio_ablation_mode = task_args.proprio_ablation_mode
    proprio_delay_steps = int(task_args.proprio_delay_steps or 0)
    rl_mode = task_args.rl_mode
    rl_actor_checkpoint = task_args.rl_actor_checkpoint
    rl_actor_enabled = rl_mode == "actor"
    rl_action_trace = bool(task_args.rl_action_trace)
    rl_action_trace_steps = max(0, int(task_args.rl_action_trace_steps or 0))
    rl_identity_tolerance = float(task_args.rl_identity_tolerance)
    rl_collect_replay_path = task_args.rl_collect_replay_path
    rl_collect_enabled = rl_collect_replay_path is not None and str(rl_collect_replay_path).strip() != ""
    rl_collect_source = task_args.rl_collect_source
    rl_collect_save_raw = bool(task_args.rl_collect_save_raw)
    rl_insert_enabled = rl_mode != "off" or rl_action_trace or rl_collect_enabled
    image_update_active = image_update_interval <= 0 or image_update_interval > 1
    if image_update_active and image_delay_steps > 0:
      raise ValueError(
        "image_update_interval and image_delay_steps are mutually exclusive for eval ablations. "
        "Please enable only one so the experiment remains interpretable."
      )
    if image_delay_steps < 0:
      raise ValueError("image_delay_steps must be >= 0.")
    if proprio_delay_steps < 0:
      raise ValueError("proprio_delay_steps must be >= 0.")
    if proprio_ablation_mode == "delay" and proprio_delay_steps <= 0:
      raise ValueError("proprio_ablation_mode='delay' requires proprio_delay_steps > 0.")
    if rl_identity_tolerance < 0:
      raise ValueError("rl_identity_tolerance must be >= 0.")
    if rl_collect_enabled and rl_collect_source not in ("base", "identity"):
      raise ValueError("rl_collect_source must be 'base' or 'identity'.")
    if rl_actor_enabled and (rl_actor_checkpoint is None or str(rl_actor_checkpoint).strip() == ""):
      raise ValueError("--rl_actor_checkpoint is required when --rl_mode=actor.")
    if rl_actor_enabled and rl_collect_enabled:
      raise ValueError(
        "Stage-2 base replay collection only supports rl_mode=off/identity. "
        "Refusing to mix actor-generated data into the base replay."
      )

    model, tokenizer, model_args, data_args, training_args = load_model_eval(
      model_args, data_args, training_args
    )
    model.to("cuda")
    model.eval()
    rl_actor_bundle = None
    rl_actor = None
    rl_actor_obs_normalizer = None
    if rl_actor_enabled:
      rl_actor_bundle = load_actor_policy(rl_actor_checkpoint, device="cuda")
      rl_actor = rl_actor_bundle["actor"]
      rl_actor_obs_normalizer = rl_actor_bundle["actor_obs_normalizer"]
      print(
        "[rl-actor] "
        f"loaded_checkpoint={rl_actor_checkpoint} "
        f"actor_obs_dim={rl_actor_bundle['checkpoint']['actor_obs_dim']} "
        f"action_dim={rl_actor_bundle['checkpoint']['action_dim']}"
      )
    if rl_collect_enabled or rl_actor_enabled:
      for param in model.parameters():
        param.requires_grad = False
      rl_feature_capture, rl_feature_handle = register_traj_decoder_input_hook(model)
    else:
      rl_feature_capture, rl_feature_handle = None, None

    data_args.sep_query_token = model_args.sep_query_token

    import random
    assert task_args.task in seed_map
    print(f"Setting seed to {seed_map[task_args.task]}")
    random.seed(seed_map[task_args.task])
    randomize_idxes = list(range(10000))
    random.shuffle(randomize_idxes)

    # train data are using idxes 0-49, test start from 50
    set_selection = "Test"
    if set_selection == "Train":
      # curr_random_idx = 0 + task_args.room_idx * task_args.num_trials * task_args.num_episodes
      assert False
    elif set_selection == "Test":
      # We used the first 100 random idxes for training
      # Starting from 101th random ides for evaluation
      randomize_total_episodes = task_args.randomize_total_episodes or task_args.num_episodes
      randomize_total_trials = task_args.randomize_total_trials or task_args.num_trials
      randomize_slice_offset = (
        int(task_args.episode_start_idx) * int(randomize_total_trials)
        + int(task_args.trial_start_idx)
      )
      curr_random_idx = 100 + (
        task_args.room_idx * 5 + task_args.table_idx
      )  * int(randomize_total_trials) * int(randomize_total_episodes)
      curr_random_idx += randomize_slice_offset

    # parse configuration
    env_cfg: BaseEnvCfg = parse_env_cfg(
        task_args.task,
        num_envs=1,
    )

    env_cfg.episode_length_s = 60 # 60 seconds episode length -> For long horizon tasks
    env_cfg.randomize = True
    _apply_task_cfg_overrides(task_args.task, env_cfg)
    # create environment
    env_cfg.spawn_background =True
    # select background
    room_idx = task_args.room_idx
    table_idx = task_args.table_idx
    env_cfg.room_idx = room_idx
    env_cfg.table_idx = table_idx
    env: BaseEnv = gym.make(
        task_args.task,
        cfg=env_cfg
    )
    base_env: BaseEnv = env.unwrapped

    base_env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
    env.reset()

    # IK controllers
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(
        left_ik_cfg, num_envs=base_env.scene.num_envs, device=base_env.sim.device
    )
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="pinv")
    right_ik_controller = DifferentialIKController(
        right_ik_cfg, num_envs=base_env.scene.num_envs, device=base_env.sim.device
    )

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(
        base_env.scene.num_envs, left_ik_controller.action_dim, device=base_env.robot.device
    )
    left_ik_commands_robot = torch.zeros(
        base_env.scene.num_envs, left_ik_controller.action_dim, device=base_env.robot.device
    )
    right_ik_commands_world = torch.zeros(
        base_env.scene.num_envs, right_ik_controller.action_dim, device=base_env.robot.device
    )
    right_ik_commands_robot = torch.zeros(
        base_env.scene.num_envs, right_ik_controller.action_dim, device=base_env.robot.device
    )
    action_dim = _get_action_dim(base_env)
    action = torch.zeros((base_env.scene.num_envs, action_dim), device=base_env.robot.device)

    save_path = os.path.join(
      task_args.video_saving_path,
      task_args.additional_label,
      f"inference_{task_args.smooth_weight}_{task_args.hand_smooth_weight}"
    )

    from pathlib import Path
    Path(save_path).mkdir(exist_ok=True, parents=True)

    import pickle
    with open("init_poses_fixed_set_100traj.pkl", "rb") as f:
       init_poses = pickle.load(f)

    task_name = task_args.task[9:-3]
    load_name = task_name

    # Collect Initial Hand and EE poses from data -> Only set the arm and hand for start
    from human_plan.ego_bench_eval.utils import TASK_INIT_EPISODE
    episode_start_idx = int(task_args.episode_start_idx)
    episode_end_idx = episode_start_idx + int(task_args.num_episodes)
    episode_list = TASK_INIT_EPISODE[task_name][episode_start_idx:episode_end_idx]

    hist_len = data_args.predict_future_step * data_args.future_index

    import numpy as np
    cam_intrinsics = np.array([
      [488.6662,   0.0000, 640.0000],
      [  0.0000, 488.6662, 360.0000],
      [  0.0000,   0.0000,   1.0000]
    ])

    padding = 0
    vision_rng = np.random.default_rng(seed_map[task_args.task])
    print(
      "[eval-ablation] "
      f"chunk_exec_len={chunk_exec_len} "
      f"image_update_interval={image_update_interval} "
      f"image_delay_steps={image_delay_steps} "
      f"proprio_ablation_mode={proprio_ablation_mode} "
      f"proprio_delay_steps={proprio_delay_steps} "
      f"rl_mode={rl_mode} "
      f"rl_action_trace={rl_action_trace} "
      f"rl_collect_enabled={rl_collect_enabled} "
      f"vision_input_mode={task_args.vision_input_mode}"
    )
    print(
      "[eval-ablation] image source key: env_results[0]['fixed_rgb'] -> raw_data_dict['image']; "
      "current eval executes one env.step per model query."
    )
    print(
      "[eval-ablation] proprio source keys: "
      f"env={PROPRIO_ENV_KEYS} -> raw_data_dict['proprio_input'] and raw proprio branches."
    )
    if chunk_exec_len is not None:
      print(
        "[eval-ablation:chunk] current eval is already single-step/receding-horizon execution; "
        "chunk_exec_len is recorded as a no-op instead of changing rollout semantics."
      )
    if rl_insert_enabled:
      print(
        "[rl-posttrain] insertion point: after temporal smoothing and before ik_step; "
        "identity sets a_exec = a_ref; actor mode uses normalized TD3+BC output."
      )
    if rl_collect_enabled:
      print(
        "[rl-collect] "
        f"path={rl_collect_replay_path} source={rl_collect_source} "
        "bc_target=canonical_action_norm_from_a_ref_raw "
        "action_normalizer=fit_minmax_on_collected_bc_target_raw "
        "reward=sparse_final_success"
      )

    replay_writer = None
    rl_optional_warned = set()
    if rl_collect_enabled:
      replay_writer = ReplayBufferWriter(
        rl_collect_replay_path,
        metadata={
          "task": task_args.task,
          "room_idx": room_idx,
          "table_idx": table_idx,
          "source": rl_collect_source,
          "actor_insert_point": "after_temporal_smoothing",
          "bc_target": "canonical_action_norm_from_a_ref_raw",
          "action_normalizer_mode": "fit_minmax_on_collected_bc_target_raw",
          "reward": "sparse_final_success",
        },
        save_raw=rl_collect_save_raw,
      )

    rl_episode_counter = 0

    # with torch.inference_mode():
    for episode_idx in episode_list:
      for trial_idx in range(int(task_args.trial_start_idx), int(task_args.trial_start_idx) + int(task_args.num_trials)):
        # seq_name = f"episode_{episode_idx}.hdf5"
        seq_name = episode_idx[0]
        rl_episode_counter += 1
        rl_episode_id = rl_episode_counter

        # 30 Hz
        rgb_obs_hist = deque(maxlen=120)
        # original video is 15fps and env is 30 fps
        action_hist_left_ee = deque(maxlen=hist_len)
        action_hist_right_ee = deque(maxlen=hist_len)
        action_hist_left_hand = deque(maxlen=hist_len)
        action_hist_right_hand = deque(maxlen=hist_len)
        image_ablation = ImageAblationBuffer(
          image_update_interval=image_update_interval,
          image_delay_steps=image_delay_steps,
          debug=task_args.eval_ablation_debug,
        )
        proprio_ablation = ProprioAblationBuffer(
          mode=proprio_ablation_mode,
          delay_steps=proprio_delay_steps,
          debug=task_args.eval_ablation_debug,
        )
        action_norm_sum = 0.0
        action_delta_norm_sum = 0.0
        action_stat_count = 0
        prev_action_for_stats = None
        logged_action_shape = False
        logged_proprio_shape = False
        action_chunk_horizon = None
        action_exec_len = 1
        rl_action_spec = None
        rl_action_normalizer = None
        rl_actor_action_dim = 0
        rl_identity_max_abs_diff = 0.0
        rl_identity_mean_abs_diff_sum = 0.0
        rl_identity_diff_count = 0
        rl_trace_count = 0
        rl_trace_lines = []
        rl_pending_transition = None
        rl_logged_obs_shapes = False
        rl_logged_actor_obs_shapes = False
        rl_actor_ref_mean_abs_sum = 0.0
        rl_actor_ref_max_abs = 0.0
        rl_actor_stat_count = 0
        rl_actor_num_clipped_dims = 0
        rl_ref_num_clipped_dims = 0

        seq_save_path = os.path.join(
          save_path,
          task_name,
          f"room_{room_idx}",
          f"table_{table_idx}",
        )
        from pathlib import Path
        Path(seq_save_path).mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(
          seq_save_path,
          f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}.mp4"
        )
        output_path = _make_non_overwriting_path(output_path)
        if task_args.save_frames:
          frames_output_path = os.path.join(
            seq_save_path,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}"
          )
          frames_output_path = _make_non_overwriting_path(frames_output_path)
          Path(frames_output_path).mkdir(exist_ok=True, parents=True)
        out = None
        if task_args.save_video:
          fps = 15
          out = cv2.VideoWriter(
            output_path,
            #  seq_save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (1280, 720)
          )

        # def init_env():
        if True:
            # reset
          curr_random_idx += 1
          base_env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
          env_results = env.reset()
          env_results = sanitize_task_success(task_args.task, base_env, env_results)
          left_ik_controller.reset()
          right_ik_controller.reset()
          padding_idx = padding
          # for padding_idx in range(padding):

          left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
          right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

          for idx in range(100):
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["left_ee"]
            right_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["right_ee"]

            ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world, 
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              left_ee_pose_traj_gt, right_ee_pose_traj_gt,
              left_dof, right_dof,
              action
            )
            env_results = env.step(action)
            rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            rgb_obs = cv2.resize(rgb_obs, (384, 384))
        env_results = sanitize_task_success(task_args.task, base_env, env_results)
        initial_rgb_obs = rgb_obs.copy()
        initial_model_rgb_obs = _get_model_rgb_obs(
          rgb_obs, task_args.vision_input_mode, initial_rgb_obs, vision_rng
        )
        image_ablation.reset(initial_model_rgb_obs)
        rgb_obs_hist.append(
          image_ablation.apply(initial_model_rgb_obs)
        )
        count = padding

        result = False
        success_streak = 0
        from human_plan.ego_bench_eval.utils import TASK_MAX_HORIZON
        max_horizon = TASK_MAX_HORIZON[task_args.task]
        if task_args.max_eval_steps is not None and task_args.max_eval_steps > 0:
          max_horizon = min(max_horizon, task_args.max_eval_steps)

        for i in tqdm.tqdm(range(max_horizon)):
          # run everything in inference mode
          # obtain quantities from simulation
          env_results = sanitize_task_success(task_args.task, base_env, env_results)
          _assert_proprio_env_keys(env_results[0])
          rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]

          from human_plan.ego_bench_eval.utils import process_proprio_input

          proprio_input, raw_proprio_inputs = process_proprio_input(
            env_results[0]["left_finger_tip_pos"].cpu().numpy(),
            env_results[0]["right_finger_tip_pos"].cpu().numpy(),
            env_results[0]["left_ee_pose"].cpu().numpy(),
            env_results[0]["right_ee_pose"].cpu().numpy(),
            env_results[0]["qpos"],
            cam_intrinsics,
            input_hand_dof=data_args.input_hand_dof
          )
          proprio_pack = _make_proprio_pack(proprio_input, raw_proprio_inputs)
          used_proprio_pack = proprio_ablation.apply(proprio_pack)
          used_proprio_input = used_proprio_pack["proprio_input"]
          used_raw_proprio_inputs = _raw_proprio_from_pack(used_proprio_pack)

          rgb_obs = cv2.resize(rgb_obs, (384, 384))
          model_rgb_obs = _get_model_rgb_obs(
            rgb_obs, task_args.vision_input_mode, initial_rgb_obs, vision_rng
          )
          rgb_obs_hist.append(
            image_ablation.apply(model_rgb_obs)
          )

          raw_language_instruction = get_language_instruction(
              task_args.task
          )

          raw_data_dict = process_input(
              rgb_obs_hist, 
              used_proprio_input.to("cuda"),
              raw_language_instruction,
              data_args, model_args, tokenizer
          )
          if "image" not in raw_data_dict:
            raise KeyError(
              "Eval image ablation expected process_input() to produce raw_data_dict['image'], "
              f"but available keys are: {sorted(raw_data_dict.keys())}"
            )

          raw_data_dict.update(used_raw_proprio_inputs)
          with torch.inference_mode():
            action_dict = ik_eval_single_step(
                raw_data_dict,
                model, tokenizer,
            )

          if action_chunk_horizon is None:
            action_chunk_horizon = int(action_dict["left_ee_pose"].shape[0])
            requested_exec_len = action_chunk_horizon if chunk_exec_len is None else min(
              chunk_exec_len, action_chunk_horizon
            )
            print(
              "[eval-ablation:chunk] "
              f"action_chunk_shape={_format_action_shape(action_dict)} "
              f"horizon_H={action_chunk_horizon} requested_m={requested_exec_len} "
              f"actual_env_steps_per_query={action_exec_len}"
            )
          if not logged_action_shape:
            print(f"[eval-ablation:image] model image tensor key='image' shape={tuple(raw_data_dict['image'].shape)}")
            logged_action_shape = True
          if not logged_proprio_shape:
            print(
              "[eval-ablation:proprio] "
              f"mode={proprio_ablation.mode} delay_steps={proprio_ablation.delay_steps} "
              f"model_key='proprio_input' shape={tuple(raw_data_dict['proprio_input'].shape)} "
              f"env_keys={PROPRIO_ENV_KEYS}"
            )
            logged_proprio_shape = True

          from human_plan.ego_bench_eval.utils import smooth_action, repeat_action
          action_hist_right_ee.append(
            repeat_action(action_dict["right_ee_pose"], data_args.future_index)
          )
          action_hist_left_ee.append(
            repeat_action(action_dict["left_ee_pose"], data_args.future_index)
          )

          action_hist_left_hand.append(
            repeat_action(action_dict["left_qpos_multi_step"], data_args.future_index)
          )
          action_hist_right_hand.append(
            repeat_action(action_dict["right_qpos_multi_step"], data_args.future_index)
          )

          action_left_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_left_ee
          )

          action_right_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_right_ee
          )

          action_left_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_left_hand
          )
          action_right_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_right_hand
          )

          trace_this_step = rl_action_trace and rl_trace_count < rl_action_trace_steps
          a_ref_dict = make_exec_action_dict(
            action_left_ee,
            action_right_ee,
            action_left_hand,
            action_right_hand,
          )
          if rl_insert_enabled:
            if rl_action_spec is None:
              rl_action_spec = ActionSpec.from_action_dict(a_ref_dict)
              if rl_actor_enabled:
                rl_action_normalizer = rl_actor_bundle["action_normalizer"]
                normalizer_dim = int(rl_action_normalizer.mean.shape[0])
                if normalizer_dim != rl_action_spec.dim:
                  raise ValueError(
                    f"Actor checkpoint action_dim={normalizer_dim} does not match "
                    f"post-smoothing a_ref dim={rl_action_spec.dim}. "
                    "Refusing to guess an action interface."
                  )
              else:
                rl_action_normalizer = AffineNormalizer.identity(rl_action_spec.dim)
              rl_actor_action_dim = rl_action_spec.dim
              _record_trace_line(
                rl_trace_lines,
                "[rl-posttrain] "
                f"actor_insert_point=after_temporal_smoothing "
                f"actor_action_dim={rl_actor_action_dim} "
                f"pack_slices={format_action_spec(rl_action_spec)}",
              )

            a_ref = pack_action(a_ref_dict, rl_action_spec)
            a_ref_norm_unclipped = rl_action_normalizer.normalize(a_ref, clip=None)
            ref_clipped_dims_this_step = int(np.sum(np.abs(a_ref_norm_unclipped) > 1.0))
            rl_ref_num_clipped_dims += ref_clipped_dims_this_step
            if rl_actor_enabled:
              a_ref_norm = np.clip(a_ref_norm_unclipped, -1.0, 1.0).astype(np.float32, copy=False)
            else:
              a_ref_norm = a_ref_norm_unclipped

            identity_roundtrip = rl_action_normalizer.denormalize(a_ref_norm_unclipped)
            identity_roundtrip_dict = unpack_action(identity_roundtrip, rl_action_spec)
            identity_diff = max_abs_action_diff(a_ref_dict, identity_roundtrip_dict, rl_action_spec)
            identity_mean_diff = mean_abs_action_diff(a_ref_dict, identity_roundtrip_dict, rl_action_spec)
            rl_identity_max_abs_diff = max(rl_identity_max_abs_diff, identity_diff)
            rl_identity_mean_abs_diff_sum += identity_mean_diff
            rl_identity_diff_count += 1
            if rl_mode == "identity" and identity_diff > rl_identity_tolerance:
              raise AssertionError(
                f"RL identity pack/unpack diff {identity_diff:.8g} exceeds "
                f"tolerance {rl_identity_tolerance:.8g}."
              )

            if rl_actor_enabled:
              if rl_actor is None or rl_actor_obs_normalizer is None:
                raise AssertionError("Actor mode expected a loaded TD3+BC actor checkpoint.")
              h_in = rl_feature_capture.value if rl_feature_capture is not None else None
              if h_in is None:
                raise RuntimeError(
                  "RL actor mode requires EgoVLA traj-decoder input latent h_in, "
                  "but the feature hook did not capture a value this step."
                )
              actor_obs, actor_report = build_actor_obs(
                h_in,
                used_proprio_pack,
                action_dict,
                rl_action_spec,
                a_ref_norm,
              )
              actor_obs_norm = rl_actor_obs_normalizer.normalize(actor_obs.reshape(1, -1), clip=None)
              expected_actor_obs_dim = int(rl_actor_bundle["checkpoint"]["actor_obs_dim"])
              if actor_obs_norm.shape != (1, expected_actor_obs_dim):
                raise ValueError(
                  f"Actor obs dim mismatch: built {actor_obs_norm.shape}, "
                  f"checkpoint expects {(1, expected_actor_obs_dim)}. "
                  "This usually means the feature hook/proprio/action summary interface changed."
                )
              actor_device = next(rl_actor.parameters()).device
              with torch.inference_mode():
                actor_input = torch.as_tensor(actor_obs_norm, dtype=torch.float32, device=actor_device)
                a_exec_norm = rl_actor(actor_input).detach().cpu().numpy().reshape(-1)
              if a_exec_norm.shape != (rl_action_spec.dim,):
                raise ValueError(
                  f"Actor output shape mismatch: got {a_exec_norm.shape}, "
                  f"expected {(rl_action_spec.dim,)}."
                )
              a_exec_norm = np.clip(a_exec_norm, -1.0, 1.0).astype(np.float32, copy=False)
              actor_ref_abs = np.abs(a_exec_norm - a_ref_norm)
              rl_actor_ref_mean_abs_sum += float(actor_ref_abs.mean())
              rl_actor_ref_max_abs = max(rl_actor_ref_max_abs, float(actor_ref_abs.max()))
              rl_actor_num_clipped_dims += int(np.sum(np.abs(a_exec_norm) >= 1.0 - 1.0e-6))
              rl_actor_stat_count += 1
              if not rl_logged_actor_obs_shapes:
                _record_trace_line(
                  rl_trace_lines,
                  "[rl-actor] "
                  f"actor_report=({actor_report.summary()}) "
                  f"actor_obs_norm_shape={actor_obs_norm.shape}",
                )
                rl_logged_actor_obs_shapes = True
            else:
              a_exec_norm = a_ref_norm

            a_exec = rl_action_normalizer.denormalize(a_exec_norm)
            a_exec_dict = unpack_action(a_exec, rl_action_spec)

            if trace_this_step:
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} raw_pred shape={shape_summary(action_dict, ['raw_pred']).get('raw_pred')}",
              )
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} action_dict fields shape="
                f"{shape_summary(action_dict, ['left_ee_pose', 'right_ee_pose', 'left_qpos_multi_step', 'right_qpos_multi_step'])}",
              )
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} smooth_action fields shape={shape_summary(a_ref_dict)}",
              )
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} pack_action(a_ref) shape={a_ref.shape} "
                f"a_ref_norm {format_stats(array_stats(a_ref_norm))} "
                f"a_ref_norm_unclipped {format_stats(array_stats(a_ref_norm_unclipped))} "
                f"ref_clipped_dims={ref_clipped_dims_this_step}",
              )
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} actor_action_dim={rl_actor_action_dim} "
                f"pack/unpack slices={format_action_spec(rl_action_spec)}",
              )
              _record_trace_line(
                rl_trace_lines,
                "[rl-action-trace] "
                f"step={i} unpack_action(a_exec) fields shape={shape_summary(a_exec_dict)} "
                f"identity_max_abs_diff={identity_diff:.8g} "
                f"identity_mean_abs_diff={identity_mean_diff:.8g}",
              )
              if rl_actor_enabled:
                actor_ref_abs = np.abs(a_exec_norm - a_ref_norm)
                _record_trace_line(
                  rl_trace_lines,
                  "[rl-actor-trace] "
                  f"step={i} a_exec_norm {format_stats(array_stats(a_exec_norm))} "
                  f"mean_abs_actor_minus_ref_norm={float(actor_ref_abs.mean()):.6f} "
                  f"max_abs_actor_minus_ref_norm={float(actor_ref_abs.max()):.6f} "
                  f"num_clipped_dims={int(np.sum(np.abs(a_exec_norm) >= 1.0 - 1.0e-6))}",
                )

            if rl_mode in ("identity", "actor") or rl_action_trace:
              action_left_ee = a_exec_dict["left_ee_pose"]
              action_right_ee = a_exec_dict["right_ee_pose"]
              action_left_hand = a_exec_dict["left_qpos"]
              action_right_hand = a_exec_dict["right_qpos"]

          if rl_collect_enabled:
            if rl_action_spec is None or rl_action_normalizer is None:
              raise AssertionError("Replay collection expected initialized RL action spec and normalizer.")
            h_in = rl_feature_capture.value if rl_feature_capture is not None else None
            if h_in is None:
              raise RuntimeError(
                "Replay collection requires EgoVLA traj-decoder input latent h_in, "
                "but the feature hook did not capture a value this step."
              )
            actor_obs, actor_report = build_actor_obs(
              h_in,
              used_proprio_pack,
              action_dict,
              rl_action_spec,
              a_ref_norm,
            )
            critic_obs, critic_report, newly_missing = build_critic_obs(
              env_results[0],
              actor_obs,
              optional_warned=rl_optional_warned,
            )
            if newly_missing:
              rl_optional_warned.update(newly_missing)
              _record_trace_line(
                rl_trace_lines,
                "[rl-collect] optional critic_obs fields missing in current task/env: "
                f"{newly_missing}"
              )
            if not rl_logged_obs_shapes:
              _record_trace_line(
                rl_trace_lines,
                "[rl-collect] "
                f"actor_report=({actor_report.summary()}) "
                f"critic_report=({critic_report.summary()})"
              )
              rl_logged_obs_shapes = True

            if rl_pending_transition is not None:
              _finalize_replay_transition(
                replay_writer,
                rl_pending_transition,
                actor_obs,
                critic_obs,
                a_ref,
              )
              rl_pending_transition = None

            rl_pending_transition = {
              "actor_obs": actor_obs,
              "critic_obs": critic_obs,
              "action_raw": a_exec,
              "bc_target_raw": a_ref,
              "source": rl_collect_source,
              "reward": 0.0,
              "done": 0.0,
              "success": 0.0,
              "timeout": 0.0,
              "episode_id": rl_episode_id,
              "episode_step": i + 1,
              "episode_length": 0.0,
              "episode_success": 0.0,
              "episode_timeout": 0.0,
              "raw": {
                "episode_id": rl_episode_id,
                "episode_label": episode_idx[0],
                "trial_idx": trial_idx,
                "step": i,
                "h_in_shape": actor_report.h_in_shape,
                "proprio_shapes": actor_report.proprio_shapes,
                "critic_shapes": critic_report.critic_shapes,
                "a_ref": a_ref.copy(),
                "a_ref_norm_temporary": a_ref_norm.copy(),
                "a_exec": a_exec.copy(),
                "a_exec_norm_temporary": a_exec_norm.copy(),
                "subtask_success": subtask_success_metrics(env_results[0]),
              },
            }

          ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world,
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              action_left_ee,
              action_right_ee,

              action_left_hand,
              action_right_hand,

              action
          )
          if trace_this_step:
            _record_trace_line(
              rl_trace_lines,
              "[rl-action-trace] "
              f"step={i} env.step final action shape={tuple(action.shape)} "
              f"action_device={action.device} action_dtype={action.dtype}",
            )
            rl_trace_count += 1
          env_results = env.step(action)
          current_action_for_stats = action.detach().clone()
          action_norm_sum += current_action_for_stats.norm(dim=-1).mean().item()
          if prev_action_for_stats is not None:
            action_delta_norm_sum += (
              current_action_for_stats - prev_action_for_stats
            ).norm(dim=-1).mean().item()
          prev_action_for_stats = current_action_for_stats
          action_stat_count += 1

          success_reached, success_streak = update_eval_success(
            task_args.task, base_env, env_results, success_streak
          )
          timeout_reached = (i == max_horizon - 1) and not success_reached
          if rl_collect_enabled and rl_pending_transition is not None:
            rl_pending_transition["success"] = 1.0 if success_reached else 0.0
            rl_pending_transition["timeout"] = 1.0 if timeout_reached else 0.0
            rl_pending_transition["done"] = 1.0 if (success_reached or timeout_reached) else 0.0
            rl_pending_transition["reward"] = 1.0 if success_reached else 0.0
            rl_pending_transition["episode_success"] = 1.0 if success_reached else 0.0
            rl_pending_transition["episode_timeout"] = 1.0 if timeout_reached else 0.0
            rl_pending_transition["raw"]["post_step_subtask_success"] = subtask_success_metrics(env_results[0])
            if success_reached or timeout_reached:
              rl_pending_transition["episode_length"] = action_stat_count
              _finalize_replay_transition(
                replay_writer,
                rl_pending_transition,
                rl_pending_transition["actor_obs"],
                rl_pending_transition["critic_obs"],
                rl_pending_transition["bc_target_raw"],
              )
              replay_writer.set_episode_result(
                rl_episode_id,
                action_stat_count,
                1.0 if success_reached else 0.0,
                1.0 if timeout_reached else 0.0,
              )
              rl_pending_transition = None

          result_img_3d = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, ::-1].copy()
          if task_args.project_trajs == 1:
            from human_plan.utils.visualization import (
              project_points
            )

            pred_3d = action_dict["pred_3d"]
            proj_2d = project_points(
              pred_3d, cam_intrinsics
            )
            proj_2d = proj_2d.reshape(-1, 2, 2)

            for fi in range(proj_2d.shape[0]-1):
              for j in range(2):
                result_img_3d = cv2.circle(
                  result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  5, (0, 255, 0), thickness=-1
                )
                if fi < proj_2d.shape[0] - 1:
                  result_img_3d = cv2.line(
                    result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  (int(proj_2d[fi + 1, j, 0]),int(proj_2d[fi + 1, j, 1])),
                    (0, 255, 0), thickness=2
                  )  

          if out is not None:
            out.write(result_img_3d)

          if task_args.save_frames:
            cv2.imwrite(
              os.path.join(frames_output_path, f"{i}.jpg"),
              result_img_3d
            )
          count += 1

          # Record the current frame before breaking so first-step successes do
          # not leave behind empty mp4 containers with no video stream.
          if success_reached:
            result = True
            break

        with open(task_args.result_saving_path, "a") as f:
          f.write(f"Task: {task_name}, Room Idx: {room_idx}, Table Idx: {table_idx}, Episode Label: {episode_idx[0]}, Trial Label: {trial_idx}, Result: {result} \n")
          avg_action_norm = action_norm_sum / max(action_stat_count, 1)
          avg_action_delta_norm = action_delta_norm_sum / max(action_stat_count - 1, 1)
          avg_actor_ref_abs = rl_actor_ref_mean_abs_sum / max(rl_actor_stat_count, 1)
          avg_identity_mean_abs_diff = rl_identity_mean_abs_diff_sum / max(rl_identity_diff_count, 1)
          f.write(
            "eval_ablation: "
            f"chunk_exec_len={chunk_exec_len} "
            f"action_chunk_horizon={action_chunk_horizon} "
            f"actual_exec_len={action_exec_len} "
            f"image_update_interval={image_update_interval} "
            f"image_delay_steps={image_delay_steps} "
            f"image_mode={image_ablation.mode} "
            f"image_key=image "
            f"proprio_ablation_mode={proprio_ablation.mode} "
            f"proprio_delay_steps={proprio_ablation.delay_steps} "
            f"proprio_key=proprio_input "
            f"episode_length={action_stat_count} "
            f"action_norm_mean={avg_action_norm:.6f} "
            f"action_delta_norm_mean={avg_action_delta_norm:.6f} "
            f"rl_mode={rl_mode} "
            f"rl_actor_action_dim={rl_actor_action_dim} "
            f"rl_identity_max_abs_diff={rl_identity_max_abs_diff:.8g} "
            f"rl_identity_mean_abs_diff={avg_identity_mean_abs_diff:.8g} "
            f"rl_actor_mean_abs_actor_minus_ref_norm={avg_actor_ref_abs:.6f} "
            f"rl_actor_max_abs_actor_minus_ref_norm={rl_actor_ref_max_abs:.6f} "
            f"rl_actor_num_clipped_dims={rl_actor_num_clipped_dims} "
            f"rl_ref_num_clipped_dims={rl_ref_num_clipped_dims} "
            f"image_mean_abs_diff={image_ablation.mean_abs_diff():.6f} "
            f"proprio_mean_abs_diff={proprio_ablation.mean_abs_diff():.6f}\n"
          )
          if rl_trace_lines:
            f.write("rl_action_trace:\n")
            for line in rl_trace_lines:
              f.write(f"{line}\n")
          subtask_string = ""
          for key in env_results[0].keys():
            if "success" in key:
              subtask_string += f"{key}: {env_results[0][key].sum().item()} "
          subtask_string += "\n"
          f.write(subtask_string)
          
        if out is not None:
          out.release()
          _convert_video_to_h264(output_path)
        # close the simulator
    if replay_writer is not None:
      replay_path = replay_writer.save()
      replay_contract_parts = []
      with np.load(replay_path, allow_pickle=False) as replay_data:
        action_dim = int(replay_data["action_norm"].shape[-1])
        actor_obs_dim = int(replay_data["actor_obs"].shape[-1])
        actor_tail_diff = float(
          np.max(np.abs(replay_data["actor_obs"][:, -action_dim:] - replay_data["bc_target_norm"]))
        )
        critic_prefix_diff = float(
          np.max(np.abs(replay_data["critic_obs"][:, :actor_obs_dim] - replay_data["actor_obs"]))
        )
        replay_contract_parts.extend(
          [
            f"action_norm_range=[{float(replay_data['action_norm'].min()):.6f},{float(replay_data['action_norm'].max()):.6f}]",
            f"bc_target_norm_range=[{float(replay_data['bc_target_norm'].min()):.6f},{float(replay_data['bc_target_norm'].max()):.6f}]",
            f"next_bc_target_norm_range=[{float(replay_data['next_bc_target_norm'].min()):.6f},{float(replay_data['next_bc_target_norm'].max()):.6f}]",
            f"actor_tail_diff={actor_tail_diff:.8g}",
            f"critic_prefix_diff={critic_prefix_diff:.8g}",
            "action_normalizer_present="
            + str(
              all(
                key in replay_data.files
                for key in ("action_normalizer_mean", "action_normalizer_scale", "action_normalizer_eps")
              )
            ),
          ]
        )
        if "episode_length" in replay_data.files:
          episode_lengths = np.asarray(replay_data["episode_length"]).reshape(-1)
          terminal_mask = np.asarray(replay_data["done"]).reshape(-1) > 0.5
          terminal_lengths = episode_lengths[terminal_mask]
          if terminal_lengths.size > 0:
            replay_contract_parts.append(
              "episode_length_terminal_range="
              f"[{float(terminal_lengths.min()):.0f},{float(terminal_lengths.max()):.0f}]"
            )
          replay_contract_parts.append(
            f"episode_length_recorded={str(bool(episode_lengths.size))}"
          )
        if "metadata_json" in replay_data.files:
          replay_metadata = json.loads(str(replay_data["metadata_json"].item()))
          replay_contract_parts.append(
            "format=" + str(replay_metadata.get("format", "unknown"))
          )
      replay_message = (
        f"[rl-collect] saved_replay={replay_path} transitions={len(replay_writer)} "
        + " ".join(replay_contract_parts)
      )
      print(replay_message)
      with open(task_args.result_saving_path, "a") as f:
        f.write(replay_message + "\n")
    if rl_feature_handle is not None:
      rl_feature_handle.remove()
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
