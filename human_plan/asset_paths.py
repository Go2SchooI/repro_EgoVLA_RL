"""Paths to bundled simulation assets in the source checkout."""
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
HAND_ACTUATION_PATH = ASSET_ROOT / "retargeting" / "hand_actuation_net.pth"
HAND_MANO_RETARGET_PATH = ASSET_ROOT / "retargeting" / "hand_mano_retarget_net.pth"
INIT_POSES_PATH = ASSET_ROOT / "initial_poses" / "init_poses_fixed_set_100traj.pkl"
LEGACY_FIXED_SET_POSES_PATH = ASSET_ROOT / "initial_poses" / "legacy" / "init_poses_fixed_set.pkl"
