import numpy as np
import pytest

from rl_posttrain.action_utils import (
    ActionSpec,
    make_exec_action_dict,
    max_abs_action_diff,
    mean_abs_action_diff,
    pack_action,
    unpack_action,
)
from rl_posttrain.normalizer import AffineNormalizer


def _sample_action():
    return make_exec_action_dict(
        np.arange(7, dtype=np.float32),
        np.arange(7, dtype=np.float32) + 10.0,
        np.arange(12, dtype=np.float32) + 20.0,
        np.arange(12, dtype=np.float32) + 40.0,
    )


def test_pack_unpack_uses_post_smoothing_action_dim():
    action = _sample_action()
    spec = ActionSpec.from_action_dict(action)

    assert spec.dim == 38
    assert [(item.name, item.start, item.end) for item in spec.slices] == [
        ("left_ee_pose", 0, 7),
        ("right_ee_pose", 7, 14),
        ("left_qpos", 14, 26),
        ("right_qpos", 26, 38),
    ]

    packed = pack_action(action, spec)
    unpacked = unpack_action(packed, spec)

    assert packed.shape == (38,)
    assert max_abs_action_diff(action, unpacked, spec) == 0.0
    assert mean_abs_action_diff(action, unpacked, spec) == 0.0


def test_identity_action_normalizer_roundtrip():
    action = _sample_action()
    spec = ActionSpec.from_action_dict(action)
    normalizer = AffineNormalizer.identity(spec.dim)

    packed = pack_action(action, spec)
    normed = normalizer.normalize(packed)
    denormed = normalizer.denormalize(normed)

    np.testing.assert_allclose(denormed, packed)


def test_pack_action_rejects_shape_changes():
    action = _sample_action()
    spec = ActionSpec.from_action_dict(action)
    action["left_qpos"] = np.zeros((2, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="shape changed"):
        pack_action(action, spec)
