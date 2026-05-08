import os
from types import SimpleNamespace

import torch


def blend_shapes(betas: torch.Tensor, shapedirs: torch.Tensor) -> torch.Tensor:
    """Minimal replacement for smplx.lbs.blend_shapes."""
    return torch.einsum("bl,vcl->bvc", betas, shapedirs)


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Minimal replacement for smplx.lbs.vertices2joints."""
    if J_regressor.dim() == 2:
        return torch.einsum("jv,bvc->bjc", J_regressor, vertices)
    if J_regressor.dim() == 3:
        return torch.einsum("bjv,bvc->bjc", J_regressor, vertices)
    raise ValueError(f"Unsupported J_regressor shape: {tuple(J_regressor.shape)}")


try:
    import smplx  # type: ignore
except Exception:
    smplx = None


if smplx is not None:

    def create_mano_model(model_path: str, is_rhand: bool, num_pca_comps: int = 15):
        return smplx.create(
            model_path,
            "mano",
            use_pca=True,
            is_rhand=is_rhand,
            num_pca_comps=num_pca_comps,
        )

else:
    from manopth.manolayer import ManoLayer

    _MANO_JOINT_MAPPING = torch.tensor(
        [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
        dtype=torch.long,
    )
    _MANO_JOINT_MAPPING_INV = torch.argsort(_MANO_JOINT_MAPPING)

    class _ManoCompat(torch.nn.Module):
        """Compatibility wrapper that mimics the subset of smplx MANO used here."""

        def __init__(self, model_path: str, is_rhand: bool, num_pca_comps: int = 15):
            super().__init__()
            mano_root = os.path.dirname(model_path)
            side = "right" if is_rhand else "left"
            self.layer = ManoLayer(
                use_pca=True,
                ncomps=num_pca_comps,
                side=side,
                mano_root=mano_root,
                flat_hand_mean=False,
            )

        @property
        def dtype(self):
            return self.layer.th_v_template.dtype

        @property
        def v_template(self):
            return self.layer.th_v_template.squeeze(0)

        @property
        def shapedirs(self):
            return self.layer.th_shapedirs

        @property
        def J_regressor(self):
            return self.layer.th_J_regressor

        def forward(
            self,
            betas=None,
            global_orient=None,
            hand_pose=None,
            transl=None,
            return_verts=True,
            **_,
        ):
            candidates = [x for x in (betas, global_orient, hand_pose, transl) if x is not None]
            batch_size = candidates[0].shape[0] if candidates else 1
            device = candidates[0].device if candidates else self.v_template.device
            dtype = candidates[0].dtype if candidates else self.dtype

            if betas is None:
                betas = torch.zeros((batch_size, 10), device=device, dtype=dtype)
            if global_orient is None:
                global_orient = torch.zeros((batch_size, 3), device=device, dtype=dtype)
            if hand_pose is None:
                hand_pose = torch.zeros((batch_size, 15), device=device, dtype=dtype)

            pose_coeffs = torch.cat([global_orient, hand_pose], dim=-1).to(device=device, dtype=dtype)
            betas = betas.to(device=device, dtype=dtype)
            transl_arg = None if transl is None else transl.to(device=device, dtype=dtype)

            verts_mm, joints_reordered_21_mm = self.layer(
                th_pose_coeffs=pose_coeffs,
                th_betas=betas,
                th_trans=transl_arg,
            )

            verts = verts_mm / 1000.0
            joints_reordered_21 = joints_reordered_21_mm / 1000.0
            joints_original_21 = joints_reordered_21[:, _MANO_JOINT_MAPPING_INV.to(joints_reordered_21.device)]
            joints_original_16 = joints_original_21[:, :16]
            return SimpleNamespace(vertices=verts, joints=joints_original_16)


    def create_mano_model(model_path: str, is_rhand: bool, num_pca_comps: int = 15):
        return _ManoCompat(model_path, is_rhand=is_rhand, num_pca_comps=num_pca_comps)
