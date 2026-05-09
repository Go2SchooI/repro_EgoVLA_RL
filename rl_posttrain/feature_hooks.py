from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CapturedTensor:
    value: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.value = None


def register_traj_decoder_input_hook(model) -> tuple[CapturedTensor, object]:
    """Capture h_in, the latent tensor passed into the trajectory decoder.

    This is intentionally passive: it does not modify model inputs or outputs.
    The eval path calls `traj_decoder.inference(...)` directly, so a normal
    forward hook would not fire. We wrap that bound method on the instance and
    restore it through the returned handle.
    """

    capture = CapturedTensor()
    traj_decoder = model.get_traj_decoder()
    original_inference = traj_decoder.inference

    def _wrapped_inference(latent, *args, **kwargs):
        if latent is None:
            capture.value = None
        elif hasattr(latent, "detach"):
            capture.value = latent.detach().clone()
        else:
            capture.value = None
        return original_inference(latent, *args, **kwargs)

    class _Handle:
        def remove(self) -> None:
            traj_decoder.inference = original_inference

    traj_decoder.inference = _wrapped_inference
    handle = _Handle()
    return capture, handle
