from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class AffineNormalizer:
    """Simple per-dimension affine normalizer.

    Stage 0/1 uses the identity instance so identity eval cannot silently change
    the baseline command. Later replay collection can fit mean/std or bounds and
    save them through the same interface.
    """

    mean: np.ndarray
    scale: np.ndarray
    eps: float = 1.0e-6

    @classmethod
    def identity(cls, dim: int) -> "AffineNormalizer":
        if dim <= 0:
            raise ValueError(f"Normalizer dim must be positive, got {dim}.")
        return cls(
            mean=np.zeros(dim, dtype=np.float32),
            scale=np.ones(dim, dtype=np.float32),
        )

    @classmethod
    def fit_standard(cls, values: np.ndarray, eps: float = 1.0e-6) -> "AffineNormalizer":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"fit_standard expects a 2D array, got shape {array.shape}.")
        mean = array.mean(axis=0).astype(np.float32)
        scale = array.std(axis=0).astype(np.float32)
        scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
        return cls(mean=mean, scale=scale, eps=eps)

    @classmethod
    def fit_minmax(cls, values: np.ndarray, eps: float = 1.0e-6) -> "AffineNormalizer":
        """Fit x_norm=(x-center)/half_range so training targets live near [-1, 1]."""

        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"fit_minmax expects a 2D array, got shape {array.shape}.")
        low = array.min(axis=0)
        high = array.max(axis=0)
        center = ((high + low) * 0.5).astype(np.float32)
        half_range = ((high - low) * 0.5).astype(np.float32)
        half_range = np.where(half_range < eps, 1.0, half_range).astype(np.float32)
        return cls(mean=center, scale=half_range, eps=eps)

    def _check(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Normalizer expected last dim {self.mean.shape[0]}, got shape {array.shape}."
            )
        return array

    def normalize(self, value: np.ndarray, clip: float | None = None) -> np.ndarray:
        array = self._check(value)
        normed = (array - self.mean) / np.maximum(self.scale, self.eps)
        if clip is not None:
            normed = np.clip(normed, -float(clip), float(clip))
        return normed.astype(np.float32, copy=False)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        array = self._check(value)
        denormed = array * np.maximum(self.scale, self.eps) + self.mean
        return denormed.astype(np.float32, copy=False)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float32, copy=True),
            "scale": self.scale.astype(np.float32, copy=True),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "AffineNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            eps=float(state.get("eps", 1.0e-6)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.state_dict())

    @classmethod
    def load(cls, path: str | Path) -> "AffineNormalizer":
        data = np.load(path)
        return cls.from_state_dict({key: data[key] for key in data.files})
