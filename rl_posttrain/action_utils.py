from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_ACTION_FIELD_ORDER = (
    "left_ee_pose",
    "right_ee_pose",
    "left_qpos",
    "right_qpos",
)


@dataclass(frozen=True)
class ActionSlice:
    name: str
    start: int
    end: int
    shape: Tuple[int, ...]

    @property
    def dim(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ActionSpec:
    slices: Tuple[ActionSlice, ...]

    @classmethod
    def from_action_dict(
        cls,
        action_dict: Mapping[str, object],
        field_order: Sequence[str] = DEFAULT_ACTION_FIELD_ORDER,
    ) -> "ActionSpec":
        start = 0
        slices = []
        missing = [name for name in field_order if name not in action_dict]
        if missing:
            raise KeyError(
                f"Cannot build ActionSpec; missing action fields {missing}. "
                f"Available fields: {sorted(action_dict.keys())}"
            )
        for name in field_order:
            value = _to_numpy(action_dict[name])
            shape = tuple(value.shape)
            if value.size <= 0:
                raise ValueError(f"Action field {name!r} is empty with shape {shape}.")
            end = start + int(value.size)
            slices.append(ActionSlice(name=name, start=start, end=end, shape=shape))
            start = end
        return cls(tuple(slices))

    @property
    def dim(self) -> int:
        if not self.slices:
            return 0
        return self.slices[-1].end

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.slices)


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pack_action(action_dict: Mapping[str, object], spec: ActionSpec) -> np.ndarray:
    parts = []
    for item in spec.slices:
        value = _to_numpy(action_dict[item.name]).astype(np.float32, copy=False)
        if tuple(value.shape) != item.shape:
            raise ValueError(
                f"Action field {item.name!r} shape changed: expected {item.shape}, "
                f"got {tuple(value.shape)}."
            )
        parts.append(value.reshape(-1))
    packed = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if packed.shape != (spec.dim,):
        raise AssertionError(f"Packed action shape mismatch: got {packed.shape}, expected {(spec.dim,)}.")
    return packed


def unpack_action(action: object, spec: ActionSpec) -> Dict[str, np.ndarray]:
    flat = _to_numpy(action).astype(np.float32, copy=False).reshape(-1)
    if flat.shape != (spec.dim,):
        raise ValueError(f"Flat action shape mismatch: got {flat.shape}, expected {(spec.dim,)}.")
    return {
        item.name: flat[item.start : item.end].reshape(item.shape).copy()
        for item in spec.slices
    }


def max_abs_action_diff(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    spec: ActionSpec,
) -> float:
    max_diff = 0.0
    for item in spec.slices:
        lhs_value = _to_numpy(lhs[item.name]).astype(np.float32, copy=False)
        rhs_value = _to_numpy(rhs[item.name]).astype(np.float32, copy=False)
        if tuple(lhs_value.shape) != tuple(rhs_value.shape):
            raise ValueError(
                f"Cannot compare field {item.name!r}: shapes differ "
                f"{tuple(lhs_value.shape)} vs {tuple(rhs_value.shape)}."
            )
        max_diff = max(max_diff, float(np.max(np.abs(lhs_value - rhs_value))))
    return max_diff


def mean_abs_action_diff(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    spec: ActionSpec,
) -> float:
    total = 0.0
    count = 0
    for item in spec.slices:
        lhs_value = _to_numpy(lhs[item.name]).astype(np.float32, copy=False)
        rhs_value = _to_numpy(rhs[item.name]).astype(np.float32, copy=False)
        if tuple(lhs_value.shape) != tuple(rhs_value.shape):
            raise ValueError(
                f"Cannot compare field {item.name!r}: shapes differ "
                f"{tuple(lhs_value.shape)} vs {tuple(rhs_value.shape)}."
            )
        diff = np.abs(lhs_value - rhs_value).reshape(-1)
        total += float(diff.sum())
        count += int(diff.size)
    if count <= 0:
        raise ValueError("Cannot compute mean diff for an empty action spec.")
    return total / count


def shape_summary(mapping: Mapping[str, object], keys: Iterable[str] | None = None) -> Dict[str, Tuple[int, ...]]:
    selected_keys = keys if keys is not None else mapping.keys()
    summary: Dict[str, Tuple[int, ...]] = {}
    for key in selected_keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if hasattr(value, "shape"):
            summary[key] = tuple(value.shape)  # type: ignore[arg-type]
        else:
            with np.errstate(all="ignore"):
                summary[key] = tuple(_to_numpy(value).shape)
    return summary


def array_stats(value: object) -> Dict[str, float]:
    array = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def format_stats(stats: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.6f}" for key, value in stats.items())


def format_action_spec(spec: ActionSpec) -> str:
    fields = [
        f"{item.name}[{item.start}:{item.end}] shape={item.shape}"
        for item in spec.slices
    ]
    return "; ".join(fields)


def make_exec_action_dict(
    left_ee_pose: object,
    right_ee_pose: object,
    left_qpos: object,
    right_qpos: object,
) -> Dict[str, object]:
    return {
        "left_ee_pose": left_ee_pose,
        "right_ee_pose": right_ee_pose,
        "left_qpos": left_qpos,
        "right_qpos": right_qpos,
    }
