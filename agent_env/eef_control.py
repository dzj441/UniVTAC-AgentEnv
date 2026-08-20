"""Lightweight EEF action routing and private planner diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np


EEFActionRouteName = Literal["move", "gripper", "all", "no_op"]


@dataclass(frozen=True)
class EEFActionRoute:
    name: EEFActionRouteName
    arm_changed: bool
    gripper_changed: bool


def classify_delta_eef_action(
    delta_position: object,
    delta_rpy: object,
    delta_gripper: object,
) -> EEFActionRoute:
    position = _finite_vector(delta_position, "delta_position")
    rotation = _finite_vector(delta_rpy, "delta_rpy")
    try:
        gripper = float(delta_gripper)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_gripper must be finite") from exc
    if not np.isfinite(gripper):
        raise ValueError("delta_gripper must be finite")

    arm_changed = bool(np.any(position != 0.0) or np.any(rotation != 0.0))
    gripper_changed = gripper != 0.0
    if arm_changed and gripper_changed:
        name: EEFActionRouteName = "all"
    elif arm_changed:
        name = "move"
    elif gripper_changed:
        name = "gripper"
    else:
        name = "no_op"
    return EEFActionRoute(name, arm_changed, gripper_changed)


def motion_gen_result_diagnostics(result: object) -> dict[str, Any]:
    status = getattr(result, "status", None)
    status_value = status.value if isinstance(status, Enum) else status
    status_name = status.name if isinstance(status, Enum) else None
    return {
        "schema_version": "univtac.curobo_motion_gen_diagnostics.v1",
        "success": _json_value(getattr(result, "success", None)),
        "status": {
            "type": type(status).__name__ if status is not None else None,
            "name": status_name,
            "value": _json_value(status_value),
            "text": str(status) if status is not None else None,
        },
        "valid_query": _json_value(getattr(result, "valid_query", None)),
        "attempts": _json_value(getattr(result, "attempts", None)),
        "trajopt_attempts": _json_value(getattr(result, "trajopt_attempts", None)),
        "used_graph": _json_value(getattr(result, "used_graph", None)),
        "goalset_index": _json_value(getattr(result, "goalset_index", None)),
        "path_buffer_last_tstep": _json_value(
            getattr(result, "path_buffer_last_tstep", None)
        ),
        "timing_seconds": {
            "solve": _json_value(getattr(result, "solve_time", None)),
            "ik": _json_value(getattr(result, "ik_time", None)),
            "graph": _json_value(getattr(result, "graph_time", None)),
            "trajopt": _json_value(getattr(result, "trajopt_time", None)),
            "finetune": _json_value(getattr(result, "finetune_time", None)),
            "total": _json_value(getattr(result, "total_time", None)),
        },
        "terminal_error": {
            "position": _json_value(getattr(result, "position_error", None)),
            "rotation": _json_value(getattr(result, "rotation_error", None)),
            "cspace": _json_value(getattr(result, "cspace_error", None)),
        },
        "debug_info_present": getattr(result, "debug_info", None) is not None,
    }


def _finite_vector(value: object, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly 3 finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly 3 finite values")
    return vector


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (RuntimeError, TypeError, ValueError):
            pass
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)
