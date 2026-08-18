"""Artifact and oracle-annotation helpers for native benchmark observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .artifacts import file_sha256, save_rgb


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_plane(value: Any) -> np.ndarray:
    """Normalize a batched camera output to one HxW plane."""

    array = to_numpy(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a camera plane, got shape {array.shape}")
    return array


def image_artifact(
    path: Path,
    array: np.ndarray,
    *,
    content_image: bool = True,
) -> dict[str, Any]:
    artifact = save_rgb(path, array)
    artifact["media_type"] = "image/png"
    artifact["content_image"] = content_image
    return artifact


def save_depth_artifacts(
    directory: Path,
    depth_value: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Save metric float depth, validity, and an explicitly ranged preview."""

    directory.mkdir(parents=True, exist_ok=True)
    depth = camera_plane(depth_value).astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("Metric depth contains no finite positive pixels")
    depth_path = directory / "depth_m.npy"
    np.save(depth_path, depth, allow_pickle=False)
    valid_rgb = np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2)
    valid_artifact = image_artifact(directory / "depth_valid_mask.png", valid_rgb)

    finite = depth[valid]
    near = float(np.quantile(finite, 0.01))
    far = float(np.quantile(finite, 0.99))
    if far <= near:
        far = near + 1e-6
    normalized = np.zeros_like(depth, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - near) / (far - near) * 255.0, 0, 255
    ).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    preview = image_artifact(directory / "depth_visualization.png", colored)
    artifact = {
        "depth_m": {
            "path": str(depth_path.resolve()),
            "sha256": file_sha256(depth_path),
            "media_type": "application/x-npy",
            "dtype": "float32",
            "shape": [int(value) for value in depth.shape],
            "unit": "metre",
        },
        "valid_mask": valid_artifact,
        "visualization": preview,
        "visualization_range_m": {"near": near, "far": far},
        "statistics": {
            "valid_fraction": float(valid.mean()),
            "finite_min_m": float(finite.min()),
            "finite_max_m": float(finite.max()),
        },
    }
    return artifact, depth, colored


def normalize_instance_mapping(info: Any) -> dict[int, str]:
    """Normalize Isaac's annotator metadata without assuming one release shape."""

    if isinstance(info, list) and len(info) == 1:
        info = info[0]
    if not isinstance(info, Mapping):
        return {}
    if "idToLabels" in info:
        info = info["idToLabels"]
    if not isinstance(info, Mapping):
        return {}
    result: dict[int, str] = {}
    for raw_id, label in info.items():
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str):
            rendered = label
        elif isinstance(label, Mapping):
            preferred = next(
                (
                    label.get(key)
                    for key in ("primPath", "prim_path", "path", "class", "label")
                    if isinstance(label.get(key), str)
                ),
                None,
            )
            rendered = preferred or json.dumps(label, sort_keys=True, default=str)
        else:
            rendered = str(label)
        result[instance_id] = rendered
    return result


def instance_role_mask(
    instance_value: Any,
    id_to_labels: Mapping[int, str],
    private_prim_name: str,
) -> tuple[np.ndarray, list[int]]:
    """Select every leaf instance below one private actor prim."""

    instance = camera_plane(instance_value)
    token = f"/{private_prim_name}/"
    suffix = f"/{private_prim_name}"
    selected = sorted(
        instance_id
        for instance_id, label in id_to_labels.items()
        if token in label or label.endswith(suffix)
    )
    if not selected:
        return np.zeros(instance.shape, dtype=bool), []
    return np.isin(instance, np.asarray(selected, dtype=instance.dtype)), selected


def bbox_xyxy_exclusive(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_annotation_artifacts(
    directory: Path,
    *,
    role: str,
    mask: np.ndarray,
    rgb: np.ndarray,
    provide_bbox: bool,
    provide_mask: bool,
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
    """Serialize only the independently enabled anonymous public features."""

    directory.mkdir(parents=True, exist_ok=True)
    public: dict[str, Any] = {
        "public_role": role,
        "visible": bool(mask.any()),
    }
    panels: list[tuple[str, np.ndarray]] = []
    bbox = bbox_xyxy_exclusive(mask)
    if provide_bbox:
        public["bbox_xyxy_exclusive"] = bbox
        overlay = Image.fromarray(rgb, mode="RGB")
        if bbox is not None:
            draw = ImageDraw.Draw(overlay)
            x1, y1, x2, y2 = bbox
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="#00ff7f", width=3)
        overlay_array = np.asarray(overlay)
        public["bbox_overlay"] = image_artifact(
            directory / f"{role}_bbox_overlay.png",
            overlay_array,
            content_image=False,
        )
        panels.append((f"{role}_bbox", overlay_array))
    if provide_mask:
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        public["mask"] = image_artifact(directory / f"{role}_mask.png", mask_rgb)
        panels.append((f"{role}_mask", mask_rgb))
    return public, panels


def quaternion_wxyz_matrix(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(q))
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def homogeneous_transform(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Position must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_matrix(quaternion_wxyz)
    transform[:3, 3] = position
    return transform


def robot_base_camera_extrinsic(
    *,
    robot_position_world: Any,
    robot_quaternion_world_wxyz: Any,
    camera_position_world: Any,
    camera_quaternion_world_ros_wxyz: Any,
) -> list[list[float]]:
    world_robot = homogeneous_transform(
        robot_position_world, robot_quaternion_world_wxyz
    )
    world_camera = homogeneous_transform(
        camera_position_world, camera_quaternion_world_ros_wxyz
    )
    robot_camera = np.linalg.inv(world_robot) @ world_camera
    if not np.isfinite(robot_camera).all():
        raise ValueError("Camera extrinsic contains non-finite values")
    return robot_camera.tolist()
