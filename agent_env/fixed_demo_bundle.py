"""Project authenticated P6 expert masters into Agent-visible ICL bundles.

The P6 master is evaluator-owned provenance.  This module copies a strict
allowlist into a new directory, rewrites every path relative to that directory,
and validates the completed bundle before publishing it.  No source manifest,
seed, checker evidence, action, or absolute host path is Agent-visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .artifacts import file_sha256
from .benchmark_observations import bbox_xyxy_exclusive
from .benchmark_profiles import (
    AnnotationCapabilities,
    ObservationProfile,
    ROBOT_STATE_FIELDS,
    get_observation_profile,
)
from .benchmark_tasks import get_benchmark_task
from .p6_expert_master import (
    P6_MASTER_SCHEMA_VERSION,
    P6_PROFILE,
    WRIST_METRIC_DEPTH_SURFACE_POLICY,
    validate_p6_observation,
)


FIXED_DEMO_ROOT_ENV = "UNIVTAC_FIXED_EXPERT_MASTER_ROOT"
PUBLIC_ROLES = ("manipulated_object", "goal_fixture")
CAMERA_NAMES = ("head", "wrist")
MAX_CONTACT_SHEET_FRAMES = 12


class FixedDemoBundleError(ValueError):
    """Raised when a master or projected bundle violates the public contract."""


@dataclass(frozen=True)
class FixedDemoAssetSpec:
    """Evaluator-private registry entry for one fixed demonstration."""

    task: str
    seed: int
    manifest_relative_path: str
    manifest_sha256: str


_FIXED_DEMO_ASSETS = {
    "pull_out_key": FixedDemoAssetSpec(
        task="pull_out_key",
        seed=0,
        manifest_relative_path=(
            "pull_out_key_seed_0_wrist_depth_v3/p6_master_manifest.json"
        ),
        manifest_sha256=(
            "ff48c0c2df6e75152960e2a78af30ba79b43aaf82905a04093c816b89918fc44"
        ),
    ),
    "put_bottle_in_shelf": FixedDemoAssetSpec(
        task="put_bottle_in_shelf",
        seed=1,
        manifest_relative_path=(
            "put_bottle_in_shelf_seed_1_wrist_depth_v3/p6_master_manifest.json"
        ),
        manifest_sha256=(
            "3e3af14ab0184a9fbfe20129e8ea765c3563dc8182bd8288f998e4a528919c96"
        ),
    ),
}


_IMAGE_DESTINATIONS = {
    "head_rgb": Path("head/rgb.png"),
    "wrist_rgb": Path("wrist/rgb.png"),
    "left_tactile_rgb": Path("tactile/left_rgb_marker.png"),
    "right_tactile_rgb": Path("tactile/right_rgb_marker.png"),
}


def get_fixed_demo_asset_spec(task: str) -> FixedDemoAssetSpec:
    """Return the frozen private registry entry for ``task``."""

    get_benchmark_task(task)
    try:
        return _FIXED_DEMO_ASSETS[task]
    except KeyError as exc:
        raise FixedDemoBundleError(
            f"No fixed demonstration is registered for task {task!r}"
        ) from exc


def resolve_fixed_demo_root(value: Path | None) -> Path:
    """Resolve the host-side master root from a CLI value or environment."""

    candidate: Path | None = value
    if candidate is None:
        configured = os.environ.get(FIXED_DEMO_ROOT_ENV)
        candidate = Path(configured) if configured else None
    if candidate is None:
        raise FixedDemoBundleError(
            "fixed_demo requires --fixed-demo-root or "
            f"the {FIXED_DEMO_ROOT_ENV} environment variable"
        )
    root = candidate.expanduser().resolve()
    if not root.is_dir():
        raise FixedDemoBundleError(f"Fixed-demo master root does not exist: {root}")
    return root


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedDemoBundleError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FixedDemoBundleError(f"{label} must be a JSON object: {path}")
    return value


def _json_text(value: Any, *, indent: int | None = None) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FixedDemoBundleError(f"Public JSON is not finite/serializable: {exc}") from exc
    return rendered + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_json_text(record) for record in records), encoding="utf-8")


def _safe_relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise FixedDemoBundleError(f"{label} must be a safe relative path: {value!r}")
    return path


def _path_inside(root: Path, relative: str, label: str) -> Path:
    safe = _safe_relative_path(relative, label)
    resolved = (root / safe).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise FixedDemoBundleError(f"{label} escaped its root: {relative!r}")
    return resolved


def _artifact_source_path(value: Any, master_root: Path, label: str) -> Path:
    if not isinstance(value, dict):
        raise FixedDemoBundleError(f"{label} artifact must be an object")
    raw_path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise FixedDemoBundleError(f"{label} artifact omitted path or SHA-256")
    path = Path(raw_path).resolve()
    if not path.is_relative_to(master_root) or not path.is_file() or path.is_symlink():
        raise FixedDemoBundleError(f"{label} artifact escaped the P6 master: {path}")
    if file_sha256(path) != digest:
        raise FixedDemoBundleError(f"{label} source artifact hash mismatch: {path}")
    return path


def _copy_artifact(
    source: dict[str, Any],
    *,
    master_root: Path,
    bundle_root: Path,
    relative_destination: Path,
    label: str,
    fields: tuple[str, ...] = ("media_type", "content_image"),
) -> dict[str, Any]:
    source_path = _artifact_source_path(source, master_root, label)
    if relative_destination.is_absolute() or ".." in relative_destination.parts:
        raise FixedDemoBundleError(f"Unsafe destination for {label}")
    destination = bundle_root / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    digest = file_sha256(destination)
    if digest != source.get("sha256"):
        raise FixedDemoBundleError(f"{label} changed while it was copied")
    public: dict[str, Any] = {
        "path": relative_destination.as_posix(),
        "sha256": digest,
    }
    for field in fields:
        if field in source:
            public[field] = source[field]
    return public


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FixedDemoBundleError(f"{label} must be a finite {size}-vector") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise FixedDemoBundleError(f"{label} must be a finite {size}-vector")
    return [float(item) for item in array]


def _public_robot_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(ROBOT_STATE_FIELDS):
        raise FixedDemoBundleError("P6 master robot state does not match the allowlist")
    width = float(value["gripper_width_m"])
    if not np.isfinite(width):
        raise FixedDemoBundleError("gripper_width_m must be finite")
    return {
        "joint_position_9d": _finite_vector(
            value["joint_position_9d"], 9, "joint_position_9d"
        ),
        "joint_velocity_9d": _finite_vector(
            value["joint_velocity_9d"], 9, "joint_velocity_9d"
        ),
        "gripper_width_m": width,
        "end_effector_pose_robot_base_wxyz_7d": _finite_vector(
            value["end_effector_pose_robot_base_wxyz_7d"],
            7,
            "end_effector_pose_robot_base_wxyz_7d",
        ),
    }


def _public_intrinsics(value: Any, camera_name: str) -> list[list[float]]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FixedDemoBundleError(f"Invalid {camera_name} intrinsics") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise FixedDemoBundleError(f"Invalid {camera_name} intrinsics")
    return [[float(item) for item in row] for row in matrix]


def _public_extrinsics(value: Any, camera_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "matrix_T_robot_base_camera_ros_4x4",
        "camera_convention",
        "translation_unit",
    }:
        raise FixedDemoBundleError(f"Invalid {camera_name} extrinsics")
    try:
        matrix = np.asarray(
            value["matrix_T_robot_base_camera_ros_4x4"], dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise FixedDemoBundleError(f"Invalid {camera_name} extrinsic matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise FixedDemoBundleError(f"Invalid {camera_name} extrinsic matrix")
    if value["translation_unit"] != "metre":
        raise FixedDemoBundleError("Camera translation unit must be metre")
    return {
        "matrix_T_robot_base_camera_ros_4x4": [
            [float(item) for item in row] for row in matrix
        ],
        "camera_convention": str(value["camera_convention"]),
        "translation_unit": "metre",
    }


def _copy_image_modality(
    *,
    name: str,
    source: dict[str, Any],
    frame_root: Path,
    frame_relative: Path,
    master_root: Path,
    bundle_root: Path,
) -> tuple[dict[str, Any], Path]:
    relative = frame_relative / _IMAGE_DESTINATIONS[name]
    public = _copy_artifact(
        source,
        master_root=master_root,
        bundle_root=bundle_root,
        relative_destination=relative,
        label=name,
    )
    return public, frame_root / _IMAGE_DESTINATIONS[name]


def _copy_depth_modality(
    *,
    name: str,
    source: dict[str, Any],
    frame_root: Path,
    frame_relative: Path,
    master_root: Path,
    bundle_root: Path,
) -> tuple[dict[str, Any], Path]:
    if set(source) != {
        "depth_m",
        "valid_mask",
        "visualization",
        "visualization_range_m",
        "statistics",
    }:
        raise FixedDemoBundleError(f"{name} source fields are not allowlisted")
    camera = name.removesuffix("_depth")
    directory = frame_relative / camera
    depth_m = _copy_artifact(
        source["depth_m"],
        master_root=master_root,
        bundle_root=bundle_root,
        relative_destination=directory / "depth_m.npy",
        label=f"{name}.depth_m",
        fields=("media_type", "dtype", "shape", "unit"),
    )
    valid_mask = _copy_artifact(
        source["valid_mask"],
        master_root=master_root,
        bundle_root=bundle_root,
        relative_destination=directory / "depth_valid_mask.png",
        label=f"{name}.valid_mask",
    )
    visualization = _copy_artifact(
        source["visualization"],
        master_root=master_root,
        bundle_root=bundle_root,
        relative_destination=directory / "depth_visualization.png",
        label=f"{name}.visualization",
    )
    range_value = source["visualization_range_m"]
    statistics = source["statistics"]
    if not isinstance(range_value, dict) or set(range_value) != {"near", "far"}:
        raise FixedDemoBundleError(f"{name} visualization range is invalid")
    if not isinstance(statistics, dict) or set(statistics) != {
        "valid_fraction",
        "finite_min_m",
        "finite_max_m",
    }:
        raise FixedDemoBundleError(f"{name} depth statistics are invalid")
    numeric = [
        float(range_value["near"]),
        float(range_value["far"]),
        float(statistics["valid_fraction"]),
        float(statistics["finite_min_m"]),
        float(statistics["finite_max_m"]),
    ]
    if not np.isfinite(numeric).all():
        raise FixedDemoBundleError(f"{name} depth metadata is not finite")
    public = {
        "depth_m": depth_m,
        "valid_mask": valid_mask,
        "visualization": visualization,
        "visualization_range_m": {"near": numeric[0], "far": numeric[1]},
        "statistics": {
            "valid_fraction": numeric[2],
            "finite_min_m": numeric[3],
            "finite_max_m": numeric[4],
        },
    }
    return public, frame_root / camera / "depth_visualization.png"


def _project_annotations(
    source: Any,
    *,
    profile: ObservationProfile,
    capabilities: AnnotationCapabilities,
    frame_relative: Path,
    master_root: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise FixedDemoBundleError("Initial P6 annotations are unavailable")
    projected: dict[str, Any] = {}
    for camera_name in profile.camera_names:
        camera_source = source.get(camera_name)
        if not isinstance(camera_source, dict) or set(camera_source) != set(PUBLIC_ROLES):
            raise FixedDemoBundleError(
                f"Initial {camera_name} annotations are incomplete"
            )
        camera_public: dict[str, Any] = {}
        for role in PUBLIC_ROLES:
            role_source = camera_source[role]
            if not isinstance(role_source, dict):
                raise FixedDemoBundleError(f"Invalid {camera_name}/{role} annotation")
            public: dict[str, Any] = {
                "public_role": role,
                "visible": bool(role_source.get("visible")),
            }
            directory = frame_relative / "annotations" / camera_name
            if capabilities.provide_bbox:
                bbox = role_source.get("bbox_xyxy_exclusive")
                if bbox is not None:
                    bbox = [int(item) for item in bbox]
                    if len(bbox) != 4:
                        raise FixedDemoBundleError("BBox must have four coordinates")
                public["bbox_xyxy_exclusive"] = bbox
                public["bbox_overlay"] = _copy_artifact(
                    role_source["bbox_overlay"],
                    master_root=master_root,
                    bundle_root=bundle_root,
                    relative_destination=directory / f"{role}_bbox_overlay.png",
                    label=f"{camera_name}/{role}.bbox_overlay",
                )
            if capabilities.provide_mask:
                public["mask"] = _copy_artifact(
                    role_source["mask"],
                    master_root=master_root,
                    bundle_root=bundle_root,
                    relative_destination=directory / f"{role}_mask.png",
                    label=f"{camera_name}/{role}.mask",
                    fields=(
                        "media_type",
                        "content_image",
                        "mode",
                        "background_value",
                        "foreground_value",
                    ),
                )
                public["mask_overlay"] = _copy_artifact(
                    role_source["mask_overlay"],
                    master_root=master_root,
                    bundle_root=bundle_root,
                    relative_destination=directory / f"{role}_mask_overlay.png",
                    label=f"{camera_name}/{role}.mask_overlay",
                )
            camera_public[role] = public
        projected[camera_name] = camera_public
    return projected


def contact_sheet_indices(frame_count: int, limit: int = MAX_CONTACT_SHEET_FRAMES) -> list[int]:
    """Return deterministic endpoint-preserving uniform sample indices."""

    if frame_count <= 0 or limit <= 0:
        raise FixedDemoBundleError("Contact-sheet frame count and limit must be positive")
    if frame_count <= limit:
        return list(range(frame_count))
    return [index * (frame_count - 1) // (limit - 1) for index in range(limit)]


def _build_contact_sheet(
    samples: list[tuple[str, Path]],
    destination: Path,
) -> None:
    if not samples:
        raise FixedDemoBundleError("Cannot build an empty contact sheet")
    columns = min(4, len(samples))
    rows = (len(samples) + columns - 1) // columns
    tile_width = 240
    image_height = 135
    label_height = 20
    tile_height = image_height + label_height
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
    draw = ImageDraw.Draw(canvas)
    for slot, (label, path) in enumerate(samples):
        with Image.open(path) as image:
            tile = ImageOps.fit(
                image.convert("RGB"),
                (tile_width, image_height),
                method=Image.Resampling.LANCZOS,
            )
        x = (slot % columns) * tile_width
        y = (slot // columns) * tile_height
        canvas.paste(tile, (x, y + label_height))
        draw.text((x + 4, y + 3), label, fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _json_artifact(relative: Path, root: Path, media_type: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "sha256": file_sha256(path),
        "media_type": media_type,
    }


def _bundle_content_integrity(root: Path) -> dict[str, Any]:
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FixedDemoBundleError(f"Agent bundle must not contain symlinks: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        entries.append((relative, file_sha256(path)))
    canonical = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    return {
        "algorithm": "sha256(relative_path_and_file_sha256_v1)",
        "file_count_excluding_manifest": len(entries),
        "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _authenticate_master(
    asset_root: Path,
    spec: FixedDemoAssetSpec,
) -> tuple[Path, Path, dict[str, Any]]:
    asset_root = asset_root.resolve()
    manifest_path = _path_inside(
        asset_root, spec.manifest_relative_path, "Fixed-demo manifest"
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FixedDemoBundleError(f"Fixed-demo manifest is missing: {manifest_path}")
    actual_digest = file_sha256(manifest_path)
    if actual_digest != spec.manifest_sha256:
        raise FixedDemoBundleError("Fixed-demo master manifest hash mismatch")
    manifest = _read_json(manifest_path, "fixed-demo P6 master manifest")
    required = {
        "schema_version",
        "task",
        "seed",
        "trajectory_representation",
        "actions_present",
        "step_eef_conversion_performed",
        "agent_ready",
        "intended_use",
        "source",
        "capture",
        "replay_verification",
        "visibility_contract",
    }
    if set(manifest) != required:
        raise FixedDemoBundleError("Fixed-demo P6 master top-level fields changed")
    if manifest.get("schema_version") != P6_MASTER_SCHEMA_VERSION:
        raise FixedDemoBundleError("Fixed-demo master must use the v3 P6 schema")
    if manifest.get("task") != spec.task or manifest.get("seed") != spec.seed:
        raise FixedDemoBundleError("Fixed-demo master task/seed registry mismatch")
    if (
        manifest.get("agent_ready") is not False
        or manifest.get("actions_present") is not False
        or manifest.get("step_eef_conversion_performed") is not False
    ):
        raise FixedDemoBundleError("Fixed-demo master action/visibility contract is unsafe")
    if manifest.get("trajectory_representation") != (
        "successful_expert_observation_waypoints"
    ):
        raise FixedDemoBundleError("Unexpected fixed-demo trajectory representation")
    replay = manifest.get("replay_verification")
    if not isinstance(replay, dict) or (
        replay.get("execution_succeeded") is not True
        or replay.get("official_task_success") is not True
    ):
        raise FixedDemoBundleError("Fixed-demo master lacks successful replay proof")
    capture = manifest.get("capture")
    if not isinstance(capture, dict):
        raise FixedDemoBundleError("Fixed-demo master capture metadata is missing")
    if capture.get("observation_profile") != P6_PROFILE.to_manifest():
        raise FixedDemoBundleError("Fixed-demo master is not complete P6")
    depth_policy = capture.get("wrist_metric_depth_surface_policy")
    if not isinstance(depth_policy, dict) or set(depth_policy) != set(
        WRIST_METRIC_DEPTH_SURFACE_POLICY
    ) or (
        depth_policy.get("schema_version")
        != WRIST_METRIC_DEPTH_SURFACE_POLICY["schema_version"]
        or depth_policy.get("rigid_gripper_and_gelsight_housing_included")
        is not True
        or depth_policy.get("deformable_optical_gel_surface_included")
        is not False
    ):
        raise FixedDemoBundleError(
            "Fixed-demo master lacks the wrist-depth surface policy"
        )
    annotation_source = capture.get("annotations")
    if not isinstance(annotation_source, dict) or (
        annotation_source.get("bbox") is not True
        or annotation_source.get("mask") is not True
        or annotation_source.get("schedule") != "initial_observation_only"
        or annotation_source.get("public_roles") != list(PUBLIC_ROLES)
        or annotation_source.get("raw_instance_metadata") is not False
    ):
        raise FixedDemoBundleError("Fixed-demo master annotation source is unsafe")
    visibility = manifest.get("visibility_contract")
    if not isinstance(visibility, dict) or (
        visibility.get("episode_outcome") != "successful expert demonstration"
        or any(
            visibility.get(field) is not False
            for field in (
                "stepwise_success",
                "checker_details",
                "actions",
                "actor_poses",
                "planner_or_ik_state",
                "contact_points",
                "raw_tactile_depth_marker_or_pose",
            )
        )
    ):
        raise FixedDemoBundleError("Fixed-demo master visibility contract is unsafe")
    return manifest_path, manifest_path.parent.resolve(), manifest


def project_fixed_demo_bundle(
    *,
    asset_root: Path,
    destination: Path,
    task: str,
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    asset_spec: FixedDemoAssetSpec | None = None,
) -> dict[str, Any]:
    """Build, validate, and atomically publish one static Agent ICL bundle.

    The returned receipt is evaluator-private and may contain absolute host
    paths.  Only ``destination`` is intended for the Agent workspace.
    """

    spec = asset_spec or get_fixed_demo_asset_spec(task)
    if spec.task != task:
        raise FixedDemoBundleError("Fixed-demo asset spec task mismatch")
    get_benchmark_task(task)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Fixed-demo destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path, master_root, master = _authenticate_master(asset_root, spec)
    capture = master["capture"]
    waypoints = capture.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise FixedDemoBundleError("Fixed-demo master has no waypoints")
    if capture.get("captured_observation_count") != len(waypoints):
        raise FixedDemoBundleError("Fixed-demo waypoint count is inconsistent")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    ).resolve()
    try:
        state_records: list[dict[str, Any]] = []
        trajectory_records: list[dict[str, Any]] = []
        extrinsic_records: list[dict[str, Any]] = []
        frame_records: list[dict[str, Any]] = []
        intrinsics: dict[str, Any] | None = None
        overview_sources: dict[str, list[tuple[str, Path]]] = {
            name: [] for name in profile.public_modalities
        }
        first_source_step = int(waypoints[0].get("source_sim_step"))

        for index, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict) or waypoint.get("index") != index:
                raise FixedDemoBundleError("Fixed-demo waypoint indices are not contiguous")
            source_step = int(waypoint.get("source_sim_step"))
            if source_step < first_source_step:
                raise FixedDemoBundleError("Fixed-demo source steps are not monotonic")
            observation_relative = waypoint.get("observation_file")
            if not isinstance(observation_relative, str):
                raise FixedDemoBundleError("Fixed-demo waypoint omitted observation_file")
            observation_path = _path_inside(
                master_root, observation_relative, "P6 observation"
            )
            if not observation_path.is_file() or observation_path.is_symlink():
                raise FixedDemoBundleError(f"P6 observation is missing: {observation_path}")
            if file_sha256(observation_path) != waypoint.get("sha256"):
                raise FixedDemoBundleError("P6 observation JSON hash mismatch")
            source_observation = _read_json(observation_path, "P6 observation")
            validate_p6_observation(
                source_observation,
                task=task,
                master_root=master_root,
                initial_observation=index == 0,
                require_initial_annotations=True,
            )

            frame_id = f"frame_{index:06d}"
            frame_relative = Path("frames") / frame_id
            frame_root = temporary / frame_relative
            modalities: dict[str, Any] = {}
            source_modalities = source_observation["modalities"]
            for name in profile.public_modalities:
                if name.endswith("_depth"):
                    public, overview = _copy_depth_modality(
                        name=name,
                        source=source_modalities[name],
                        frame_root=frame_root,
                        frame_relative=frame_relative,
                        master_root=master_root,
                        bundle_root=temporary,
                    )
                else:
                    public, overview = _copy_image_modality(
                        name=name,
                        source=source_modalities[name],
                        frame_root=frame_root,
                        frame_relative=frame_relative,
                        master_root=master_root,
                        bundle_root=temporary,
                    )
                modalities[name] = public
                overview_sources[name].append((frame_id, overview))

            relative_step = source_step - first_source_step
            state = _public_robot_state(source_observation["robot_state"])
            state_records.append(
                {
                    "observation_id": frame_id,
                    "record_index": index,
                    "relative_sim_step": relative_step,
                    **state,
                }
            )
            trajectory_records.append(
                {
                    "frame_index": index,
                    "relative_sim_step": relative_step,
                    "observation": f"frames/{frame_id}/observation.json",
                    "state_record_index": index,
                    "representation": "observed_expert_waypoint",
                }
            )

            observation: dict[str, Any] = {
                "schema_version": "univtac.agent_observation.v1",
                "observation_id": frame_id,
                "task": task,
                "observation_profile": profile.name,
                "modalities": modalities,
                "state_ref": {"file": "state.jsonl", "record_index": index},
            }
            if profile.expose_camera_intrinsics:
                frame_intrinsics = {
                    camera_name: {
                        "intrinsic_matrix_3x3": _public_intrinsics(
                            source_observation["camera_calibration"][camera_name][
                                "intrinsic_matrix_3x3"
                            ],
                            camera_name,
                        )
                    }
                    for camera_name in profile.camera_names
                }
                if intrinsics is None:
                    intrinsics = frame_intrinsics
                elif frame_intrinsics != intrinsics:
                    raise FixedDemoBundleError("Camera intrinsics changed within the demo")
                calibration_ref: dict[str, Any] = {
                    "intrinsics_file": "camera_intrinsics.json"
                }
                if profile.expose_camera_extrinsics:
                    extrinsic_records.append(
                        {
                            "observation_id": frame_id,
                            "record_index": index,
                            "cameras": {
                                camera_name: {
                                    "extrinsics": _public_extrinsics(
                                        source_observation["camera_calibration"][
                                            camera_name
                                        ]["extrinsics"],
                                        camera_name,
                                    )
                                }
                                for camera_name in profile.camera_names
                            },
                        }
                    )
                    calibration_ref.update(
                        {
                            "extrinsics_file": "camera_extrinsics.jsonl",
                            "extrinsics_record_index": index,
                        }
                    )
                observation["camera_calibration_ref"] = calibration_ref
            if index == 0 and annotations.enabled_features:
                observation["annotations"] = _project_annotations(
                    source_observation.get("annotations"),
                    profile=profile,
                    capabilities=annotations,
                    frame_relative=frame_relative,
                    master_root=master_root,
                    bundle_root=temporary,
                )
            observation_file = frame_relative / "observation.json"
            _write_json(temporary / observation_file, observation)
            frame_records.append(
                {
                    "frame_index": index,
                    "observation_id": frame_id,
                    "observation": _json_artifact(
                        observation_file, temporary, "application/json"
                    ),
                }
            )

        _write_jsonl(temporary / "state.jsonl", state_records)
        _write_jsonl(temporary / "trajectory.jsonl", trajectory_records)
        calibration_manifest: dict[str, Any] | None = None
        if profile.expose_camera_intrinsics:
            assert intrinsics is not None
            _write_json(
                temporary / "camera_intrinsics.json",
                {
                    "schema_version": "univtac.camera_intrinsics.v1",
                    "cameras": intrinsics,
                },
            )
            calibration_manifest = {
                "intrinsics": _json_artifact(
                    Path("camera_intrinsics.json"), temporary, "application/json"
                )
            }
            if profile.expose_camera_extrinsics:
                _write_jsonl(
                    temporary / "camera_extrinsics.jsonl", extrinsic_records
                )
                calibration_manifest["extrinsics"] = _json_artifact(
                    Path("camera_extrinsics.jsonl"),
                    temporary,
                    "application/x-ndjson",
                )

        sampled_indices = contact_sheet_indices(len(waypoints))
        contact_sheets: dict[str, Any] = {}
        for modality, sources in overview_sources.items():
            selected = [sources[index] for index in sampled_indices]
            relative = Path("overview/contact_sheets") / f"{modality}.png"
            _build_contact_sheet(selected, temporary / relative)
            contact_sheets[modality] = _json_artifact(
                relative, temporary, "image/png"
            )
            contact_sheets[modality]["content_image"] = True

        manifest: dict[str, Any] = {
            "schema_version": "univtac.fixed_expert_demo_bundle.v1",
            "task": task,
            "icl_condition": "fixed_demo",
            "observation_profile": profile.to_manifest(),
            "annotations": annotations.to_manifest(),
            "demonstration": {
                "episode_outcome": "successful expert demonstration",
                "task_start_condition": "ungrasped",
                "representation": "observed_expert_waypoints",
                "actions_present": False,
                "step_eef_actions_present": False,
                "frame_count": len(waypoints),
                "timebase": {
                    "field": "relative_sim_step",
                    "unit": "physics_step",
                    "origin": "first demonstrated waypoint",
                },
            },
            "trajectory": _json_artifact(
                Path("trajectory.jsonl"), temporary, "application/x-ndjson"
            ),
            "state": _json_artifact(
                Path("state.jsonl"), temporary, "application/x-ndjson"
            ),
            "frames": frame_records,
            "overview": {
                "sampling": {
                    "rule": "uniform_endpoint_preserving_v1",
                    "maximum_frames_per_sheet": MAX_CONTACT_SHEET_FRAMES,
                    "sampled_frame_indices": sampled_indices,
                },
                "contact_sheets": contact_sheets,
            },
        }
        if calibration_manifest is not None:
            manifest["camera_calibration"] = calibration_manifest
        manifest["integrity"] = _bundle_content_integrity(temporary)
        _write_json(temporary / "manifest.json", manifest)
        validate_fixed_demo_bundle(
            temporary,
            expected_task=task,
            expected_profile=profile,
            expected_annotations=annotations,
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    public_manifest = _read_json(destination / "manifest.json", "projected bundle")
    return {
        "schema_version": "univtac.fixed_demo_projection_receipt.v1",
        "task": task,
        "profile": profile.to_manifest(),
        "annotations": annotations.to_manifest(),
        "source_master_manifest": {
            "path": str(manifest_path),
            "sha256": spec.manifest_sha256,
        },
        "agent_bundle": {
            "path": str(destination),
            "manifest_sha256": file_sha256(destination / "manifest.json"),
            "content_integrity": public_manifest["integrity"],
        },
    }


def _validate_relative_artifact(
    value: Any,
    *,
    bundle_root: Path,
    referenced: set[Path],
    label: str,
    expected_media_type: str | None = None,
    expected_fields: set[str] | None = None,
) -> Path:
    if not isinstance(value, dict):
        raise FixedDemoBundleError(f"{label} must be an artifact object")
    if expected_fields is not None and set(value) != expected_fields:
        raise FixedDemoBundleError(f"{label} artifact fields do not match")
    raw_path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise FixedDemoBundleError(f"{label} omitted path or SHA-256")
    relative = _safe_relative_path(raw_path, label)
    path = (bundle_root / relative).resolve()
    if not path.is_relative_to(bundle_root) or not path.is_file() or path.is_symlink():
        raise FixedDemoBundleError(f"{label} escaped the Agent bundle")
    if expected_media_type is not None and value.get("media_type") != expected_media_type:
        raise FixedDemoBundleError(f"{label} has the wrong media type")
    if file_sha256(path) != digest:
        raise FixedDemoBundleError(f"{label} hash mismatch")
    referenced.add(path)
    return path


def _assert_no_absolute_strings(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_absolute_strings(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_absolute_strings(child, f"{label}[{index}]")
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("file://")):
        raise FixedDemoBundleError(f"Absolute host path leaked at {label}")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FixedDemoBundleError(f"Could not read {label}: {exc}") from exc
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FixedDemoBundleError(f"Invalid {label} line {index}: {exc}") from exc
        if not isinstance(value, dict):
            raise FixedDemoBundleError(f"{label} line {index} must be an object")
        records.append(value)
    return records


def validate_fixed_demo_bundle(
    bundle_root: Path,
    *,
    expected_task: str,
    expected_profile: ObservationProfile,
    expected_annotations: AnnotationCapabilities,
) -> dict[str, Any]:
    """Validate exact fields, paths, hashes, and physical Profile projection."""

    bundle_root = bundle_root.resolve()
    manifest_path = bundle_root / "manifest.json"
    manifest = _read_json(manifest_path, "Agent-visible fixed-demo manifest")
    _assert_no_absolute_strings(manifest, "manifest")
    required_manifest_fields = {
        "schema_version",
        "task",
        "icl_condition",
        "observation_profile",
        "annotations",
        "demonstration",
        "trajectory",
        "state",
        "frames",
        "overview",
        "integrity",
    }
    if expected_profile.expose_camera_intrinsics:
        required_manifest_fields.add("camera_calibration")
    if set(manifest) != required_manifest_fields:
        raise FixedDemoBundleError("Agent-visible manifest fields do not match")
    if manifest.get("schema_version") != "univtac.fixed_expert_demo_bundle.v1":
        raise FixedDemoBundleError("Unsupported Agent-visible fixed-demo schema")
    if manifest.get("task") != expected_task or manifest.get("icl_condition") != "fixed_demo":
        raise FixedDemoBundleError("Agent-visible fixed-demo task/condition mismatch")
    if manifest.get("observation_profile") != expected_profile.to_manifest():
        raise FixedDemoBundleError("Agent-visible fixed-demo Profile mismatch")
    if manifest.get("annotations") != expected_annotations.to_manifest():
        raise FixedDemoBundleError("Agent-visible fixed-demo annotation mismatch")
    demonstration = manifest.get("demonstration")
    if not isinstance(demonstration, dict) or set(demonstration) != {
        "episode_outcome",
        "task_start_condition",
        "representation",
        "actions_present",
        "step_eef_actions_present",
        "frame_count",
        "timebase",
    }:
        raise FixedDemoBundleError("Agent-visible demonstration summary fields changed")
    if (
        demonstration.get("episode_outcome") != "successful expert demonstration"
        or demonstration.get("task_start_condition") != "ungrasped"
        or demonstration.get("representation") != "observed_expert_waypoints"
        or demonstration.get("actions_present") is not False
        or demonstration.get("step_eef_actions_present") is not False
    ):
        raise FixedDemoBundleError("Agent-visible demonstration semantics are unsafe")
    if demonstration.get("timebase") != {
        "field": "relative_sim_step",
        "unit": "physics_step",
        "origin": "first demonstrated waypoint",
    }:
        raise FixedDemoBundleError("Agent-visible demonstration timebase changed")
    frame_count = demonstration.get("frame_count")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise FixedDemoBundleError("Agent-visible demonstration frame count is invalid")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise FixedDemoBundleError("Agent-visible frame index is incomplete")

    referenced: set[Path] = {manifest_path}
    state_path = _validate_relative_artifact(
        manifest["state"],
        bundle_root=bundle_root,
        referenced=referenced,
        label="state",
        expected_media_type="application/x-ndjson",
        expected_fields={"path", "sha256", "media_type"},
    )
    trajectory_path = _validate_relative_artifact(
        manifest["trajectory"],
        bundle_root=bundle_root,
        referenced=referenced,
        label="trajectory",
        expected_media_type="application/x-ndjson",
        expected_fields={"path", "sha256", "media_type"},
    )
    state_records = _read_jsonl(state_path, "state.jsonl")
    trajectory_records = _read_jsonl(trajectory_path, "trajectory.jsonl")
    if len(state_records) != frame_count or len(trajectory_records) != frame_count:
        raise FixedDemoBundleError("Agent-visible JSONL record counts are incomplete")
    previous_relative_step = -1
    for record_index, state in enumerate(state_records):
        _assert_no_absolute_strings(state, f"state[{record_index}]")
        expected_state_fields = {
            "observation_id",
            "record_index",
            "relative_sim_step",
            *ROBOT_STATE_FIELDS,
        }
        if set(state) != expected_state_fields:
            raise FixedDemoBundleError("Agent-visible state fields do not match")
        if state.get("observation_id") != f"frame_{record_index:06d}":
            raise FixedDemoBundleError("Agent-visible state observation_id mismatch")
        if state.get("record_index") != record_index:
            raise FixedDemoBundleError("Agent-visible state indices are not contiguous")
        relative_step = state.get("relative_sim_step")
        if (
            not isinstance(relative_step, int)
            or relative_step < 0
            or relative_step < previous_relative_step
        ):
            raise FixedDemoBundleError("Agent-visible relative simulator steps are invalid")
        previous_relative_step = relative_step
        _public_robot_state({field: state[field] for field in ROBOT_STATE_FIELDS})

    calibration = manifest.get("camera_calibration")
    intrinsics_records: dict[str, Any] | None = None
    extrinsics_records: list[dict[str, Any]] | None = None
    if expected_profile.expose_camera_intrinsics:
        if not isinstance(calibration, dict):
            raise FixedDemoBundleError("Agent-visible calibration manifest is missing")
        expected_calibration_fields = {"intrinsics"}
        if expected_profile.expose_camera_extrinsics:
            expected_calibration_fields.add("extrinsics")
        if set(calibration) != expected_calibration_fields:
            raise FixedDemoBundleError("Agent-visible calibration fields do not match")
        intrinsics_path = _validate_relative_artifact(
            calibration["intrinsics"],
            bundle_root=bundle_root,
            referenced=referenced,
            label="camera intrinsics",
            expected_media_type="application/json",
            expected_fields={"path", "sha256", "media_type"},
        )
        intrinsics_document = _read_json(intrinsics_path, "camera intrinsics")
        _assert_no_absolute_strings(intrinsics_document, "camera_intrinsics")
        if set(intrinsics_document) != {"schema_version", "cameras"} or (
            intrinsics_document.get("schema_version") != "univtac.camera_intrinsics.v1"
        ):
            raise FixedDemoBundleError("Agent-visible intrinsics schema is invalid")
        intrinsics_records = intrinsics_document.get("cameras")
        if not isinstance(intrinsics_records, dict) or set(intrinsics_records) != set(
            expected_profile.camera_names
        ):
            raise FixedDemoBundleError("Agent-visible intrinsics cameras do not match")
        for camera_name, camera in intrinsics_records.items():
            if not isinstance(camera, dict) or set(camera) != {"intrinsic_matrix_3x3"}:
                raise FixedDemoBundleError("Agent-visible intrinsics fields changed")
            _public_intrinsics(camera["intrinsic_matrix_3x3"], camera_name)
        if expected_profile.expose_camera_extrinsics:
            extrinsics_path = _validate_relative_artifact(
                calibration["extrinsics"],
                bundle_root=bundle_root,
                referenced=referenced,
                label="camera extrinsics",
                expected_media_type="application/x-ndjson",
                expected_fields={"path", "sha256", "media_type"},
            )
            extrinsics_records = _read_jsonl(extrinsics_path, "camera_extrinsics.jsonl")
            if len(extrinsics_records) != frame_count:
                raise FixedDemoBundleError("Agent-visible extrinsics are incomplete")

    for index, (frame, trajectory) in enumerate(zip(frames, trajectory_records)):
        frame_id = f"frame_{index:06d}"
        if not isinstance(frame, dict) or set(frame) != {
            "frame_index",
            "observation_id",
            "observation",
        }:
            raise FixedDemoBundleError("Agent-visible frame index fields changed")
        if frame.get("frame_index") != index or frame.get("observation_id") != frame_id:
            raise FixedDemoBundleError("Agent-visible frame indices are not contiguous")
        observation_path = _validate_relative_artifact(
            frame["observation"],
            bundle_root=bundle_root,
            referenced=referenced,
            label=f"frame {index} observation",
            expected_media_type="application/json",
            expected_fields={"path", "sha256", "media_type"},
        )
        observation = _read_json(observation_path, f"frame {index} observation")
        _assert_no_absolute_strings(observation, f"frame[{index}]")
        expected_observation_fields = {
            "schema_version",
            "observation_id",
            "task",
            "observation_profile",
            "modalities",
            "state_ref",
        }
        if expected_profile.expose_camera_intrinsics:
            expected_observation_fields.add("camera_calibration_ref")
        if index == 0 and expected_annotations.enabled_features:
            expected_observation_fields.add("annotations")
        if set(observation) != expected_observation_fields:
            raise FixedDemoBundleError(f"Frame {index} observation fields do not match")
        if (
            observation.get("schema_version") != "univtac.agent_observation.v1"
            or observation.get("observation_id") != frame_id
            or observation.get("task") != expected_task
            or observation.get("observation_profile") != expected_profile.name
        ):
            raise FixedDemoBundleError(f"Frame {index} observation identity mismatch")
        state_ref = observation.get("state_ref")
        if state_ref != {"file": "state.jsonl", "record_index": index}:
            raise FixedDemoBundleError(f"Frame {index} state reference mismatch")
        modalities = observation.get("modalities")
        if not isinstance(modalities, dict) or set(modalities) != set(
            expected_profile.public_modalities
        ):
            raise FixedDemoBundleError(f"Frame {index} modalities do not match Profile")
        for name, artifact in modalities.items():
            if name.endswith("_depth"):
                if not isinstance(artifact, dict) or set(artifact) != {
                    "depth_m",
                    "valid_mask",
                    "visualization",
                    "visualization_range_m",
                    "statistics",
                }:
                    raise FixedDemoBundleError(f"Frame {index} {name} fields changed")
                depth_path = _validate_relative_artifact(
                    artifact["depth_m"],
                    bundle_root=bundle_root,
                    referenced=referenced,
                    label=f"frame {index} {name} metric depth",
                    expected_media_type="application/x-npy",
                    expected_fields={
                        "path",
                        "sha256",
                        "media_type",
                        "dtype",
                        "shape",
                        "unit",
                    },
                )
                depth = np.load(depth_path, allow_pickle=False)
                if depth.shape != (270, 480) or depth.dtype != np.float32:
                    raise FixedDemoBundleError("Projected metric depth shape/dtype changed")
                if (
                    artifact["depth_m"].get("dtype") != "float32"
                    or artifact["depth_m"].get("shape") != [270, 480]
                    or artifact["depth_m"].get("unit") != "metre"
                ):
                    raise FixedDemoBundleError("Projected metric depth metadata changed")
                for preview in ("valid_mask", "visualization"):
                    _validate_relative_artifact(
                        artifact[preview],
                        bundle_root=bundle_root,
                        referenced=referenced,
                        label=f"frame {index} {name} {preview}",
                        expected_media_type="image/png",
                        expected_fields={
                            "path",
                            "sha256",
                            "media_type",
                            "content_image",
                        },
                    )
                    if artifact[preview].get("content_image") is not True:
                        raise FixedDemoBundleError("Depth display artifact is not public")
            else:
                _validate_relative_artifact(
                    artifact,
                    bundle_root=bundle_root,
                    referenced=referenced,
                    label=f"frame {index} {name}",
                    expected_media_type="image/png",
                    expected_fields={
                        "path",
                        "sha256",
                        "media_type",
                        "content_image",
                    },
                )
                if artifact.get("content_image") is not True:
                    raise FixedDemoBundleError("Profile image is not marked public")
        if expected_profile.expose_camera_intrinsics:
            calibration_ref = observation.get("camera_calibration_ref")
            expected_ref: dict[str, Any] = {
                "intrinsics_file": "camera_intrinsics.json"
            }
            if expected_profile.expose_camera_extrinsics:
                expected_ref.update(
                    {
                        "extrinsics_file": "camera_extrinsics.jsonl",
                        "extrinsics_record_index": index,
                    }
                )
                assert extrinsics_records is not None
                extrinsic_record = extrinsics_records[index]
                _assert_no_absolute_strings(
                    extrinsic_record, f"camera_extrinsics[{index}]"
                )
                if not isinstance(extrinsic_record, dict) or set(extrinsic_record) != {
                    "observation_id",
                    "record_index",
                    "cameras",
                }:
                    raise FixedDemoBundleError("Agent-visible extrinsic fields changed")
                if (
                    extrinsic_record.get("observation_id") != frame_id
                    or extrinsic_record.get("record_index") != index
                ):
                    raise FixedDemoBundleError("Agent-visible extrinsic index mismatch")
                cameras = extrinsic_record.get("cameras")
                if not isinstance(cameras, dict) or set(cameras) != set(
                    expected_profile.camera_names
                ):
                    raise FixedDemoBundleError("Agent-visible extrinsic cameras mismatch")
                for camera_name, camera in cameras.items():
                    if not isinstance(camera, dict) or set(camera) != {"extrinsics"}:
                        raise FixedDemoBundleError("Agent-visible extrinsic fields changed")
                    _public_extrinsics(camera["extrinsics"], camera_name)
            if calibration_ref != expected_ref:
                raise FixedDemoBundleError(f"Frame {index} calibration reference mismatch")
        if index == 0 and expected_annotations.enabled_features:
            public_annotations = observation.get("annotations")
            if not isinstance(public_annotations, dict) or set(public_annotations) != set(
                expected_profile.camera_names
            ):
                raise FixedDemoBundleError("Initial public annotation cameras mismatch")
            for camera_name, camera in public_annotations.items():
                if not isinstance(camera, dict) or set(camera) != set(PUBLIC_ROLES):
                    raise FixedDemoBundleError("Initial public annotation roles mismatch")
                for role, annotation in camera.items():
                    expected_fields = {"public_role", "visible"}
                    if expected_annotations.provide_bbox:
                        expected_fields.update(
                            {"bbox_xyxy_exclusive", "bbox_overlay"}
                        )
                    if expected_annotations.provide_mask:
                        expected_fields.update({"mask", "mask_overlay"})
                    if not isinstance(annotation, dict) or set(annotation) != expected_fields:
                        raise FixedDemoBundleError("Initial annotation fields mismatch")
                    if annotation.get("public_role") != role:
                        raise FixedDemoBundleError("Initial annotation public role mismatch")
                    if not isinstance(annotation.get("visible"), bool):
                        raise FixedDemoBundleError("Initial annotation visibility is invalid")
                    if expected_annotations.provide_bbox:
                        bbox = annotation.get("bbox_xyxy_exclusive")
                        if bbox is not None and (
                            not isinstance(bbox, list)
                            or len(bbox) != 4
                            or any(not isinstance(item, int) for item in bbox)
                            or not (0 <= bbox[0] < bbox[2] <= 480)
                            or not (0 <= bbox[1] < bbox[3] <= 270)
                        ):
                            raise FixedDemoBundleError("Initial public bbox is invalid")
                        if annotation["visible"] != (bbox is not None):
                            raise FixedDemoBundleError("Initial bbox visibility mismatch")
                        _validate_relative_artifact(
                            annotation["bbox_overlay"],
                            bundle_root=bundle_root,
                            referenced=referenced,
                            label=f"{camera_name}/{role} bbox overlay",
                            expected_media_type="image/png",
                            expected_fields={
                                "path",
                                "sha256",
                                "media_type",
                                "content_image",
                            },
                        )
                        if annotation["bbox_overlay"].get("content_image") is not True:
                            raise FixedDemoBundleError("BBox overlay is not marked public")
                    if expected_annotations.provide_mask:
                        mask_path = _validate_relative_artifact(
                            annotation["mask"],
                            bundle_root=bundle_root,
                            referenced=referenced,
                            label=f"{camera_name}/{role} mask",
                            expected_media_type="image/png",
                            expected_fields={
                                "path",
                                "sha256",
                                "media_type",
                                "content_image",
                                "mode",
                                "background_value",
                                "foreground_value",
                            },
                        )
                        with Image.open(mask_path) as mask_image:
                            mask_array = np.asarray(mask_image)
                            if mask_image.mode != "L" or mask_image.size != (480, 270):
                                raise FixedDemoBundleError("Public mask must be single-channel")
                        if not set(np.unique(mask_array)).issubset({0, 255}):
                            raise FixedDemoBundleError("Public mask must be binary")
                        mask = mask_array == 255
                        if annotation["visible"] != bool(mask.any()):
                            raise FixedDemoBundleError("Initial mask visibility mismatch")
                        if expected_annotations.provide_bbox and (
                            annotation["bbox_xyxy_exclusive"]
                            != bbox_xyxy_exclusive(mask)
                        ):
                            raise FixedDemoBundleError("Initial bbox and mask disagree")
                        if (
                            annotation["mask"].get("content_image") is not False
                            or annotation["mask"].get("mode") != "L"
                            or annotation["mask"].get("background_value") != 0
                            or annotation["mask"].get("foreground_value") != 255
                        ):
                            raise FixedDemoBundleError("Public mask metadata changed")
                        _validate_relative_artifact(
                            annotation["mask_overlay"],
                            bundle_root=bundle_root,
                            referenced=referenced,
                            label=f"{camera_name}/{role} mask overlay",
                            expected_media_type="image/png",
                            expected_fields={
                                "path",
                                "sha256",
                                "media_type",
                                "content_image",
                            },
                        )
                        if annotation["mask_overlay"].get("content_image") is not True:
                            raise FixedDemoBundleError("Mask overlay is not marked public")
        elif "annotations" in observation:
            raise FixedDemoBundleError("Annotations leaked after the initial frame")

        expected_trajectory = {
            "frame_index": index,
            "relative_sim_step": state_records[index]["relative_sim_step"],
            "observation": f"frames/{frame_id}/observation.json",
            "state_record_index": index,
            "representation": "observed_expert_waypoint",
        }
        if trajectory != expected_trajectory:
            raise FixedDemoBundleError("Agent-visible trajectory index mismatch")

    overview = manifest.get("overview")
    if not isinstance(overview, dict) or set(overview) != {
        "sampling",
        "contact_sheets",
    }:
        raise FixedDemoBundleError("Agent-visible overview fields changed")
    if overview.get("sampling") != {
        "rule": "uniform_endpoint_preserving_v1",
        "maximum_frames_per_sheet": MAX_CONTACT_SHEET_FRAMES,
        "sampled_frame_indices": contact_sheet_indices(frame_count),
    }:
        raise FixedDemoBundleError("Agent-visible contact-sheet sampling changed")
    contact_sheets = overview.get("contact_sheets")
    if not isinstance(contact_sheets, dict) or set(contact_sheets) != set(
        expected_profile.public_modalities
    ):
        raise FixedDemoBundleError("Agent-visible contact sheets do not match Profile")
    for modality, artifact in contact_sheets.items():
        _validate_relative_artifact(
            artifact,
            bundle_root=bundle_root,
            referenced=referenced,
            label=f"{modality} contact sheet",
            expected_media_type="image/png",
            expected_fields={
                "path",
                "sha256",
                "media_type",
                "content_image",
            },
        )
        if artifact.get("content_image") is not True:
            raise FixedDemoBundleError("Contact sheet is not marked public")

    actual_files = {
        path.resolve()
        for path in bundle_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != referenced:
        extras = sorted(str(path.relative_to(bundle_root)) for path in actual_files - referenced)
        missing = sorted(str(path.relative_to(bundle_root)) for path in referenced - actual_files)
        raise FixedDemoBundleError(
            f"Agent bundle file allowlist mismatch; extras={extras}, missing={missing}"
        )
    if manifest.get("integrity") != _bundle_content_integrity(bundle_root):
        raise FixedDemoBundleError("Agent bundle aggregate integrity mismatch")
    return manifest
