"""Capture and freeze sensor-enhanced P6 expert observation trajectories.

The frozen scripted expert remains the motion/provenance source.  This module
records only observations at its original waypoints; it never invents or
exports ``step_eef`` actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .artifacts import (
    file_sha256,
    marker_grid_health,
    require_initial_tactile_health,
    save_composite,
    tensor_to_rgb,
)
from .benchmark_observations import (
    bbox_xyxy_exclusive,
    camera_plane,
    image_artifact,
    instance_role_mask,
    normalize_instance_mapping,
    robot_base_camera_extrinsic,
    save_annotation_artifacts,
    save_depth_artifacts,
    to_numpy,
)
from .benchmark_profiles import get_observation_profile
from .benchmark_tasks import get_benchmark_task


P6_PROFILE = get_observation_profile(6)
P6_MODALITIES = frozenset(P6_PROFILE.public_modalities)
P6_ROBOT_STATE = frozenset(P6_PROFILE.public_robot_state)
INSTANCE_DATA_TYPE = "instance_id_segmentation_fast"


class P6ExpertMasterError(ValueError):
    pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P6ExpertMasterError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise P6ExpertMasterError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _camera_info_for_instance(camera: Any) -> Any:
    info: Any = camera.data.info
    if isinstance(info, list):
        if len(info) != 1:
            raise P6ExpertMasterError(
                "Expected exactly one environment in camera metadata"
            )
        info = info[0]
    if isinstance(info, dict):
        return info.get(INSTANCE_DATA_TYPE, {})
    return {}


def validate_fixed_expert_source(
    *,
    manifest_path: Path,
    source_hdf5: Path,
    task: str,
    seed: int,
) -> dict[str, Any]:
    """Authenticate one replay-proven frozen source without modifying it."""

    manifest_path = manifest_path.resolve()
    source_hdf5 = source_hdf5.resolve()
    if not manifest_path.is_file():
        raise P6ExpertMasterError(
            f"Frozen expert manifest does not exist: {manifest_path}"
        )
    if not source_hdf5.is_file():
        raise P6ExpertMasterError(f"Source HDF5 does not exist: {source_hdf5}")
    manifest = _read_json(manifest_path, "frozen expert manifest")
    if manifest.get("schema_version") != "univtac.fixed_expert_trajectory.v1":
        raise P6ExpertMasterError("Frozen expert manifest has an unsupported schema")
    if manifest.get("task") != task or manifest.get("seed") != seed:
        raise P6ExpertMasterError("Frozen expert task/seed does not match the replay")
    if manifest.get("positive_example") is not True:
        raise P6ExpertMasterError("Frozen expert is not a positive demonstration")
    if manifest.get("agent_visible_checker_details") is not False:
        raise P6ExpertMasterError("Frozen expert checker-visibility contract is unsafe")
    replay = manifest.get("replay_verification")
    if not isinstance(replay, dict) or replay.get("official_task_success") is not True:
        raise P6ExpertMasterError("Frozen expert lacks successful replay verification")
    source = manifest.get("source")
    trajectory = source.get("trajectory") if isinstance(source, dict) else None
    if not isinstance(trajectory, dict):
        raise P6ExpertMasterError("Frozen expert omitted source trajectory provenance")
    recorded_path = trajectory.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != source_hdf5:
        raise P6ExpertMasterError("Frozen expert refers to a different source HDF5")
    actual_sha256 = file_sha256(source_hdf5)
    if trajectory.get("sha256") != actual_sha256:
        raise P6ExpertMasterError("Source HDF5 hash does not match the frozen expert")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "source_hdf5": source_hdf5,
        "source_hdf5_sha256": actual_sha256,
    }


def capture_p6_observation(
    *,
    task: Any,
    task_name: str,
    observation_id: str,
    observation_root: Path,
    initial_observation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize one actual simulator waypoint with the public P6 schema."""

    raw = task._get_observations()
    if raw.get("actor"):
        raise P6ExpertMasterError("P6 master capture must not request actor observations")
    obs_dir = observation_root.resolve() / observation_id
    modalities: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    public_annotations: dict[str, Any] = {}
    panels: list[tuple[str, np.ndarray]] = []
    task_spec = get_benchmark_task(task_name)

    robot = task._robot_manager.robot
    robot_position = to_numpy(robot.data.root_link_pos_w[0])
    robot_quaternion = to_numpy(robot.data.root_link_quat_w[0])

    for camera_name in P6_PROFILE.camera_names:
        camera_observation = raw.get("observation", {}).get(camera_name, {})
        if set(camera_observation) != {"rgb", "depth"}:
            raise P6ExpertMasterError(
                f"P6 {camera_name} observation must contain exactly RGB and depth"
            )
        rgb = tensor_to_rgb(camera_observation["rgb"])
        modalities[f"{camera_name}_rgb"] = image_artifact(
            obs_dir / camera_name / "rgb.png", rgb
        )
        panels.append((f"{camera_name}_rgb", rgb))
        depth_artifact, _, depth_preview = save_depth_artifacts(
            obs_dir / camera_name,
            camera_observation["depth"],
        )
        modalities[f"{camera_name}_depth"] = depth_artifact
        panels.append((f"{camera_name}_depth_m", depth_preview))

        camera = task._camera_manager.cameras[camera_name]
        matrix = to_numpy(camera.data.intrinsic_matrices[0]).astype(np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise P6ExpertMasterError(f"Invalid {camera_name} camera intrinsics")
        camera_position = to_numpy(camera.data.pos_w[0])
        camera_quaternion_ros = to_numpy(camera.data.quat_w_ros[0])
        extrinsic = robot_base_camera_extrinsic(
            robot_position_world=robot_position,
            robot_quaternion_world_wxyz=robot_quaternion,
            camera_position_world=camera_position,
            camera_quaternion_world_ros_wxyz=camera_quaternion_ros,
        )
        calibration[camera_name] = {
            "intrinsic_matrix_3x3": matrix.tolist(),
            "extrinsics": {
                "matrix_T_robot_base_camera_ros_4x4": extrinsic,
                "camera_convention": "ROS optical: +Z forward, -Y up",
                "translation_unit": "metre",
            },
        }
        if camera_name == "wrist":
            ee_position = np.asarray(
                task._robot_manager.get_ee_pose().p, dtype=np.float64
            )
            distance = float(np.linalg.norm(camera_position - ee_position))
            if distance >= 0.20:
                raise P6ExpertMasterError(
                    "Wrist extrinsics appear stale: camera-to-EEF distance "
                    f"is {distance:.4f} m"
                )

        if initial_observation:
            instance = camera_plane(camera.data.output[INSTANCE_DATA_TYPE])
            mapping = normalize_instance_mapping(_camera_info_for_instance(camera))
            if not mapping:
                raise P6ExpertMasterError(
                    f"{camera_name} instance metadata is unavailable"
                )
            camera_annotations: dict[str, Any] = {}
            for role, private_name in (
                ("manipulated_object", task_spec.manipulated_prim_name),
                ("goal_fixture", task_spec.goal_prim_name),
            ):
                mask, _ = instance_role_mask(instance, mapping, private_name)
                annotation, annotation_panels = save_annotation_artifacts(
                    obs_dir / "annotations" / camera_name,
                    role=role,
                    mask=mask,
                    rgb=rgb,
                    provide_bbox=True,
                    provide_mask=True,
                )
                camera_annotations[role] = annotation
                panels.extend(
                    (f"{camera_name}_{label}", panel)
                    for label, panel in annotation_panels
                )
            public_annotations[camera_name] = camera_annotations

    tactile_health: dict[str, Any] = {}
    for internal_name, public_name in (
        ("left_tactile", "left_tactile_rgb"),
        ("right_tactile", "right_tactile_rgb"),
    ):
        tactile_observation = raw.get("tactile", {}).get(internal_name, {})
        if set(tactile_observation) != {"rgb_marker"}:
            raise P6ExpertMasterError(
                f"P6 {internal_name} must contain only rgb_marker"
            )
        tactile = tensor_to_rgb(tactile_observation["rgb_marker"])
        modalities[public_name] = image_artifact(
            obs_dir / internal_name / "rgb_marker.png", tactile
        )
        panels.append((public_name, tactile))
        tactile_health[internal_name] = marker_grid_health(tactile)
    require_initial_tactile_health(
        tactile_health,
        initial_observation=initial_observation,
    )

    joint_position = to_numpy(robot.data.joint_pos[0]).astype(np.float64)
    joint_velocity = to_numpy(robot.data.joint_vel[0]).astype(np.float64)
    if joint_position.shape != (9,) or joint_velocity.shape != (9,):
        raise P6ExpertMasterError("Expected a 9-DoF Franka robot state")
    ee_pose = np.asarray(
        task._robot_manager.get_ee_pose().totensor(), dtype=np.float64
    )
    robot_state = {
        "joint_position_9d": joint_position.tolist(),
        "joint_velocity_9d": joint_velocity.tolist(),
        "gripper_width_m": float(joint_position[-2] + joint_position[-1]),
        "end_effector_pose_robot_base_wxyz_7d": ee_pose.tolist(),
    }
    composite = save_composite(panels, obs_dir / "composite.png")
    composite["media_type"] = "image/png"
    composite["content_image"] = False
    observation = {
        "schema_version": "univtac.embodied_observation.v1",
        "observation_id": observation_id,
        "task": task_name,
        "observation_profile": P6_PROFILE.name,
        "modalities": modalities,
        "robot_state": robot_state,
        "camera_calibration": calibration,
        "artifacts": {"composite": composite},
    }
    if public_annotations:
        observation["annotations"] = public_annotations
    _write_json(obs_dir / "observation.json", observation)
    return observation, tactile_health


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise P6ExpertMasterError(f"{label} must be a finite {size}-vector") from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise P6ExpertMasterError(f"{label} must be a finite {size}-vector")
    return vector


def _validate_artifact(
    value: Any,
    *,
    master_root: Path,
    expected_media_type: str,
) -> Path:
    if not isinstance(value, dict):
        raise P6ExpertMasterError("Observation artifact must be an object")
    raw_path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(sha256, str):
        raise P6ExpertMasterError("Observation artifact omitted path or SHA-256")
    path = Path(raw_path).resolve()
    if not path.is_relative_to(master_root) or not path.is_file():
        raise P6ExpertMasterError(f"Observation artifact escaped the master: {path}")
    if value.get("media_type") != expected_media_type:
        raise P6ExpertMasterError(f"Unexpected artifact media type: {value}")
    if file_sha256(path) != sha256:
        raise P6ExpertMasterError(f"Observation artifact hash mismatch: {path}")
    return path


def validate_p6_observation(
    observation: dict[str, Any],
    *,
    task: str,
    master_root: Path,
    initial_observation: bool = False,
    require_initial_annotations: bool = False,
) -> None:
    """Fail closed if a captured waypoint is incomplete or exceeds P6."""

    master_root = master_root.resolve()
    required_top_level = {
        "schema_version",
        "observation_id",
        "task",
        "observation_profile",
        "modalities",
        "robot_state",
        "camera_calibration",
        "artifacts",
    }
    if initial_observation and require_initial_annotations:
        required_top_level.add("annotations")
    if set(observation) != required_top_level:
        raise P6ExpertMasterError("P6 observation top-level fields do not match")
    if observation.get("schema_version") != "univtac.embodied_observation.v1":
        raise P6ExpertMasterError("P6 observation has an unsupported schema")
    if observation.get("task") != task:
        raise P6ExpertMasterError("P6 observation task mismatch")
    if observation.get("observation_profile") != P6_PROFILE.name:
        raise P6ExpertMasterError("P6 observation profile mismatch")
    observation_id = observation.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise P6ExpertMasterError("P6 observation omitted observation_id")

    modalities = observation.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != P6_MODALITIES:
        raise P6ExpertMasterError("P6 observation modalities are incomplete")
    for name in (
        "head_rgb",
        "wrist_rgb",
        "left_tactile_rgb",
        "right_tactile_rgb",
    ):
        _validate_artifact(
            modalities[name],
            master_root=master_root,
            expected_media_type="image/png",
        )
    for name in ("head_depth", "wrist_depth"):
        depth = modalities[name]
        if not isinstance(depth, dict) or set(depth) != {
            "depth_m",
            "valid_mask",
            "visualization",
            "visualization_range_m",
            "statistics",
        }:
            raise P6ExpertMasterError(f"{name} artifact is incomplete")
        depth_path = _validate_artifact(
            depth["depth_m"],
            master_root=master_root,
            expected_media_type="application/x-npy",
        )
        depth_array = np.load(depth_path, allow_pickle=False)
        if depth_array.shape != (270, 480) or depth_array.dtype != np.float32:
            raise P6ExpertMasterError(f"{name} must be a 270x480 float32 array")
        if not np.any(np.isfinite(depth_array) & (depth_array > 0)):
            raise P6ExpertMasterError(f"{name} contains no valid metric depth")
        for preview in ("valid_mask", "visualization"):
            _validate_artifact(
                depth[preview],
                master_root=master_root,
                expected_media_type="image/png",
            )

    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict) or set(robot_state) != P6_ROBOT_STATE:
        raise P6ExpertMasterError("P6 robot state is incomplete")
    _finite_vector(robot_state["joint_position_9d"], 9, "joint_position_9d")
    _finite_vector(robot_state["joint_velocity_9d"], 9, "joint_velocity_9d")
    _finite_vector(
        robot_state["end_effector_pose_robot_base_wxyz_7d"],
        7,
        "end_effector_pose_robot_base_wxyz_7d",
    )
    if not np.isfinite(float(robot_state["gripper_width_m"])):
        raise P6ExpertMasterError("gripper_width_m must be finite")

    calibration = observation.get("camera_calibration")
    if not isinstance(calibration, dict) or set(calibration) != {"head", "wrist"}:
        raise P6ExpertMasterError("P6 camera calibration is incomplete")
    for camera_name in ("head", "wrist"):
        camera = calibration[camera_name]
        if not isinstance(camera, dict) or set(camera) != {
            "intrinsic_matrix_3x3",
            "extrinsics",
        }:
            raise P6ExpertMasterError(f"{camera_name} calibration is incomplete")
        intrinsic = np.asarray(camera["intrinsic_matrix_3x3"], dtype=np.float64)
        extrinsic = camera["extrinsics"]
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise P6ExpertMasterError(f"Invalid {camera_name} intrinsics")
        if not isinstance(extrinsic, dict):
            raise P6ExpertMasterError(f"Invalid {camera_name} extrinsics")
        matrix = np.asarray(
            extrinsic.get("matrix_T_robot_base_camera_ros_4x4"), dtype=np.float64
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise P6ExpertMasterError(f"Invalid {camera_name} extrinsic matrix")

    artifacts = observation.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"composite"}:
        raise P6ExpertMasterError("P6 observation composite is incomplete")
    _validate_artifact(
        artifacts["composite"],
        master_root=master_root,
        expected_media_type="image/png",
    )

    if initial_observation and require_initial_annotations:
        annotations = observation.get("annotations")
        if not isinstance(annotations, dict) or set(annotations) != {"head", "wrist"}:
            raise P6ExpertMasterError("Initial P6 annotations are incomplete")
        for camera_name, camera_annotations in annotations.items():
            if not isinstance(camera_annotations, dict) or set(camera_annotations) != {
                "manipulated_object",
                "goal_fixture",
            }:
                raise P6ExpertMasterError(
                    f"Initial {camera_name} annotation roles are incomplete"
                )
            for role, annotation in camera_annotations.items():
                required_annotation_fields = {
                    "public_role",
                    "visible",
                    "bbox_xyxy_exclusive",
                    "bbox_overlay",
                    "mask",
                    "mask_overlay",
                }
                if not isinstance(annotation, dict) or set(annotation) != required_annotation_fields:
                    raise P6ExpertMasterError(
                        f"Initial {camera_name}/{role} annotation is incomplete"
                    )
                if annotation.get("public_role") != role:
                    raise P6ExpertMasterError("Initial annotation public role mismatch")
                mask_path = _validate_artifact(
                    annotation["mask"],
                    master_root=master_root,
                    expected_media_type="image/png",
                )
                with Image.open(mask_path) as mask_image:
                    if mask_image.mode != "L" or mask_image.size != (480, 270):
                        raise P6ExpertMasterError(
                            "Initial annotation mask must be a 480x270 single-channel PNG"
                        )
                    mask_array = np.asarray(mask_image)
                if not set(np.unique(mask_array)).issubset({0, 255}):
                    raise P6ExpertMasterError("Initial annotation mask is not binary")
                mask = mask_array == 255
                if annotation.get("visible") != bool(mask.any()):
                    raise P6ExpertMasterError("Initial annotation visibility mismatch")
                if annotation.get("bbox_xyxy_exclusive") != bbox_xyxy_exclusive(mask):
                    raise P6ExpertMasterError("Initial annotation bbox does not match mask")
                for overlay_name in ("bbox_overlay", "mask_overlay"):
                    _validate_artifact(
                        annotation[overlay_name],
                        master_root=master_root,
                        expected_media_type="image/png",
                    )


def build_p6_master_manifest(
    *,
    task: str,
    seed: int,
    fixed_expert_manifest_path: Path,
    source_hdf5: Path,
    master_root: Path,
    waypoint_records: list[dict[str, Any]],
    physics_action_count: int,
    execution_succeeded: bool,
    official_task_success: bool,
    evaluator_checks: dict[str, Any],
    replay_video: Path,
) -> dict[str, Any]:
    """Freeze a complete P6 master only after replay and artifact validation."""

    source = validate_fixed_expert_source(
        manifest_path=fixed_expert_manifest_path,
        source_hdf5=source_hdf5,
        task=task,
        seed=seed,
    )
    master_root = master_root.resolve()
    if not execution_succeeded:
        raise P6ExpertMasterError("P6 replay did not execute the complete trajectory")
    if not official_task_success:
        raise P6ExpertMasterError("P6 replay did not pass terminal evaluation")
    if evaluator_checks.get("base_task_success") is not True:
        raise P6ExpertMasterError("P6 replay did not pass the base task checker")
    if evaluator_checks.get("settle_steps", 0) < 60:
        raise P6ExpertMasterError("P6 replay stability window was shorter than 60 steps")
    if task == "put_bottle_in_shelf" and (
        evaluator_checks.get("released") is not True
        or evaluator_checks.get("stable_after_release") is not True
    ):
        raise P6ExpertMasterError("P6 bottle replay did not verify release and stability")

    with h5py.File(source_hdf5, "r") as handle:
        source_steps = np.asarray(handle["step"][()]).reshape(-1)
    expected_count = len(source_steps)
    if len(waypoint_records) != expected_count:
        raise P6ExpertMasterError("P6 master did not capture every source waypoint")
    expected_physics_actions = int(source_steps[-1] - source_steps[0] + 1)
    if physics_action_count != expected_physics_actions:
        raise P6ExpertMasterError("P6 replay did not preserve recorded source timing")

    frozen_waypoints: list[dict[str, Any]] = []
    for index, (record, source_step) in enumerate(zip(waypoint_records, source_steps)):
        if record.get("source_frame_index") != index:
            raise P6ExpertMasterError("P6 waypoint indices are not contiguous")
        if record.get("source_sim_step") != int(source_step):
            raise P6ExpertMasterError("P6 waypoint source timestamps do not match")
        observation_file_value = record.get("observation_file")
        if not isinstance(observation_file_value, str):
            raise P6ExpertMasterError("P6 waypoint omitted its observation file")
        observation_file = Path(observation_file_value).resolve()
        if not observation_file.is_relative_to(master_root) or not observation_file.is_file():
            raise P6ExpertMasterError("P6 waypoint observation escaped the master")
        observation = _read_json(observation_file, "P6 observation")
        expected_id = f"demo_obs_{index:03d}"
        if observation.get("observation_id") != expected_id:
            raise P6ExpertMasterError("P6 waypoint observation_id mismatch")
        validate_p6_observation(
            observation,
            task=task,
            master_root=master_root,
            initial_observation=index == 0,
            require_initial_annotations=True,
        )
        frozen_waypoints.append(
            {
                "index": index,
                "source_sim_step": int(source_step),
                "observation_id": expected_id,
                "observation_file": str(observation_file.relative_to(master_root)),
                "sha256": file_sha256(observation_file),
            }
        )

    replay_video = replay_video.resolve()
    if not replay_video.is_relative_to(master_root) or not replay_video.is_file():
        raise P6ExpertMasterError("P6 replay video is missing or escaped the master")
    return {
        "schema_version": "univtac.fixed_expert_p6_master.v2",
        "task": task,
        "seed": int(seed),
        "trajectory_representation": "successful_expert_observation_waypoints",
        "actions_present": False,
        "step_eef_conversion_performed": False,
        "agent_ready": False,
        "intended_use": "host-side source for a later ICL projection contract",
        "source": {
            "fixed_expert_manifest": {
                "path": str(source["manifest_path"]),
                "sha256": source["manifest_sha256"],
            },
            "trajectory_hdf5": {
                "path": str(source["source_hdf5"]),
                "sha256": source["source_hdf5_sha256"],
            },
        },
        "capture": {
            "observation_profile": P6_PROFILE.to_manifest(),
            "annotations": {
                "bbox": True,
                "mask": True,
                "schedule": "initial_observation_only",
                "public_roles": ["manipulated_object", "goal_fixture"],
                "raw_instance_metadata": False,
            },
            "source_waypoint_sampling": "all recorded source frames",
            "source_frame_count": expected_count,
            "captured_observation_count": len(frozen_waypoints),
            "trajectory_timing": "recorded_step_interpolation",
            "motion_source": "forced recorded qpos; no agent action/tool call",
            "privileged_pre_move_executed_before_replay": False,
            "waypoints": frozen_waypoints,
        },
        "replay_verification": {
            "execution_succeeded": True,
            "physics_action_count": int(physics_action_count),
            "official_task_success": True,
            "evaluator_checks": evaluator_checks,
            "video": {
                "path": str(replay_video),
                "sha256": file_sha256(replay_video),
            },
        },
        "visibility_contract": {
            "episode_outcome": "successful expert demonstration",
            "stepwise_success": False,
            "checker_details": False,
            "actions": False,
            "actor_poses": False,
            "planner_or_ik_state": False,
            "contact_points": False,
            "raw_tactile_depth_marker_or_pose": False,
        },
    }
