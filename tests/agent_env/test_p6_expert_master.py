from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from agent_env.artifacts import file_sha256, save_composite
from agent_env.benchmark_observations import (
    image_artifact,
    save_annotation_artifacts,
    save_depth_artifacts,
)
from agent_env.benchmark_profiles import get_observation_profile
from agent_env.p6_expert_master import (
    P6ExpertMasterError,
    build_p6_collection_candidate_manifest,
    build_p6_master_manifest,
    configure_p6_capture_cfg,
    validate_fixed_expert_source,
    validate_gelsight_wrist_depth_surface_policy,
)


class _VisibilityAttribute:
    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return True

    def IsValid(self) -> bool:
        return True

    def Get(self) -> bool:
        return self.value


class _VisibilityPrim:
    def __init__(self, value: bool) -> None:
        self.attribute = _VisibilityAttribute(value)

    def GetAttribute(self, name: str) -> _VisibilityAttribute:
        assert name == "primvars:invisibleToSecondaryRays"
        return self.attribute


class _VisibilityStage:
    def __init__(self, values: dict[str, bool]) -> None:
        self.prims = {
            path: _VisibilityPrim(value) for path, value in values.items()
        }

    def GetPrimAtPath(self, path: str) -> _VisibilityPrim:
        return self.prims[path]


def visibility_task() -> SimpleNamespace:
    rigid = [f"/rigid_{index}" for index in range(4)]
    gelpads = [f"/gelpad_{index}" for index in range(2)]
    stage = _VisibilityStage(
        {
            **{path: False for path in rigid},
            **{path: True for path in gelpads},
        }
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(tactile_sensor_type="gsmini"),
        num_envs=1,
        scene=SimpleNamespace(stage=stage),
        _gelsight_depth_visibility={
            "schema_version": "univtac.gelsight_wrist_depth_visibility.v1",
            "rigid_case_plate_visible": rigid,
            "deformable_gelpads_hidden": gelpads,
        },
    )


def test_p6_capture_accepts_validated_gelsight_depth_surface_policy() -> None:
    validate_gelsight_wrist_depth_surface_policy(visibility_task())


def test_p6_capture_requires_gelsight_depth_visibility_report() -> None:
    task = visibility_task()
    del task._gelsight_depth_visibility

    with pytest.raises(P6ExpertMasterError, match="lacks a validated"):
        validate_gelsight_wrist_depth_surface_policy(task)


def test_p6_capture_rejects_duplicate_gelsight_depth_targets() -> None:
    task = visibility_task()
    rigid = task._gelsight_depth_visibility["rigid_case_plate_visible"]
    rigid[-1] = rigid[0]

    with pytest.raises(P6ExpertMasterError, match="invalid or duplicate"):
        validate_gelsight_wrist_depth_surface_policy(task)


def test_p6_capture_rechecks_composed_gelsight_visibility_values() -> None:
    task = visibility_task()
    rigid = task._gelsight_depth_visibility["rigid_case_plate_visible"]
    task.scene.stage.prims[rigid[0]].attribute.value = True

    with pytest.raises(P6ExpertMasterError, match="violates"):
        validate_gelsight_wrist_depth_surface_policy(task)


def test_configure_p6_capture_cfg_uses_one_canonical_sensor_contract() -> None:
    cameras = [
        SimpleNamespace(name="head"),
        SimpleNamespace(name="wrist"),
    ]
    cfg = SimpleNamespace(
        reset_time_limit=120.0,
        obs_data_type={"actor": True},
        cameras=cameras,
        random_texture=True,
    )
    configure_p6_capture_cfg(cfg)

    assert cfg.reset_time_limit == 900.0
    assert cfg.obs_data_type == {
        "camera": ["rgb", "depth"],
        "tactile": ["rgb_marker"],
        "embodiment": ["joint", "ee"],
    }
    assert cfg.random_texture is False
    for camera in cameras:
        assert camera.data_types == [
            "rgb",
            "depth",
            "instance_id_segmentation_fast",
        ]
        assert camera.update_latest_camera_pose is True
        assert camera.colorize_instance_id_segmentation is False


def write_source(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=np.asarray([30, 50], dtype=np.int64))
        handle.create_dataset("embodiment/joint", data=np.zeros((2, 9), np.float32))
        ee = np.zeros((2, 7), np.float32)
        ee[:, 3] = 1.0
        handle.create_dataset("embodiment/ee", data=ee)
        handle.create_dataset(
            "collection/phase", data=np.asarray(["pre_move", "task"], dtype="S16")
        )


def write_frozen_manifest(path: Path, source: Path, *, task: str, seed: int) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "univtac.fixed_expert_trajectory.v1",
                "task": task,
                "seed": seed,
                "positive_example": True,
                "agent_visible_checker_details": False,
                "source": {
                    "trajectory": {
                        "path": str(source.resolve()),
                        "sha256": file_sha256(source),
                    }
                },
                "replay_verification": {"official_task_success": True},
            }
        ),
        encoding="utf-8",
    )


def write_p6_observation(root: Path, index: int, task: str) -> Path:
    observation_id = f"demo_obs_{index:03d}"
    directory = root / "p6_observations" / observation_id
    rgb = np.full((270, 480, 3), 20 + index, dtype=np.uint8)
    modalities = {
        "head_rgb": image_artifact(directory / "head" / "rgb.png", rgb),
        "wrist_rgb": image_artifact(directory / "wrist" / "rgb.png", rgb),
        "left_tactile_rgb": image_artifact(
            directory / "left_tactile" / "rgb_marker.png", rgb
        ),
        "right_tactile_rgb": image_artifact(
            directory / "right_tactile" / "rgb_marker.png", rgb
        ),
    }
    depth = np.full((270, 480), 0.5 + index * 0.01, dtype=np.float32)
    panels = [(name, rgb) for name in modalities]
    for camera in ("head", "wrist"):
        artifact, _, preview = save_depth_artifacts(directory / camera, depth)
        modalities[f"{camera}_depth"] = artifact
        panels.append((f"{camera}_depth_m", preview))
    calibration = {
        camera: {
            "intrinsic_matrix_3x3": np.eye(3).tolist(),
            "extrinsics": {
                "matrix_T_robot_base_camera_ros_4x4": np.eye(4).tolist(),
                "camera_convention": "ROS optical: +Z forward, -Y up",
                "translation_unit": "metre",
            },
        }
        for camera in ("head", "wrist")
    }
    annotations = None
    if index == 0:
        annotations = {}
        mask = np.zeros((270, 480), dtype=bool)
        mask[20:80, 30:90] = True
        for camera in ("head", "wrist"):
            annotations[camera] = {}
            for role in ("manipulated_object", "goal_fixture"):
                annotation, annotation_panels = save_annotation_artifacts(
                    directory / "annotations" / camera,
                    role=role,
                    mask=mask,
                    rgb=rgb,
                    provide_bbox=True,
                    provide_mask=True,
                )
                annotations[camera][role] = annotation
                panels.extend(
                    (f"{camera}_{label}", panel)
                    for label, panel in annotation_panels
                )
    composite = save_composite(panels, directory / "composite.png")
    composite.update({"media_type": "image/png", "content_image": False})
    observation = {
        "schema_version": "univtac.embodied_observation.v1",
        "observation_id": observation_id,
        "task": task,
        "observation_profile": get_observation_profile(6).name,
        "modalities": modalities,
        "robot_state": {
            "joint_position_9d": [0.0] * 9,
            "joint_velocity_9d": [0.0] * 9,
            "gripper_width_m": 0.04,
            "end_effector_pose_robot_base_wxyz_7d": [0.4, 0.0, 0.3, 1, 0, 0, 0],
        },
        "camera_calibration": calibration,
        "artifacts": {"composite": composite},
    }
    if annotations is not None:
        observation["annotations"] = annotations
    observation_file = directory / "observation.json"
    observation_file.write_text(json.dumps(observation), encoding="utf-8")
    return observation_file


def master_fixture(tmp_path: Path, *, task: str = "pull_out_key") -> dict:
    source = tmp_path / "source.hdf5"
    write_source(source)
    frozen = tmp_path / "frozen.json"
    write_frozen_manifest(frozen, source, task=task, seed=7)
    observation_files = [
        write_p6_observation(tmp_path, index, task) for index in range(2)
    ]
    video = tmp_path / "video" / "7_success.mp4"
    video.parent.mkdir()
    video.write_bytes(b"test replay video")
    return {
        "task": task,
        "seed": 7,
        "fixed_expert_manifest_path": frozen,
        "source_hdf5": source,
        "master_root": tmp_path,
        "waypoint_records": [
            {
                "source_frame_index": index,
                "source_sim_step": step,
                "observation_file": str(path),
            }
            for index, (step, path) in enumerate(zip((30, 50), observation_files))
        ],
        "physics_action_count": 21,
        "execution_succeeded": True,
        "official_task_success": True,
        "evaluator_checks": {"base_task_success": True, "settle_steps": 60},
        "replay_video": video,
    }


def test_p6_master_freezes_complete_observation_waypoints(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    manifest = build_p6_master_manifest(**inputs)

    assert manifest["schema_version"] == "univtac.fixed_expert_p6_master.v3"
    assert manifest["actions_present"] is False
    assert manifest["step_eef_conversion_performed"] is False
    assert manifest["agent_ready"] is False
    assert manifest["capture"]["captured_observation_count"] == 2
    assert manifest["capture"]["observation_profile"]["index"] == 6
    assert manifest["capture"]["annotations"]["bbox"] is True
    assert manifest["capture"]["annotations"]["mask"] is True
    assert manifest["capture"]["wrist_metric_depth_surface_policy"] == {
        "schema_version": "univtac.wrist_metric_depth_surface_policy.v1",
        "rigid_gripper_and_gelsight_housing_included": True,
        "deformable_optical_gel_surface_included": False,
    }
    assert (
        manifest["capture"]["annotations"]["schedule"]
        == "initial_observation_only"
    )


def test_collection_candidate_retains_p6_without_claiming_replay(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    manifest = build_p6_collection_candidate_manifest(
        task=str(inputs["task"]),
        seed=int(inputs["seed"]),
        source_hdf5=Path(inputs["source_hdf5"]),
        collection_video=Path(inputs["replay_video"]),
        candidate_root=tmp_path,
        waypoint_records=list(inputs["waypoint_records"]),
    )

    assert manifest["schema_version"] == (
        "univtac.fixed_expert_p6_collection_candidate.v1"
    )
    assert manifest["agent_ready"] is False
    assert manifest["replay_verified"] is False
    assert manifest["verification"] == {
        "collection_checker_success": True,
        "independent_replay_required": True,
        "official_task_success": None,
    }
    assert manifest["capture"]["captured_observation_count"] == 2
    assert manifest["capture"]["annotations"]["schedule"] == (
        "initial_observation_only"
    )


def test_p6_master_rejects_annotations_after_initial_frame(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    first = Path(inputs["waypoint_records"][0]["observation_file"])
    later = Path(inputs["waypoint_records"][1]["observation_file"])
    first_observation = json.loads(first.read_text(encoding="utf-8"))
    later_observation = json.loads(later.read_text(encoding="utf-8"))
    later_observation["annotations"] = first_observation["annotations"]
    later.write_text(json.dumps(later_observation), encoding="utf-8")

    with pytest.raises(P6ExpertMasterError, match="top-level fields"):
        build_p6_master_manifest(**inputs)


def test_p6_master_rejects_agent_action_fields(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    observation_file = Path(inputs["waypoint_records"][0]["observation_file"])
    observation = json.loads(observation_file.read_text(encoding="utf-8"))
    observation["action"] = {"step_eef": [0.0] * 7}
    observation_file.write_text(json.dumps(observation), encoding="utf-8")

    with pytest.raises(P6ExpertMasterError, match="top-level fields"):
        build_p6_master_manifest(**inputs)


def test_p6_master_requires_terminally_successful_replay(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    inputs["official_task_success"] = False
    with pytest.raises(P6ExpertMasterError, match="terminal evaluation"):
        build_p6_master_manifest(**inputs)


def test_p6_master_authenticates_frozen_source_hash(tmp_path: Path) -> None:
    inputs = master_fixture(tmp_path)
    source = Path(inputs["source_hdf5"])
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(P6ExpertMasterError, match="hash"):
        validate_fixed_expert_source(
            manifest_path=Path(inputs["fixed_expert_manifest_path"]),
            source_hdf5=source,
            task=str(inputs["task"]),
            seed=int(inputs["seed"]),
        )
