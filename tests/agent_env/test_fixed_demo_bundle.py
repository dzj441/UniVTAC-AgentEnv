from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from agent_env.artifacts import file_sha256, save_composite
from agent_env.benchmark_observations import (
    image_artifact,
    save_annotation_artifacts,
    save_depth_artifacts,
)
from agent_env.benchmark_profiles import AnnotationCapabilities, get_observation_profile
from agent_env.fixed_demo_bundle import (
    FixedDemoAssetSpec,
    FixedDemoBundleError,
    contact_sheet_indices,
    get_fixed_demo_asset_spec,
    project_fixed_demo_bundle,
    validate_fixed_demo_bundle,
)
from agent_env.p6_expert_master import build_p6_master_manifest


@pytest.mark.parametrize(
    "task",
    [
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
    ],
)
def test_six_promoted_tasks_have_seed_zero_fixed_demo_assets(task: str) -> None:
    spec = get_fixed_demo_asset_spec(task)
    assert spec.task == task
    assert spec.seed == 0
    assert len(spec.manifest_sha256) == 64


def _write_source(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=np.asarray([30, 50], dtype=np.int64))
        handle.create_dataset("embodiment/joint", data=np.zeros((2, 9), np.float32))
        ee = np.zeros((2, 7), np.float32)
        ee[:, 3] = 1.0
        handle.create_dataset("embodiment/ee", data=ee)
        handle.create_dataset(
            "collection/phase", data=np.asarray(["pre_move", "task"], dtype="S16")
        )


def _write_frozen_manifest(path: Path, source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "univtac.fixed_expert_trajectory.v1",
                "task": "pull_out_key",
                "seed": 7,
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


def _write_observation(root: Path, index: int) -> Path:
    observation_id = f"demo_obs_{index:03d}"
    directory = root / "p6_observations" / observation_id
    rgb = np.full((270, 480, 3), 32 + index, dtype=np.uint8)
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
    panels = [(name, rgb) for name in modalities]
    depth = np.full((270, 480), 0.5 + index * 0.01, dtype=np.float32)
    for camera in ("head", "wrist"):
        artifact, _, preview = save_depth_artifacts(directory / camera, depth)
        modalities[f"{camera}_depth"] = artifact
        panels.append((f"{camera}_depth", preview))
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
        mask = np.zeros((270, 480), dtype=bool)
        mask[20:80, 30:90] = True
        annotations = {}
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
                panels.extend(annotation_panels)
    composite = save_composite(panels, directory / "composite.png")
    composite.update({"media_type": "image/png", "content_image": False})
    observation = {
        "schema_version": "univtac.embodied_observation.v1",
        "observation_id": observation_id,
        "task": "pull_out_key",
        "observation_profile": get_observation_profile(6).name,
        "modalities": modalities,
        "robot_state": {
            "joint_position_9d": [float(index)] * 9,
            "joint_velocity_9d": [0.0] * 9,
            "gripper_width_m": 0.04,
            "end_effector_pose_robot_base_wxyz_7d": [
                0.4,
                0.0,
                0.3,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        "camera_calibration": calibration,
        "artifacts": {"composite": composite},
    }
    if annotations is not None:
        observation["annotations"] = annotations
    path = directory / "observation.json"
    path.write_text(json.dumps(observation), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def synthetic_master(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, FixedDemoAssetSpec]:
    asset_root = tmp_path_factory.mktemp("fixed-demo-master")
    master_root = asset_root / "registered_demo"
    master_root.mkdir()
    source = master_root / "source.hdf5"
    _write_source(source)
    frozen = master_root / "frozen.json"
    _write_frozen_manifest(frozen, source)
    observations = [_write_observation(master_root, index) for index in range(2)]
    video = master_root / "video" / "7_success.mp4"
    video.parent.mkdir()
    video.write_bytes(b"synthetic replay video")
    manifest = build_p6_master_manifest(
        task="pull_out_key",
        seed=7,
        fixed_expert_manifest_path=frozen,
        source_hdf5=source,
        master_root=master_root,
        waypoint_records=[
            {
                "source_frame_index": index,
                "source_sim_step": step,
                "observation_file": str(path),
            }
            for index, (step, path) in enumerate(zip((30, 50), observations))
        ],
        physics_action_count=21,
        execution_succeeded=True,
        official_task_success=True,
        evaluator_checks={"base_task_success": True, "settle_steps": 60},
        replay_video=video,
    )
    manifest_path = master_root / "p6_master_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = FixedDemoAssetSpec(
        task="pull_out_key",
        seed=7,
        manifest_relative_path="registered_demo/p6_master_manifest.json",
        manifest_sha256=file_sha256(manifest_path),
    )
    return asset_root, spec


@pytest.mark.parametrize("profile_index", range(1, 7))
@pytest.mark.parametrize(
    ("provide_bbox", "provide_mask"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_projection_matrix_physically_matches_profile_and_annotations(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
    profile_index: int,
    provide_bbox: bool,
    provide_mask: bool,
) -> None:
    asset_root, spec = synthetic_master
    profile = get_observation_profile(profile_index)
    annotations = AnnotationCapabilities(provide_bbox, provide_mask)
    destination = tmp_path / "expert_demo"
    project_fixed_demo_bundle(
        asset_root=asset_root,
        destination=destination,
        task="pull_out_key",
        profile=profile,
        annotations=annotations,
        asset_spec=spec,
    )
    manifest = validate_fixed_demo_bundle(
        destination,
        expected_task="pull_out_key",
        expected_profile=profile,
        expected_annotations=annotations,
    )
    assert manifest["demonstration"]["episode_outcome"] == (
        "successful expert demonstration"
    )
    assert manifest["demonstration"]["actions_present"] is False
    assert "seed" not in manifest
    assert ("camera_calibration" in manifest) is profile.expose_camera_intrinsics

    first = json.loads(
        (destination / "frames/frame_000000/observation.json").read_text(
            encoding="utf-8"
        )
    )
    later = json.loads(
        (destination / "frames/frame_000001/observation.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(first["modalities"]) == set(profile.public_modalities)
    assert set(later["modalities"]) == set(profile.public_modalities)
    assert ("annotations" in first) is bool(annotations.enabled_features)
    assert "annotations" not in later
    assert (destination / "frames/frame_000000/wrist").exists() is (
        "wrist_rgb" in profile.public_modalities
        or "wrist_depth" in profile.public_modalities
    )
    assert (destination / "frames/frame_000000/tactile").exists() is (
        profile.expose_tactile
    )
    if annotations.enabled_features:
        assert set(first["annotations"]) == set(profile.camera_names)
        role = first["annotations"][profile.camera_names[0]]["manipulated_object"]
        assert ("bbox_xyxy_exclusive" in role) is provide_bbox
        assert ("mask" in role) is provide_mask

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    assert str(asset_root) not in public_text
    for forbidden in (
        '"seed"',
        "checker_details",
        "official_task_success",
        "actor_pose",
        "planner_target",
        "contact_point",
        "source_hdf5",
    ):
        assert forbidden not in public_text


def test_projection_is_content_deterministic(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
) -> None:
    asset_root, spec = synthetic_master
    receipts = []
    for name in ("one", "two"):
        receipts.append(
            project_fixed_demo_bundle(
                asset_root=asset_root,
                destination=tmp_path / name,
                task="pull_out_key",
                profile=get_observation_profile(6),
                annotations=AnnotationCapabilities(True, True),
                asset_spec=spec,
            )
        )
    assert receipts[0]["agent_bundle"]["content_integrity"] == receipts[1][
        "agent_bundle"
    ]["content_integrity"]
    assert receipts[0]["agent_bundle"]["manifest_sha256"] == receipts[1][
        "agent_bundle"
    ]["manifest_sha256"]


def test_projection_authenticates_registry_manifest(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
) -> None:
    asset_root, spec = synthetic_master
    bad_spec = FixedDemoAssetSpec(
        task=spec.task,
        seed=spec.seed,
        manifest_relative_path=spec.manifest_relative_path,
        manifest_sha256="0" * 64,
    )
    with pytest.raises(FixedDemoBundleError, match="manifest hash"):
        project_fixed_demo_bundle(
            asset_root=asset_root,
            destination=tmp_path / "expert_demo",
            task="pull_out_key",
            profile=get_observation_profile(1),
            annotations=AnnotationCapabilities(),
            asset_spec=bad_spec,
        )


@pytest.mark.parametrize(
    "mutation",
    ("legacy_schema", "missing_depth_policy", "integer_depth_policy"),
)
def test_projection_rejects_master_without_v3_depth_contract(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
    mutation: str,
) -> None:
    source_root, source_spec = synthetic_master
    asset_root = tmp_path / "asset_root"
    shutil.copytree(source_root, asset_root)
    manifest_path = asset_root / source_spec.manifest_relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "legacy_schema":
        manifest["schema_version"] = "univtac.fixed_expert_p6_master.v2"
    elif mutation == "missing_depth_policy":
        del manifest["capture"]["wrist_metric_depth_surface_policy"]
    else:
        policy = manifest["capture"]["wrist_metric_depth_surface_policy"]
        policy["rigid_gripper_and_gelsight_housing_included"] = 1
        policy["deformable_optical_gel_surface_included"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = FixedDemoAssetSpec(
        task=source_spec.task,
        seed=source_spec.seed,
        manifest_relative_path=source_spec.manifest_relative_path,
        manifest_sha256=file_sha256(manifest_path),
    )

    with pytest.raises(FixedDemoBundleError, match="v3 P6|wrist-depth"):
        project_fixed_demo_bundle(
            asset_root=asset_root,
            destination=tmp_path / "expert_demo",
            task="pull_out_key",
            profile=get_observation_profile(6),
            annotations=AnnotationCapabilities(),
            asset_spec=spec,
        )


def test_bundle_validation_rejects_tampering_and_extra_files(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
) -> None:
    asset_root, spec = synthetic_master
    destination = tmp_path / "expert_demo"
    profile = get_observation_profile(1)
    annotations = AnnotationCapabilities()
    project_fixed_demo_bundle(
        asset_root=asset_root,
        destination=destination,
        task="pull_out_key",
        profile=profile,
        annotations=annotations,
        asset_spec=spec,
    )
    (destination / "unexpected.txt").write_text("not allowlisted", encoding="utf-8")
    with pytest.raises(FixedDemoBundleError, match="file allowlist"):
        validate_fixed_demo_bundle(
            destination,
            expected_task="pull_out_key",
            expected_profile=profile,
            expected_annotations=annotations,
        )


def test_projection_refuses_to_overlay_existing_destination(
    tmp_path: Path,
    synthetic_master: tuple[Path, FixedDemoAssetSpec],
) -> None:
    asset_root, spec = synthetic_master
    destination = tmp_path / "expert_demo"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        project_fixed_demo_bundle(
            asset_root=asset_root,
            destination=destination,
            task="pull_out_key",
            profile=get_observation_profile(1),
            annotations=AnnotationCapabilities(),
            asset_spec=spec,
        )


def test_contact_sheet_sampling_is_endpoint_preserving() -> None:
    assert contact_sheet_indices(3, 12) == [0, 1, 2]
    assert contact_sheet_indices(28, 12) == [0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27]
