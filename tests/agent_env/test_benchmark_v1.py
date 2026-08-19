from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from agent_env.benchmark_capabilities import (
    BenchmarkCapabilityGateway,
    benchmark_capability_manifest,
    build_benchmark_tool_registry,
)
from agent_env.benchmark_contract import (
    public_benchmark_command_schema,
    validate_benchmark_command_fields,
)
from agent_env.benchmark_observations import (
    bbox_xyxy_exclusive,
    instance_role_mask,
    normalize_instance_mapping,
    robot_base_camera_extrinsic,
    save_annotation_artifacts,
    save_depth_artifacts,
)
from agent_env.benchmark_profiles import (
    AnnotationCapabilities,
    get_observation_profile,
    list_observation_profiles,
)
from agent_env.benchmark_protocol import (
    BenchmarkEpisodeProtocol,
    BenchmarkProtocolError,
)
from agent_env.benchmark_tasks import get_benchmark_task, list_benchmark_tasks
from agent_env.capabilities import CapabilityViolation
from scripts.run_codex_benchmark import operator_prompt


def decision(source: str = "head_rgb") -> dict:
    return {
        "evidence": [
            {
                "source": source,
                "finding": "The target shifted in the latest image.",
                "implication": "A smaller correction is appropriate.",
            }
        ],
        "alternatives_considered": ["Hold position and observe again."],
        "uncertainty": 0.25,
        "expected_effect": "Move toward the visible target.",
        "parameter_rationale": "One centimetre limits overshoot.",
        "rationale": "Use the latest visible displacement.",
    }


def test_six_observation_profiles_match_the_main_axis() -> None:
    profiles = list_observation_profiles()
    assert [profile.index for profile in profiles] == [1, 2, 3, 4, 5, 6]
    assert get_observation_profile(1).public_modalities == ("head_rgb",)
    assert get_observation_profile(2).public_modalities == ("head_rgb", "wrist_rgb")
    assert get_observation_profile(3).public_modalities == (
        "head_rgb",
        "left_tactile_rgb",
        "right_tactile_rgb",
    )
    assert get_observation_profile(6).public_modalities == (
        "head_rgb",
        "wrist_rgb",
        "left_tactile_rgb",
        "right_tactile_rgb",
        "head_depth",
        "wrist_depth",
    )
    assert get_observation_profile(6).expose_camera_intrinsics
    assert get_observation_profile(6).expose_camera_extrinsics


@pytest.mark.parametrize(
    ("bbox", "mask", "features"),
    [
        (False, False, []),
        (True, False, ["bbox"]),
        (False, True, ["mask"]),
        (True, True, ["bbox", "mask"]),
    ],
)
def test_annotation_switches_are_independently_composable(
    bbox: bool, mask: bool, features: list[str]
) -> None:
    manifest = AnnotationCapabilities(bbox, mask).to_manifest()
    assert manifest["enabled_features"] == features
    assert manifest["schedule"] == "initial_observation_only"
    assert manifest["raw_instance_ids"] is False
    assert manifest["raw_labels"] is False


def test_only_two_new_tasks_are_registered() -> None:
    assert [task.name for task in list_benchmark_tasks()] == [
        "pull_out_key",
        "put_bottle_in_shelf",
    ]
    bottle = get_benchmark_task("put_bottle_in_shelf")
    assert "pick up" in bottle.instruction.lower()
    default_manifest = bottle.to_manifest()
    assert default_manifest["start_condition"] == "ungrasped"
    assert default_manifest["pre_move_enabled"] is False
    assert "fixed default home" in default_manifest["initial_state"]
    legacy_manifest = bottle.to_manifest(pre_move=True)
    assert legacy_manifest["start_condition"] == "pregrasped"
    assert legacy_manifest["pre_move_enabled"] is True
    assert "already-grasped" in legacy_manifest["instruction"]


def test_operator_prompt_discloses_configured_episode_token_budget() -> None:
    unlimited = operator_prompt("pull_out_key", pre_move=False)
    assert "cumulative output tokens" not in unlimited

    limited = operator_prompt(
        "pull_out_key",
        pre_move=False,
        max_output_tokens=12_000,
    )
    assert "12000 cumulative output tokens" in limited
    assert "already includes reasoning output" in limited
    assert "makes the benchmark result a failure" in limited


def test_protocol_accepts_zero_step_and_enforces_all_bounds() -> None:
    protocol = BenchmarkEpisodeProtocol(get_observation_profile(6))
    assert protocol.MAX_STEPS == 50
    assert protocol.contract_manifest()["action_budget"] == {"step_eef": 50}
    protocol.start("obs_000")
    dp, dr, dg = protocol.prepare_step(
        observation_id="obs_000",
        delta_position=[0, 0, 0],
        delta_rpy=[0, 0, 0],
        delta_gripper=0,
    )
    assert dp.tolist() == [0, 0, 0]
    assert dr.tolist() == [0, 0, 0]
    assert dg == 0
    protocol.complete_step("obs_001")
    with pytest.raises(BenchmarkProtocolError, match="latest id"):
        protocol.prepare_step(
            observation_id="obs_000",
            delta_position=[0, 0, 0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )
    with pytest.raises(BenchmarkProtocolError, match="norm"):
        protocol.prepare_step(
            observation_id="obs_001",
            delta_position=[0.04, 0.04, 0.04],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )
    protocol.finish("obs_001")
    assert protocol.terminal


def test_dynamic_registry_contains_exactly_three_tools() -> None:
    registry = build_benchmark_tool_registry(
        get_observation_profile(6), AnnotationCapabilities(True, True)
    )
    assert list(registry) == ["start_episode", "step_eef", "finish_episode"]
    sources = registry["step_eef"].input_schema["properties"]["decision_record"][
        "properties"
    ]["evidence"]["items"]["properties"]["source"]["enum"]
    assert "object_bbox" in sources
    assert "object_mask" in sources
    assert "camera_extrinsics" in sources


def test_capability_manifest_is_deterministic_and_task_specific() -> None:
    manifest = benchmark_capability_manifest(
        get_benchmark_task("pull_out_key"),
        get_observation_profile(6),
        AnnotationCapabilities(True, True),
    )
    digest = manifest.pop("sha256")
    import json

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["task"]["name"] == "pull_out_key"
    assert manifest["task"]["start_condition"] == "ungrasped"

    legacy = benchmark_capability_manifest(
        get_benchmark_task("pull_out_key"),
        get_observation_profile(6),
        AnnotationCapabilities(True, True),
        pre_move=True,
    )
    assert legacy["task"]["start_condition"] == "pregrasped"
    assert legacy["sha256"] != digest


def test_public_wire_schema_excludes_host_close() -> None:
    titles = [variant["title"] for variant in public_benchmark_command_schema()["oneOf"]]
    assert titles == ["start", "step", "finish"]
    with pytest.raises(ValueError, match="Missing required"):
        validate_benchmark_command_fields({"command": "step"})


def test_gateway_strips_paths_attaches_images_and_hides_success(tmp_path: Path) -> None:
    image = tmp_path / "observations" / "obs_000" / "head.png"
    image.parent.mkdir(parents=True)
    # Small valid PNG header/content produced without optional image libraries.
    from PIL import Image

    Image.new("RGB", (2, 2), "black").save(image)
    sha = hashlib.sha256(image.read_bytes()).hexdigest()

    responses = iter(
        [
            {
                "status": "rollout_started",
                "observation": {
                    "observation_id": "obs_000",
                    "modalities": {
                        "head_rgb": {
                            "path": str(image),
                            "sha256": sha,
                            "media_type": "image/png",
                            "content_image": True,
                        }
                    },
                    "robot_state": {},
                },
            },
            {
                "status": "action_complete",
                "task_success": False,
                "observation": {"observation_id": "obs_001"},
            },
        ]
    )
    gateway = BenchmarkCapabilityGateway(
        task=get_benchmark_task("pull_out_key"),
        profile=get_observation_profile(1),
        annotations=AnnotationCapabilities(),
        simulator_request=lambda _: next(responses),
        simulator_run_dir=tmp_path,
    )
    started = gateway.execute("start_episode", {"agent_note": "test"})
    assert started.success
    assert "path" not in str(started.public_response)
    assert any(item["type"] == "inputImage" for item in started.content_items)
    with pytest.raises(CapabilityViolation, match="success leaked"):
        gateway.execute(
            "step_eef",
            {
                "observation_id": "obs_000",
                "delta_position": [0, 0, 0],
                "delta_rpy": [0, 0, 0],
                "delta_gripper": 0,
                "decision_record": decision(),
            },
        )


def test_native_depth_and_anonymous_annotation_artifacts(tmp_path: Path) -> None:
    import numpy as np

    depth = np.asarray([[0.5, np.inf], [0.7, 0.9]], dtype=np.float32)
    depth_artifact, packed, preview = save_depth_artifacts(tmp_path / "depth", depth)
    assert packed.dtype == np.float32
    assert depth_artifact["statistics"]["valid_fraction"] == 0.75
    assert depth_artifact["depth_m"]["unit"] == "metre"
    assert preview.shape == (2, 2, 3)

    mapping = normalize_instance_mapping(
        {"idToLabels": {"4": "/World/envs/env_0/key/mesh", "7": "slot"}}
    )
    instance = np.asarray([[0, 4], [4, 0]], dtype=np.int32)
    mask, ids = instance_role_mask(instance, mapping, "key")
    assert ids == [4]
    assert bbox_xyxy_exclusive(mask) == [0, 0, 2, 2]
    public, panels = save_annotation_artifacts(
        tmp_path / "annotation",
        role="manipulated_object",
        mask=mask,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        provide_bbox=True,
        provide_mask=True,
    )
    assert set(public) == {
        "public_role",
        "visible",
        "bbox_xyxy_exclusive",
        "bbox_overlay",
        "mask",
        "mask_overlay",
    }
    assert len(panels) == 2
    assert public["bbox_overlay"]["content_image"] is True
    assert public["mask"]["content_image"] is False
    assert public["mask_overlay"]["content_image"] is True
    with Image.open(public["mask"]["path"]) as image:
        assert image.mode == "L"
        assert set(image.getdata()) == {0, 255}
    assert "key" not in str(public)


def test_extrinsic_is_expressed_in_robot_base_frame() -> None:
    matrix = robot_base_camera_extrinsic(
        robot_position_world=[1, 2, 3],
        robot_quaternion_world_wxyz=[1, 0, 0, 0],
        camera_position_world=[1.1, 2.2, 3.3],
        camera_quaternion_world_ros_wxyz=[1, 0, 0, 0],
    )
    assert [row[3] for row in matrix[:3]] == pytest.approx([0.1, 0.2, 0.3])
