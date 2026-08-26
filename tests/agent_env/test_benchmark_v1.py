from __future__ import annotations

import hashlib
import sys
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
from agent_env.benchmark_tasks import (
    benchmark_task_parameters,
    get_benchmark_task,
    list_benchmark_tasks,
)
from agent_env.capabilities import CapabilityViolation
from agent_env.contract import EVALUATOR_SEED_ENV
from agent_env.icl import get_icl_condition, list_icl_conditions
from scripts.run_codex_benchmark import (
    BASE_INSTRUCTIONS,
    CodexTurnFailedError,
    DEVELOPER_INSTRUCTIONS,
    codex_sandbox_policy,
    effective_codex_network_access,
    operator_prompt,
    parse_args,
    raise_for_failed_codex_turn,
    validate_fixed_demo_evaluation_seed,
)


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


def test_all_eight_tasks_are_registered() -> None:
    assert [task.name for task in list_benchmark_tasks()] == [
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
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

    grasp = get_benchmark_task("grasp_classify")
    assert grasp.instruction == (
        "Grasp the center prism and place it upright on the green pad."
    )
    assert "rough" not in grasp.instruction.lower()
    assert "plain" not in grasp.instruction.lower()


def test_generic_grasp_classify_uses_a_fixed_green_target() -> None:
    assert benchmark_task_parameters("grasp_classify") == {
        "target_pad": {
            "mode": "fixed_color",
            "color": "green",
            "material_classification_required": False,
        }
    }


def test_key_initial_yaw_defaults_to_legacy_random_and_accepts_fixed_override() -> None:
    default = benchmark_task_parameters("pull_out_key")
    assert default["key_initial_relative_yaw"]["mode"] == "legacy_random"
    assert default["key_initial_relative_yaw"]["range_rad"] == pytest.approx(
        [-1.5707963267948966, -0.7853981633974483]
    )

    fixed = benchmark_task_parameters(
        "pull_out_key", key_initial_relative_yaw_rad=-1.5707963267948966
    )
    assert fixed["key_initial_relative_yaw"]["mode"] == "fixed"
    assert fixed["key_initial_relative_yaw"]["value_rad"] == pytest.approx(
        -1.5707963267948966
    )

    with pytest.raises(ValueError, match="only for pull_out_key"):
        benchmark_task_parameters(
            "put_bottle_in_shelf", key_initial_relative_yaw_rad=-1.0
        )
    with pytest.raises(ValueError, match="must be finite"):
        benchmark_task_parameters(
            "pull_out_key", key_initial_relative_yaw_rad=float("nan")
        )


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


def test_operator_prompt_is_only_task_instruction_and_optional_condition_notice() -> None:
    single = operator_prompt(
        "pull_out_key",
        pre_move=False,
        interaction_mode="single_turn",
    )
    multi = operator_prompt(
        "pull_out_key",
        pre_move=False,
        interaction_mode="action_per_turn",
    )
    assert single == multi
    assert single == "Grasp the key and pull it completely out of the slot.\n"
    assert "start_episode" not in multi
    assert "decision_record" not in multi


def test_base_instruction_contains_only_robot_lifecycle_and_terminal_feedback() -> None:
    assert BASE_INSTRUCTIONS == (
        "Control the robot through start_episode, step_eef, and finish_episode. After each\n"
        "robot-control tool call, wait until its resulting observation is provided before\n"
        "calling another robot-control tool. Task success is returned only by finish_episode.\n"
    )
    assert "runtime capabilities" not in BASE_INSTRUCTIONS


def test_developer_instruction_requests_public_progress_without_strategy_fields() -> None:
    assert "concise commentary update" in DEVELOPER_INSTRUCTIONS
    assert "observable" in DEVELOPER_INSTRUCTIONS
    assert "hidden chain-of-thought" in DEVELOPER_INSTRUCTIONS
    assert "decision_record" not in DEVELOPER_INSTRUCTIONS
    assert "rationale" not in DEVELOPER_INSTRUCTIONS


def test_icl_axis_and_demo_relation_notice() -> None:
    assert [condition.name for condition in list_icl_conditions()] == [
        "none",
        "fixed_demo",
    ]
    without_demo = operator_prompt(
        "pull_out_key",
        pre_move=False,
        icl_condition="none",
    )
    with_demo = operator_prompt(
        "pull_out_key",
        pre_move=False,
        icl_condition="fixed_demo",
    )
    assert "benchmark_inputs/expert_demo" not in without_demo
    assert with_demo == (
        "Grasp the key and pull it completely out of the slot.\n"
        "\nA verified successful demonstration from a separate episode of the same "
        "task is available at benchmark_inputs/expert_demo/. The current scene "
        "configuration and object or goal poses may differ.\n"
    )
    assert "observed expert waypoints" not in with_demo
    assert "step_eef actions" not in with_demo


def test_codex_sandbox_and_network_policy_are_explicit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert codex_sandbox_policy(
        "workspace-write",
        workspace=workspace,
        network_access=True,
    ) == {
        "type": "workspaceWrite",
        "writableRoots": [str(workspace.resolve())],
        "networkAccess": True,
    }
    assert effective_codex_network_access("read-only", False) is False
    assert effective_codex_network_access("workspace-write", True) is True
    assert effective_codex_network_access("danger-full-access", False) is True
    assert codex_sandbox_policy(
        "danger-full-access",
        workspace=workspace,
        network_access=True,
    ) == {"type": "dangerFullAccess"}


def test_reference_runner_defaults_to_multiturn_and_network_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_codex_benchmark.py",
            "--task",
            "pull_out_key",
            "--profile",
            "1",
            "--dry-run",
        ],
    )
    defaults = parse_args()
    assert defaults.interaction_mode == "action_per_turn"
    assert defaults.codex_sandbox == "danger-full-access"
    assert defaults.codex_network_access is True
    assert defaults.effort == "high"
    assert defaults.key_initial_relative_yaw_rad is None
    assert defaults.sim_step_recorder is True
    assert defaults.sim_step_recorder_fps == pytest.approx(10.0)
    assert defaults.post_action_settle_steps == 60
    assert defaults.sim_step_recorder_post_action_steps == 60

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_codex_benchmark.py",
            "--task",
            "pull_out_key",
            "--profile",
            "1",
            "--no-codex-network-access",
            "--dry-run",
        ],
    )
    assert parse_args().codex_network_access is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_codex_benchmark.py",
            "--task",
            "pull_out_key",
            "--profile",
            "1",
            "--key-initial-relative-yaw-rad",
            "-1.5707963267948966",
            "--dry-run",
        ],
    )
    assert parse_args().key_initial_relative_yaw_rad == pytest.approx(
        -1.5707963267948966
    )


def test_failed_codex_turn_preserves_infrastructure_error_before_action_count() -> None:
    turn_result = {
        "thread_id": "thread-network-failure",
        "turn_id": "turn-network-failure",
        "turn": {
            "status": "failed",
            "error": {
                "message": "stream disconnected before completion",
                "codexErrorInfo": "other",
                "additionalDetails": None,
            },
        },
        "dynamic_tool_call_count": 0,
    }

    with pytest.raises(
        CodexTurnFailedError,
        match="stream disconnected before completion",
    ) as failure:
        raise_for_failed_codex_turn(turn_result)

    assert failure.value.to_manifest() == {
        "component": "codex_app_server",
        "phase": "turn",
        "thread_id": "thread-network-failure",
        "turn_id": "turn-network-failure",
        "error": {
            "message": "stream disconnected before completion",
            "codexErrorInfo": "other",
            "additionalDetails": None,
        },
    }


def test_completed_codex_turn_is_not_classified_as_infrastructure_failure() -> None:
    raise_for_failed_codex_turn(
        {
            "thread_id": "thread-ok",
            "turn_id": "turn-ok",
            "turn": {"status": "completed", "error": None},
            "dynamic_tool_call_count": 0,
        }
    )


def test_recorder_tail_defaults_to_and_cannot_exceed_action_settling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = [
        "run_codex_benchmark.py",
        "--task",
        "pull_out_key",
        "--profile",
        "1",
        "--post-action-settle-steps",
        "24",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", common)
    inherited = parse_args()
    assert inherited.post_action_settle_steps == 24
    assert inherited.sim_step_recorder_post_action_steps == 24

    monkeypatch.setattr(
        sys,
        "argv",
        [*common[:-1], "--sim-step-recorder-post-action-steps", "12", "--dry-run"],
    )
    shorter = parse_args()
    assert shorter.post_action_settle_steps == 24
    assert shorter.sim_step_recorder_post_action_steps == 12

    monkeypatch.setattr(
        sys,
        "argv",
        [*common[:-1], "--sim-step-recorder-post-action-steps", "25", "--dry-run"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_fixed_demo_excludes_its_seed_from_evaluation() -> None:
    fixed_demo = get_icl_condition("fixed_demo")
    with pytest.raises(ValueError, match="must differ"):
        validate_fixed_demo_evaluation_seed(
            task_name="pull_out_key",
            icl_condition=fixed_demo,
            environ={EVALUATOR_SEED_ENV: "0"},
        )
    validate_fixed_demo_evaluation_seed(
        task_name="pull_out_key",
        icl_condition=fixed_demo,
        environ={EVALUATOR_SEED_ENV: "10000000"},
    )
    validate_fixed_demo_evaluation_seed(
        task_name="put_bottle_in_shelf",
        icl_condition=get_icl_condition("none"),
        environ={EVALUATOR_SEED_ENV: "1"},
    )


def test_protocol_accepts_unbounded_finite_arm_deltas_and_physical_gripper_targets() -> None:
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
    dp, dr, dg = protocol.prepare_step(
        observation_id="obs_001",
        delta_position=[0.4, -0.5, 0.6],
        delta_rpy=[1.2, -0.9, 3.0],
        delta_gripper=0,
    )
    assert dp.tolist() == [0.4, -0.5, 0.6]
    assert dr.tolist() == [1.2, -0.9, 3.0]
    assert dg == 0
    _, _, dg = protocol.prepare_step(
        observation_id="obs_001",
        delta_position=[0, 0, 0],
        delta_rpy=[0, 0, 0],
        delta_gripper=0.019,
        current_gripper_qpos=0.02,
    )
    assert dg == pytest.approx(0.019)
    with pytest.raises(BenchmarkProtocolError, match="physical per-finger"):
        protocol.prepare_step(
            observation_id="obs_001",
            delta_position=[0, 0, 0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0.02,
            current_gripper_qpos=0.02,
        )
    contract = protocol.contract_manifest()["action_bounds"]
    assert contract["translation_benchmark_limit"] is None
    assert contract["rotation_benchmark_limit"] is None
    assert contract["gripper_target_qpos_range_m"] == [0.0, 0.039]
    assert "max_abs_translation_component_m" not in contract
    assert "max_abs_rotation_component_rad" not in contract
    protocol.finish("obs_001")
    assert protocol.terminal


def test_dynamic_registry_contains_exactly_three_tools() -> None:
    registry = build_benchmark_tool_registry(
        get_observation_profile(6), AnnotationCapabilities(True, True)
    )
    assert list(registry) == ["start_episode", "step_eef", "finish_episode"]
    assert registry["start_episode"].input_schema == {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {},
    }
    assert set(registry["step_eef"].input_schema["properties"]) == {
        "observation_id",
        "delta_position",
        "delta_rpy",
        "delta_gripper",
    }
    assert set(registry["step_eef"].input_schema["required"]) == {
        "observation_id",
        "delta_position",
        "delta_rpy",
        "delta_gripper",
    }
    assert registry["finish_episode"].input_schema["required"] == ["observation_id"]
    assert set(registry["finish_episode"].input_schema["properties"]) == {
        "observation_id"
    }
    assert all(not tool.requires_decision_record for tool in registry.values())
    step_schema = registry["step_eef"].input_schema["properties"]
    assert "maximum" not in step_schema["delta_position"]["items"]
    assert "maximum" not in step_schema["delta_rpy"]["items"]
    assert step_schema["delta_gripper"]["minimum"] == -0.039
    assert step_schema["delta_gripper"]["maximum"] == 0.039

    fixed_demo_registry = build_benchmark_tool_registry(
        get_observation_profile(6),
        AnnotationCapabilities(True, True),
        get_icl_condition("fixed_demo"),
    )
    assert (
        fixed_demo_registry["step_eef"].input_schema
        == registry["step_eef"].input_schema
    )


def test_gateway_forwards_large_arm_delta_but_rejects_physical_gripper_overrun(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def observation(observation_id: str) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "modalities": {},
            "robot_state": {
                "joint_position_9d": [0.0] * 7 + [0.02, 0.02],
                "joint_velocity_9d": [0.0] * 9,
                "gripper_width_m": 0.04,
                "end_effector_pose_robot_base_wxyz_7d": [0.0] * 7,
            },
        }

    def simulator(command: dict[str, object]) -> dict[str, object]:
        calls.append(command)
        if command["command"] == "start":
            return {"status": "rollout_started", "observation": observation("obs_000")}
        return {"status": "action_complete", "observation": observation("obs_001")}

    gateway = BenchmarkCapabilityGateway(
        task=get_benchmark_task("pull_out_key"),
        profile=get_observation_profile(1),
        annotations=AnnotationCapabilities(False, False),
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
    )
    assert gateway.execute("start_episode", {}).success
    moved = gateway.execute(
        "step_eef",
        {
            "observation_id": "obs_000",
            "delta_position": [0.4, -0.5, 0.6],
            "delta_rpy": [1.2, -0.9, 3.0],
            "delta_gripper": 0,
        },
    )
    assert moved.success
    assert calls[-1]["delta_position"] == [0.4, -0.5, 0.6]
    assert calls[-1]["delta_rpy"] == [1.2, -0.9, 3.0]

    rejected = gateway.execute(
        "step_eef",
        {
            "observation_id": "obs_001",
            "delta_position": [0, 0, 0],
            "delta_rpy": [0, 0, 0],
            "delta_gripper": 0.02,
        },
    )
    assert not rejected.success
    assert rejected.execution_target == "rejected"
    assert "physical target qpos" in rejected.raw_response["message"]
    assert len(calls) == 2


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
    assert manifest["icl"] == {
        "name": "none",
        "fixed_demo_available": False,
    }
    assert not any(
        "shell" in capability
        for capability in manifest["forbidden_agent_capabilities"]
    )

    legacy = benchmark_capability_manifest(
        get_benchmark_task("pull_out_key"),
        get_observation_profile(6),
        AnnotationCapabilities(True, True),
        pre_move=True,
    )
    assert legacy["task"]["start_condition"] == "pregrasped"
    assert legacy["sha256"] != digest

    fixed_demo = benchmark_capability_manifest(
        get_benchmark_task("pull_out_key"),
        get_observation_profile(6),
        AnnotationCapabilities(True, True),
        icl_condition="fixed_demo",
    )
    assert fixed_demo["icl"]["workspace_path"] == "benchmark_inputs/expert_demo"
    assert fixed_demo["sha256"] != digest


def test_public_wire_schema_excludes_host_close() -> None:
    variants = {
        variant["title"]: variant
        for variant in public_benchmark_command_schema()["oneOf"]
    }
    titles = list(variants)
    assert titles == ["start", "step", "finish"]
    assert variants["start"]["properties"] == {"command": {"const": "start"}}
    assert set(variants["step"]["properties"]) == {
        "command",
        "observation_id",
        "delta_position",
        "delta_rpy",
        "delta_gripper",
    }
    assert set(variants["finish"]["properties"]) == {"command", "observation_id"}
    with pytest.raises(ValueError, match="Missing required"):
        validate_benchmark_command_fields({"command": "step"})
    with pytest.raises(ValueError, match="Unknown field.*rationale"):
        validate_benchmark_command_fields(
            {
                "command": "step",
                "observation_id": "obs_000",
                "delta_position": [0, 0, 0],
                "delta_rpy": [0, 0, 0],
                "delta_gripper": 0,
                "rationale": "legacy narration",
            }
        )


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
    started = gateway.execute("start_episode", {})
    assert started.success
    assert "path" not in str(started.public_response)
    assert any(item["type"] == "inputImage" for item in started.content_items)
    assert {
        item["text"]
        for item in started.content_items
        if item["type"] == "inputText"
    } >= {
        "The next image corresponds to public JSON field "
        "observation.modalities.head_rgb."
    }
    with pytest.raises(CapabilityViolation, match="success leaked"):
        gateway.execute(
            "step_eef",
            {
                "observation_id": "obs_000",
                "delta_position": [0, 0, 0],
                "delta_rpy": [0, 0, 0],
                "delta_gripper": 0,
            },
        )


def test_gateway_image_labels_are_complete_public_json_paths(tmp_path: Path) -> None:
    from PIL import Image

    observation_root = tmp_path / "observations" / "obs_000"
    observation_root.mkdir(parents=True)

    def artifact(relative: str) -> dict[str, object]:
        path = observation_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), "black").save(path)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "media_type": "image/png",
            "content_image": True,
        }

    response = {
        "status": "rollout_started",
        "observation": {
            "observation_id": "obs_000",
            "modalities": {
                "head_depth": {
                    "valid_mask": artifact("head/depth_valid_mask.png"),
                    "visualization": artifact("head/depth_visualization.png"),
                },
                "wrist_depth": {
                    "valid_mask": artifact("wrist/depth_valid_mask.png"),
                },
            },
            "annotations": {
                "wrist": {
                    "goal_fixture": {
                        "bbox_overlay": artifact(
                            "annotations/wrist/goal_fixture_bbox_overlay.png"
                        ),
                    }
                }
            },
            "robot_state": {},
        },
    }
    gateway = BenchmarkCapabilityGateway(
        task=get_benchmark_task("pull_out_key"),
        profile=get_observation_profile(6),
        annotations=AnnotationCapabilities(provide_bbox=True),
        simulator_request=lambda _: response,
        simulator_run_dir=tmp_path,
    )

    started = gateway.execute("start_episode", {})
    labels = [
        item["text"]
        for item in started.content_items
        if item["type"] == "inputText"
        and item["text"].startswith("The next image corresponds")
    ]
    assert labels == [
        "The next image corresponds to public JSON field "
        "observation.modalities.head_depth.valid_mask.",
        "The next image corresponds to public JSON field "
        "observation.modalities.head_depth.visualization.",
        "The next image corresponds to public JSON field "
        "observation.modalities.wrist_depth.valid_mask.",
        "The next image corresponds to public JSON field "
        "observation.annotations.wrist.goal_fixture.bbox_overlay.",
    ]


def test_gateway_keeps_terminal_checker_details_private(tmp_path: Path) -> None:
    responses = iter(
        [
            {
                "status": "rollout_started",
                "observation": {
                    "observation_id": "obs_000",
                    "modalities": {},
                    "robot_state": {},
                },
            },
            {
                "status": "rollout_finished",
                "official_task_success": True,
                "seed_reveal": 123456789,
                "salt_reveal": "private-commitment-opening",
                "evaluator_checks": {
                    "base_task_success": True,
                    "settle_steps": 60,
                    "private_threshold_m": 0.01,
                },
                "observation": {
                    "observation_id": "obs_001",
                    "modalities": {},
                    "robot_state": {},
                },
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
    gateway.execute("start_episode", {})
    finished = gateway.execute(
        "finish_episode",
        {
            "observation_id": "obs_000",
        },
    )

    assert finished.raw_response["evaluator_checks"]["settle_steps"] == 60
    assert finished.raw_response["seed_reveal"] == 123456789
    assert finished.raw_response["salt_reveal"] == "private-commitment-opening"
    assert finished.public_response["official_task_success"] is True
    assert "evaluator_checks" not in finished.public_response
    assert "seed_reveal" not in finished.public_response
    assert "salt_reveal" not in finished.public_response
    public_content = str(finished.content_items)
    assert "private_threshold_m" not in public_content
    assert "123456789" not in public_content
    assert "private-commitment-opening" not in public_content


def test_gateway_publishes_only_current_metric_depth_and_raw_masks(
    tmp_path: Path,
) -> None:
    import json
    import numpy as np

    run_dir = tmp_path / "evaluator_run"
    workspace = tmp_path / "agent_workspace"
    workspace.mkdir()

    def observation(observation_id: str, *, include_mask: bool) -> dict:
        directory = run_dir / "observations" / observation_id
        depth_path = directory / "head" / "depth_m.npy"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, np.full((2, 2), 0.5, dtype=np.float32))
        depth_sha = hashlib.sha256(depth_path.read_bytes()).hexdigest()
        payload = {
            "observation_id": observation_id,
            "modalities": {
                "head_depth": {
                    "depth_m": {
                        "path": str(depth_path),
                        "sha256": depth_sha,
                        "media_type": "application/x-npy",
                        "dtype": "float32",
                        "shape": [2, 2],
                        "unit": "metre",
                    }
                }
            },
            "robot_state": {},
        }
        if include_mask:
            mask_path = (
                directory
                / "annotations"
                / "head"
                / "manipulated_object_mask.png"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (2, 2), 255).save(mask_path)
            payload["annotations"] = {
                "head": {
                    "manipulated_object": {
                        "mask": {
                            "path": str(mask_path),
                            "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                            "media_type": "image/png",
                            "content_image": False,
                        }
                    }
                }
            }
        return payload

    responses = iter(
        [
            {
                "status": "rollout_started",
                "observation": observation("obs_000", include_mask=True),
            },
            {
                "status": "action_complete",
                "observation": observation("obs_001", include_mask=False),
            },
        ]
    )
    gateway = BenchmarkCapabilityGateway(
        task=get_benchmark_task("pull_out_key"),
        profile=get_observation_profile(6),
        annotations=AnnotationCapabilities(False, True),
        simulator_request=lambda _: next(responses),
        simulator_run_dir=run_dir,
        agent_workspace=workspace,
    )
    started = gateway.execute("start_episode", {})
    depth = started.public_response["observation"]["modalities"]["head_depth"][
        "depth_m"
    ]
    assert depth["workspace_path"] == (
        "benchmark_inputs/current_observation/head/depth_m.npy"
    )
    old_mask = (
        workspace
        / "benchmark_inputs/current_observation/annotations/head/"
        "manipulated_object_mask.png"
    )
    assert old_mask.is_file()
    initial_manifest = json.loads(
        (
            workspace / "benchmark_inputs/current_observation/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert initial_manifest["observation_id"] == "obs_000"

    stepped = gateway.execute(
        "step_eef",
        {
            "observation_id": "obs_000",
            "delta_position": [0.0, 0.0, 0.0],
            "delta_rpy": [0.0, 0.0, 0.0],
            "delta_gripper": 0.0,
        },
    )
    assert stepped.success
    assert not old_mask.exists()
    latest_manifest = json.loads(
        (
            workspace / "benchmark_inputs/current_observation/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert latest_manifest["observation_id"] == "obs_001"
    assert latest_manifest["retention"] == "current_observation_only"


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
    candidate_mapping = normalize_instance_mapping(
        {
            "idToLabels": {
                "4": "/World/envs/env_0/green_pad/mesh",
                "7": "/World/envs/env_0/orange_pad/mesh",
            }
        }
    )
    candidate_instance = np.asarray([[4, 0], [0, 7]], dtype=np.int32)
    candidate_mask, candidate_ids = instance_role_mask(
        candidate_instance,
        candidate_mapping,
        ("green_pad", "orange_pad"),
    )
    assert candidate_ids == [4, 7]
    assert candidate_mask.tolist() == [[True, False], [False, True]]
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
