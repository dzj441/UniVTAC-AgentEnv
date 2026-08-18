from __future__ import annotations

import json

import pytest

from agent_env.profiles import get_profile
from agent_env.visibility import (
    AgentVisibilityError,
    assert_agent_visible_response,
    project_simulator_response,
)
from scripts.run_codex_agent_env import operator_prompt


def _artifact(name: str) -> dict[str, str]:
    return {"artifact_id": f"observations/obs_000/{name}.png", "sha256": "a" * 64}


def _robot_state() -> dict[str, object]:
    return {
        "joint_position_8d": [0.0] * 8,
        "gripper_qpos": 0.007,
        "end_effector_pose_robot_base_7d": [0.0] * 7,
    }


def test_task_prompt_contains_semantics_and_generic_placement_guidance() -> None:
    prompts = {operator_prompt(level) for level in (1, 2, 3)}
    assert len(prompts) == 1
    prompt = prompts.pop()
    assert prompt == (
        "Task: grasp_classify\n\n"
        "Determine whether the grasped prism is rough or plain. Place a rough prism on\n"
        "the orange pad and a plain prism on the green pad.\n"
        "For a stable placement, align the prism upright over the center of the target\n"
        "pad before releasing it; avoid releasing near the pad edge.\n"
    )
    for forbidden in (
        "targetward",
        "world y",
        "probe",
        "commit",
        "Level",
        "task_success",
        "action",
        "metres",
        "maximum",
    ):
        assert forbidden not in prompt


@pytest.mark.parametrize("level", [1, 2, 3])
def test_projection_strips_host_protocol_metadata_at_every_level(level: int) -> None:
    profile = get_profile(level)
    raw = {
        "status": "rollout_started",
        "level": level,
        "profile": profile.name,
        "seed_commitment_sha256": "b" * 64,
        "stage": "classification",
        "required_before_translation": "submit_prediction",
        "optional_before_prediction": "up to 2 probes",
        "task_success_feedback": "withheld",
        "observation": {
            "observation_id": "obs_000",
            "level": level,
            "profile": profile.name,
            "stage": "classification",
            "probe_count": 0,
            "post_prediction_action_count": 0,
            "modalities": {
                name: _artifact(name) for name in profile.public_modalities
            },
            "robot_state": _robot_state(),
            "tactile_health": {
                "left": {"dark_pixel_count": 6200},
                "right": {"dark_pixel_count": 6200},
            },
            "artifacts": {"composite": _artifact("composite")},
        },
    }
    projected = project_simulator_response(raw, profile)
    rendered = json.dumps(projected)
    assert set(projected) == {"status", "observation"}
    assert set(projected["observation"]) == {
        "observation_id",
        "modalities",
        "robot_state",
    }
    for forbidden in (
        "targetward_world_y_sign",
        "cumulative_y_m",
        "target_halfspace_locked",
        "guidance_unlocked",
        "stage",
        "remaining_probes",
        "remaining_post_prediction_actions",
        "tactile_health",
        "task_success_feedback",
        "seed_commitment_sha256",
        "dark_pixel_count",
    ):
        assert forbidden not in rendered


def test_fail_closed_audit_rejects_a_future_agent_visible_hint() -> None:
    profile = get_profile(1)
    response = {
        "status": "rollout_started",
        "observation": {
            "observation_id": "obs_000",
            "modalities": {
                "head_rgb": _artifact("head_rgb"),
                "wrist_rgb": _artifact("wrist_rgb"),
            },
            "robot_state": _robot_state(),
        },
        "targetward_world_y_sign": -1.0,
    }
    with pytest.raises(AgentVisibilityError, match="unexpected field"):
        assert_agent_visible_response(response, profile)


def test_only_level3_may_receive_post_commit_task_success() -> None:
    raw = {
        "status": "action_complete",
        "feedback": {"execution_succeeded": True, "task_success": False},
        "observation": {
            "observation_id": "obs_001",
            "modalities": {},
            "robot_state": _robot_state(),
        },
    }
    level1 = get_profile(1)
    raw["observation"]["modalities"] = {
        name: _artifact(name) for name in level1.public_modalities
    }
    projected_level1 = project_simulator_response(raw, level1)
    assert "feedback" not in projected_level1

    level3 = get_profile(3)
    raw["observation"]["modalities"] = {
        name: _artifact(name) for name in level3.public_modalities
    }
    projected_level3 = project_simulator_response(raw, level3)
    assert projected_level3["feedback"] == {
        "task_success": False,
    }
