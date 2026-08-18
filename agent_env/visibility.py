"""Fail-closed projection from host protocol data to an agent-visible result.

The simulator keeps rich records for replay and evaluator auditing.  Those
records are not the agent interface.  This module defines the much smaller
projection that may be returned to an embodied agent: sensor frames, basic
robot state, and the one Level-specific feedback bit explicitly declared by
the benchmark.
"""

from __future__ import annotations

import copy
from typing import Any

from .profiles import AgentEnvProfile


JsonDict = dict[str, Any]


class AgentVisibilityError(RuntimeError):
    """Raised when a projected response exceeds the declared agent surface."""


_SIMULATOR_STATUSES = {
    "rollout_started",
    "probe_complete",
    "prediction_submitted",
    "action_complete",
    "rollout_finished",
    "bridge_status",
    "command_error",
}

_FORBIDDEN_AGENT_KEYS = {
    "active",
    "committed_target",
    "commitment_verified",
    "cumulative_y_m",
    "execution_duration_seconds",
    "execution_succeeded",
    "expected_target",
    "final_observation_id",
    "guidance_unlocked",
    "level",
    "official_task_success",
    "optional_before_prediction",
    "post_prediction_action_count",
    "predicted_class",
    "probe_count",
    "profile",
    "remaining_post_prediction_actions",
    "remaining_probes",
    "required_before_translation",
    "reward",
    "run_dir",
    "salt_reveal",
    "seed_commitment_sha256",
    "seed_reveal",
    "stage",
    "success_seen_before_prediction",
    "target_halfspace_locked",
    "targetward_world_y_sign",
    "task_success_feedback",
    "terminal_reason",
    "true_class",
    "wall_seconds",
}


def _observation(value: Any, profile: AgentEnvProfile) -> JsonDict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AgentVisibilityError("Simulator observation must be an object")
    observation_id = value.get("observation_id")
    modalities = value.get("modalities")
    robot_state = value.get("robot_state")
    if not isinstance(observation_id, str):
        raise AgentVisibilityError("Simulator observation omitted observation_id")
    if not isinstance(modalities, dict) or set(modalities) != set(
        profile.public_modalities
    ):
        raise AgentVisibilityError("Simulator observation modalities do not match profile")
    if not isinstance(robot_state, dict) or set(robot_state) != set(
        profile.public_robot_state
    ):
        raise AgentVisibilityError("Simulator robot state does not match profile")
    return {
        "observation_id": observation_id,
        "modalities": {
            name: copy.deepcopy(modalities[name]) for name in profile.public_modalities
        },
        "robot_state": {
            name: copy.deepcopy(robot_state[name]) for name in profile.public_robot_state
        },
    }


def _task_feedback(value: Any, profile: AgentEnvProfile) -> JsonDict:
    if not isinstance(value, dict):
        raise AgentVisibilityError("Simulator feedback must be an object")
    if not profile.expose_task_success_after_prediction:
        raise AgentVisibilityError("Task feedback is unavailable outside Level 3")
    success = value.get("task_success")
    if not isinstance(success, bool):
        raise AgentVisibilityError("Level 3 feedback omitted task_success")
    return {"task_success": success}


def project_simulator_response(
    response: JsonDict,
    profile: AgentEnvProfile,
) -> JsonDict:
    """Return only information that the embodied agent is allowed to observe."""

    status = response.get("status")
    if status not in _SIMULATOR_STATUSES:
        raise AgentVisibilityError(f"Unrecognized simulator status {status!r}")
    projected: JsonDict = {"status": status}

    if status == "rollout_started":
        projected["observation"] = _observation(response.get("observation"), profile)
    elif status in {"probe_complete", "action_complete"}:
        projected["observation"] = _observation(response.get("observation"), profile)
        if status == "action_complete" and profile.expose_task_success_after_prediction:
            projected["feedback"] = _task_feedback(response.get("feedback"), profile)
    elif status == "prediction_submitted":
        observation_id = response.get("observation_id")
        if not isinstance(observation_id, str):
            raise AgentVisibilityError("Prediction acknowledgement omitted observation_id")
        projected["observation_id"] = observation_id
    elif status == "rollout_finished":
        observation = _observation(response.get("observation"), profile)
        if observation is not None:
            projected["observation"] = observation
        last_action = response.get("last_action")
        if (
            profile.expose_task_success_after_prediction
            and isinstance(last_action, dict)
            and last_action.get("feedback") is not None
        ):
            projected["feedback"] = _task_feedback(last_action["feedback"], profile)
    elif status == "command_error":
        # Exact exception text is retained by the host transcript.  Returning it
        # would let an agent query hidden protocol state through rejected calls.
        projected["message"] = "Command rejected by the environment contract."
    # bridge_status is host-only.  A status-only projection remains fail-safe if
    # an internal caller accidentally routes it through this function.

    return projected


def assert_agent_visible_response(
    response: JsonDict,
    profile: AgentEnvProfile,
) -> None:
    """Verify the final, path-scrubbed payload sent to the model."""

    status = response.get("status")
    expected_top_level: dict[str, set[str]] = {
        "rollout_started": {"status", "observation"},
        "probe_complete": {"status", "observation"},
        "prediction_submitted": {"status", "observation_id"},
        "action_complete": {"status", "feedback", "observation"},
        "rollout_finished": {"status", "feedback", "observation"},
        "bridge_status": {"status"},
        "command_error": {"status", "message"},
    }
    allowed = expected_top_level.get(str(status))
    if allowed is None:
        raise AgentVisibilityError(f"Unrecognized projected status {status!r}")
    unknown = set(response) - allowed
    if unknown:
        raise AgentVisibilityError(
            "Agent response contains unexpected field(s): " + ", ".join(sorted(unknown))
        )
    required = set(allowed)
    if status == "action_complete" and not profile.expose_task_success_after_prediction:
        required.discard("feedback")
    if status != "rollout_finished" and set(response) != required:
        missing = required - set(response)
        if missing:
            raise AgentVisibilityError(
                "Agent response omitted field(s): " + ", ".join(sorted(missing))
            )

    observation = response.get("observation")
    if observation is not None:
        if not isinstance(observation, dict) or set(observation) != {
            "observation_id",
            "modalities",
            "robot_state",
        }:
            raise AgentVisibilityError("Agent observation exceeds the minimal surface")
        modalities = observation.get("modalities")
        if not isinstance(modalities, dict) or set(modalities) != set(
            profile.public_modalities
        ):
            raise AgentVisibilityError("Agent modalities do not match profile")
        for artifact in modalities.values():
            if not isinstance(artifact, dict) or set(artifact) != {
                "artifact_id",
                "sha256",
            }:
                raise AgentVisibilityError("Agent image metadata exceeds transport fields")
        robot_state = observation.get("robot_state")
        if not isinstance(robot_state, dict) or set(robot_state) != set(
            profile.public_robot_state
        ):
            raise AgentVisibilityError("Agent robot state does not match profile")

    feedback = response.get("feedback")
    if feedback is not None:
        expected_feedback = {"task_success"}
        if not profile.expose_task_success_after_prediction:
            raise AgentVisibilityError("Agent feedback is unavailable outside Level 3")
        if not isinstance(feedback, dict) or set(feedback) != expected_feedback:
            raise AgentVisibilityError("Agent feedback exceeds the selected Level")
        if not all(isinstance(value, bool) for value in feedback.values()):
            raise AgentVisibilityError("Agent feedback values must be boolean")

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
            return
        if not isinstance(value, dict):
            return
        forbidden = _FORBIDDEN_AGENT_KEYS & set(value)
        if forbidden:
            raise AgentVisibilityError(
                "Agent response leaked forbidden field(s): "
                + ", ".join(sorted(forbidden))
            )
        if "task_success" in value and not profile.expose_task_success_after_prediction:
            raise AgentVisibilityError("task_success leaked outside Level 3")
        for child in value.values():
            visit(child)

    visit(response)
