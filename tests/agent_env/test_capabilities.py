from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_env.capabilities import (
    CapabilityGateway,
    CapabilityViolation,
    ToolCallLoopError,
    build_tool_registry,
    capability_manifest,
)
from agent_env.profiles import get_profile


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def decision(source: str = "head_rgb") -> dict[str, Any]:
    return {
        "evidence": [
            {
                "source": source,
                "finding": "The object edge remains centered.",
                "implication": "A small bounded motion should preserve the grasp.",
            }
        ],
        "alternatives_considered": ["Wait instead, but it would add no new evidence."],
        "uncertainty": 0.25,
        "expected_effect": "The object should move toward the selected pad.",
        "parameter_rationale": "One centimetre is visible but below the maximum bound.",
        "rationale": "Use the latest public image to make a conservative move.",
    }


class FakeSimulator:
    def __init__(self, run_dir: Path, level: int) -> None:
        self.run_dir = run_dir
        self.profile = get_profile(level)
        self.calls: list[dict[str, Any]] = []
        self.observation_index = 0
        self.leak_task_success = False

    def _artifact(self, observation_id: str, modality: str) -> dict[str, str]:
        path = self.run_dir / "observations" / observation_id / f"{modality}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_1X1)
        return {"path": str(path), "sha256": hashlib.sha256(PNG_1X1).hexdigest()}

    def observation(self) -> dict[str, Any]:
        observation_id = f"obs_{self.observation_index:03d}"
        self.observation_index += 1
        observation = {
            "observation_id": observation_id,
            "level": self.profile.level,
            "profile": self.profile.name,
            "stage": "classification" if observation_id == "obs_000" else "post_prediction_control",
            "probe_count": 0,
            "post_prediction_action_count": 0,
            "modalities": {
                modality: self._artifact(observation_id, modality)
                for modality in self.profile.public_modalities
            },
            "robot_state": {
                "joint_position_8d": [0.0] * 8,
                "gripper_qpos": 0.007,
                "end_effector_pose_robot_base_7d": [0.0] * 7,
            },
        }
        if self.profile.expose_tactile:
            observation["tactile_health"] = {
                side: {
                    "healthy": True,
                    "expected_markers": 63,
                    "plausible_marker_components": 61,
                    "all_dark_components": 64,
                    "dark_pixel_count": 6200,
                }
                for side in ("left", "right")
            }
        return observation

    def __call__(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(command)
        name = command["command"]
        if name == "start":
            return {
                "status": "rollout_started",
                "level": self.profile.level,
                "profile": self.profile.name,
                "seed_commitment_sha256": "a" * 64,
                "stage": "classification",
                "required_before_translation": "submit_prediction",
                "optional_before_prediction": "up to 2 probes",
                "task_success_feedback": "withheld",
                "observation": self.observation(),
            }
        if name == "submit_prediction":
            return {
                "status": "prediction_submitted",
                "predicted_class": command["predicted_class"],
                "committed_target": command["target_pad"],
                "irreversible": True,
                "observation_id": "obs_000",
                "rationale": command["rationale"],
                "success_seen_before_prediction": False,
                "stage": "post_prediction_control",
                "task_success_feedback": "withheld",
                "remaining_post_prediction_actions": 10,
            }
        if name in {"act", "wait", "probe"}:
            response: dict[str, Any] = {
                "status": "probe_complete" if name == "probe" else "action_complete",
                "feedback": {"execution_succeeded": True},
                "observation": self.observation(),
            }
            if name != "probe" and (
                self.profile.expose_task_success_after_prediction
                or self.leak_task_success
            ):
                response["feedback"]["task_success"] = False
            return response
        if name == "finish":
            return {
                "status": "rollout_finished",
                "true_class": "rough",
                "expected_target": "orange",
                "official_task_success": False,
            }
        if name == "status":
            return {
                "status": "bridge_status",
                "stage": "classification",
                "latest_observation_id": "obs_000",
                "terminal": False,
                "run_dir": str(self.run_dir),
            }
        raise AssertionError(name)


def started_gateway(tmp_path: Path, level: int = 1) -> tuple[CapabilityGateway, FakeSimulator]:
    simulator = FakeSimulator(tmp_path, level)
    gateway = CapabilityGateway(
        level=level,
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
    )
    execution = gateway.execute("start_episode", {"agent_note": "independent run"})
    assert execution.success is True
    return gateway, simulator


def test_level_registry_exposes_only_bounded_embodied_primitives() -> None:
    expected = {
        "start_episode",
        "probe_gripper",
        "commit_classification",
        "act_delta_ee",
        "wait_physics",
        "finish_episode",
    }
    for level in (1, 2, 3):
        registry = build_tool_registry(level)
        assert set(registry) == expected
        assert not any(
            forbidden in name
            for name in registry
            for forbidden in ("ik", "joint_target", "planner", "shell", "python")
        )
        assert registry["act_delta_ee"].simulator_command == "act"
        assert registry["act_delta_ee"].effect == "world_mutating"
        act_schema = registry["act_delta_ee"].input_schema["properties"]
        probe_schema = registry["probe_gripper"].input_schema["properties"]
        assert act_schema["delta_gripper"]["maximum"] == 0.005
        assert probe_schema["delta_gripper"]["maximum"] == 0.002
        assert "only after commit_classification" in registry["act_delta_ee"].description
        assert "only after commit_classification" in registry["wait_physics"].description
        assert "at most two probes" in registry["probe_gripper"].description


def test_consecutive_rejected_calls_are_bounded_without_public_counter(
    tmp_path: Path,
) -> None:
    gateway, _ = started_gateway(tmp_path)
    assert gateway.MAX_CONSECUTIVE_REJECTED_CALLS == 6
    arguments = {
        "observation_id": "obs_000",
        "steps": 1,
        "decision_record": decision(),
    }
    for _ in range(gateway.MAX_CONSECUTIVE_REJECTED_CALLS):
        rejected = gateway.execute("wait_physics", arguments)
        assert rejected.success is False
        assert "remaining" not in json.dumps(rejected.public_response)
    with pytest.raises(ToolCallLoopError, match="limit exceeded"):
        gateway.execute("wait_physics", arguments)


def test_schema_rejection_is_actionable_without_exposing_private_state(
    tmp_path: Path,
) -> None:
    gateway, _ = started_gateway(tmp_path)
    invalid = decision()
    invalid["evidence"] = []
    rejected = gateway.execute(
        "probe_gripper",
        {
            "observation_id": "not-the-latest-id",
            "delta_gripper": -0.001,
            "decision_record": invalid,
        },
    )
    assert rejected.success is False
    assert rejected.public_response == {
        "status": "tool_rejected",
        "tool": "probe_gripper",
        "message": "decision_record.evidence must contain at least one item.",
        "observation_id": "obs_000",
    }
    rendered = json.dumps(rejected.public_response)
    assert "stage" not in rendered
    assert "remaining" not in rendered
    assert "success" not in rendered


def test_decision_schema_sources_are_level_scoped() -> None:
    def sources(level: int) -> set[str]:
        schema = build_tool_registry(level)["act_delta_ee"].input_schema
        return set(
            schema["properties"]["decision_record"]["properties"]["evidence"]
            ["items"]["properties"]["source"]["enum"]
        )

    assert sources(1) == {"head_rgb", "wrist_rgb", "robot_state"}
    assert sources(2) == sources(3) == {
        "head_rgb",
        "wrist_rgb",
        "left_tactile_marker",
        "right_tactile_marker",
        "robot_state",
    }


def test_capability_manifest_is_stable_and_distinguishes_levels() -> None:
    manifests = [capability_manifest(level) for level in (1, 2, 3)]
    assert len({manifest["sha256"] for manifest in manifests}) == 3
    for manifest in manifests:
        unsigned = dict(manifest)
        digest = unsigned.pop("sha256")
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == digest


def test_gateway_returns_only_level_images_and_never_host_paths(tmp_path: Path) -> None:
    gateway, _ = started_gateway(tmp_path, level=1)
    first = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision(),
        },
    )
    assert first.success is True
    assert first.public_response == {
        "status": "prediction_submitted",
        "observation_id": "obs_000",
    }

    # The original start call delivered one text payload plus two labeled image pairs.
    fresh_gateway = CapabilityGateway(
        level=1,
        simulator_request=FakeSimulator(tmp_path / "fresh", 1),
        simulator_run_dir=tmp_path / "fresh",
    )
    start = fresh_gateway.execute("start_episode", {"agent_note": "fresh"})
    assert len(start.content_items) == 5
    assert sum(item["type"] == "inputImage" for item in start.content_items) == 2
    rendered = json.dumps(start.public_response)
    assert str((tmp_path / "fresh").resolve()) not in rendered
    assert "artifact_id" in rendered
    assert set(start.public_response) == {"status", "observation"}
    assert set(start.public_response["observation"]) == {
        "observation_id",
        "modalities",
        "robot_state",
    }
    for forbidden in (
        "level",
        "profile",
        "stage",
        "probe_count",
        "post_prediction_action_count",
        "seed_commitment_sha256",
        "required_before_translation",
        "optional_before_prediction",
        "task_success_feedback",
    ):
        assert forbidden not in rendered


def test_unregistered_ik_and_unknown_ik_fields_never_reach_simulator(tmp_path: Path) -> None:
    gateway, simulator = started_gateway(tmp_path)
    before = len(simulator.calls)
    with pytest.raises(CapabilityViolation, match="Unregistered"):
        gateway.execute("solve_ik", {"target_pose": [0] * 7})
    assert len(simulator.calls) == before

    rejected = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "ik_solution": [0.0] * 7,
            "decision_record": decision(),
        },
    )
    assert rejected.success is False
    assert rejected.public_response["message"] == (
        "Tool arguments do not satisfy the published input schema."
    )
    assert rejected.raw_response is not None
    assert "Unknown tool field" in rejected.raw_response["message"]
    assert "stage" not in rejected.public_response
    assert "latest_observation_id" not in rejected.public_response
    assert len(simulator.calls) == before


def test_level1_cannot_claim_tactile_evidence(tmp_path: Path) -> None:
    gateway, simulator = started_gateway(tmp_path, level=1)
    rejected = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision("left_tactile_marker"),
        },
    )
    assert rejected.success is False
    assert "Level 1" not in rejected.public_response["message"]
    assert rejected.raw_response is not None
    assert "unavailable at Level 1" in rejected.raw_response["message"]
    assert len(simulator.calls) == 1


def test_level2_can_use_tactile_but_cannot_receive_success(tmp_path: Path) -> None:
    gateway, simulator = started_gateway(tmp_path, level=2)
    committed = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision("left_tactile_marker"),
        },
    )
    assert committed.success is True
    simulator.leak_task_success = True
    with pytest.raises(CapabilityViolation, match="task_success"):
        gateway.execute(
            "wait_physics",
            {
                "observation_id": "obs_000",
                "steps": 10,
                "decision_record": decision("right_tactile_marker"),
            },
        )


def test_level3_receives_success_only_after_irreversible_commit(tmp_path: Path) -> None:
    gateway, _ = started_gateway(tmp_path, level=3)
    committed = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision("left_tactile_marker"),
        },
    )
    assert committed.success is True
    waited = gateway.execute(
        "wait_physics",
        {
            "observation_id": "obs_000",
            "steps": 10,
            "decision_record": decision("right_tactile_marker"),
        },
    )
    assert waited.success is True
    assert waited.public_response["feedback"]["task_success"] is False


def test_tactile_health_statistics_are_host_only(tmp_path: Path) -> None:
    gateway, _ = started_gateway(tmp_path, level=2)
    start = CapabilityGateway(
        level=2,
        simulator_request=FakeSimulator(tmp_path / "second", 2),
        simulator_run_dir=tmp_path / "second",
    ).execute("start_episode", {"agent_note": "fresh"})
    assert start.raw_response is not None
    assert "tactile_health" in start.raw_response["observation"]
    assert "tactile_health" not in start.public_response["observation"]
    assert set(start.public_response["observation"]["modalities"]) == {
        "head_rgb",
        "wrist_rgb",
        "left_tactile_marker",
        "right_tactile_marker",
    }


def test_stale_observation_and_out_of_bounds_action_are_host_rejected(tmp_path: Path) -> None:
    gateway, simulator = started_gateway(tmp_path, level=1)
    gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "plain",
            "target_pad": "green",
            "decision_record": decision(),
        },
    )
    before = len(simulator.calls)
    stale = gateway.execute(
        "act_delta_ee",
        {
            "observation_id": "obs_999",
            "delta_position": [0, 0, 0],
            "delta_rpy": [0, 0, 0],
            "delta_gripper": 0,
            "decision_record": decision(),
        },
    )
    oversized = gateway.execute(
        "act_delta_ee",
        {
            "observation_id": "obs_000",
            "delta_position": [0.05, 0, 0],
            "delta_rpy": [0, 0, 0],
            "delta_gripper": 0,
            "decision_record": decision(),
        },
    )
    assert stale.success is oversized.success is False
    assert stale.decision_record == decision()
    assert oversized.decision_record == decision()
    assert len(simulator.calls) == before
    assert stale.public_response["observation_id"] == "obs_000"
    assert oversized.public_response["observation_id"] == "obs_000"
    assert stale.public_response["message"].startswith(
        "Use the exact observation_id"
    )
    assert oversized.public_response["message"] == (
        "Tool arguments do not satisfy the published input schema."
    )
    for response in (stale.public_response, oversized.public_response):
        rendered = json.dumps(response)
        assert "stage" not in rendered
        assert "remaining" not in rendered


def test_terminal_truth_is_kept_out_of_agent_result(tmp_path: Path) -> None:
    gateway, _ = started_gateway(tmp_path, level=1)
    gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision(),
        },
    )
    finished = gateway.execute(
        "finish_episode",
        {
            "observation_id": "obs_000",
            "final_note": "done",
            "decision_record": decision(),
        },
    )
    assert finished.public_response == {"status": "rollout_finished"}
    assert finished.raw_response is not None
    assert finished.raw_response["true_class"] == "rough"
    assert finished.raw_response["official_task_success"] is False


def test_artifact_escape_is_rejected(tmp_path: Path) -> None:
    simulator = FakeSimulator(tmp_path, 1)
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(PNG_1X1)

    def escaped(_: dict[str, Any]) -> dict[str, Any]:
        observation = simulator.observation()
        observation["modalities"]["head_rgb"] = {
            "path": str(outside),
            "sha256": hashlib.sha256(PNG_1X1).hexdigest(),
        }
        return {"status": "rollout_started", "observation": observation}

    gateway = CapabilityGateway(
        level=1, simulator_request=escaped, simulator_run_dir=tmp_path
    )
    with pytest.raises(CapabilityViolation, match="outside"):
        gateway.execute("start_episode", {"agent_note": "escape test"})


def test_future_simulator_field_fails_closed(tmp_path: Path) -> None:
    simulator = FakeSimulator(tmp_path, 1)

    def future_response(command: dict[str, Any]) -> dict[str, Any]:
        response = simulator(command)
        response["new_privileged_hint"] = "do not expose"
        return response

    gateway = CapabilityGateway(
        level=1,
        simulator_request=future_response,
        simulator_run_dir=tmp_path,
    )
    with pytest.raises(CapabilityViolation, match="unregistered top-level"):
        gateway.execute("start_episode", {"agent_note": "future field test"})


def test_future_nested_observation_field_fails_closed(tmp_path: Path) -> None:
    simulator = FakeSimulator(tmp_path, 1)

    def future_observation(command: dict[str, Any]) -> dict[str, Any]:
        response = simulator(command)
        response["observation"]["object_pose"] = [0.0] * 7
        return response

    gateway = CapabilityGateway(
        level=1,
        simulator_request=future_observation,
        simulator_run_dir=tmp_path,
    )
    with pytest.raises(CapabilityViolation, match="observation contains unregistered"):
        gateway.execute("start_episode", {"agent_note": "nested future field"})
