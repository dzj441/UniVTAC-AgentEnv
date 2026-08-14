"""Level-scoped tool contracts for an embodied Codex operator.

The simulator protocol is deliberately not handed to the model as a generic
JSON pipe.  Instead, this module projects it into a small immutable registry of
typed tools.  The registry is the host-owned authority: prompts may describe
tools, but they cannot add one, widen a schema, or bypass stage checks.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .profiles import AgentEnvProfile, get_profile


JsonDict = dict[str, Any]
SimulatorRequest = Callable[[JsonDict], JsonDict]


class CapabilityViolation(RuntimeError):
    """An attempt to invoke something outside the registered tool surface."""


class ToolInputError(ValueError):
    """A malformed or stage-invalid call to an otherwise registered tool."""


@dataclass(frozen=True)
class EmbodiedToolSpec:
    """One agent-visible atomic capability."""

    name: str
    description: str
    input_schema: JsonDict
    simulator_command: str
    effect: str
    allowed_stages: frozenset[str]
    requires_decision_record: bool = True

    def to_dynamic_tool(self) -> JsonDict:
        """Convert to the Codex app-server dynamic-tool representation."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "inputSchema": copy.deepcopy(self.input_schema),
        }

    def to_manifest(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "effect": self.effect,
            "allowed_stages": sorted(self.allowed_stages),
            "simulator_command": self.simulator_command,
            "requires_decision_record": self.requires_decision_record,
            "input_schema": copy.deepcopy(self.input_schema),
        }


@dataclass(frozen=True)
class GatewayExecution:
    """Result of one dynamic tool call, including private relay evidence."""

    tool: str
    success: bool
    simulator_command: JsonDict | None
    raw_response: JsonDict | None
    public_response: JsonDict
    content_items: tuple[JsonDict, ...]
    decision_record: JsonDict | None
    prior_observation_id: str | None
    next_observation_id: str | None


def _string_schema(description: str) -> JsonDict:
    return {"type": "string", "minLength": 1, "description": description}


def _decision_schema(profile: AgentEnvProfile) -> JsonDict:
    sources = [*profile.public_modalities, "robot_state"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "evidence",
            "alternatives_considered",
            "uncertainty",
            "expected_effect",
            "parameter_rationale",
            "rationale",
        ],
        "properties": {
            "evidence": {
                "type": "array",
                "minItems": 1,
                "description": "Only evidence present in the latest public observation.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "finding", "implication"],
                    "properties": {
                        "source": {"enum": sources},
                        "finding": _string_schema("What was directly observed."),
                        "implication": _string_schema(
                            "How that observation affects this decision."
                        ),
                    },
                },
            },
            "alternatives_considered": {
                "type": "array",
                "minItems": 1,
                "items": _string_schema("A concrete alternative and why it was rejected."),
            },
            "uncertainty": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "0 is certain and 1 is maximally uncertain.",
            },
            "expected_effect": _string_schema(
                "Expected visible or tactile effect of the proposed operation."
            ),
            "parameter_rationale": _string_schema(
                "Why these exact magnitudes were selected over nearby alternatives."
            ),
            "rationale": _string_schema(
                "Concise decision rationale suitable for the simulator audit trail."
            ),
        },
    }


def _object_schema(
    properties: Mapping[str, JsonDict],
    required: tuple[str, ...],
) -> JsonDict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": copy.deepcopy(dict(properties)),
    }


def build_tool_registry(level: int | str) -> dict[str, EmbodiedToolSpec]:
    """Build the immutable, level-specific agent tool surface."""

    profile = get_profile(level)
    decision = _decision_schema(profile)
    observation_id = _string_schema("The latest observation_id returned by a tool.")
    vector3 = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    tools = (
        EmbodiedToolSpec(
            name="start_episode",
            description=(
                "Initialize the one-shot grasp_classify episode and return the first "
                f"Level {profile.level} observation. Call exactly once."
            ),
            input_schema=_object_schema(
                {"agent_note": _string_schema("Purpose of this independent rollout.")},
                ("agent_note",),
            ),
            simulator_command="start",
            effect="world_mutating",
            allowed_stages=frozenset({"ready"}),
            requires_decision_record=False,
        ),
        EmbodiedToolSpec(
            name="probe_gripper",
            description=(
                "Apply one bounded gripper-only probe before classification and return a "
                "fresh observation. No Cartesian translation or rotation is available. "
                "Units are metres: the maximum absolute delta is 0.002 m = 2 mm; "
                "values such as 0.003, 0.01, or 0.02 are invalid."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "delta_gripper": {
                        "type": "number",
                        "minimum": -0.002,
                        "maximum": 0.002,
                        "description": (
                            "Nonzero gripper delta in metres; absolute maximum "
                            "0.002 m = 2 mm."
                        ),
                    },
                    "decision_record": decision,
                },
                ("observation_id", "delta_gripper", "decision_record"),
            ),
            simulator_command="probe",
            effect="world_mutating",
            allowed_stages=frozenset({"classification"}),
        ),
        EmbodiedToolSpec(
            name="commit_classification",
            description=(
                "Irreversibly classify the grasped surface and commit to its mapped pad. "
                "rough maps to orange; plain maps to green."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "predicted_class": {"enum": ["rough", "plain"]},
                    "target_pad": {"enum": ["orange", "green"]},
                    "decision_record": decision,
                },
                (
                    "observation_id",
                    "predicted_class",
                    "target_pad",
                    "decision_record",
                ),
            ),
            simulator_command="submit_prediction",
            effect="irreversible_commitment",
            allowed_stages=frozenset({"classification"}),
        ),
        EmbodiedToolSpec(
            name="act_delta_ee",
            description=(
                "Execute one bounded world-frame end-effector delta after commitment. "
                "This is the only Cartesian control interface; no IK, joint target, "
                "object pose, trajectory planner, or privileged state is exposed. "
                "Position units are metres (4 cm per component, 6 cm norm maximum); "
                "gripper maximum is 0.005 m = 5 mm."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "delta_position": {
                        **vector3,
                        "description": (
                            "World XYZ metres; each component <=0.04 and norm <=0.06."
                        ),
                    },
                    "delta_rpy": {
                        **vector3,
                        "description": "World roll/pitch/yaw radians; each <=0.35.",
                    },
                    "delta_gripper": {
                        "type": "number",
                        "minimum": -0.005,
                        "maximum": 0.005,
                        "description": (
                            "Gripper delta in metres; absolute maximum 0.005 m = 5 mm. "
                            "Values such as 0.01 or 0.02 are invalid."
                        ),
                    },
                    "decision_record": decision,
                },
                (
                    "observation_id",
                    "delta_position",
                    "delta_rpy",
                    "delta_gripper",
                    "decision_record",
                ),
            ),
            simulator_command="act",
            effect="world_mutating",
            allowed_stages=frozenset({"post_prediction_control"}),
        ),
        EmbodiedToolSpec(
            name="wait_physics",
            description=(
                "Advance physics without issuing a robot-control delta, then return a "
                "fresh observation."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "steps": {"type": "integer", "minimum": 1, "maximum": 60},
                    "decision_record": decision,
                },
                ("observation_id", "steps", "decision_record"),
            ),
            simulator_command="wait",
            effect="world_mutating",
            allowed_stages=frozenset({"post_prediction_control"}),
        ),
        EmbodiedToolSpec(
            name="finish_episode",
            description="End the rollout at the current state and request terminal evaluation.",
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "final_note": _string_schema("Terminal summary of the attempted task."),
                    "decision_record": decision,
                },
                ("observation_id", "final_note", "decision_record"),
            ),
            simulator_command="finish",
            effect="terminal",
            allowed_stages=frozenset({"post_prediction_control"}),
        ),
        EmbodiedToolSpec(
            name="inspect_episode_status",
            description=(
                "Read public protocol status only. It never exposes object pose, truth, "
                "reward, hidden seed, or locked success feedback."
            ),
            input_schema=_object_schema({}, ()),
            simulator_command="status",
            effect="read_only",
            allowed_stages=frozenset(
                {"ready", "classification", "post_prediction_control", "terminal"}
            ),
            requires_decision_record=False,
        ),
    )
    return {tool.name: tool for tool in tools}


def capability_manifest(level: int | str) -> JsonDict:
    profile = get_profile(level)
    registry = build_tool_registry(level)
    payload: JsonDict = {
        "schema_version": "univtac.codex_capabilities.v1",
        "level": profile.level,
        "profile": profile.to_manifest(),
        "tools": [tool.to_manifest() for tool in registry.values()],
        "forbidden_agent_capabilities": [
            "shell or arbitrary code execution",
            "filesystem reads or writes",
            "web, apps, plugins, skills, or subagents",
            "direct IK or joint-target commands",
            "trajectory-planner access",
            "simulator internals, object/pad poses, reward, or checker state",
            "camera depth, intrinsics, or extrinsics",
        ],
        "host_enforcement": [
            "only registered dynamic tools are relayed",
            "tool input is validated again outside the model",
            "every world-changing call must cite the latest observation",
            "simulator protocol independently revalidates bounds and stages",
            "model-visible artifacts contain ids and hashes, never host paths",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ToolInputError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ToolInputError(f"{name} must be a finite number")
    return result


def _finite_vector3(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ToolInputError(f"{name} must contain exactly three finite numbers")
    return [_finite_number(item, name) for item in value]


def _validate_decision_record(
    value: Any,
    profile: AgentEnvProfile,
) -> JsonDict:
    if not isinstance(value, dict):
        raise ToolInputError("decision_record must be an object")
    expected = {
        "evidence",
        "alternatives_considered",
        "uncertainty",
        "expected_effect",
        "parameter_rationale",
        "rationale",
    }
    if set(value) != expected:
        raise ToolInputError(
            "decision_record fields must be exactly: " + ", ".join(sorted(expected))
        )
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ToolInputError("decision_record.evidence must be a non-empty array")
    allowed_sources = {*profile.public_modalities, "robot_state"}
    normalized_evidence: list[JsonDict] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {
            "source",
            "finding",
            "implication",
        }:
            raise ToolInputError(
                f"decision_record.evidence[{index}] has invalid fields"
            )
        source = _nonempty_string(item["source"], "evidence.source")
        if source not in allowed_sources:
            raise ToolInputError(
                f"Evidence source {source!r} is unavailable at Level {profile.level}"
            )
        normalized_evidence.append(
            {
                "source": source,
                "finding": _nonempty_string(item["finding"], "evidence.finding"),
                "implication": _nonempty_string(
                    item["implication"], "evidence.implication"
                ),
            }
        )
    alternatives = value["alternatives_considered"]
    if not isinstance(alternatives, list) or not alternatives:
        raise ToolInputError(
            "decision_record.alternatives_considered must be a non-empty array"
        )
    uncertainty = _finite_number(value["uncertainty"], "decision_record.uncertainty")
    if not 0.0 <= uncertainty <= 1.0:
        raise ToolInputError("decision_record.uncertainty must be in [0, 1]")
    return {
        "evidence": normalized_evidence,
        "alternatives_considered": [
            _nonempty_string(item, "alternatives_considered item")
            for item in alternatives
        ],
        "uncertainty": uncertainty,
        "expected_effect": _nonempty_string(
            value["expected_effect"], "decision_record.expected_effect"
        ),
        "parameter_rationale": _nonempty_string(
            value["parameter_rationale"], "decision_record.parameter_rationale"
        ),
        "rationale": _nonempty_string(
            value["rationale"], "decision_record.rationale"
        ),
    }


class CapabilityGateway:
    """Validate dynamic calls, relay allowed commands, and scrub host paths."""

    def __init__(
        self,
        *,
        level: int | str,
        simulator_request: SimulatorRequest,
        simulator_run_dir: Path,
    ) -> None:
        self.profile = get_profile(level)
        self.registry = build_tool_registry(level)
        self._simulator_request = simulator_request
        self.simulator_run_dir = simulator_run_dir.resolve()
        self.stage = "ready"
        self.latest_observation_id: str | None = None
        self.terminal = False
        self._lock = threading.Lock()

    def dynamic_tools(self) -> list[JsonDict]:
        return [tool.to_dynamic_tool() for tool in self.registry.values()]

    def execute(self, tool_name: str, arguments: Any) -> GatewayExecution:
        """Execute exactly one registered tool under a closed-loop lock."""

        with self._lock:
            if tool_name not in self.registry:
                raise CapabilityViolation(f"Unregistered embodied tool: {tool_name!r}")
            spec = self.registry[tool_name]
            if not isinstance(arguments, dict):
                return self._rejected(tool_name, "Tool arguments must be an object")
            try:
                command, decision = self._build_command(spec, arguments)
            except ToolInputError as exc:
                # Preserve a structurally valid decision even when a later stage,
                # observation-id, or numeric-bound check rejects the operation.  The
                # rejected attempt is part of the agent's externally meaningful
                # behavior and must remain available for replay.
                rejected_decision: JsonDict | None = None
                if spec.requires_decision_record and "decision_record" in arguments:
                    try:
                        rejected_decision = _validate_decision_record(
                            arguments["decision_record"], self.profile
                        )
                    except ToolInputError:
                        pass
                return self._rejected(
                    tool_name,
                    str(exc),
                    decision_record=rejected_decision,
                )

            prior_observation_id = self.latest_observation_id
            raw_response = self._simulator_request(command)
            if not isinstance(raw_response, dict):
                raise RuntimeError("Simulator returned a non-object response")
            self._assert_no_capability_leak(raw_response)
            public_response, images = self._publicize_response(raw_response)
            self._update_state(raw_response)
            success = raw_response.get("status") != "command_error"
            content_items: list[JsonDict] = [
                {
                    "type": "inputText",
                    "text": json.dumps(public_response, ensure_ascii=False),
                }
            ]
            for modality, image_path, sha256 in images:
                content_items.append(
                    {
                        "type": "inputText",
                        "text": (
                            f"The next image is modality={modality!r}, "
                            f"sha256={sha256}."
                        ),
                    }
                )
                content_items.append(
                    {
                        "type": "inputImage",
                        "imageUrl": self._png_data_url(image_path, sha256),
                    }
                )
            return GatewayExecution(
                tool=tool_name,
                success=success,
                simulator_command=command,
                raw_response=raw_response,
                public_response=public_response,
                content_items=tuple(content_items),
                decision_record=decision,
                prior_observation_id=prior_observation_id,
                next_observation_id=self.latest_observation_id,
            )

    def _rejected(
        self,
        tool_name: str,
        message: str,
        *,
        decision_record: JsonDict | None = None,
    ) -> GatewayExecution:
        response = {
            "status": "tool_rejected",
            "tool": tool_name,
            "message": message,
            "stage": self.stage,
            "latest_observation_id": self.latest_observation_id,
        }
        return GatewayExecution(
            tool=tool_name,
            success=False,
            simulator_command=None,
            raw_response=None,
            public_response=response,
            content_items=(
                {
                    "type": "inputText",
                    "text": json.dumps(response, ensure_ascii=False),
                },
            ),
            decision_record=decision_record,
            prior_observation_id=self.latest_observation_id,
            next_observation_id=self.latest_observation_id,
        )

    def _build_command(
        self,
        spec: EmbodiedToolSpec,
        arguments: JsonDict,
    ) -> tuple[JsonDict, JsonDict | None]:
        schema = spec.input_schema
        required = set(schema["required"])
        allowed = set(schema["properties"])
        missing = required - set(arguments)
        unknown = set(arguments) - allowed
        if missing:
            raise ToolInputError("Missing tool field(s): " + ", ".join(sorted(missing)))
        if unknown:
            raise ToolInputError("Unknown tool field(s): " + ", ".join(sorted(unknown)))
        if self.stage not in spec.allowed_stages:
            raise ToolInputError(
                f"{spec.name} is unavailable during stage {self.stage!r}"
            )

        decision: JsonDict | None = None
        if spec.requires_decision_record:
            decision = _validate_decision_record(
                arguments.get("decision_record"), self.profile
            )
        command: JsonDict = {"command": spec.simulator_command}
        if spec.name == "start_episode":
            command["agent_note"] = _nonempty_string(
                arguments["agent_note"], "agent_note"
            )
        elif spec.name == "inspect_episode_status":
            pass
        else:
            observation_id = _nonempty_string(
                arguments["observation_id"], "observation_id"
            )
            if observation_id != self.latest_observation_id:
                raise ToolInputError(
                    f"observation_id must be the latest id {self.latest_observation_id!r}"
                )
            assert decision is not None
            command["observation_id"] = observation_id
            command["rationale"] = (
                f"{decision['rationale']} Exact-parameter basis: "
                f"{decision['parameter_rationale']}"
            )

        if spec.name == "probe_gripper":
            delta = _finite_number(arguments["delta_gripper"], "delta_gripper")
            if delta == 0.0 or abs(delta) > 0.002:
                raise ToolInputError("delta_gripper must be nonzero and within +/-0.002")
            command["delta_gripper"] = delta
        elif spec.name == "commit_classification":
            predicted = arguments["predicted_class"]
            target = arguments["target_pad"]
            mapping = {"rough": "orange", "plain": "green"}
            if predicted not in mapping or target != mapping[predicted]:
                raise ToolInputError("Classification-to-pad mapping is inconsistent")
            command.update({"predicted_class": predicted, "target_pad": target})
        elif spec.name == "act_delta_ee":
            position = _finite_vector3(arguments["delta_position"], "delta_position")
            rotation = _finite_vector3(arguments["delta_rpy"], "delta_rpy")
            gripper = _finite_number(arguments["delta_gripper"], "delta_gripper")
            if max(map(abs, position)) > 0.04 or math.sqrt(
                sum(value * value for value in position)
            ) > 0.06:
                raise ToolInputError("delta_position exceeds the public action bound")
            if max(map(abs, rotation)) > 0.35:
                raise ToolInputError("delta_rpy exceeds the public action bound")
            if abs(gripper) > 0.005:
                raise ToolInputError("delta_gripper exceeds the public action bound")
            command.update(
                {
                    "delta_position": position,
                    "delta_rpy": rotation,
                    "delta_gripper": gripper,
                }
            )
        elif spec.name == "wait_physics":
            steps = arguments["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 60:
                raise ToolInputError("steps must be an integer in [1, 60]")
            command["steps"] = steps
        elif spec.name == "finish_episode":
            command["final_note"] = _nonempty_string(
                arguments["final_note"], "final_note"
            )
        return command, decision

    def _update_state(self, response: JsonDict) -> None:
        status = response.get("status")
        observation = response.get("observation")
        if isinstance(observation, dict) and isinstance(
            observation.get("observation_id"), str
        ):
            self.latest_observation_id = observation["observation_id"]
        if status == "rollout_started":
            self.stage = "classification"
        elif status == "prediction_submitted":
            self.stage = "post_prediction_control"
        elif status == "bridge_status":
            public_stage = response.get("stage")
            if isinstance(public_stage, str):
                self.stage = public_stage
            latest = response.get("latest_observation_id")
            if isinstance(latest, str):
                self.latest_observation_id = latest
            self.terminal = bool(response.get("terminal"))
        elif status == "rollout_finished":
            self.stage = "terminal"
            self.terminal = True

    def _assert_no_capability_leak(self, response: JsonDict) -> None:
        status = response.get("status")
        terminal_response = status == "rollout_finished"
        expected_top_level: dict[str, set[str]] = {
            "rollout_started": {
                "status",
                "level",
                "profile",
                "seed_commitment_sha256",
                "stage",
                "required_before_translation",
                "optional_before_prediction",
                "task_success_feedback",
                "observation",
            },
            "probe_complete": {
                "status",
                "probe_index",
                "prior_observation_id",
                "rationale",
                "action",
                "feedback",
                "execution_duration_seconds",
                "remaining_probes",
                "observation",
            },
            "prediction_submitted": {
                "status",
                "predicted_class",
                "committed_target",
                "irreversible",
                "guidance_unlocked",
                "observation_id",
                "rationale",
                "success_seen_before_prediction",
                "stage",
                "task_success_feedback",
                "remaining_post_prediction_actions",
                "targetward_world_y_sign",
            },
            "action_complete": {
                "status",
                "post_prediction_action_index",
                "prior_observation_id",
                "rationale",
                "action",
                "committed_target",
                "cumulative_y_m",
                "target_halfspace_locked",
                "feedback",
                "execution_duration_seconds",
                "observation",
                "remaining_post_prediction_actions",
            },
            "bridge_status": {
                "status",
                "profile",
                "level",
                "started",
                "active",
                "terminal",
                "stage",
                "guidance_unlocked",
                "probe_count",
                "post_prediction_action_count",
                "predicted_class",
                "committed_target",
                "latest_observation_id",
                "seed_commitment_sha256",
                "run_dir",
            },
            "rollout_finished": {
                "status",
                "timestamp_utc",
                "terminal_reason",
                "level",
                "profile",
                "predicted_class",
                "true_class",
                "classification_correct",
                "committed_target",
                "expected_target",
                "committed_pad_correct",
                "official_task_success",
                "probe_count",
                "post_prediction_action_count",
                "sim_action_count",
                "wall_seconds",
                "final_observation_id",
                "final_note",
                "seed_reveal",
                "salt_reveal",
                "seed_commitment_sha256",
                "commitment_verified",
                "replay_video",
                "observation",
                "last_action",
            },
            "command_error": {"status", "error_type", "message"},
        }
        if status not in expected_top_level:
            raise CapabilityViolation(
                f"Simulator returned an unrecognized response status {status!r}"
            )
        unknown_top_level = set(response) - expected_top_level[status]
        if unknown_top_level:
            raise CapabilityViolation(
                "Simulator response contains unregistered top-level field(s): "
                + ", ".join(sorted(unknown_top_level))
            )
        self._assert_registered_nested_fields(response)
        always_forbidden = {
            "actor",
            "actor_pose",
            "object_pose",
            "pad_pose",
            "reward",
            "depth",
            "intrinsics",
            "extrinsics",
            "contact_force",
        }

        def visit(value: Any, *, key: str = "") -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item, key=key)
                return
            if not isinstance(value, dict):
                return
            forbidden = always_forbidden & set(value)
            if forbidden:
                raise CapabilityViolation(
                    "Simulator response contains forbidden field(s): "
                    + ", ".join(sorted(forbidden))
                )
            if not terminal_response and {
                "true_class",
                "expected_target",
                "seed_reveal",
                "salt_reveal",
            } & set(value):
                raise CapabilityViolation("Terminal evaluator truth leaked before termination")
            if (
                "task_success" in value
                and (
                    not self.profile.expose_task_success_after_prediction
                    or self.stage != "post_prediction_control"
                )
                and not terminal_response
            ):
                raise CapabilityViolation(
                    "task_success is unavailable at "
                    f"Level {self.profile.level} in stage {self.stage}"
                )
            modalities = value.get("modalities")
            if isinstance(modalities, dict) and set(modalities) != set(
                self.profile.public_modalities
            ):
                raise CapabilityViolation(
                    "Observation modalities do not exactly match the selected Level"
                )
            robot_state = value.get("robot_state")
            if isinstance(robot_state, dict) and set(robot_state) != set(
                self.profile.public_robot_state
            ):
                raise CapabilityViolation(
                    "Observation robot state does not exactly match the selected Level"
                )
            if not self.profile.expose_tactile and "tactile_health" in value:
                raise CapabilityViolation("Tactile metadata leaked into Level 1")
            for child_key, child_value in value.items():
                visit(child_value, key=child_key)

        visit(response)

    def _assert_registered_nested_fields(self, response: JsonDict) -> None:
        """Fail closed if a future simulator field appears in a public structure."""

        allowed: dict[str, set[str]] = {
            "observation": {
                "observation_id",
                "level",
                "profile",
                "stage",
                "probe_count",
                "post_prediction_action_count",
                "modalities",
                "robot_state",
                "tactile_health",
                "artifacts",
            },
            "robot_state": set(self.profile.public_robot_state),
            "artifact": {"path", "artifact_id", "sha256"},
            "tactile_sensor_health": {
                "healthy",
                "expected_markers",
                "plausible_marker_components",
                "all_dark_components",
                "dark_pixel_count",
            },
            "feedback": {"execution_succeeded", "task_success"},
            "action": {
                "primitive",
                "delta_gripper",
                "delta_position_world_m",
                "delta_rpy_world_rad",
                "physics_steps",
            },
            "last_action": {
                "post_prediction_action_index",
                "prior_observation_id",
                "rationale",
                "action",
                "committed_target",
                "cumulative_y_m",
                "target_halfspace_locked",
                "feedback",
                "execution_duration_seconds",
                "observation",
            },
            "replay_video": {
                "path",
                "artifact_id",
                "sha256",
                "codec_name",
                "profile",
                "width",
                "height",
                "pix_fmt",
                "nb_frames",
                "error_type",
                "message",
            },
        }

        def require_keys(value: Any, kind: str) -> None:
            if not isinstance(value, dict):
                raise CapabilityViolation(f"Simulator {kind} must be an object")
            unknown = set(value) - allowed[kind]
            if unknown:
                raise CapabilityViolation(
                    f"Simulator {kind} contains unregistered field(s): "
                    + ", ".join(sorted(unknown))
                )

        def check_observation(value: Any) -> None:
            if value is None:
                return
            require_keys(value, "observation")
            assert isinstance(value, dict)
            modalities = value.get("modalities")
            if not isinstance(modalities, dict):
                raise CapabilityViolation("Simulator observation.modalities must be an object")
            for artifact in modalities.values():
                require_keys(artifact, "artifact")
            require_keys(value.get("robot_state"), "robot_state")
            tactile_health = value.get("tactile_health")
            if tactile_health is not None:
                if not isinstance(tactile_health, dict) or set(tactile_health) != {
                    "left",
                    "right",
                }:
                    raise CapabilityViolation(
                        "Simulator tactile_health must contain exactly left and right"
                    )
                for sensor in tactile_health.values():
                    require_keys(sensor, "tactile_sensor_health")
            artifacts = value.get("artifacts")
            if artifacts is not None:
                if not isinstance(artifacts, dict):
                    raise CapabilityViolation("Simulator observation.artifacts must be an object")
                for artifact in artifacts.values():
                    require_keys(artifact, "artifact")

        def check_action_record(value: Any) -> None:
            if value is None:
                return
            require_keys(value, "last_action")
            assert isinstance(value, dict)
            check_observation(value.get("observation"))
            feedback = value.get("feedback")
            action = value.get("action")
            if feedback is not None:
                require_keys(feedback, "feedback")
            if action is not None:
                require_keys(action, "action")

        check_observation(response.get("observation"))
        feedback = response.get("feedback")
        action = response.get("action")
        if feedback is not None:
            require_keys(feedback, "feedback")
        if action is not None:
            require_keys(action, "action")
        check_action_record(response.get("last_action"))
        replay = response.get("replay_video")
        if replay is not None:
            require_keys(replay, "replay_video")

    def _publicize_response(
        self,
        response: JsonDict,
    ) -> tuple[JsonDict, list[tuple[str, Path, str]]]:
        images: list[tuple[str, Path, str]] = []
        seen_paths: set[Path] = set()

        def visit(value: Any, *, key: str = "") -> Any:
            if isinstance(value, list):
                return [visit(item, key=key) for item in value]
            if not isinstance(value, dict):
                return copy.deepcopy(value)
            output: JsonDict = {}
            artifact_path: Path | None = None
            artifact_hash = value.get("sha256")
            raw_path = value.get("path")
            if isinstance(raw_path, str):
                artifact_path = Path(raw_path).resolve()
                if not artifact_path.is_relative_to(self.simulator_run_dir):
                    raise CapabilityViolation(
                        "Simulator attempted to expose an artifact outside its run directory"
                    )
                output["artifact_id"] = str(
                    artifact_path.relative_to(self.simulator_run_dir)
                )
            for child_key, child_value in value.items():
                if child_key in {"path", "run_dir"}:
                    continue
                output[child_key] = visit(child_value, key=child_key)
            if (
                key in self.profile.public_modalities
                and artifact_path is not None
                and isinstance(artifact_hash, str)
                and artifact_path not in seen_paths
            ):
                images.append((key, artifact_path, artifact_hash))
                seen_paths.add(artifact_path)
            return output

        return visit(response), images

    @staticmethod
    def _png_data_url(path: Path, expected_sha256: str) -> str:
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise CapabilityViolation(
                f"Artifact hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
            )
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CapabilityViolation(f"Agent image is not a PNG: {path.name}")
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
