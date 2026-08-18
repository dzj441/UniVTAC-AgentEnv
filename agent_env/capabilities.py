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

from .perception_profiles import PerceptionProfile, get_perception_profile
from .perception_runtime import (
    PerceptionRuntime,
    PerceptionServiceError,
)
from .profiles import AgentEnvProfile, get_profile
from .visibility import (
    AgentVisibilityError,
    assert_agent_visible_response,
    project_simulator_response,
)


JsonDict = dict[str, Any]
SimulatorRequest = Callable[[JsonDict], JsonDict]
PUBLIC_RGB_WIDTH = 480
PUBLIC_RGB_HEIGHT = 270


class CapabilityViolation(RuntimeError):
    """An attempt to invoke something outside the registered tool surface."""


class ToolInputError(ValueError):
    """A malformed or stage-invalid call to an otherwise registered tool."""

    def __init__(self, message: str, *, public_message: str | None = None) -> None:
        super().__init__(message)
        self.public_message = (
            public_message
            or "Tool arguments do not satisfy the published input schema."
        )


class ToolCallLoopError(RuntimeError):
    """Raised when an agent repeats rejected calls without making progress."""


@dataclass(frozen=True)
class EmbodiedToolSpec:
    """One agent-visible atomic capability."""

    name: str
    description: str
    input_schema: JsonDict
    simulator_command: str | None
    effect: str
    allowed_stages: frozenset[str]
    requires_decision_record: bool = True
    execution_target: str = "simulator"

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
            "execution_target": self.execution_target,
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
    execution_target: str
    simulator_command: JsonDict | None
    backend_request: JsonDict | None
    raw_response: JsonDict | None
    public_response: JsonDict
    content_items: tuple[JsonDict, ...]
    decision_record: JsonDict | None
    prior_observation_id: str | None
    next_observation_id: str | None


def _string_schema(description: str) -> JsonDict:
    return {"type": "string", "minLength": 1, "description": description}


def _decision_schema(
    profile: AgentEnvProfile,
    perception_profile: PerceptionProfile,
) -> JsonDict:
    sources = [
        *profile.public_modalities,
        "robot_state",
        *perception_profile.evidence_sources,
    ]
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


def build_tool_registry(
    level: int | str,
    perception_profile: str | PerceptionProfile = "none",
) -> dict[str, EmbodiedToolSpec]:
    """Build the immutable, level-specific agent tool surface."""

    profile = get_profile(level)
    semantic = get_perception_profile(perception_profile)
    decision = _decision_schema(profile, semantic)
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
                "Call exactly once before every other embodied tool; starts the task "
                "and returns the first observation."
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
                "Optional only after start_episode and before commit_classification. "
                "Apply a gripper-only delta in metres and return a new observation; "
                "at most two probes may be accepted."
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
                "After start_episode and any optional probes, irreversibly submit the "
                "surface classification and selected pad. This is required before "
                "act_delta_ee, wait_physics, or finish_episode."
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
                "Available only after commit_classification. Apply world-frame "
                "end-effector XYZ/RPY and gripper deltas, then return a new observation. "
                "Together, act_delta_ee and wait_physics may be accepted at most 20 times."
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
                "Available only after commit_classification. Advance physics without "
                "robot motion and return a new observation. Together, wait_physics and "
                "act_delta_ee may be accepted at most 20 times."
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
            description=(
                "Available only after commit_classification. End the episode at its "
                "current state."
            ),
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
    )
    perception_tools: list[EmbodiedToolSpec] = []
    if semantic.expose_sam3:
        perception_tools.append(
            EmbodiedToolSpec(
                name="sam3_segment",
                description=(
                    "Run SAM3 on the latest public head/wrist RGB. Text mode uses a "
                    "concise open-vocabulary phrase; points mode uses 1-64 foreground/"
                    "background pixels in the original 480x270 image. Returns masks, "
                    "boxes, scores, and a candidate contact sheet without changing the world. "
                    f"At most {semantic.max_sam3_calls_per_observation} calls are allowed "
                    "for each observation."
                ),
                input_schema=_object_schema(
                    {
                        "observation_id": observation_id,
                        "camera": {"enum": ["head_rgb", "wrist_rgb"]},
                        "mode": {"enum": ["text", "points"]},
                        "prompt": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "description": "Required in text mode; concise English visual phrase.",
                        },
                        "points": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["x", "y", "label"],
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "exclusiveMaximum": 480},
                                    "y": {"type": "number", "minimum": 0, "exclusiveMaximum": 270},
                                    "label": {"enum": [0, 1]},
                                },
                            },
                            "description": "Required in points mode; label 1 foreground, 0 background.",
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Text-mode threshold; defaults to 0.5.",
                        },
                        "decision_record": decision,
                    },
                    ("observation_id", "camera", "mode", "decision_record"),
                ),
                simulator_command=None,
                effect="read_only",
                allowed_stages=frozenset({"classification", "post_prediction_control"}),
                execution_target="perception",
            )
        )
    if semantic.expose_unidepth_v2:
        perception_tools.append(
            EmbodiedToolSpec(
                name="estimate_metric_depth",
                description=(
                    "Estimate a monocular metric-depth prior for the latest public head/"
                    "wrist RGB with UniDepth V2. Returns a depth/confidence visualization, "
                    "global statistics, and optional 3x3-median samples. This is predicted "
                    "depth, never simulator ground truth or final collision evidence. "
                    f"At most {semantic.max_unidepth_v2_calls_per_observation} calls are "
                    "allowed for each observation."
                ),
                input_schema=_object_schema(
                    {
                        "observation_id": observation_id,
                        "camera": {"enum": ["head_rgb", "wrist_rgb"]},
                        "resolution_level": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 9,
                            "description": "Higher is more image resolution/compute; default 4.",
                        },
                        "sample_points": {
                            "type": "array",
                            "maxItems": 16,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "exclusiveMaximum": 480},
                                    "y": {"type": "number", "minimum": 0, "exclusiveMaximum": 270},
                                },
                            },
                            "description": "Optional original-image pixels to sample numerically.",
                        },
                        "decision_record": decision,
                    },
                    ("observation_id", "camera", "decision_record"),
                ),
                simulator_command=None,
                effect="read_only",
                allowed_stages=frozenset({"classification", "post_prediction_control"}),
                execution_target="perception",
            )
        )
    return {tool.name: tool for tool in (*tools, *perception_tools)}


def capability_manifest(
    level: int | str,
    perception_profile: str | PerceptionProfile = "none",
) -> JsonDict:
    profile = get_profile(level)
    semantic = get_perception_profile(perception_profile)
    registry = build_tool_registry(level, semantic)
    payload: JsonDict = {
        "schema_version": "univtac.codex_capabilities.v2",
        "level": profile.level,
        "profile": profile.to_manifest(),
        "perception_profile": semantic.to_manifest(),
        "tools": [tool.to_manifest() for tool in registry.values()],
        "forbidden_agent_capabilities": [
            "shell or arbitrary code execution",
            "filesystem reads or writes",
            "web, apps, plugins, skills, or subagents",
            "direct IK or joint-target commands",
            "trajectory-planner access",
            "simulator internals, object/pad poses, reward, or checker state",
            "raw simulator camera depth, intrinsics, or extrinsics",
        ],
        "host_enforcement": [
            "only registered dynamic tools are relayed",
            "tool input is validated again outside the model",
            "every world-changing call must cite the latest observation",
            "simulator protocol independently revalidates bounds and stages",
            "model-visible artifacts contain ids and hashes, never host paths",
            "semantic tools resolve only the latest public RGB by observation id",
            "derived depth uses host calibration but never exposes calibration or raw depth",
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


def _validate_pixel_points(
    value: Any,
    *,
    labels: bool,
    maximum: int,
) -> list[JsonDict]:
    """Normalize agent-supplied pixels in the fixed public 480x270 RGB frame."""

    if not isinstance(value, list):
        raise ToolInputError("pixel points must be an array")
    if labels and not value:
        raise ToolInputError("SAM3 points mode requires at least one point")
    if len(value) > maximum:
        raise ToolInputError(f"pixel points exceed the maximum of {maximum}")
    normalized: list[JsonDict] = []
    expected = {"x", "y", "label"} if labels else {"x", "y"}
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != expected:
            raise ToolInputError(f"pixel point {index} has invalid fields")
        x = _finite_number(item["x"], f"pixel point {index}.x")
        y = _finite_number(item["y"], f"pixel point {index}.y")
        if not 0.0 <= x < 480.0 or not 0.0 <= y < 270.0:
            raise ToolInputError(f"pixel point {index} is outside the 480x270 image")
        point: JsonDict = {"x": x, "y": y}
        if labels:
            label = item["label"]
            if isinstance(label, bool) or label not in {0, 1}:
                raise ToolInputError(f"pixel point {index}.label must be 0 or 1")
            point["label"] = int(label)
        normalized.append(point)
    return normalized


def _validate_decision_record(
    value: Any,
    profile: AgentEnvProfile,
    perception_profile: PerceptionProfile,
) -> JsonDict:
    if not isinstance(value, dict):
        raise ToolInputError(
            "decision_record must be an object",
            public_message="decision_record must be an object.",
        )
    expected = {
        "evidence",
        "alternatives_considered",
        "uncertainty",
        "expected_effect",
        "parameter_rationale",
        "rationale",
    }
    if set(value) != expected:
        fields = ", ".join(sorted(expected))
        raise ToolInputError(
            "decision_record fields must be exactly: " + fields,
            public_message="decision_record fields must be exactly: " + fields + ".",
        )
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ToolInputError(
            "decision_record.evidence must be a non-empty array",
            public_message="decision_record.evidence must contain at least one item.",
        )
    allowed_sources = {
        *profile.public_modalities,
        "robot_state",
        *perception_profile.evidence_sources,
    }
    normalized_evidence: list[JsonDict] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {
            "source",
            "finding",
            "implication",
        }:
            raise ToolInputError(
                f"decision_record.evidence[{index}] has invalid fields",
                public_message=(
                    f"decision_record.evidence[{index}] must contain source, finding, "
                    "and implication."
                ),
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
            "decision_record.alternatives_considered must be a non-empty array",
            public_message=(
                "decision_record.alternatives_considered must contain at least one item."
            ),
        )
    uncertainty = _finite_number(value["uncertainty"], "decision_record.uncertainty")
    if not 0.0 <= uncertainty <= 1.0:
        raise ToolInputError(
            "decision_record.uncertainty must be in [0, 1]",
            public_message="decision_record.uncertainty must be in [0, 1].",
        )
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

    MAX_CONSECUTIVE_REJECTED_CALLS = 6

    def __init__(
        self,
        *,
        level: int | str,
        simulator_request: SimulatorRequest,
        simulator_run_dir: Path,
        perception_profile: str | PerceptionProfile = "none",
        perception_runtime: PerceptionRuntime | None = None,
    ) -> None:
        self.profile = get_profile(level)
        self.perception_profile = get_perception_profile(perception_profile)
        self.registry = build_tool_registry(level, self.perception_profile)
        self._simulator_request = simulator_request
        self.simulator_run_dir = simulator_run_dir.resolve()
        self._perception_runtime = perception_runtime
        if (
            self.perception_profile.name != "none"
            and self._perception_runtime is None
        ):
            raise ValueError("Enabled perception profile requires a PerceptionRuntime")
        self.stage = "ready"
        self.latest_observation_id: str | None = None
        self.terminal = False
        self._latest_rgb: dict[str, tuple[Path, str]] = {}
        self._available_semantic_evidence: set[tuple[str, str]] = set()
        self._perception_call_counts: dict[tuple[str, str], int] = {}
        self._consecutive_rejected_calls = 0
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
            if spec.execution_target == "perception":
                return self._execute_perception(spec, arguments)
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
                            arguments["decision_record"],
                            self.profile,
                            self.perception_profile,
                        )
                    except ToolInputError:
                        pass
                return self._rejected(
                    tool_name,
                    str(exc),
                    decision_record=rejected_decision,
                    public_message=exc.public_message,
                    public_observation_id=self.latest_observation_id,
                )

            prior_observation_id = self.latest_observation_id
            raw_response = self._simulator_request(command)
            if not isinstance(raw_response, dict):
                raise RuntimeError("Simulator returned a non-object response")
            self._assert_no_capability_leak(raw_response)
            self._remember_latest_rgb(raw_response)
            public_response, images = self._publicize_response(raw_response)
            self._update_state(raw_response)
            success = raw_response.get("status") != "command_error"
            if success:
                self._consecutive_rejected_calls = 0
            else:
                self._record_rejected_call()
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
                        "text": f"The next image is modality={modality!r}.",
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
                execution_target="simulator",
                simulator_command=command,
                backend_request=command,
                raw_response=raw_response,
                public_response=public_response,
                content_items=tuple(content_items),
                decision_record=decision,
                prior_observation_id=prior_observation_id,
                next_observation_id=self.latest_observation_id,
            )

    def _execute_perception(
        self,
        spec: EmbodiedToolSpec,
        arguments: JsonDict,
    ) -> GatewayExecution:
        try:
            request, decision = self._build_perception_request(spec, arguments)
        except ToolInputError as exc:
            rejected_decision: JsonDict | None = None
            if "decision_record" in arguments:
                try:
                    rejected_decision = _validate_decision_record(
                        arguments["decision_record"],
                        self.profile,
                        self.perception_profile,
                    )
                except ToolInputError:
                    pass
            return self._rejected(
                spec.name,
                str(exc),
                decision_record=rejected_decision,
            )

        observation_id = request["observation_id"]
        camera = request["camera"]
        budget = self._perception_budget(spec.name)
        budget_key = (observation_id, spec.name)
        used = self._perception_call_counts.get(budget_key, 0)
        if used >= budget:
            return self._rejected(
                spec.name,
                f"Per-observation call budget exhausted ({budget}) for {observation_id}",
                decision_record=decision,
            )
        # Accepted attempts consume budget even when the optional backend fails.
        # This prevents an unavailable model from becoming an unbounded retry loop.
        call_index = used + 1
        self._perception_call_counts[budget_key] = call_index
        remaining_calls = budget - call_index
        image = self._latest_rgb.get(camera)
        if image is None:
            return self._rejected(
                spec.name,
                f"Latest observation has no registered {camera!r} RGB artifact",
                decision_record=decision,
            )
        assert self._perception_runtime is not None
        image_path, image_sha256 = image
        try:
            if spec.name == "sam3_segment":
                result = self._perception_runtime.segment_sam3(
                    observation_id=observation_id,
                    camera=camera,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    mode=request["mode"],
                    prompt=request.get("prompt"),
                    points=request.get("points"),
                    confidence_threshold=request["confidence_threshold"],
                )
            elif spec.name == "estimate_metric_depth":
                result = self._perception_runtime.estimate_unidepth_v2(
                    observation_id=observation_id,
                    camera=camera,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    intrinsics=self._load_host_intrinsics(observation_id, camera),
                    resolution_level=request["resolution_level"],
                    sample_points=request["sample_points"],
                )
            else:  # pragma: no cover - registry and dispatch are co-located.
                raise CapabilityViolation(f"Unknown perception tool {spec.name!r}")
        except Exception as exc:
            # Perception is read-only and optional. Fail this call closed instead
            # of terminating the embodied rollout, and do not expose exception text
            # because it can contain host paths or transport details.
            host_error = {
                "status": "semantic_perception_error",
                "tool": spec.name,
                "observation_id": observation_id,
                "camera": camera,
                "error_type": (
                    type(exc).__name__
                    if isinstance(exc, PerceptionServiceError)
                    else "PerceptionHostError"
                ),
                "message": "Configured perception service failed or returned invalid data.",
                "call_index_for_observation": call_index,
                "remaining_calls_for_observation": remaining_calls,
            }
            public = {
                "status": "semantic_perception_error",
                "tool": spec.name,
                "observation_id": observation_id,
                "message": "Configured perception service failed.",
            }
            return GatewayExecution(
                tool=spec.name,
                success=False,
                execution_target="perception",
                simulator_command=None,
                backend_request={
                    key: value
                    for key, value in request.items()
                    if key != "decision_record"
                },
                raw_response=host_error,
                public_response=public,
                content_items=(
                    {"type": "inputText", "text": json.dumps(public)},
                ),
                decision_record=decision,
                prior_observation_id=self.latest_observation_id,
                next_observation_id=self.latest_observation_id,
            )

        host_result = {
            **result.public_response,
            "call_index_for_observation": call_index,
            "remaining_calls_for_observation": remaining_calls,
        }
        public_result = copy.deepcopy(result.public_response)
        content_items: list[JsonDict] = [
            {
                "type": "inputText",
                "text": json.dumps(public_result, ensure_ascii=False),
            }
        ]
        for label, path, sha256 in result.content_images:
            content_items.extend(
                [
                    {
                        "type": "inputText",
                        "text": f"The next image is derived_artifact={label!r}, sha256={sha256}.",
                    },
                    {
                        "type": "inputImage",
                        "imageUrl": self._png_data_url(path, sha256),
                    },
                ]
            )
        if result.success:
            source = (
                "sam3_result"
                if spec.name == "sam3_segment"
                else "unidepth_v2_result"
            )
            self._available_semantic_evidence.add((observation_id, source))
        return GatewayExecution(
            tool=spec.name,
            success=result.success,
            execution_target="perception",
            simulator_command=None,
            backend_request=result.backend_request,
            raw_response=host_result,
            public_response=public_result,
            content_items=tuple(content_items),
            decision_record=decision,
            prior_observation_id=self.latest_observation_id,
            next_observation_id=self.latest_observation_id,
        )

    def _perception_budget(self, tool_name: str) -> int:
        if tool_name == "sam3_segment":
            return self.perception_profile.max_sam3_calls_per_observation
        if tool_name == "estimate_metric_depth":
            return self.perception_profile.max_unidepth_v2_calls_per_observation
        raise CapabilityViolation(f"Unknown perception tool {tool_name!r}")

    def _build_perception_request(
        self,
        spec: EmbodiedToolSpec,
        arguments: JsonDict,
    ) -> tuple[JsonDict, JsonDict]:
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
            raise ToolInputError(f"{spec.name} is unavailable during stage {self.stage!r}")
        observation_id = _nonempty_string(arguments["observation_id"], "observation_id")
        if observation_id != self.latest_observation_id:
            raise ToolInputError(
                f"observation_id must be the latest id {self.latest_observation_id!r}"
            )
        camera = _nonempty_string(arguments["camera"], "camera")
        if camera not in {"head_rgb", "wrist_rgb"}:
            raise ToolInputError("camera must be exactly 'head_rgb' or 'wrist_rgb'")
        decision = _validate_decision_record(
            arguments.get("decision_record"),
            self.profile,
            self.perception_profile,
        )
        self._validate_semantic_evidence_availability(decision, observation_id)
        request: JsonDict = {
            "observation_id": observation_id,
            "camera": camera,
        }
        if spec.name == "sam3_segment":
            mode = _nonempty_string(arguments["mode"], "mode").lower()
            if mode not in {"text", "points"}:
                raise ToolInputError("mode must be exactly 'text' or 'points'")
            request["mode"] = mode
            if mode == "text":
                if "points" in arguments:
                    raise ToolInputError("points is unavailable in SAM3 text mode")
                prompt = _nonempty_string(arguments.get("prompt"), "prompt")
                if len(prompt) > 256:
                    raise ToolInputError("prompt exceeds 256 characters")
                threshold = _finite_number(
                    arguments.get("confidence_threshold", 0.5),
                    "confidence_threshold",
                )
                if not 0.0 <= threshold <= 1.0:
                    raise ToolInputError("confidence_threshold must be in [0, 1]")
                request.update(
                    {"prompt": prompt, "confidence_threshold": threshold}
                )
            else:
                if "prompt" in arguments or "confidence_threshold" in arguments:
                    raise ToolInputError(
                        "prompt/confidence_threshold are unavailable in SAM3 points mode"
                    )
                request["points"] = _validate_pixel_points(
                    arguments.get("points"), labels=True, maximum=64
                )
                request["confidence_threshold"] = 0.5
        elif spec.name == "estimate_metric_depth":
            level = arguments.get("resolution_level", 4)
            if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < 10:
                raise ToolInputError("resolution_level must be an integer in [0, 9]")
            request["resolution_level"] = level
            request["sample_points"] = _validate_pixel_points(
                arguments.get("sample_points", []), labels=False, maximum=16
            )
        return request, decision

    def _validate_semantic_evidence_availability(
        self,
        decision: JsonDict,
        observation_id: str,
    ) -> None:
        for evidence in decision["evidence"]:
            source = evidence["source"]
            if source in {"sam3_result", "unidepth_v2_result"} and (
                observation_id,
                source,
            ) not in self._available_semantic_evidence:
                raise ToolInputError(
                    f"Evidence source {source!r} has not been produced for {observation_id}"
                )

    def _remember_latest_rgb(self, response: JsonDict) -> None:
        observation = response.get("observation")
        if not isinstance(observation, dict):
            return
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str):
            return
        modalities = observation.get("modalities")
        if not isinstance(modalities, dict):
            return
        remembered: dict[str, tuple[Path, str]] = {}
        for camera in ("head_rgb", "wrist_rgb"):
            artifact = modalities.get(camera)
            if not isinstance(artifact, dict):
                raise CapabilityViolation(f"Observation omitted required RGB {camera}")
            raw_path, sha256 = artifact.get("path"), artifact.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(sha256, str):
                raise CapabilityViolation(f"Observation RGB {camera} has invalid artifact metadata")
            path = Path(raw_path).resolve()
            if not path.is_relative_to(self.simulator_run_dir):
                raise CapabilityViolation("RGB artifact path is outside the simulator run directory")
            remembered[camera] = (path, sha256)
        self._latest_rgb = remembered

    def _load_host_intrinsics(self, observation_id: str, camera: str) -> JsonDict:
        if not observation_id.startswith("obs_") or not observation_id[4:].isdigit():
            raise PerceptionServiceError("invalid observation id for calibration lookup")
        path = (
            self.simulator_run_dir
            / ".host_sensor_metadata"
            / f"{observation_id}.json"
        )
        if not path.is_file():
            raise PerceptionServiceError("host camera calibration sidecar is unavailable")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PerceptionServiceError("host camera calibration sidecar is invalid") from exc
        if not isinstance(payload, dict) or payload.get("observation_id") != observation_id:
            raise PerceptionServiceError("host camera calibration provenance mismatch")
        cameras = payload.get("cameras")
        value = cameras.get(camera) if isinstance(cameras, dict) else None
        if not isinstance(value, dict):
            raise PerceptionServiceError("host camera calibration is missing")
        width, height = value.get("width"), value.get("height")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or width != PUBLIC_RGB_WIDTH
            or height != PUBLIC_RGB_HEIGHT
        ):
            raise PerceptionServiceError("host camera dimensions do not match public RGB")
        intrinsics = value.get("intrinsics") if isinstance(value, dict) else None
        if not isinstance(intrinsics, dict) or set(intrinsics) != {"fx", "fy", "cx", "cy"}:
            raise PerceptionServiceError("host camera intrinsics are malformed")
        result: JsonDict = {}
        for key in ("fx", "fy", "cx", "cy"):
            try:
                number = _finite_number(intrinsics[key], f"intrinsics.{key}")
            except ToolInputError as exc:
                raise PerceptionServiceError("host camera intrinsics are malformed") from exc
            if key in {"fx", "fy"} and number <= 0:
                raise PerceptionServiceError("host focal length is invalid")
            if key == "cx" and not 0 <= number < width:
                raise PerceptionServiceError("host principal point is invalid")
            if key == "cy" and not 0 <= number < height:
                raise PerceptionServiceError("host principal point is invalid")
            result[key] = number
        return result

    def _rejected(
        self,
        tool_name: str,
        message: str,
        *,
        decision_record: JsonDict | None = None,
        public_message: str | None = None,
        public_observation_id: str | None = None,
    ) -> GatewayExecution:
        self._record_rejected_call()
        host_response = {
            "status": "tool_rejected",
            "tool": tool_name,
            "message": message,
            "stage": self.stage,
            "latest_observation_id": self.latest_observation_id,
        }
        response = {
            "status": "tool_rejected",
            "tool": tool_name,
            "message": (
                public_message
                or "Tool call rejected by the environment contract."
            ),
        }
        if public_observation_id is not None:
            response["observation_id"] = public_observation_id
        return GatewayExecution(
            tool=tool_name,
            success=False,
            execution_target="rejected",
            simulator_command=None,
            backend_request=None,
            raw_response=host_response,
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

    def _record_rejected_call(self) -> None:
        """Abort an unproductive retry loop without exposing a live counter."""

        self._consecutive_rejected_calls += 1
        if (
            self._consecutive_rejected_calls
            > self.MAX_CONSECUTIVE_REJECTED_CALLS
        ):
            raise ToolCallLoopError(
                "Consecutive rejected embodied-tool call limit exceeded"
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
                f"{spec.name} is unavailable during stage {self.stage!r}",
                public_message=(
                    "This tool is unavailable at the current API lifecycle; follow "
                    "the lifecycle preconditions in its description."
                ),
            )

        decision: JsonDict | None = None
        if spec.requires_decision_record:
            decision = _validate_decision_record(
                arguments.get("decision_record"),
                self.profile,
                self.perception_profile,
            )
        command: JsonDict = {"command": spec.simulator_command}
        if spec.name == "start_episode":
            command["agent_note"] = _nonempty_string(
                arguments["agent_note"], "agent_note"
            )
        else:
            observation_id = _nonempty_string(
                arguments["observation_id"], "observation_id"
            )
            if observation_id != self.latest_observation_id:
                raise ToolInputError(
                    f"observation_id must be the latest id {self.latest_observation_id!r}",
                    public_message=(
                        "Use the exact observation_id repeated in this rejection or "
                        "returned by the latest successful embodied tool."
                    ),
                )
            assert decision is not None
            self._validate_semantic_evidence_availability(decision, observation_id)
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
                "observation_id",
                "rationale",
                "success_seen_before_prediction",
                "stage",
                "task_success_feedback",
                "remaining_post_prediction_actions",
            },
            "action_complete": {
                "status",
                "post_prediction_action_index",
                "prior_observation_id",
                "rationale",
                "action",
                "committed_target",
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
        try:
            projected = project_simulator_response(response, self.profile)
        except AgentVisibilityError as exc:
            raise CapabilityViolation(str(exc)) from exc
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

        public = visit(projected)
        try:
            assert_agent_visible_response(public, self.profile)
        except AgentVisibilityError as exc:
            raise CapabilityViolation(str(exc)) from exc
        return public, images

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
