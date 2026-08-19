"""Closed, task-agnostic Codex capability gateway for AgentEnv v1."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import file_sha256
from .benchmark_profiles import AnnotationCapabilities, ObservationProfile
from .benchmark_tasks import BenchmarkTaskSpec
from .capabilities import (
    CapabilityViolation,
    EmbodiedToolSpec,
    GatewayExecution,
    ToolCallLoopError,
    ToolInputError,
)
from .icl import ICLCondition, get_icl_condition


JsonDict = dict[str, Any]
SimulatorRequest = Callable[[JsonDict], JsonDict]


def _string_schema(description: str) -> JsonDict:
    return {"type": "string", "minLength": 1, "description": description}


def _object_schema(
    properties: Mapping[str, JsonDict], required: tuple[str, ...]
) -> JsonDict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": copy.deepcopy(dict(properties)),
    }


def _evidence_sources(
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    icl_condition: ICLCondition,
) -> tuple[str, ...]:
    sources = [*profile.public_modalities, "robot_state"]
    if profile.expose_camera_intrinsics:
        sources.append("camera_intrinsics")
    if profile.expose_camera_extrinsics:
        sources.append("camera_extrinsics")
    if annotations.provide_bbox:
        sources.append("object_bbox")
    if annotations.provide_mask:
        sources.append("object_mask")
    if icl_condition.fixed_demo_available:
        sources.append("expert_demo")
    return tuple(sources)


def _decision_schema(
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    icl_condition: ICLCondition,
) -> JsonDict:
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
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "finding", "implication"],
                    "properties": {
                        "source": {
                            "enum": list(
                                _evidence_sources(profile, annotations, icl_condition)
                            )
                        },
                        "finding": _string_schema("What was directly observed."),
                        "implication": _string_schema("How it affects this decision."),
                    },
                },
            },
            "alternatives_considered": {
                "type": "array",
                "minItems": 1,
                "items": _string_schema("A considered alternative."),
            },
            "uncertainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "expected_effect": _string_schema("Expected observable effect."),
            "parameter_rationale": _string_schema("Why these exact numeric values."),
            "rationale": _string_schema("Concise auditable reasoning summary."),
        },
    }


def build_benchmark_tool_registry(
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    icl_condition: ICLCondition | str = "none",
) -> dict[str, EmbodiedToolSpec]:
    if isinstance(icl_condition, str):
        icl_condition = get_icl_condition(icl_condition)
    decision = _decision_schema(profile, annotations, icl_condition)
    observation_id = _string_schema("Latest observation_id returned by a successful tool.")
    vector3 = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    tools = (
        EmbodiedToolSpec(
            name="start_episode",
            description="Start exactly one episode and return its first observation.",
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
            name="step_eef",
            description=(
                "Apply one bounded world-frame end-effector XYZ/RPY and gripper delta, "
                "then return a fresh observation. Zero deltas are allowed to let physics "
                "settle. At most 50 accepted steps are available."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "delta_position": {
                        **vector3,
                        "description": "World XYZ metres; each <=0.04, norm <=0.06.",
                    },
                    "delta_rpy": {
                        **vector3,
                        "description": "World RPY radians; each absolute value <=0.35.",
                    },
                    "delta_gripper": {
                        "type": "number",
                        "minimum": -0.005,
                        "maximum": 0.005,
                        "description": "Per-finger qpos delta in metres; abs <=0.005.",
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
            simulator_command="step",
            effect="world_mutating",
            allowed_stages=frozenset({"active"}),
        ),
        EmbodiedToolSpec(
            name="finish_episode",
            description=(
                "End the episode at the current state. The evaluator settles physics and "
                "reveals official task success only in this terminal response."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "final_note": _string_schema("Terminal summary of the attempt."),
                    "decision_record": decision,
                },
                ("observation_id", "final_note", "decision_record"),
            ),
            simulator_command="finish",
            effect="terminal",
            allowed_stages=frozenset({"active"}),
        ),
    )
    return {tool.name: tool for tool in tools}


def benchmark_capability_manifest(
    task: BenchmarkTaskSpec,
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    *,
    pre_move: bool = False,
    icl_condition: ICLCondition | str = "none",
) -> JsonDict:
    if isinstance(icl_condition, str):
        icl_condition = get_icl_condition(icl_condition)
    tools = build_benchmark_tool_registry(profile, annotations, icl_condition)
    payload: JsonDict = {
        "schema_version": "univtac.embodied_codex_capabilities.v1",
        "task": task.to_manifest(pre_move=pre_move),
        "observation_profile": profile.to_manifest(),
        "annotations": annotations.to_manifest(),
        "icl": icl_condition.to_manifest(),
        "tools": [tool.to_manifest() for tool in tools.values()],
        "forbidden_agent_capabilities": [
            "direct IK, joint targets, or trajectory-planner access",
            "raw simulator instance IDs, labels, or USD prim paths",
            "task-success feedback before finish_episode",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


class BenchmarkCapabilityGateway:
    """Validate the three-tool surface and scrub all host artifact paths."""

    MAX_CONSECUTIVE_REJECTED_CALLS = 6

    def __init__(
        self,
        *,
        task: BenchmarkTaskSpec,
        profile: ObservationProfile,
        annotations: AnnotationCapabilities,
        simulator_request: SimulatorRequest,
        simulator_run_dir: Path,
        icl_condition: ICLCondition | str = "none",
        agent_workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.profile = profile
        self.annotations = annotations
        if isinstance(icl_condition, str):
            icl_condition = get_icl_condition(icl_condition)
        self.icl_condition = icl_condition
        self.registry = build_benchmark_tool_registry(
            profile, annotations, icl_condition
        )
        self._simulator_request = simulator_request
        self.simulator_run_dir = simulator_run_dir.resolve()
        self.agent_workspace = (
            agent_workspace.resolve() if agent_workspace is not None else None
        )
        self.stage = "ready"
        self.latest_observation_id: str | None = None
        self.terminal = False
        self._consecutive_rejected_calls = 0
        self._lock = threading.Lock()

    def dynamic_tools(self) -> list[JsonDict]:
        return [tool.to_dynamic_tool() for tool in self.registry.values()]

    def execute(self, tool_name: str, arguments: Any) -> GatewayExecution:
        with self._lock:
            if tool_name not in self.registry:
                raise CapabilityViolation(f"Unregistered embodied tool: {tool_name!r}")
            spec = self.registry[tool_name]
            if not isinstance(arguments, dict):
                return self._rejected(tool_name, "Tool arguments must be an object")
            try:
                command, decision = self._build_command(spec, arguments)
            except ToolInputError as exc:
                rejected_decision = None
                if spec.requires_decision_record and "decision_record" in arguments:
                    try:
                        rejected_decision = self._validate_decision(arguments["decision_record"])
                    except ToolInputError:
                        pass
                return self._rejected(
                    tool_name,
                    str(exc),
                    decision_record=rejected_decision,
                    public_observation_id=self.latest_observation_id,
                )

            prior = self.latest_observation_id
            raw = self._simulator_request(command)
            if not isinstance(raw, dict):
                raise RuntimeError("Simulator returned a non-object response")
            self._assert_no_leak(raw)
            current_artifacts = self._publish_current_artifacts(raw)
            public, images = self._publicize(
                raw,
                current_artifacts=current_artifacts,
            )
            self._update_state(raw)
            success = raw.get("status") != "command_error"
            if success:
                self._consecutive_rejected_calls = 0
            else:
                self._record_rejected_call()
            content: list[JsonDict] = [
                {"type": "inputText", "text": json.dumps(public, ensure_ascii=False)}
            ]
            for label, path, sha256 in images:
                content.extend(
                    [
                        {"type": "inputText", "text": f"The next image is {label}."},
                        {
                            "type": "inputImage",
                            "imageUrl": self._png_data_url(path, sha256),
                        },
                    ]
                )
            return GatewayExecution(
                tool=tool_name,
                success=success,
                execution_target="simulator",
                simulator_command=command,
                backend_request=command,
                raw_response=raw,
                public_response=public,
                content_items=tuple(content),
                decision_record=decision,
                prior_observation_id=prior,
                next_observation_id=self.latest_observation_id,
            )

    def _build_command(
        self, spec: EmbodiedToolSpec, arguments: JsonDict
    ) -> tuple[JsonDict, JsonDict | None]:
        required = set(spec.input_schema["required"])
        allowed = set(spec.input_schema["properties"])
        missing, unknown = required - set(arguments), set(arguments) - allowed
        if missing:
            raise ToolInputError("Missing tool field(s): " + ", ".join(sorted(missing)))
        if unknown:
            raise ToolInputError("Unknown tool field(s): " + ", ".join(sorted(unknown)))
        if self.stage not in spec.allowed_stages:
            raise ToolInputError(f"{spec.name} is unavailable during stage {self.stage!r}")
        decision = None
        command: JsonDict = {"command": spec.simulator_command}
        if spec.name == "start_episode":
            command["agent_note"] = self._nonempty(arguments["agent_note"], "agent_note")
            return command, decision

        observation_id = self._nonempty(arguments["observation_id"], "observation_id")
        if observation_id != self.latest_observation_id:
            raise ToolInputError(
                f"observation_id must be the latest id {self.latest_observation_id!r}"
            )
        decision = self._validate_decision(arguments["decision_record"])
        command.update(
            {
                "observation_id": observation_id,
                "rationale": (
                    f"{decision['rationale']} Exact-parameter basis: "
                    f"{decision['parameter_rationale']}"
                ),
            }
        )
        if spec.name == "step_eef":
            position = self._vector3(arguments["delta_position"], "delta_position")
            rotation = self._vector3(arguments["delta_rpy"], "delta_rpy")
            gripper = self._finite(arguments["delta_gripper"], "delta_gripper")
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
        elif spec.name == "finish_episode":
            command["final_note"] = self._nonempty(arguments["final_note"], "final_note")
        return command, decision

    def _validate_decision(self, value: Any) -> JsonDict:
        if not isinstance(value, dict):
            raise ToolInputError("decision_record must be an object")
        required = {
            "evidence",
            "alternatives_considered",
            "uncertainty",
            "expected_effect",
            "parameter_rationale",
            "rationale",
        }
        if set(value) != required:
            raise ToolInputError("decision_record fields do not match the published schema")
        evidence = value["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ToolInputError("decision_record.evidence must be a non-empty array")
        allowed_sources = set(
            _evidence_sources(
                self.profile,
                self.annotations,
                self.icl_condition,
            )
        )
        normalized_evidence: list[JsonDict] = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"source", "finding", "implication"}:
                raise ToolInputError("Each evidence item must match the published schema")
            source = item["source"]
            if source not in allowed_sources:
                raise ToolInputError(f"Evidence source {source!r} is unavailable")
            normalized_evidence.append(
                {
                    "source": source,
                    "finding": self._nonempty(item["finding"], "evidence.finding"),
                    "implication": self._nonempty(item["implication"], "evidence.implication"),
                }
            )
        alternatives = value["alternatives_considered"]
        if not isinstance(alternatives, list) or not alternatives:
            raise ToolInputError("alternatives_considered must be a non-empty array")
        uncertainty = self._finite(value["uncertainty"], "uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ToolInputError("uncertainty must be in [0, 1]")
        return {
            "evidence": normalized_evidence,
            "alternatives_considered": [
                self._nonempty(item, "alternatives_considered item")
                for item in alternatives
            ],
            "uncertainty": uncertainty,
            "expected_effect": self._nonempty(value["expected_effect"], "expected_effect"),
            "parameter_rationale": self._nonempty(
                value["parameter_rationale"], "parameter_rationale"
            ),
            "rationale": self._nonempty(value["rationale"], "rationale"),
        }

    def _assert_no_leak(self, response: JsonDict) -> None:
        terminal = response.get("status") == "rollout_finished"
        forbidden_always = {
            "idToLabels",
            "instance_id",
            "prim_path",
            "usd_path",
            "actor_pose",
            "object_pose",
            "reward",
        }

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            leaked = forbidden_always & set(value)
            if leaked:
                raise CapabilityViolation(
                    "Simulator response leaked forbidden field(s): " + ", ".join(sorted(leaked))
                )
            if not terminal and {"task_success", "official_task_success"} & set(value):
                raise CapabilityViolation("Task success leaked before finish_episode")
            for child in value.values():
                visit(child)

        visit(response)

    def _publish_current_artifacts(self, response: JsonDict) -> dict[Path, str]:
        """Expose only current metric arrays/raw masks in the Agent workspace."""

        if self.agent_workspace is None:
            return {}
        observation = response.get("observation")
        if not isinstance(observation, dict):
            return {}
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise CapabilityViolation("Current artifact publication requires observation_id")

        selected: list[tuple[Path, str, str]] = []

        def collect(value: Any, *, key: str = "artifact") -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item, key=key)
                return
            if not isinstance(value, dict):
                return
            raw_path = value.get("path")
            digest = value.get("sha256")
            if (
                isinstance(raw_path, str)
                and isinstance(digest, str)
                and (
                    value.get("media_type") == "application/x-npy"
                    or (
                        key == "mask"
                        and value.get("media_type") == "image/png"
                        and value.get("content_image") is False
                    )
                )
            ):
                source = Path(raw_path).resolve()
                if not source.is_relative_to(self.simulator_run_dir) or not source.is_file():
                    raise CapabilityViolation(
                        "Current artifact escaped the simulator run directory"
                    )
                if file_sha256(source) != digest:
                    raise CapabilityViolation("Current artifact hash mismatch")
                relative = source.relative_to(self.simulator_run_dir)
                expected_prefix = Path("observations") / observation_id
                try:
                    payload_relative = relative.relative_to(expected_prefix)
                except ValueError as exc:
                    raise CapabilityViolation(
                        "Current artifact does not belong to the returned observation"
                    ) from exc
                if not payload_relative.parts or ".." in payload_relative.parts:
                    raise CapabilityViolation("Current artifact path is unsafe")
                selected.append((source, payload_relative.as_posix(), digest))
            for child_key, child in value.items():
                collect(child, key=child_key)

        collect(observation)
        inputs_root = self.agent_workspace / "benchmark_inputs"
        current_root = inputs_root / "current_observation"
        inputs_root.mkdir(parents=True, exist_ok=True)
        if not selected:
            if current_root.exists():
                shutil.rmtree(current_root)
            return {}

        temporary = Path(
            tempfile.mkdtemp(prefix=".current_observation-", dir=inputs_root)
        ).resolve()
        published: dict[Path, str] = {}
        files: list[dict[str, str]] = []
        try:
            for source, relative, digest in selected:
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if file_sha256(destination) != digest:
                    raise CapabilityViolation("Current artifact changed while copying")
                workspace_relative = (
                    Path("benchmark_inputs") / "current_observation" / relative
                ).as_posix()
                published[source] = workspace_relative
                files.append({"path": relative, "sha256": digest})
            manifest = {
                "schema_version": "univtac.current_observation_artifacts.v1",
                "observation_id": observation_id,
                "retention": "current_observation_only",
                "files": files,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if current_root.exists():
                shutil.rmtree(current_root)
            temporary.replace(current_root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return published

    def _publicize(
        self,
        response: JsonDict,
        *,
        current_artifacts: Mapping[Path, str] | None = None,
    ) -> tuple[JsonDict, list[tuple[str, Path, str]]]:
        images: list[tuple[str, Path, str]] = []
        seen: set[Path] = set()
        current_artifacts = current_artifacts or {}

        def visit(value: Any, *, key: str = "artifact") -> Any:
            if isinstance(value, list):
                return [visit(item, key=key) for item in value]
            if not isinstance(value, dict):
                return copy.deepcopy(value)
            output: JsonDict = {}
            raw_path = value.get("path")
            path: Path | None = None
            if isinstance(raw_path, str):
                path = Path(raw_path).resolve()
                if not path.is_relative_to(self.simulator_run_dir):
                    raise CapabilityViolation("Artifact path escaped the simulator run directory")
                output["artifact_id"] = str(path.relative_to(self.simulator_run_dir))
                if path in current_artifacts:
                    output["workspace_path"] = current_artifacts[path]
            for child_key, child_value in value.items():
                if child_key in {"path", "run_dir", "content_image"}:
                    continue
                output[child_key] = visit(child_value, key=child_key)
            sha256 = value.get("sha256")
            if (
                path is not None
                and value.get("media_type") == "image/png"
                and value.get("content_image") is True
                and isinstance(sha256, str)
                and path not in seen
            ):
                images.append((key, path, sha256))
                seen.add(path)
            return output

        return visit(response), images

    def _update_state(self, response: JsonDict) -> None:
        observation = response.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("observation_id"), str):
            self.latest_observation_id = observation["observation_id"]
        status = response.get("status")
        if status == "rollout_started":
            self.stage = "active"
        elif status == "rollout_finished":
            self.stage = "terminal"
            self.terminal = True

    def _rejected(
        self,
        tool_name: str,
        message: str,
        *,
        decision_record: JsonDict | None = None,
        public_observation_id: str | None = None,
    ) -> GatewayExecution:
        self._record_rejected_call()
        public: JsonDict = {
            "status": "tool_rejected",
            "tool": tool_name,
            "message": "Tool call rejected by the environment contract.",
        }
        if public_observation_id is not None:
            public["observation_id"] = public_observation_id
        return GatewayExecution(
            tool=tool_name,
            success=False,
            execution_target="rejected",
            simulator_command=None,
            backend_request=None,
            raw_response={
                "status": "tool_rejected",
                "tool": tool_name,
                "message": message,
                "stage": self.stage,
                "latest_observation_id": self.latest_observation_id,
            },
            public_response=public,
            content_items=(
                {"type": "inputText", "text": json.dumps(public, ensure_ascii=False)},
            ),
            decision_record=decision_record,
            prior_observation_id=self.latest_observation_id,
            next_observation_id=self.latest_observation_id,
        )

    def _record_rejected_call(self) -> None:
        self._consecutive_rejected_calls += 1
        if self._consecutive_rejected_calls > self.MAX_CONSECUTIVE_REJECTED_CALLS:
            raise ToolCallLoopError("Consecutive rejected embodied-tool call limit exceeded")

    @staticmethod
    def _nonempty(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ToolInputError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ToolInputError(f"{name} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolInputError(f"{name} must be a finite number") from exc
        if not math.isfinite(number):
            raise ToolInputError(f"{name} must be a finite number")
        return number

    @classmethod
    def _vector3(cls, value: Any, name: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ToolInputError(f"{name} must contain exactly three numbers")
        return [cls._finite(item, f"{name} item") for item in value]

    @staticmethod
    def _png_data_url(path: Path, expected_sha256: str) -> str:
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise CapabilityViolation("Artifact hash mismatch")
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CapabilityViolation("Agent image is not a PNG")
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
