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
from .benchmark_protocol import BenchmarkEpisodeProtocol
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


def build_benchmark_tool_registry(
    profile: ObservationProfile,
    annotations: AnnotationCapabilities,
    icl_condition: ICLCondition | str = "none",
) -> dict[str, EmbodiedToolSpec]:
    if isinstance(icl_condition, str):
        icl_condition = get_icl_condition(icl_condition)
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
            input_schema=_object_schema({}, ()),
            simulator_command="start",
            effect="world_mutating",
            allowed_stages=frozenset({"ready"}),
            requires_decision_record=False,
        ),
        EmbodiedToolSpec(
            name="step_eef",
            description=(
                "Apply one finite world-frame end-effector XYZ/RPY and gripper delta, "
                "then return a fresh observation. XYZ/RPY deltas have no benchmark "
                "magnitude limit. The resulting per-finger gripper qpos must remain "
                "within its physical range [0, 0.039] metres. Zero deltas are allowed "
                "to let physics settle. At most 50 accepted steps are available."
            ),
            input_schema=_object_schema(
                {
                    "observation_id": observation_id,
                    "delta_position": {
                        **vector3,
                        "description": "Finite world-frame XYZ delta in metres.",
                    },
                    "delta_rpy": {
                        **vector3,
                        "description": "Finite world-frame RPY delta in radians.",
                    },
                    "delta_gripper": {
                        "type": "number",
                        "minimum": -BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M,
                        "maximum": BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M,
                        "description": (
                            "Per-finger qpos delta in metres. The resulting target qpos "
                            "must remain in the physical range [0, 0.039]."
                        ),
                    },
                },
                (
                    "observation_id",
                    "delta_position",
                    "delta_rpy",
                    "delta_gripper",
                ),
            ),
            simulator_command="step",
            effect="world_mutating",
            allowed_stages=frozenset({"active"}),
            requires_decision_record=False,
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
                },
                ("observation_id",),
            ),
            simulator_command="finish",
            effect="terminal",
            allowed_stages=frozenset({"active"}),
            requires_decision_record=False,
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
        self._latest_gripper_qpos_m: float | None = None
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
                return self._rejected(
                    tool_name,
                    str(exc),
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
                        {
                            "type": "inputText",
                            "text": (
                                "The next image corresponds to public JSON field "
                                f"{label}."
                            ),
                        },
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
            return command, decision

        observation_id = self._nonempty(arguments["observation_id"], "observation_id")
        if observation_id != self.latest_observation_id:
            raise ToolInputError(
                f"observation_id must be the latest id {self.latest_observation_id!r}"
            )
        command["observation_id"] = observation_id
        if spec.name == "step_eef":
            position = self._vector3(arguments["delta_position"], "delta_position")
            rotation = self._vector3(arguments["delta_rpy"], "delta_rpy")
            gripper = self._finite(arguments["delta_gripper"], "delta_gripper")
            if abs(gripper) > BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M:
                raise ToolInputError("delta_gripper exceeds the physical opening range")
            if gripper != 0.0:
                if self._latest_gripper_qpos_m is None:
                    raise ToolInputError(
                        "latest gripper qpos is unavailable for target validation"
                    )
                target_qpos = self._latest_gripper_qpos_m + gripper
                tolerance = 1e-6
                if not (
                    BenchmarkEpisodeProtocol.GRIPPER_MIN_QPOS_M - tolerance
                    <= target_qpos
                    <= BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M + tolerance
                ):
                    raise ToolInputError(
                        "delta_gripper would exceed the physical target qpos range"
                    )
            command.update(
                {
                    "delta_position": position,
                    "delta_rpy": rotation,
                    "delta_gripper": gripper,
                }
            )
        return command, decision

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
            if not terminal and "evaluator_checks" in value:
                raise CapabilityViolation("Evaluator checks leaked before finish_episode")
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

        def visit(
            value: Any,
            *,
            public_path: tuple[str, ...] = (),
        ) -> Any:
            if isinstance(value, list):
                return [
                    visit(item, public_path=(*public_path, str(index)))
                    for index, item in enumerate(value)
                ]
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
                output[child_key] = visit(
                    child_value,
                    public_path=(*public_path, child_key),
                )
            sha256 = value.get("sha256")
            if (
                path is not None
                and value.get("media_type") == "image/png"
                and value.get("content_image") is True
                and isinstance(sha256, str)
                and path not in seen
            ):
                label = ".".join(public_path) or "artifact"
                images.append((label, path, sha256))
                seen.add(path)
            return output

        # The simulator's terminal response doubles as the evaluator record, so it
        # intentionally contains detailed checker diagnostics and the commitment
        # opening.  Those fields remain available through ``raw_response`` and
        # evaluator-private files but are not part of the Agent contract: only the
        # official terminal success bit is public.
        public_source = response
        if response.get("status") == "rollout_finished":
            evaluator_private_fields = {
                "evaluator_checks",
                "salt_reveal",
                "seed_reveal",
            }
            public_source = {
                key: value
                for key, value in response.items()
                if key not in evaluator_private_fields
            }
        return visit(public_source), images

    def _update_state(self, response: JsonDict) -> None:
        observation = response.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("observation_id"), str):
            self.latest_observation_id = observation["observation_id"]
            robot_state = observation.get("robot_state")
            if isinstance(robot_state, dict):
                joint_position = robot_state.get("joint_position_9d")
                if isinstance(joint_position, list) and len(joint_position) == 9:
                    qpos = joint_position[-2]
                    if isinstance(qpos, (int, float)) and math.isfinite(float(qpos)):
                        self._latest_gripper_qpos_m = float(qpos)
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
