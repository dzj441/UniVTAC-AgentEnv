"""Read-only HTTP backend for replaying AgentEnv/Codex interaction records.

The viewer intentionally reads the existing append-only artifacts instead of
introducing another logging format.  It can therefore open completed runs and
refresh an in-progress run while its JSONL streams are still growing.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


TOOL_LABELS = {
    "start_episode": "启动 episode",
    "probe_gripper": "触觉/夹爪探测",
    "commit_classification": "锁定分类与目标",
    "act_delta_ee": "执行末端增量动作",
    "step_eef": "执行末端增量动作",
    "wait_physics": "等待物理稳定",
    "finish_episode": "提交终局评测",
    "inspect_episode_status": "读取公开状态",
    "sam3_segment": "SAM 3 语义分割",
    "estimate_metric_depth": "UniDepth V2 预测深度",
    "start": "启动 episode",
    "probe": "夹爪探测",
    "submit_prediction": "锁定分类与目标",
    "act": "执行末端增量动作",
    "wait": "等待物理稳定",
    "finish": "提交终局评测",
    "status": "读取公开状态",
}

TRANSCRIPT_TOOL_NAMES = {
    "start": "start_episode",
    "probe": "probe_gripper",
    "submit_prediction": "commit_classification",
    "act": "act_delta_ee",
    "step": "step_eef",
    "wait": "wait_physics",
    "finish": "finish_episode",
    "status": "inspect_episode_status",
}

STATIC_FILENAMES = frozenset({"index.html", "app.js", "styles.css", "favicon.svg"})


class ViewerDataError(RuntimeError):
    """A malformed or unsafe viewer data request."""


class RunNotFound(ViewerDataError):
    """The requested run does not exist below the configured root."""


@dataclass(frozen=True)
class RunFiles:
    """Canonical paths for one discovered run."""

    run_id: str
    directory: Path
    kind: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all complete JSONL records, tolerating a live partial final line."""

    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _iso_elapsed(timestamp: Any, origin: Any) -> float:
    if not isinstance(timestamp, str) or not isinstance(origin, str):
        return 0.0

    def parse(value: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)

    try:
        return max(0.0, (parse(timestamp) - parse(origin)).total_seconds())
    except ValueError:
        return 0.0


def _profile_from(manifest: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    for source in (manifest, fallback):
        for key in ("observation_profile", "profile"):
            profile = source.get(key)
            if isinstance(profile, dict):
                return profile
    return {}


def _task_name(value: Any, default: str = "grasp_classify") -> str:
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"]
    if isinstance(value, str) and value:
        return value
    return default


def _viewable_artifact(run_dir: Path, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    artifact = _relative_artifact(
        run_dir,
        value.get("artifact_id") or value.get("path"),
    )
    if artifact is None:
        return None
    return {
        "artifact": artifact,
        "sha256": value.get("sha256"),
        "media_type": value.get("media_type"),
    }


def _relative_artifact(run_dir: Path, value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (run_dir / raw).resolve()
    try:
        relative = candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _normalize_observation(value: Any, run_dir: Path) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("observation_id"), str):
        return None
    # Protocol acknowledgements such as commit_classification also carry the
    # current observation_id and stage, but they are not sensor observations.
    # Requiring the two stable observation payload fields prevents those thin
    # status envelopes from replacing the complete frame indexed by that id.
    raw_modalities = value.get("modalities")
    raw_robot_state = value.get("robot_state")
    if not isinstance(raw_modalities, dict) or not isinstance(raw_robot_state, dict):
        return None
    modalities: dict[str, dict[str, Any]] = {}
    for name, metadata in raw_modalities.items():
        if not isinstance(metadata, dict):
            continue
        viewable = _viewable_artifact(run_dir, metadata)
        if viewable is None and isinstance(metadata.get("visualization"), dict):
            viewable = _viewable_artifact(run_dir, metadata["visualization"])
            if viewable is not None:
                viewable["visualization_range_m"] = metadata.get(
                    "visualization_range_m"
                )
                viewable["statistics"] = metadata.get("statistics")
        if viewable is not None:
            modalities[str(name)] = viewable

    raw_annotations = value.get("annotations")
    if isinstance(raw_annotations, dict):
        for camera, roles in raw_annotations.items():
            if not isinstance(roles, dict):
                continue
            for role, annotation in roles.items():
                if not isinstance(annotation, dict):
                    continue
                for feature, suffix in (("bbox_overlay", "bbox"), ("mask", "mask")):
                    viewable = _viewable_artifact(run_dir, annotation.get(feature))
                    if viewable is not None:
                        modalities[f"{camera}_{role}_{suffix}"] = viewable

    composite: str | None = None
    artifacts = value.get("artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("composite"), dict):
        metadata = artifacts["composite"]
        composite = _relative_artifact(
            run_dir,
            metadata.get("artifact_id") or metadata.get("path"),
        )

    normalized = {
        "observation_id": value["observation_id"],
        "level": value.get("level"),
        "profile": value.get("observation_profile") or value.get("profile"),
        "task": value.get("task"),
        "stage": value.get("stage"),
        "probe_count": value.get("probe_count"),
        "post_prediction_action_count": value.get("post_prediction_action_count"),
        "modalities": modalities,
        "composite": composite,
        "robot_state": raw_robot_state,
        "camera_calibration": value.get("camera_calibration")
        if isinstance(value.get("camera_calibration"), dict)
        else {},
        "annotations": raw_annotations if isinstance(raw_annotations, dict) else {},
        "tactile_health": value.get("tactile_health")
        if isinstance(value.get("tactile_health"), dict)
        else {},
    }
    if "task_success" in value:
        normalized["task_success"] = value.get("task_success")
    return normalized


def _observation_completeness(value: dict[str, Any]) -> tuple[int, ...]:
    """Rank duplicate records so a partial frame cannot replace a rich one."""

    modalities = value.get("modalities")
    robot_state = value.get("robot_state")
    tactile_health = value.get("tactile_health")
    return (
        len(modalities) if isinstance(modalities, dict) else 0,
        len(robot_state) if isinstance(robot_state, dict) else 0,
        int(bool(value.get("composite"))),
        len(tactile_health) if isinstance(tactile_health, dict) else 0,
        int("task_success" in value),
        sum(
            value.get(key) is not None
            for key in ("level", "profile", "stage", "probe_count", "post_prediction_action_count")
        ),
    )


def _collect_observations(value: Any, run_dir: Path, output: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, list):
        for child in value:
            _collect_observations(child, run_dir, output)
        return
    if not isinstance(value, dict):
        return
    observation = _normalize_observation(value, run_dir)
    if observation is not None:
        observation_id = observation["observation_id"]
        existing = output.get(observation_id)
        if existing is None or _observation_completeness(
            observation
        ) > _observation_completeness(existing):
            output[observation_id] = observation
    for child in value.values():
        _collect_observations(child, run_dir, output)


def _collect_derived_artifacts(value: Any, run_dir: Path) -> list[dict[str, str]]:
    """Collect model-derived viewable images without treating them as observations."""

    found: list[dict[str, str]] = []

    def visit(child: Any, label: str = "derived") -> None:
        if isinstance(child, list):
            for index, item in enumerate(child):
                visit(item, f"{label}_{index}")
            return
        if not isinstance(child, dict):
            return
        raw = child.get("artifact_id") or child.get("path")
        relative = _relative_artifact(run_dir, raw)
        if relative and Path(relative).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            found.append({"label": label, "artifact": relative})
        for key, item in child.items():
            if key not in {"artifact_id", "path", "modalities", "observation"}:
                visit(item, str(key))

    visit(value)
    deduplicated: list[dict[str, str]] = []
    seen: set[str] = set()
    preferred = sorted(
        found,
        key=lambda item: (
            0
            if item["label"] in {"candidate_contact_sheet", "visualization"}
            else 1,
            item["label"],
        ),
    )
    for item in preferred:
        if item["artifact"] not in seen:
            seen.add(item["artifact"])
            deduplicated.append(item)
    return deduplicated[:8]


def _normalize_message(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("text")
    if value is None:
        value = record.get("summary")
    if value is None:
        value = record.get("content")
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, dict):
        parts = [json.dumps(value, ensure_ascii=False, indent=2)]
    elif value is None:
        parts = []
    else:
        text = str(value).strip()
        parts = [text] if text else []
    if not parts:
        return None
    return {
        "kind": str(record.get("kind") or "message"),
        "parts": parts,
        "elapsed_seconds": _as_float(record.get("elapsed_seconds")),
        "timestamp_utc": record.get("timestamp_utc"),
    }


def _compact_environment_response(value: Any) -> Any:
    """Remove duplicated images and local absolute paths from a response."""

    if isinstance(value, list):
        return [_compact_environment_response(child) for child in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, child in value.items():
        if key in {"observation", "run_dir", "replay_video"}:
            continue
        if key == "path" and isinstance(child, str) and Path(child).is_absolute():
            continue
        output[key] = _compact_environment_response(child)
    return output


def _numeric_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < length:
        return None
    result: list[float] = []
    for item in value[:length]:
        number = _as_float(item, float("nan"))
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def _state_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    if not before or not after or before.get("observation_id") == after.get("observation_id"):
        return {}
    before_state = before.get("robot_state")
    after_state = after.get("robot_state")
    if not isinstance(before_state, dict) or not isinstance(after_state, dict):
        return {}
    result: dict[str, Any] = {}
    before_gripper = before_state.get("gripper_width_m", before_state.get("gripper_qpos"))
    after_gripper = after_state.get("gripper_width_m", after_state.get("gripper_qpos"))
    if isinstance(before_gripper, (int, float)) and isinstance(after_gripper, (int, float)):
        delta = float(after_gripper) - float(before_gripper)
        result["gripper_delta_m"] = delta
        result["gripper_delta_mm"] = delta * 1000.0
    before_pose = _numeric_vector(
        before_state.get(
            "end_effector_pose_robot_base_wxyz_7d",
            before_state.get("end_effector_pose_robot_base_7d"),
        ),
        3,
    )
    after_pose = _numeric_vector(
        after_state.get(
            "end_effector_pose_robot_base_wxyz_7d",
            after_state.get("end_effector_pose_robot_base_7d"),
        ),
        3,
    )
    if before_pose is not None and after_pose is not None:
        delta_xyz = [after_pose[i] - before_pose[i] for i in range(3)]
        result["end_effector_delta_xyz_m"] = delta_xyz
        result["end_effector_translation_m"] = math.sqrt(
            sum(component * component for component in delta_xyz)
        )
    before_joint_value = before_state.get("joint_position_9d", before_state.get("joint_position_8d"))
    after_joint_value = after_state.get("joint_position_9d", after_state.get("joint_position_8d"))
    joint_length = 9 if isinstance(before_joint_value, list) and len(before_joint_value) >= 9 else 8
    before_joints = _numeric_vector(before_joint_value, joint_length)
    after_joints = _numeric_vector(after_joint_value, joint_length)
    if before_joints is not None and after_joints is not None:
        result["max_abs_joint_delta"] = max(
            abs(after_joints[i] - before_joints[i]) for i in range(joint_length)
        )
    return result


def _decision_for_call(
    call: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    call_id = str(call.get("call_id") or "")
    validated = decisions.get(call_id, {}).get("decision_record")
    if isinstance(validated, dict):
        return validated, "host_validated"
    arguments = call.get("arguments")
    submitted = arguments.get("decision_record") if isinstance(arguments, dict) else None
    if isinstance(submitted, dict):
        return submitted, "submitted_unvalidated"
    return None, None


def _arguments_without_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: child for key, child in value.items() if key != "decision_record"}


def _build_codex_steps(
    run_dir: Path,
    calls: list[dict[str, Any]],
    decision_records: list[dict[str, Any]],
    message_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    observations: dict[str, dict[str, Any]] = {}
    for call in calls:
        _collect_observations(
            call.get("environment_response", call.get("simulator_response")),
            run_dir,
            observations,
        )
    decisions = {
        str(record.get("call_id")): record
        for record in decision_records
        if record.get("call_id") is not None
    }
    messages = [message for record in message_records if (message := _normalize_message(record))]
    messages.sort(key=lambda item: item["elapsed_seconds"])
    calls = sorted(calls, key=lambda item: _as_float(item.get("elapsed_seconds")))

    steps: list[dict[str, Any]] = []
    message_cursor = 0
    for index, call in enumerate(calls):
        call_elapsed = _as_float(call.get("elapsed_seconds"))
        step_messages: list[dict[str, Any]] = []
        while (
            message_cursor < len(messages)
            and messages[message_cursor]["elapsed_seconds"] <= call_elapsed + 1e-6
        ):
            step_messages.append(messages[message_cursor])
            message_cursor += 1
        prior_id = call.get("prior_observation_id")
        next_id = call.get("next_observation_id")
        before = observations.get(str(prior_id)) if prior_id is not None else None
        after = observations.get(str(next_id)) if next_id is not None else None
        decision, decision_source = _decision_for_call(call, decisions)
        response = call.get("environment_response", call.get("simulator_response"))
        tool = str(call.get("tool") or "unknown")
        steps.append(
            {
                "index": index,
                "sequence": call.get("sequence", index),
                "call_id": call.get("call_id"),
                "tool": tool,
                "tool_label": TOOL_LABELS.get(tool, tool),
                "success": call.get("success") is True,
                "elapsed_seconds": call_elapsed,
                "duration_seconds": call.get("duration_seconds"),
                "timestamp_utc": call.get("timestamp_utc"),
                "prior_observation_id": prior_id,
                "next_observation_id": next_id,
                "input_observation": before,
                "output_observation": after,
                "fresh_observation": bool(after and next_id != prior_id),
                "state_delta": _state_delta(before, after),
                "messages": step_messages,
                "decision": decision,
                "decision_source": decision_source,
                "arguments": _arguments_without_decision(call.get("arguments")),
                "execution_target": call.get("execution_target")
                or ("simulator" if call.get("simulator_command") is not None else "rejected"),
                "backend_request": _compact_environment_response(
                    call.get("backend_request", call.get("simulator_command"))
                ),
                "simulator_command": _compact_environment_response(
                    call.get("simulator_command")
                ),
                "environment_response": _compact_environment_response(response),
                "derived_artifacts": _collect_derived_artifacts(response, run_dir),
            }
        )
    return steps, messages[message_cursor:], observations


def _build_capture_steps(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    observations: dict[str, dict[str, Any]] = {}
    for record in records:
        _collect_observations(record, run_dir, observations)
    origin = next(
        (record.get("timestamp_utc") for record in records if record.get("timestamp_utc")),
        None,
    )
    steps: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if record.get("kind") != "command" or not isinstance(record.get("command"), dict):
            continue
        command = record["command"]
        raw_tool = str(command.get("command") or "unknown")
        if raw_tool == "close":
            continue
        response: dict[str, Any] | None = None
        response_timestamp: Any = None
        for candidate in records[record_index + 1 :]:
            if candidate.get("kind") == "command":
                break
            if candidate.get("kind") == "response" and isinstance(
                candidate.get("response"), dict
            ):
                response = candidate["response"]
                response_timestamp = candidate.get("timestamp_utc")
                break
        response = response or {}
        tool = TRANSCRIPT_TOOL_NAMES.get(raw_tool, raw_tool)
        prior_id = command.get("observation_id")
        response_observation = _normalize_observation(response.get("observation"), run_dir)
        next_id = (
            response_observation.get("observation_id")
            if response_observation is not None
            else response.get("final_observation_id") or prior_id
        )
        before = observations.get(str(prior_id)) if prior_id is not None else None
        after = observations.get(str(next_id)) if next_id is not None else None
        status = str(response.get("status") or "")
        elapsed = _iso_elapsed(response_timestamp or record.get("timestamp_utc"), origin)
        steps.append(
            {
                "index": len(steps),
                "sequence": len(steps),
                "call_id": None,
                "tool": tool,
                "tool_label": TOOL_LABELS.get(tool, tool),
                "success": "error" not in status.lower(),
                "elapsed_seconds": elapsed,
                "duration_seconds": response.get("execution_duration_seconds"),
                "timestamp_utc": response_timestamp or record.get("timestamp_utc"),
                "prior_observation_id": prior_id,
                "next_observation_id": next_id,
                "input_observation": before,
                "output_observation": after,
                "fresh_observation": bool(after and next_id != prior_id),
                "state_delta": _state_delta(before, after),
                "messages": [],
                "decision": None,
                "decision_source": None,
                "unstructured_rationale": command.get("rationale"),
                "arguments": {
                    key: value
                    for key, value in command.items()
                    if key not in {"command", "rationale"}
                },
                "execution_target": "simulator",
                "backend_request": _compact_environment_response(command),
                "simulator_command": _compact_environment_response(command),
                "environment_response": _compact_environment_response(response),
                "derived_artifacts": [],
            }
        )
    return steps, observations


def _selected_outcome(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "timestamp_utc",
        "terminal_reason",
        "task",
        "start_condition",
        "pre_move_enabled",
        "observation_profile",
        "profile_index",
        "annotations",
        "level",
        "profile",
        "predicted_class",
        "true_class",
        "classification_correct",
        "committed_target",
        "expected_target",
        "committed_pad_correct",
        "official_task_success",
        "evaluator_checks",
        "step_eef_count",
        "probe_count",
        "post_prediction_action_count",
        "sim_action_count",
        "wall_seconds",
        "final_observation_id",
        "final_note",
        "commitment_verified",
    )
    return {key: value.get(key) for key in keys if key in value}


class RunRepository:
    """Discover and normalize run artifacts below one fixed root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ViewerDataError(f"Run root is not a directory: {self.root}")

    def _within_root(self, candidate: Path) -> Path:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ViewerDataError("Requested path leaves the configured run root") from exc
        return resolved

    def discover(self) -> list[RunFiles]:
        candidates: dict[Path, str] = {}
        for marker in self.root.rglob("codex_tool_calls.jsonl"):
            directory = self._within_root(marker.parent)
            candidates[directory] = "codex"
        for marker in self.root.rglob("agent_transcript.jsonl"):
            directory = self._within_root(marker.parent)
            candidates.setdefault(directory, "capture")
        runs = [
            RunFiles(
                run_id=directory.relative_to(self.root).as_posix(),
                directory=directory,
                kind=kind,
            )
            for directory, kind in candidates.items()
        ]
        return sorted(runs, key=lambda run: (run.directory.stat().st_mtime, run.run_id), reverse=True)

    def resolve_run(self, run_id: str) -> RunFiles:
        if not run_id or "\x00" in run_id or Path(run_id).is_absolute():
            raise RunNotFound("Invalid run id")
        directory = self._within_root(self.root / run_id)
        if not directory.is_dir():
            raise RunNotFound(f"Run not found: {run_id}")
        canonical_id = directory.relative_to(self.root).as_posix()
        if canonical_id != Path(run_id).as_posix():
            raise RunNotFound("Run id is not canonical")
        kind = "codex" if (directory / "codex_tool_calls.jsonl").is_file() else "capture"
        if not (directory / "agent_transcript.jsonl").is_file() and kind != "codex":
            raise RunNotFound(f"Not an AgentEnv run: {run_id}")
        return RunFiles(canonical_id, directory, kind)

    def resolve_artifact(self, run_id: str, artifact: str) -> Path:
        run = self.resolve_run(run_id)
        if not artifact or "\x00" in artifact or Path(artifact).is_absolute():
            raise ViewerDataError("Invalid artifact path")
        path = (run.directory / artifact).resolve()
        try:
            path.relative_to(run.directory)
        except ValueError as exc:
            raise ViewerDataError("Artifact path leaves the selected run") from exc
        if not path.is_file():
            raise RunNotFound(f"Artifact not found: {artifact}")
        canonical = path.relative_to(run.directory).as_posix()
        if canonical != Path(artifact).as_posix():
            raise ViewerDataError("Artifact path is not canonical")
        if canonical not in self._public_artifact_ids(run):
            raise ViewerDataError("Artifact is not part of the public run record")
        return path

    def _public_artifact_ids(self, run: RunFiles) -> set[str]:
        """Allow only artifacts explicitly exposed by the normalized viewer API.

        A run can also contain host-only sidecars (for example camera intrinsics),
        simulator audit files, and logs.  Path confinement prevents traversal but
        is not sufficient to keep those files private, so the artifact endpoint
        uses exactly the files referenced by the UI payload as an allowlist.
        """

        detail = self.detail(run.run_id)
        allowed: set[str] = set()

        def add(value: Any) -> None:
            if isinstance(value, str) and value:
                allowed.add(Path(value).as_posix())

        for item in detail.get("artifacts", []):
            if isinstance(item, dict):
                add(item.get("path"))
        for step in detail.get("steps", []):
            if not isinstance(step, dict):
                continue
            for key in ("input_observation", "output_observation"):
                observation = step.get(key)
                if not isinstance(observation, dict):
                    continue
                modalities = observation.get("modalities")
                if isinstance(modalities, dict):
                    for metadata in modalities.values():
                        if isinstance(metadata, dict):
                            add(metadata.get("artifact"))
                add(observation.get("composite"))
            derived = step.get("derived_artifacts")
            if isinstance(derived, list):
                for item in derived:
                    if isinstance(item, dict):
                        add(item.get("artifact"))
        return allowed

    def summary(self, run: RunFiles) -> dict[str, Any]:
        codex_manifest = _read_json(run.directory / "codex_run_manifest.json")
        env_manifest = _read_json(run.directory / "manifest.json")
        codex_outcome = _read_json(run.directory / "codex_run_outcome.json")
        evaluator = _read_json(run.directory / "evaluator_outcome.json")
        profile = _profile_from(codex_manifest, env_manifest)
        level = (
            codex_manifest.get("level")
            or profile.get("index")
            or profile.get("level")
            or evaluator.get("profile_index")
            or evaluator.get("level")
            or codex_outcome.get("profile_index")
            or codex_outcome.get("level")
        )
        created = (
            codex_manifest.get("created_utc")
            or env_manifest.get("created_utc")
            or evaluator.get("timestamp_utc")
        )
        runtime = codex_outcome.get("codex_runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        token_usage = runtime.get("token_usage")
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        total_usage = token_usage.get("total")
        total_usage = total_usage if isinstance(total_usage, dict) else {}
        tool_calls = _read_jsonl(run.directory / "codex_tool_calls.jsonl")
        messages = _read_jsonl(run.directory / "codex_messages.jsonl")
        if run.kind == "capture":
            tool_count = sum(
                record.get("kind") == "command"
                and record.get("command", {}).get("command") != "close"
                for record in _read_jsonl(run.directory / "agent_transcript.jsonl")
                if isinstance(record.get("command", {}), dict)
            )
        else:
            tool_count = len(tool_calls)
        observation_count = len(
            [path for path in (run.directory / "observations").glob("obs_*") if path.is_dir()]
        ) if (run.directory / "observations").is_dir() else 0
        status = codex_outcome.get("status")
        if not status:
            status = "completed_capture" if evaluator else "in_progress"
        parent = Path(run.run_id).parent.as_posix()
        task_manifest = codex_manifest.get("task") or env_manifest.get("task")
        task_manifest = task_manifest if isinstance(task_manifest, dict) else {}
        return {
            "id": run.run_id,
            "name": run.directory.name,
            "group": "" if parent == "." else parent,
            "kind": run.kind,
            "task": _task_name(
                codex_manifest.get("task") or env_manifest.get("task")
            ),
            "start_condition": (
                codex_manifest.get("start_condition")
                or env_manifest.get("start_condition")
                or task_manifest.get("start_condition")
                or evaluator.get("start_condition")
            ),
            "pre_move_enabled": (
                codex_manifest.get("pre_move_enabled")
                if "pre_move_enabled" in codex_manifest
                else env_manifest.get("pre_move_enabled")
                if "pre_move_enabled" in env_manifest
                else task_manifest.get("pre_move_enabled")
            ),
            "level": level,
            "profile": (
                profile.get("name")
                or evaluator.get("observation_profile")
                or evaluator.get("profile")
            ),
            "profile_description": profile.get("description"),
            "modalities": profile.get("public_modalities") or [],
            "created_utc": created,
            "status": status,
            "valid_for_scoring": codex_outcome.get("valid_for_scoring"),
            "model": runtime.get("actual_model") or codex_manifest.get("model"),
            "effort": runtime.get("reasoning_effort") or codex_manifest.get("effort"),
            "total_tokens": total_usage.get("totalTokens"),
            "wall_seconds": codex_outcome.get("total_wall_seconds") or evaluator.get("wall_seconds"),
            "tool_count": tool_count,
            "message_count": len(messages),
            "observation_count": observation_count,
            "has_video": (run.directory / "agent_observations_h264.mp4").is_file(),
            "has_trace": (run.directory / "CODEX_TRACE.md").is_file(),
            "outcome": _selected_outcome(evaluator),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.summary(run) for run in self.discover()]

    def detail(self, run_id: str) -> dict[str, Any]:
        run = self.resolve_run(run_id)
        summary = self.summary(run)
        codex_manifest = _read_json(run.directory / "codex_run_manifest.json")
        env_manifest = _read_json(run.directory / "manifest.json")
        codex_outcome = _read_json(run.directory / "codex_run_outcome.json")
        evaluator = _read_json(run.directory / "evaluator_outcome.json")
        if run.kind == "codex":
            steps, tail_messages, observations = _build_codex_steps(
                run.directory,
                _read_jsonl(run.directory / "codex_tool_calls.jsonl"),
                _read_jsonl(run.directory / "codex_decisions.jsonl"),
                _read_jsonl(run.directory / "codex_messages.jsonl"),
            )
        else:
            steps, observations = _build_capture_steps(
                run.directory,
                _read_jsonl(run.directory / "agent_transcript.jsonl"),
            )
            tail_messages = []
        prompt_path = run.directory / "codex_operator_prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else None
        profile = _profile_from(codex_manifest, env_manifest)
        runtime = codex_outcome.get("codex_runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        artifacts = []
        for path, label, media_type in (
            ("agent_observations_h264.mp4", "完整 Observation 回放", "video"),
            ("CODEX_TRACE.md", "中文 Codex Trace", "text"),
            ("codex_operator_prompt.txt", "Codex task prompt", "text"),
            ("codex_base_instructions.txt", "Codex base instructions", "text"),
            (
                "codex_developer_instructions.txt",
                "Codex developer instructions",
                "text",
            ),
            ("codex_run_manifest.json", "运行 manifest", "json"),
            ("codex_run_outcome.json", "Codex 运行结果", "json"),
            ("evaluator_outcome.json", "终局评测", "json"),
        ):
            if (run.directory / path).is_file():
                artifacts.append({"path": path, "label": label, "type": media_type})
        return {
            "summary": summary,
            "profile": profile,
            "task_prompt": prompt,
            "steps": steps,
            "tail_messages": tail_messages,
            "observation_ids": sorted(observations),
            "runtime": {
                "actual_model": runtime.get("actual_model"),
                "model_provider": runtime.get("model_provider"),
                "reasoning_effort": runtime.get("reasoning_effort"),
                "reasoning_summary": runtime.get("reasoning_summary"),
                "token_usage": runtime.get("token_usage"),
                "tool_call_summary": codex_outcome.get("tool_call_summary"),
                "event_audit": codex_outcome.get("event_audit"),
            },
            "outcome": _selected_outcome(evaluator),
            "artifacts": artifacts,
            "recording_note": (
                "展示 Codex 协议公开的 reasoning summary、显式 decision record、"
                "实际工具调用和环境反馈；不包含模型未公开的隐藏思维链。"
                if run.kind == "codex"
                else "该目录是标准化环境采集，没有 Codex reasoning/decision stream。"
            ),
        }


def public_viewer_url(port: int, template: str | None = None) -> str | None:
    """Resolve the code-server proxy URL template for one selected port."""

    value = (template if template is not None else os.environ.get("VSCODE_PROXY_URI", "")).strip()
    if not value:
        return None
    if "{{port}}" in value:
        value = value.replace("{{port}}", str(port))
    elif "{port}" in value:
        value = value.replace("{port}", str(port))
    else:
        return None
    return value if value.endswith("/") else value + "/"


def _security_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.send_header("X-Frame-Options", "SAMEORIGIN")
    handler.send_header(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; media-src 'self'; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'self'",
    )


def _route_suffix(path: str, suffix: str) -> bool:
    normalized = path.rstrip("/")
    return normalized == suffix or normalized.endswith(suffix)


def make_handler(
    repository: RunRepository,
    static_root: Path,
) -> type[BaseHTTPRequestHandler]:
    static_root = static_root.resolve()

    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "UniVTACRunViewer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[viewer] {self.address_string()} {fmt % args}", flush=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch(head_only=True)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(head_only=False)

        def _dispatch(self, *, head_only: bool) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if _route_suffix(path, "/api/health"):
                    self._send_json(
                        {
                            "status": "ok",
                            "run_count": len(repository.discover()),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        },
                        head_only=head_only,
                    )
                elif _route_suffix(path, "/api/runs"):
                    self._send_json(
                        {"runs": repository.list_runs()},
                        head_only=head_only,
                    )
                elif _route_suffix(path, "/api/run"):
                    run_id = query.get("run", [""])[0]
                    self._send_json(repository.detail(run_id), head_only=head_only)
                elif _route_suffix(path, "/api/artifact"):
                    run_id = query.get("run", [""])[0]
                    artifact = query.get("path", [""])[0]
                    self._send_file(
                        repository.resolve_artifact(run_id, artifact),
                        head_only=head_only,
                        download=query.get("download", ["0"])[0] == "1",
                    )
                else:
                    filename = Path(path.rstrip("/")).name
                    if not filename or path.endswith("/"):
                        filename = "index.html"
                    if filename not in STATIC_FILENAMES:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    static_path = (static_root / filename).resolve()
                    if static_path.parent != static_root or not static_path.is_file():
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_file(static_path, head_only=head_only, download=False)
            except RunNotFound as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc), head_only=head_only)
            except ViewerDataError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc), head_only=head_only)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self.log_error("Unhandled viewer error: %r", exc)
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "The viewer could not read this run.",
                    head_only=head_only,
                )

        def _send_json(self, value: Any, *, head_only: bool) -> None:
            payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            _security_headers(self)
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def _send_error_json(
            self,
            status: HTTPStatus,
            message: str,
            *,
            head_only: bool,
        ) -> None:
            payload = json.dumps(
                {"error": message, "status": int(status)}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            _security_headers(self)
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def _send_file(self, path: Path, *, head_only: bool, download: bool) -> None:
            stat = path.stat()
            size = stat.st_size
            etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                _security_headers(self)
                self.end_headers()
                return

            start = 0
            end = max(0, size - 1)
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match or size == 0:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    _security_headers(self)
                    self.end_headers()
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                elif last:
                    suffix = int(last)
                    start = max(0, size - suffix)
                    end = size - 1
                if start >= size or end < start:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    _security_headers(self)
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

            length = 0 if size == 0 else end - start + 1
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".md":
                content_type = "text/markdown; charset=utf-8"
            elif path.suffix in {".json", ".jsonl"}:
                content_type = "application/json; charset=utf-8"
            elif path.suffix in {".txt", ".log"}:
                content_type = "text/plain; charset=utf-8"

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, max-age=60")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            _security_headers(self)
            self.end_headers()
            if head_only or length == 0:
                return
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return ViewerHandler


def create_server(
    host: str,
    port: int,
    runs_root: Path,
    *,
    static_root: Path | None = None,
) -> ThreadingHTTPServer:
    repository = RunRepository(runs_root)
    assets = static_root or Path(__file__).with_name("viewer_static")
    handler = make_handler(repository, assets)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def describe_server_urls(host: str, port: int, proxy_template: str | None = None) -> list[str]:
    urls = []
    public = public_viewer_url(port, proxy_template)
    if public:
        urls.append(f"Code-server URL: {public}")
    local_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    urls.append(f"Local URL: http://{local_host}:{port}/")
    return urls
