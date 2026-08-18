"""Append-only artifacts for auditable Codex embodied rollouts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EventRecorder:
    """Record raw protocol traffic and normalized decisions as separate chains."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.run_dir / "codex_app_server_events.jsonl"
        self.tool_path = self.run_dir / "codex_tool_calls.jsonl"
        self.decision_path = self.run_dir / "codex_decisions.jsonl"
        self.message_path = self.run_dir / "codex_messages.jsonl"
        self.violation_path = self.run_dir / "capability_violations.jsonl"
        self._sequence: dict[Path, int] = {}
        self._chain: dict[Path, str] = {}
        self._lock = threading.Lock()
        self.started_monotonic = time.monotonic()

    def record_raw(self, direction: str, message: dict[str, Any]) -> None:
        self._append(
            self.raw_path,
            {
                "direction": direction,
                "message": _redact_credentials(message),
            },
        )

    def record_tool_call(self, payload: dict[str, Any]) -> None:
        self._append(self.tool_path, payload)

    def record_decision(self, payload: dict[str, Any]) -> None:
        self._append(self.decision_path, payload)

    def record_message(self, payload: dict[str, Any]) -> None:
        self._append(self.message_path, payload)

    def record_violation(self, payload: dict[str, Any]) -> None:
        self._append(self.violation_path, payload)

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        with self._lock:
            sequence = self._sequence.get(path, 0)
            previous = self._chain.get(path, "0" * 64)
            event = {
                "sequence": sequence,
                "timestamp_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 6),
                "previous_event_sha256": previous,
                **payload,
            }
            digest = hashlib.sha256(canonical_json(event).encode("utf-8")).hexdigest()
            event["event_sha256"] = digest
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._sequence[path] = sequence + 1
            self._chain[path] = digest

    def summary(self) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        for name, path in (
            ("app_server_events", self.raw_path),
            ("tool_calls", self.tool_path),
            ("decisions", self.decision_path),
            ("messages", self.message_path),
            ("capability_violations", self.violation_path),
        ):
            if not path.exists():
                continue
            artifacts[name] = {
                "file": path.name,
                "event_count": self._sequence.get(path, 0),
                "terminal_chain_sha256": self._chain.get(path),
                "file_sha256": _file_sha256(path),
            }
        return artifacts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_credentials(value: Any, *, key: str = "") -> Any:
    sensitive_keys = {
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "apikey",
        "api_key",
        "authorization",
        "password",
        "clientsecret",
        "client_secret",
    }
    normalized = key.replace("-", "").replace("_", "").lower()
    if normalized in {item.replace("_", "") for item in sensitive_keys}:
        return "<redacted>"
    if isinstance(value, dict):
        return {
            child_key: _redact_credentials(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_credentials(item, key=key) for item in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        header, encoded = value.split(",", 1)
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            return "<invalid-data-url-redacted>"
        return (
            f"{header},<payload-omitted bytes={len(data)} "
            f"sha256={hashlib.sha256(data).hexdigest()}>"
        )
    return value


def audit_codex_events(raw_path: Path) -> dict[str, Any]:
    """Classify agent behavior that lies outside an embodied-only rollout."""

    allowed_item_types = {
        "userMessage",
        "agentMessage",
        "reasoning",
        "dynamicToolCall",
        "contextCompaction",
        "plan",
        "sleep",
    }
    forbidden_item_types = {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "imageGeneration",
        "hookPrompt",
        "enteredReviewMode",
        "exitedReviewMode",
    }
    forbidden_methods = {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
        "tool/requestUserInput",
    }
    violations: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    if not raw_path.exists():
        return {
            "passed": False,
            "violations": [{"reason": "missing_codex_event_stream"}],
            "event_counts": {},
        }
    expected_sequence = 0
    expected_previous = "0" * 64
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        event_digest = record.get("event_sha256")
        unhashed = dict(record)
        unhashed.pop("event_sha256", None)
        computed_digest = hashlib.sha256(
            canonical_json(unhashed).encode("utf-8")
        ).hexdigest()
        if record.get("sequence") != expected_sequence:
            violations.append(
                {
                    "line": line_number,
                    "reason": "event_sequence_mismatch",
                    "expected": expected_sequence,
                    "actual": record.get("sequence"),
                }
            )
        if record.get("previous_event_sha256") != expected_previous:
            violations.append(
                {
                    "line": line_number,
                    "reason": "event_chain_mismatch",
                    "expected": expected_previous,
                    "actual": record.get("previous_event_sha256"),
                }
            )
        if event_digest != computed_digest:
            violations.append(
                {
                    "line": line_number,
                    "reason": "event_hash_mismatch",
                    "expected": computed_digest,
                    "actual": event_digest,
                }
            )
        expected_sequence += 1
        expected_previous = str(event_digest)
        message = record.get("message", {})
        method = message.get("method")
        if isinstance(method, str):
            event_counts[method] = event_counts.get(method, 0) + 1
            if method in forbidden_methods:
                violations.append(
                    {"line": line_number, "reason": "forbidden_method", "method": method}
                )
            elif (
                record.get("direction") == "server_to_host"
                and "id" in message
                and method != "item/tool/call"
            ):
                violations.append(
                    {
                        "line": line_number,
                        "reason": "unknown_server_request_fail_closed",
                        "method": method,
                    }
                )
        params = message.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if isinstance(item, dict) and item.get("type") in forbidden_item_types:
            violations.append(
                {
                    "line": line_number,
                    "reason": "forbidden_item_type",
                    "item_type": item["type"],
                    "item_id": item.get("id"),
                }
            )
        elif isinstance(item, dict) and item.get("type") not in allowed_item_types:
            violations.append(
                {
                    "line": line_number,
                    "reason": "unknown_item_type_fail_closed",
                    "item_type": item.get("type"),
                    "item_id": item.get("id"),
                }
            )
    return {
        "passed": not violations,
        "violations": violations,
        "event_counts": event_counts,
        "event_count": expected_sequence,
        "terminal_chain_sha256": expected_previous,
    }


def summarize_codex_runtime(raw_path: Path) -> dict[str, Any]:
    """Extract terminal model/settings and cumulative usage from App Server events."""

    summary: dict[str, Any] = {
        "actual_model": None,
        "model_provider": None,
        "service_tier": None,
        "reasoning_effort": None,
        "reasoning_summary": None,
        "token_usage": None,
    }
    if not raw_path.is_file():
        return summary
    for record in _read_jsonl(raw_path):
        if record.get("direction") != "server_to_host":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue

        # thread/start returns the server-resolved model even when the caller
        # intentionally requested the configured default.
        result = message.get("result")
        if isinstance(result, dict):
            if isinstance(result.get("model"), str):
                summary["actual_model"] = result["model"]
            if isinstance(result.get("modelProvider"), str):
                summary["model_provider"] = result["modelProvider"]
            if result.get("serviceTier") is not None:
                summary["service_tier"] = result["serviceTier"]

        params = message.get("params")
        if not isinstance(params, dict):
            continue
        if message.get("method") == "thread/settings/updated":
            settings = params.get("threadSettings")
            if isinstance(settings, dict):
                if isinstance(settings.get("model"), str):
                    summary["actual_model"] = settings["model"]
                if isinstance(settings.get("modelProvider"), str):
                    summary["model_provider"] = settings["modelProvider"]
                if settings.get("serviceTier") is not None:
                    summary["service_tier"] = settings["serviceTier"]
                if isinstance(settings.get("effort"), str):
                    summary["reasoning_effort"] = settings["effort"]
                if isinstance(settings.get("summary"), str):
                    summary["reasoning_summary"] = settings["summary"]
        elif message.get("method") == "thread/tokenUsage/updated":
            usage = params.get("tokenUsage")
            if isinstance(usage, dict):
                summary["token_usage"] = usage
    return summary


def summarize_tool_calls(tool_path: Path) -> dict[str, int]:
    """Count relayed, host-rejected, successful, and failed tool attempts."""

    rows = _read_jsonl(tool_path)
    def target(item: dict[str, Any]) -> str:
        explicit = item.get("execution_target")
        if isinstance(explicit, str):
            return explicit
        return "simulator" if item.get("simulator_command") is not None else "rejected"

    relayed = sum(target(item) == "simulator" for item in rows)
    perception = sum(target(item) == "perception" for item in rows)
    host_rejected = sum(target(item) == "rejected" for item in rows)
    successful = sum(item.get("success") is True for item in rows)
    failed = sum(item.get("success") is not True for item in rows)
    return {
        "total": len(rows),
        "relayed_to_simulator": relayed,
        "relayed_to_perception": perception,
        "host_rejected": host_rejected,
        "successful_results": successful,
        "failed_results": failed,
    }


def build_human_trace(
    run_dir: Path,
    *,
    runtime_summary: dict[str, Any] | None = None,
) -> Path:
    """Render tool/decision/message streams into a compact Chinese review log."""

    run_dir = run_dir.resolve()
    manifest = _read_json_if_present(run_dir / "codex_run_manifest.json")
    evaluator = _read_json_if_present(run_dir / "evaluator_outcome.json")
    calls = _read_jsonl(run_dir / "codex_tool_calls.jsonl")
    decisions = {
        str(item.get("call_id")): item
        for item in _read_jsonl(run_dir / "codex_decisions.jsonl")
    }
    messages = _read_jsonl(run_dir / "codex_messages.jsonl")
    runtime = runtime_summary or {}
    token_total = runtime.get("token_usage")
    if isinstance(token_total, dict):
        token_total = token_total.get("total")
    if not isinstance(token_total, dict):
        token_total = {}
    lines = [
        "# Codex Agent Rollout 复盘",
        "",
        "> 本文件仅整理 Codex 协议公开的消息、reasoning summary 与显式 "
        "decision record；不包含、也不声称包含模型未公开的隐藏思维链。",
        "",
        "## 运行信息",
        "",
        f"- Task：`{manifest.get('task', 'grasp_classify')}`",
        f"- Level：`{manifest.get('level', 'unknown')}`",
        f"- Requested model：`{manifest.get('model', 'unknown')}`",
        f"- Actual model：`{runtime.get('actual_model') or 'unknown'}`",
        f"- Effort：`{manifest.get('effort', 'unknown')}`",
        f"- Total / cached input tokens：`{token_total.get('totalTokens', 'unknown')}` / "
        f"`{token_total.get('cachedInputTokens', 'unknown')}`",
        f"- Output / reasoning output tokens：`{token_total.get('outputTokens', 'unknown')}` / "
        f"`{token_total.get('reasoningOutputTokens', 'unknown')}`",
        f"- Capability manifest：`{manifest.get('capability_manifest_sha256', 'unknown')}`",
        "",
        "## 多轮交互时间线",
        "",
    ]
    timeline: list[tuple[float, int, str, dict[str, Any]]] = []
    for index, call in enumerate(calls):
        timeline.append((_elapsed(call), index, "tool", call))
    for index, message in enumerate(messages):
        timeline.append((_elapsed(message), index, "message", message))
    timeline.sort(key=lambda item: (item[0], 0 if item[2] == "message" else 1, item[1]))

    tool_index = 0
    message_index = 0
    for elapsed, _, event_type, event in timeline:
        if event_type == "message":
            message_index += 1
            kind = str(event.get("kind", "message"))
            label = {
                "published_reasoning_summary": "Codex 公开 reasoning summary",
                "agent_message": "Codex agent message",
                "app_server_error": "Codex app-server error",
            }.get(kind, kind)
            lines.extend(
                [
                    f"### M{message_index}. {label}（+{elapsed:g} s）",
                    "",
                    _render_message(event),
                    "",
                ]
            )
            continue

        tool_index += 1
        call = event
        call_id = str(call.get("call_id", ""))
        decision_entry = decisions.get(call_id, {})
        decision = decision_entry.get("decision_record")
        decision_source = "宿主校验后的 decision stream"
        if not isinstance(decision, dict):
            arguments = call.get("arguments")
            submitted = (
                arguments.get("decision_record")
                if isinstance(arguments, dict)
                else None
            )
            if isinstance(submitted, dict):
                decision = submitted
                decision_source = "原始 tool argument（未进入校验后的 decision stream）"
        before = call.get("prior_observation_id") or "无"
        after = call.get("next_observation_id") or "无"
        lines.extend(
            [
                f"### A{tool_index}. `{call.get('tool', 'unknown')}`（+{elapsed:g} s）",
                "",
                f"- Call ID：`{call_id or 'unknown'}`",
                f"- Observation：`{before}` → `{after}`",
                f"- Tool success：`{call.get('success')}`",
                f"- 执行耗时：`{call.get('duration_seconds', '?')} s`",
            ]
        )
        if isinstance(decision, dict):
            lines.extend(
                [
                    f"- Decision 来源：{decision_source}",
                    f"- 不确定度：`{decision.get('uncertainty')}`",
                    f"- 总体理由：{decision.get('rationale', '')}",
                    f"- 参数理由：{decision.get('parameter_rationale', '')}",
                    f"- 预期效果：{decision.get('expected_effect', '')}",
                    "",
                    "证据：",
                    "",
                ]
            )
            for evidence in decision.get("evidence", []):
                lines.append(
                    f"- `{evidence.get('source')}`：{evidence.get('finding')}；"
                    f"含义：{evidence.get('implication')}"
                )
            lines.extend(["", "考虑过的替代方案：", ""])
            for alternative in decision.get("alternatives_considered", []):
                lines.append(f"- {alternative}")
        execution_target = call.get("execution_target")
        if not isinstance(execution_target, str):
            execution_target = (
                "simulator" if call.get("simulator_command") is not None else "rejected"
            )
        if execution_target == "rejected":
            lines.extend(["", "后端调用：`未发送（宿主验证拒绝）`", ""])
        elif execution_target == "perception":
            lines.extend(["", "实际发送给宿主感知服务的请求（图像载荷已省略）：", "", "```json"])
            lines.append(
                json.dumps(call.get("backend_request"), ensure_ascii=False, indent=2)
            )
            lines.extend(["```", ""])
        else:
            lines.extend(["", "实际发送给仿真的命令：", "", "```json"])
            lines.append(
                json.dumps(call.get("simulator_command"), ensure_ascii=False, indent=2)
            )
            lines.extend(["```", ""])
        response = call.get("environment_response", call.get("simulator_response"))
        lines.extend(["返回摘要：", "", "```json"])
        lines.append(json.dumps(_compact_response(response), ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
        for modality, artifact in _observation_artifacts(response):
            lines.extend(
                [
                    f"`{modality}`：",
                    "",
                    f"![{modality}]({artifact})",
                    "",
                ]
            )

    if evaluator:
        lines.extend(
            [
                "## 终局评测",
                "",
                f"- Predicted / true：`{evaluator.get('predicted_class')}` / "
                f"`{evaluator.get('true_class')}`",
                f"- Committed / expected target：`{evaluator.get('committed_target')}` / "
                f"`{evaluator.get('expected_target')}`",
                f"- Classification correct：`{evaluator.get('classification_correct')}`",
                f"- Official task success：`{evaluator.get('official_task_success')}`",
                f"- Wall time：`{evaluator.get('wall_seconds')}` s",
                "",
            ]
        )
    output = run_dir / "CODEX_TRACE.md"
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def _elapsed(record: dict[str, Any]) -> float:
    value = record.get("elapsed_seconds")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _render_message(message: dict[str, Any]) -> str:
    value = message.get("text")
    if value is None:
        value = message.get("summary")
    if value is None:
        value = message.get("content")
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
    return str(value or "")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def _compact_response(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_response(item) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, child in value.items():
        if key == "modalities" and isinstance(child, dict):
            output[key] = {
                name: {
                    "artifact_id": artifact.get("artifact_id"),
                    "sha256": artifact.get("sha256"),
                }
                for name, artifact in child.items()
                if isinstance(artifact, dict)
            }
        elif key == "observation":
            output[key] = _compact_response(child)
        elif key in {"last_action", "feedback", "robot_state", "tactile_health"}:
            output[key] = _compact_response(child)
        elif key not in {"artifacts"}:
            output[key] = _compact_response(child)
    return output


def _observation_artifacts(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, list):
        for item in value:
            found.extend(_observation_artifacts(item))
        return found
    if not isinstance(value, dict):
        return found
    modalities = value.get("modalities")
    if isinstance(modalities, dict):
        for name, artifact in modalities.items():
            if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str):
                found.append((str(name), artifact["artifact_id"]))
    artifact_id = value.get("artifact_id")
    if isinstance(artifact_id, str):
        label = str(value.get("kind") or value.get("label") or Path(artifact_id).stem)
        found.append((label, artifact_id))
    for key, child in value.items():
        if key != "modalities":
            found.extend(_observation_artifacts(child))
    deduplicated: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            deduplicated.append(item)
    return deduplicated
