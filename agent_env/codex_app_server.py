"""Minimal Codex app-server client for embodied-only dynamic-tool rollouts."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .capabilities import CapabilityGateway, CapabilityViolation, GatewayExecution
from .codex_events import EventRecorder


class CodexAppServerError(RuntimeError):
    pass


FORBIDDEN_ITEM_TYPES = {
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

ALLOWED_ITEM_TYPES = {
    "userMessage",
    "agentMessage",
    "reasoning",
    "dynamicToolCall",
    "contextCompaction",
    "plan",
    "sleep",
}

FORBIDDEN_SERVER_REQUESTS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
    "tool/requestUserInput",
}


class CodexAppServerClient:
    """Drive one ephemeral Codex thread and service its embodied tool calls."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        recorder: EventRecorder,
        stderr_path: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.command = tuple(command)
        self.cwd = cwd.resolve()
        self.recorder = recorder
        self.timeout_seconds = timeout_seconds
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.capability_violation = False
        self._request_id = 0
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_handle = stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            list(self.command),
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            bufsize=1,
        ) # launch codex here
        assert self.process.stdout is not None
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.recorder.record_violation(
                    {
                        "reason": "non_json_app_server_stdout",
                        "line": line.rstrip("\n")[:1000],
                    }
                )
                self.capability_violation = True
                continue
            if isinstance(message, dict):
                self.recorder.record_raw("server_to_host", message)
                self._messages.put(message)

    def _send(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise CodexAppServerError(
                f"Codex app-server exited with {self.process.returncode}"
            )
        assert self.process.stdin is not None
        self.recorder.record_raw("host_to_server", message)
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id()
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout_seconds
        deferred: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            message = self._get_message(deadline)
            if message.get("id") == request_id and "method" not in message:
                for item in deferred:
                    self._messages.put(item)
                if "error" in message:
                    raise CodexAppServerError(
                        f"Codex {method} failed: {message['error']}"
                    )
                result = message.get("result", {})
                if not isinstance(result, dict):
                    raise CodexAppServerError(
                        f"Codex {method} returned a non-object result"
                    )
                return result
            deferred.append(message)
        raise CodexAppServerError(f"Timed out waiting for {method}")

    def _get_message(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexAppServerError("Timed out waiting for Codex app-server")
        try:
            return self._messages.get(timeout=min(remaining, 10.0))
        except queue.Empty as exc:
            if self.process.poll() is not None:
                raise CodexAppServerError(
                    f"Codex app-server exited with {self.process.returncode}"
                ) from exc
            return self._get_message(deadline)

    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "univtac-agentenv",
                    "title": "UniVTAC Agentic Embodied Benchmark",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self._send({"method": "initialized", "params": {}})
        return result

    def start_thread(
        self,
        *,
        gateway: CapabilityGateway,
        model: str | None,
        base_instructions: str,
        developer_instructions: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": str(self.cwd),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
            "experimentalRawEvents": False,
            "dynamicTools": gateway.dynamic_tools(),
            "baseInstructions": base_instructions,
            "developerInstructions": developer_instructions,
            "environments": [],
            "runtimeWorkspaceRoots": [str(self.cwd)],
            "serviceName": "univtac-agentenv",
        }
        if model:
            params["model"] = model
        result = self._request("thread/start", params)
        thread = result.get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise CodexAppServerError(f"thread/start omitted a thread id: {result}")
        self.thread_id = thread["id"]
        return result

    def run_turn(
        self,
        *,
        gateway: CapabilityGateway,
        prompt: str,
        model: str | None,
        effort: str,
        reasoning_summary: str = "detailed",
    ) -> dict[str, Any]:
        if self.thread_id is None:
            raise CodexAppServerError("start_thread must be called first")
        request_id = self._next_id()
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": str(self.cwd),
            "approvalPolicy": "never",
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
            "runtimeWorkspaceRoots": [str(self.cwd)],
            "environments": [],
            "effort": effort,
            "summary": reasoning_summary,
            "responsesapiClientMetadata": {
                "benchmark": "univtac_agentenv",
                "task": "grasp_classify",
                "level": str(gateway.profile.level),
            },
        }
        if model:
            params["model"] = model
        self._send({"id": request_id, "method": "turn/start", "params": params})
        deadline = time.monotonic() + self.timeout_seconds
        start_response: dict[str, Any] | None = None
        terminal_notification: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            message = self._get_message(deadline)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise CodexAppServerError(
                        f"turn/start failed: {message['error']}"
                    )
                start_response = message.get("result", {})
                turn = start_response.get("turn") if isinstance(start_response, dict) else None
                if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                    self.turn_id = turn["id"]
                continue

            method = message.get("method")
            message_params = message.get("params")
            if isinstance(message_params, dict) and isinstance(
                message_params.get("turnId"), str
            ):
                self.turn_id = message_params["turnId"]
            if method == "item/tool/call":
                self._handle_dynamic_tool(message, gateway)
                continue
            if isinstance(method, str) and method in FORBIDDEN_SERVER_REQUESTS:
                self._fail_capability(
                    reason="forbidden_server_request",
                    evidence={"method": method, "params": message.get("params")},
                )
                self._send(
                    {
                        "id": message.get("id"),
                        "error": {
                            "code": -32001,
                            "message": "Capability denied by embodied benchmark host",
                        },
                    }
                )
                raise CapabilityViolation(f"Codex requested forbidden capability {method}")
            if isinstance(method, str) and "id" in message:
                self._fail_capability(
                    reason="unknown_server_request_fail_closed",
                    evidence={"method": method, "params": message.get("params")},
                )
                self._send(
                    {
                        "id": message.get("id"),
                        "error": {
                            "code": -32001,
                            "message": "Unrecognized capability request denied by host",
                        },
                    }
                )
                raise CapabilityViolation(
                    f"Codex emitted an unrecognized server request {method!r}"
                )

            if method in {"item/started", "item/completed"}:
                self._inspect_item(message)
            if method == "turn/completed":
                terminal_notification = message
                break
            if method == "error":
                self.recorder.record_message(
                    {"kind": "app_server_error", "params": message.get("params")}
                )

        if terminal_notification is None:
            self.interrupt_turn()
            raise CodexAppServerError("Codex turn exceeded the rollout timeout")
        params_out = terminal_notification.get("params", {})
        turn = params_out.get("turn", {}) if isinstance(params_out, dict) else {}
        return {
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "turn": turn,
            "start_response": start_response,
            "gateway_terminal": gateway.terminal,
            "capability_violation": self.capability_violation,
        }

    def _handle_dynamic_tool(
        self,
        message: dict[str, Any],
        gateway: CapabilityGateway,
    ) -> None:
        request_id = message.get("id")
        params = message.get("params", {})
        if not isinstance(params, dict):
            raise CodexAppServerError("item/tool/call params must be an object")
        tool = params.get("tool")
        arguments = params.get("arguments")
        call_id = params.get("callId")
        started = time.monotonic()
        try:
            execution = gateway.execute(str(tool), arguments)
        except CapabilityViolation as exc:
            self._fail_capability(
                reason="unregistered_or_escaped_dynamic_tool",
                evidence={"tool": tool, "arguments": arguments, "message": str(exc)},
            )
            response = {
                "success": False,
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "Capability violation: the requested tool is unavailable.",
                    }
                ],
            }
            self._send({"id": request_id, "result": response})
            raise

        self._record_execution(
            execution,
            call_id=str(call_id),
            arguments=arguments,
            duration_seconds=time.monotonic() - started,
        )
        self._send(
            {
                "id": request_id,
                "result": {
                    "success": execution.success,
                    "contentItems": list(execution.content_items),
                },
            }
        )

    def _record_execution(
        self,
        execution: GatewayExecution,
        *,
        call_id: str,
        arguments: Any,
        duration_seconds: float,
    ) -> None:
        common = {
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "call_id": call_id,
            "tool": execution.tool,
            "success": execution.success,
            "arguments": arguments,
            "execution_target": execution.execution_target,
            "backend_request": execution.backend_request,
            "simulator_command": execution.simulator_command,
            "environment_response": execution.public_response,
            "host_response": execution.raw_response,
            # Backward-compatible alias consumed by existing traces/viewers.
            "simulator_response": execution.public_response,
            "prior_observation_id": execution.prior_observation_id,
            "next_observation_id": execution.next_observation_id,
            "duration_seconds": round(duration_seconds, 6),
        }
        self.recorder.record_tool_call(common)
        if execution.decision_record is not None:
            self.recorder.record_decision(
                {
                    "thread_id": self.thread_id,
                    "turn_id": self.turn_id,
                    "call_id": call_id,
                    "tool": execution.tool,
                    "observation_id": execution.prior_observation_id,
                    "decision_record": execution.decision_record,
                    "execution_target": execution.execution_target,
                    "chosen_action": execution.backend_request,
                    "returned_observation_id": execution.next_observation_id,
                    "tool_success": execution.success,
                }
            )

    def _inspect_item(self, message: dict[str, Any]) -> None:
        params = message.get("params", {})
        item = params.get("item") if isinstance(params, dict) else None
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type in FORBIDDEN_ITEM_TYPES:
            self._fail_capability(
                reason="forbidden_codex_item",
                evidence={
                    "method": message.get("method"),
                    "item_type": item_type,
                    "item_id": item.get("id"),
                },
            )
            self.interrupt_turn()
            raise CapabilityViolation(f"Codex emitted forbidden item type {item_type}")
        if not isinstance(item_type, str) or item_type not in ALLOWED_ITEM_TYPES:
            self._fail_capability(
                reason="unknown_codex_item_fail_closed",
                evidence={
                    "method": message.get("method"),
                    "item_type": item_type,
                    "item_id": item.get("id"),
                },
            )
            self.interrupt_turn()
            raise CapabilityViolation(
                f"Codex emitted an unrecognized item type {item_type!r}"
            )
        if message.get("method") != "item/completed":
            return
        if item_type == "agentMessage":
            self.recorder.record_message(
                {
                    "kind": "agent_message",
                    "thread_id": params.get("threadId"),
                    "turn_id": params.get("turnId"),
                    "item_id": item.get("id"),
                    "phase": item.get("phase"),
                    "text": item.get("text", ""),
                }
            )
        elif item_type == "reasoning":
            self.recorder.record_message(
                {
                    "kind": "published_reasoning_summary",
                    "thread_id": params.get("threadId"),
                    "turn_id": params.get("turnId"),
                    "item_id": item.get("id"),
                    "summary": item.get("summary"),
                    "content": item.get("content"),
                }
            )

    def _fail_capability(self, *, reason: str, evidence: dict[str, Any]) -> None:
        self.capability_violation = True
        self.recorder.record_violation({"reason": reason, "evidence": evidence})

    def interrupt_turn(self) -> None:
        if self.thread_id is None or self.turn_id is None or self.process.poll() is not None:
            return
        try:
            self._send(
                {
                    "id": self._next_id(),
                    "method": "turn/interrupt",
                    "params": {"threadId": self.thread_id, "turnId": self.turn_id},
                }
            )
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self._reader.join(timeout=2)
            self._stderr_handle.close()
