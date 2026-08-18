"""Synchronous newline-JSON client for the simulator-owned AgentEnv process."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import IO, Any, Sequence


RESULT_PREFIX = "AGENT_ENV_RESULT "


class SimulatorBridgeError(RuntimeError):
    pass


class SimulatorProcessClient:
    """Own one AgentEnv subprocess and serialize every request/response pair."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 360.0,
        startup_log_path: Path | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.startup_log_path = startup_log_path
        self.process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._output_tail: deque[str] = deque(maxlen=80)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self.ready = self._result("ready")

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        log: IO[str] | None = None
        try:
            if self.startup_log_path is not None:
                self.startup_log_path.parent.mkdir(parents=True, exist_ok=True)
                log = self.startup_log_path.open("a", encoding="utf-8")
            for line in self.process.stdout:
                self._output_tail.append(line)
                if log is not None:
                    log.write(line)
                    log.flush()
                if line.startswith(RESULT_PREFIX):
                    payload = json.loads(line[len(RESULT_PREFIX) :])
                    if isinstance(payload, dict):
                        self._responses.put(payload)
        finally:
            if log is not None:
                log.close()
            self._responses.put(None)

    def _diagnostic_tail(self) -> str:
        text = "".join(self._output_tail).strip()
        return f"\n--- simulator output tail ---\n{text}" if text else ""

    def _result(self, expected_status: str | None = None) -> dict[str, Any]:
        try:
            payload = self._responses.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            code = self.process.poll()
            raise SimulatorBridgeError(
                f"Timed out waiting for AgentEnv response; process exit={code}"
                f"{self._diagnostic_tail()}"
            ) from exc
        if payload is None:
            raise SimulatorBridgeError(
                "AgentEnv process closed stdout before returning the requested response; "
                f"process exit={self.process.poll()}{self._diagnostic_tail()}"
            )
        if expected_status is not None and payload.get("status") != expected_status:
            raise SimulatorBridgeError(
                f"Expected AgentEnv status {expected_status!r}, got {payload!r}"
            )
        return payload

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise SimulatorBridgeError(
                f"AgentEnv process already exited with {self.process.returncode}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        return self._result()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self._reader.join(timeout=2)

    def wait(self, timeout_seconds: float = 120.0) -> int:
        return self.process.wait(timeout=timeout_seconds)
