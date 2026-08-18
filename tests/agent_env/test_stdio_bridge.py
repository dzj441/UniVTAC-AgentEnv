from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent_env.stdio_bridge import SimulatorBridgeError, SimulatorProcessClient


def test_simulator_bridge_reports_early_process_exit_without_waiting_for_timeout(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    with pytest.raises(SimulatorBridgeError, match="closed stdout") as captured:
        SimulatorProcessClient(
            [sys.executable, "-c", "print('startup failed before ready')"],
            cwd=tmp_path,
            timeout_seconds=30.0,
        )
    assert time.monotonic() - started < 5.0
    assert "startup failed before ready" in str(captured.value)


def test_simulator_bridge_accepts_ready_then_closes_cleanly(tmp_path: Path) -> None:
    code = (
        "import json,sys,time; "
        "print('AGENT_ENV_RESULT '+json.dumps({'status':'ready'}), flush=True); "
        "time.sleep(60)"
    )
    client = SimulatorProcessClient(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        timeout_seconds=5.0,
    )
    try:
        assert client.ready == {"status": "ready"}
    finally:
        client.close()
