from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from agent_env.capabilities import CapabilityGateway
from agent_env.codex_app_server import CodexAppServerClient
from agent_env.codex_events import EventRecorder, audit_codex_events


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _decision() -> dict[str, Any]:
    return {
        "evidence": [
            {
                "source": "head_rgb",
                "finding": "The visible face appears uniform.",
                "implication": "The plain class is more likely.",
            }
        ],
        "alternatives_considered": ["Probe, but the current visual evidence is sufficient."],
        "uncertainty": 0.2,
        "expected_effect": "The target commitment should unlock bounded control.",
        "parameter_rationale": "This commitment has no continuous motion magnitude.",
        "rationale": "Commit plain based on the latest public image.",
    }


class TerminalSimulator:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(command)
        if command["command"] == "start":
            modalities = {}
            for name in ("head_rgb", "wrist_rgb"):
                path = self.run_dir / "observations" / "obs_000" / f"{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(PNG_1X1)
                modalities[name] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(PNG_1X1).hexdigest(),
                }
            return {
                "status": "rollout_started",
                "observation": {
                    "observation_id": "obs_000",
                    "level": 1,
                    "profile": "vision_only_control",
                    "stage": "classification",
                    "probe_count": 0,
                    "post_prediction_action_count": 0,
                    "modalities": modalities,
                    "robot_state": {
                        "joint_position_8d": [0.0] * 8,
                        "gripper_qpos": 0.007,
                        "end_effector_pose_robot_base_7d": [0.0] * 7,
                    },
                },
            }
        if command["command"] == "submit_prediction":
            return {
                "status": "prediction_submitted",
                "predicted_class": "plain",
                "committed_target": "green",
                "observation_id": "obs_000",
            }
        if command["command"] == "finish":
            return {
                "status": "rollout_finished",
                "predicted_class": "plain",
                "true_class": "rough",
                "expected_target": "orange",
                "official_task_success": False,
            }
        raise AssertionError(command)


FAKE_SERVER = r'''#!/usr/bin/env python3
import json
import sys

def receive():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)

def send(payload):
    print(json.dumps(payload), flush=True)

initialize = receive()
send({"id": initialize["id"], "result": {"userAgent": "fake"}})
assert receive()["method"] == "initialized"
thread = receive()
assert thread["method"] == "thread/start"
tools = {item["name"] for item in thread["params"]["dynamicTools"]}
assert tools == {
    "start_episode", "probe_gripper", "commit_classification",
    "act_delta_ee", "wait_physics", "finish_episode",
}
send({"id": thread["id"], "result": {"thread": {"id": "thread-test"}}})
turn = receive()
assert turn["method"] == "turn/start"
send({"id": turn["id"], "result": {"turn": {"id": "turn-test", "status": "inProgress", "items": []}}})

send({"id": 101, "method": "item/tool/call", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "callId": "call-start",
    "tool": "start_episode", "arguments": {"agent_note": "isolated test"},
}})
start_result = receive()
assert start_result["id"] == 101 and start_result["result"]["success"]
start_payload = json.loads(start_result["result"]["contentItems"][0]["text"])
observation_id = start_payload["observation"]["observation_id"]

decision = {
    "evidence": [{
        "source": "head_rgb", "finding": "The face appears uniform.",
        "implication": "The plain class is more likely.",
    }],
    "alternatives_considered": ["Probe, but visual evidence is sufficient."],
    "uncertainty": 0.2,
    "expected_effect": "Commitment unlocks bounded control.",
    "parameter_rationale": "There is no continuous motion magnitude.",
    "rationale": "Commit plain from public RGB.",
}
send({"id": 102, "method": "item/tool/call", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "callId": "call-commit",
    "tool": "commit_classification", "arguments": {
        "observation_id": observation_id, "predicted_class": "plain",
        "target_pad": "green", "decision_record": decision,
    },
}})
commit_result = receive()
assert commit_result["id"] == 102 and commit_result["result"]["success"]

rejected_decision = dict(decision)
rejected_decision["parameter_rationale"] = "Twenty millimetres was proposed but exceeds the host bound."
send({"id": 103, "method": "item/tool/call", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "callId": "call-rejected",
    "tool": "act_delta_ee", "arguments": {
        "observation_id": observation_id, "delta_position": [0.0, 0.0, 0.0],
        "delta_rpy": [0.0, 0.0, 0.0], "delta_gripper": 0.02,
        "decision_record": rejected_decision,
    },
}})
rejected_result = receive()
assert rejected_result["id"] == 103 and not rejected_result["result"]["success"]

finish_decision = dict(decision)
finish_decision["expected_effect"] = "Terminate without additional disturbance."
send({"id": 104, "method": "item/tool/call", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "callId": "call-finish",
    "tool": "finish_episode", "arguments": {
        "observation_id": observation_id, "final_note": "Best attainable state.",
        "decision_record": finish_decision,
    },
}})
finish_result = receive()
assert finish_result["id"] == 104 and finish_result["result"]["success"]

send({"method": "item/completed", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "completedAtMs": 1,
    "item": {"id": "reason-1", "type": "reasoning", "summary": ["Used only public RGB."]},
}})
send({"method": "item/completed", "params": {
    "threadId": "thread-test", "turnId": "turn-test", "completedAtMs": 2,
    "item": {"id": "message-1", "type": "agentMessage", "phase": "final_answer", "text": "Rollout finished."},
}})
send({"method": "turn/completed", "params": {
    "threadId": "thread-test", "turn": {"id": "turn-test", "status": "completed", "items": []},
}})
for _ in sys.stdin:
    pass
'''


def test_app_server_loop_records_dynamic_tools_decisions_and_messages(
    tmp_path: Path,
) -> None:
    server_path = tmp_path / "fake_app_server.py"
    server_path.write_text(FAKE_SERVER, encoding="utf-8")
    recorder = EventRecorder(tmp_path)
    simulator = TerminalSimulator(tmp_path)
    gateway = CapabilityGateway(
        level=1,
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
    )
    client = CodexAppServerClient(
        [sys.executable, str(server_path)],
        cwd=tmp_path,
        recorder=recorder,
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=10,
    )
    try:
        client.initialize()
        client.start_thread(
            gateway=gateway,
            model=None,
            base_instructions="Use only dynamic tools.",
            developer_instructions="No delegation.",
        )
        result = client.run_turn(
            gateway=gateway,
            prompt="Complete one episode.",
            model=None,
            effort="high",
        )
    finally:
        client.close()

    assert result["gateway_terminal"] is True
    assert result["capability_violation"] is False
    assert [call["command"] for call in simulator.calls] == [
        "start",
        "submit_prediction",
        "finish",
    ]
    tool_rows = [
        json.loads(line)
        for line in recorder.tool_path.read_text(encoding="utf-8").splitlines()
    ]
    decision_rows = [
        json.loads(line)
        for line in recorder.decision_path.read_text(encoding="utf-8").splitlines()
    ]
    message_rows = [
        json.loads(line)
        for line in recorder.message_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["tool"] for row in tool_rows] == [
        "start_episode",
        "commit_classification",
        "act_delta_ee",
        "finish_episode",
    ]
    assert [row["tool"] for row in decision_rows] == [
        "commit_classification",
        "act_delta_ee",
        "finish_episode",
    ]
    assert tool_rows[2]["success"] is False
    assert decision_rows[1]["chosen_action"] is None
    assert decision_rows[1]["tool_success"] is False
    assert {row["kind"] for row in message_rows} == {
        "published_reasoning_summary",
        "agent_message",
    }
    assert audit_codex_events(recorder.raw_path)["passed"] is True
