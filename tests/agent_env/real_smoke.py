#!/usr/bin/env python3
"""Black-box real-Isaac acceptance client for grasp_classify AgentEnv v0."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "AGENT_ENV_RESULT "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


class Client:
    def __init__(self, process: subprocess.Popen[str], timeout: float):
        self.process = process
        self.deadline = time.monotonic() + timeout
        self.output_lines: list[str] = []
        self._lines: queue.Queue[str | None] = queue.Queue()
        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()

    def _read_output(self) -> None:
        if self.process.stdout is None:
            self._lines.put(None)
            return
        for line in self.process.stdout:
            self.output_lines.append(line)
            self._lines.put(line)
        self._lines.put(None)

    def diagnostic_tail(self, line_count: int = 40) -> str:
        tail = "".join(self.output_lines[-line_count:]).strip()
        return f"\n--- runner output tail ---\n{tail}" if tail else ""

    def send(self, command: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Runner stdin is unavailable")
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()

    def result(self, *allowed_statuses: str) -> dict[str, Any]:
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {allowed_statuses}")
            try:
                line = self._lines.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Runner exited with {self.process.returncode} before {allowed_statuses}"
                        f"{self.diagnostic_tail()}"
                    )
                continue
            if line is None:
                raise RuntimeError(
                    f"Runner closed stdout with code {self.process.poll()} before {allowed_statuses}"
                    f"{self.diagnostic_tail()}"
                )
            if not line.startswith(RESULT_PREFIX):
                continue
            payload = json.loads(line[len(RESULT_PREFIX) :])
            status = payload.get("status")
            if status == "command_error":
                raise RuntimeError(f"Runner command error: {payload}")
            if status not in allowed_statuses:
                raise RuntimeError(f"Expected {allowed_statuses}, got {payload}")
            return payload


def assert_observation_contract(level: int, observation: dict[str, Any]) -> None:
    modalities = set(observation["modalities"])
    expected = {"head_rgb", "wrist_rgb"}
    if level >= 2:
        expected |= {"left_tactile_marker", "right_tactile_marker"}
    assert modalities == expected, (modalities, expected)
    assert set(observation["robot_state"]) == {
        "joint_position_8d",
        "gripper_qpos",
        "end_effector_pose_robot_base_7d",
    }
    serialized = repr(observation)
    assert "depth" not in serialized
    assert "actor" not in serialized
    assert "task_success" not in serialized


def main() -> None:
    args = parse_args()
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "agent_runs" / f"accept_agentenv_l{args.level}_{stamp}"
    else:
        run_dir = args.run_dir.resolve()

    command = [
        str(REPO_ROOT / "scripts" / "launch_agent_env.sh"),
        "--level",
        str(args.level),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    client = Client(process, args.timeout)
    try:
        ready = client.result("ready")
        assert ready["level"] == args.level
        variants = {
            variant["title"]: variant
            for variant in ready["command_schema"]["oneOf"]
        }
        assert "delta_position" in variants["act"]["required"]
        assert variants["act"]["additionalProperties"] is False
        assert "final_note" in variants["finish"]["required"]
        private_audit = run_dir / "evaluator_private_audit.json"
        assert not private_audit.exists()

        client.send({"command": "start", "agent_note": "real acceptance smoke"})
        started = client.result("rollout_started")
        assert_observation_contract(args.level, started["observation"])
        assert not private_audit.exists()
        observation_id = started["observation"]["observation_id"]

        client.send(
            {
                "command": "submit_prediction",
                "observation_id": observation_id,
                "predicted_class": "rough",
                "target_pad": "orange",
                "rationale": "Acceptance test exercises the irreversible prediction boundary.",
            }
        )
        prediction = client.result("prediction_submitted")
        assert prediction["irreversible"] is True
        assert not private_audit.exists()

        client.send(
            {
                "command": "wait",
                "observation_id": observation_id,
                "steps": 1,
                "rationale": "Acceptance test requests one post-prediction feedback cycle.",
            }
        )
        action = client.result("action_complete", "rollout_finished")
        feedback = (
            action["last_action"]["feedback"]
            if action["status"] == "rollout_finished"
            else action["feedback"]
        )
        assert set(feedback) == (
            {"execution_succeeded", "task_success"}
            if args.level == 3
            else {"execution_succeeded"}
        )

        if action["status"] == "action_complete":
            assert not private_audit.exists()
            next_observation = action["observation"]
            assert_observation_contract(args.level, next_observation)
            client.send(
                {
                    "command": "finish",
                    "observation_id": next_observation["observation_id"],
                    "final_note": "Real acceptance smoke completed the public protocol.",
                }
            )
            outcome = client.result("rollout_finished")
        else:
            outcome = action

        assert isinstance(outcome["official_task_success"], bool)
        assert outcome["commitment_verified"] is True
        client.send({"command": "close"})
        client.result("closing")
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait(timeout=60)
        assert return_code == 0

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        saved_outcome = json.loads(
            (run_dir / "evaluator_outcome.json").read_text(encoding="utf-8")
        )
        assert manifest["profile"]["level"] == args.level
        assert manifest["initialization_reset_time_limit_seconds"] == 240.0
        nvidia = manifest["nvidia_userspace"]
        bundle_root = Path(nvidia["bundle_root"])
        assert nvidia["all_nvidia_userspace_from_bundle"] is True
        assert Path(nvidia["library_dir"]) == bundle_root / "runtime-libs-full"
        mapped_driver_libraries = [
            Path(path) for path in nvidia["mapped_driver_libraries"]
        ]
        assert mapped_driver_libraries
        assert all(path.is_relative_to(bundle_root) for path in mapped_driver_libraries)
        assert any(path.name.startswith("libcuda.so") for path in mapped_driver_libraries)
        assert saved_outcome["commitment_verified"] is True
        assert (run_dir / "agent_transcript.jsonl").is_file()
        assert private_audit.stat().st_mode & 0o777 == 0o600
        video = saved_outcome["replay_video"]
        assert video.get("codec_name") == "h264", video
        assert video.get("pix_fmt") == "yuv420p", video
        print(f"REAL_AGENT_ENV_SMOKE_OK level={args.level} run_dir={run_dir}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        client.reader_thread.join(timeout=2)
        if run_dir.is_dir() and client.output_lines:
            (run_dir / "acceptance_process.log").write_text(
                "".join(client.output_lines), encoding="utf-8"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"REAL_AGENT_ENV_SMOKE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
