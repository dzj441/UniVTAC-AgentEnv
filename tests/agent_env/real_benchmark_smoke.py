#!/usr/bin/env python3
"""Black-box real-Isaac acceptance client for generic AgentEnv v1."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "AGENT_ENV_RESULT "
TASKS = ("pull_out_key", "put_bottle_in_shelf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--profile", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--provide-bbox", action="store_true")
    parser.add_argument("--provide-mask", action="store_true")
    parser.add_argument("--pre-move", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=float, default=900.0)
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
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output_lines.append(line)
            self._lines.put(line)
        self._lines.put(None)

    def send(self, command: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()

    def result(self, *statuses: str) -> dict[str, Any]:
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {statuses}")
            try:
                line = self._lines.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Runner exited with {self.process.returncode}")
                continue
            if line is None:
                tail = "".join(self.output_lines[-60:])
                raise RuntimeError(f"Runner closed before {statuses}:\n{tail}")
            if not line.startswith(RESULT_PREFIX):
                continue
            payload = json.loads(line[len(RESULT_PREFIX) :])
            if payload.get("status") == "command_error":
                raise RuntimeError(f"Runner command error: {payload}")
            if payload.get("status") not in statuses:
                raise RuntimeError(f"Expected {statuses}, got {payload}")
            return payload


def assert_artifact(run_dir: Path, value: dict[str, Any], media_type: str) -> Path:
    path = Path(value["path"])
    assert path.is_relative_to(run_dir)
    assert path.is_file()
    assert value["media_type"] == media_type
    assert len(value["sha256"]) == 64
    return path


def assert_full_p6_observation(
    run_dir: Path,
    task: str,
    observation: dict[str, Any],
    *,
    bbox: bool,
    mask: bool,
) -> None:
    assert observation["task"] == task
    assert observation["observation_profile"].startswith("head_wrist_tactile_depth")
    assert set(observation["modalities"]) == {
        "head_rgb",
        "wrist_rgb",
        "left_tactile_rgb",
        "right_tactile_rgb",
        "head_depth",
        "wrist_depth",
    }
    for name in ("head_rgb", "wrist_rgb", "left_tactile_rgb", "right_tactile_rgb"):
        assert_artifact(run_dir, observation["modalities"][name], "image/png")
    for name in ("head_depth", "wrist_depth"):
        depth = observation["modalities"][name]
        depth_path = assert_artifact(run_dir, depth["depth_m"], "application/x-npy")
        array = np.load(depth_path, allow_pickle=False)
        assert array.shape == (270, 480)
        assert array.dtype == np.float32
        assert np.isfinite(array).any()
        assert_artifact(run_dir, depth["valid_mask"], "image/png")
        assert_artifact(run_dir, depth["visualization"], "image/png")

    assert set(observation["robot_state"]) == {
        "joint_position_9d",
        "joint_velocity_9d",
        "gripper_width_m",
        "end_effector_pose_robot_base_wxyz_7d",
    }
    assert len(observation["robot_state"]["joint_position_9d"]) == 9
    assert len(observation["robot_state"]["joint_velocity_9d"]) == 9
    assert len(observation["robot_state"]["end_effector_pose_robot_base_wxyz_7d"]) == 7
    assert set(observation["camera_calibration"]) == {"head", "wrist"}
    for camera in ("head", "wrist"):
        calibration = observation["camera_calibration"][camera]
        assert np.asarray(calibration["intrinsic_matrix_3x3"]).shape == (3, 3)
        extrinsic = np.asarray(
            calibration["extrinsics"]["matrix_T_robot_base_camera_ros_4x4"]
        )
        assert extrinsic.shape == (4, 4)
        assert np.isfinite(extrinsic).all()

    if bbox or mask:
        assert set(observation["annotations"]) == {"head", "wrist"}
        for camera in ("head", "wrist"):
            assert set(observation["annotations"][camera]) == {
                "manipulated_object",
                "goal_fixture",
            }
            for annotation in observation["annotations"][camera].values():
                assert set(annotation).issuperset({"public_role", "visible"})
                if bbox:
                    assert "bbox_xyxy_exclusive" in annotation
                    assert_artifact(run_dir, annotation["bbox_overlay"], "image/png")
                else:
                    assert "bbox_xyxy_exclusive" not in annotation
                if mask:
                    mask_path = assert_artifact(
                        run_dir, annotation["mask"], "image/png"
                    )
                    assert_artifact(
                        run_dir, annotation["mask_overlay"], "image/png"
                    )
                    with Image.open(mask_path) as mask_image:
                        assert mask_image.mode == "L"
                else:
                    assert "mask" not in annotation
    else:
        assert "annotations" not in observation

    serialized = json.dumps(observation)
    for forbidden in (
        "task_success",
        "official_task_success",
        "idToLabels",
        "/World/",
        "raw_id_to_labels",
        "prim_path",
    ):
        assert forbidden not in serialized


def main() -> None:
    args = parse_args()
    if args.profile != 6:
        raise ValueError("This full-feature acceptance client currently requires profile 6")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else REPO_ROOT
        / "agent_runs"
        / (
            f"accept_{args.task}_p6_"
            f"{'pregrasped' if args.pre_move else 'ungrasped'}_{stamp}"
        )
    )
    command = [
        str(REPO_ROOT / "scripts" / "launch_agent_env.sh"),
        "--task",
        args.task,
        "--profile",
        str(args.profile),
        "--device",
        args.device,
        "--run-dir",
        str(run_dir),
    ]
    if args.provide_bbox:
        command.append("--provide-bbox")
    if args.provide_mask:
        command.append("--provide-mask")
    if args.pre_move:
        command.append("--pre-move")
    environment = os.environ.copy()
    environment["UNIVTAC_EVALUATOR_SEED"] = str(args.seed)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    client = Client(process, args.timeout)
    try:
        ready = client.result("ready")
        assert ready["task"] == args.task
        assert ready["pre_move_enabled"] is args.pre_move
        assert ready["start_condition"] == (
            "pregrasped" if args.pre_move else "ungrasped"
        )
        assert ready["observation_profile"]["index"] == 6
        assert ready["annotations"]["bbox"] is args.provide_bbox
        assert ready["annotations"]["mask"] is args.provide_mask
        assert ready["agent_tools"] == ["start_episode", "step_eef", "finish_episode"]
        assert [variant["title"] for variant in ready["command_schema"]["oneOf"]] == [
            "start",
            "step",
            "finish",
        ]
        private_audit = run_dir / "evaluator_private_audit.json"
        assert not private_audit.exists()

        client.send({"command": "start", "agent_note": "full-feature real smoke"})
        started = client.result("rollout_started")
        assert started["start_condition"] == ready["start_condition"]
        initial_gripper_width = started["observation"]["robot_state"]["gripper_width_m"]
        if args.pre_move:
            assert initial_gripper_width < 0.03
            assert "already-grasped" in started["instruction"]
        else:
            assert initial_gripper_width >= 0.035
            assert started["instruction"].startswith(("Grasp", "Pick up"))
        assert_full_p6_observation(
            run_dir,
            args.task,
            started["observation"],
            bbox=args.provide_bbox,
            mask=args.provide_mask,
        )
        assert not private_audit.exists()

        client.send(
            {
                "command": "step",
                "observation_id": started["observation"]["observation_id"],
                "delta_position": [0, 0, 0],
                "delta_rpy": [0, 0, 0],
                "delta_gripper": 0,
                "rationale": "Acceptance test advances one bounded zero-delta EEF cycle.",
            }
        )
        stepped = client.result("action_complete")
        assert isinstance(stepped["execution_succeeded"], bool)
        assert "task_success" not in json.dumps(stepped)
        assert_full_p6_observation(
            run_dir,
            args.task,
            stepped["observation"],
            bbox=False,
            mask=False,
        )
        assert not private_audit.exists()

        client.send(
            {
                "command": "finish",
                "observation_id": stepped["observation"]["observation_id"],
                "final_note": "Full-feature real smoke reached terminal evaluation.",
            }
        )
        outcome = client.result("rollout_finished")
        assert isinstance(outcome["official_task_success"], bool)
        assert outcome["commitment_verified"] is True
        assert_full_p6_observation(
            run_dir,
            args.task,
            outcome["observation"],
            bbox=False,
            mask=False,
        )
        client.send({"command": "close"})
        client.result("closing")
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=120) == 0

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        saved = json.loads((run_dir / "evaluator_outcome.json").read_text(encoding="utf-8"))
        assert manifest["public_tools"] == ["start_episode", "step_eef", "finish_episode"]
        assert manifest["action_budget"]["step_eef"] == 50
        assert manifest["pre_move_enabled"] is args.pre_move
        assert manifest["start_condition"] == ready["start_condition"]
        assert manifest["annotations"]["bbox"] is args.provide_bbox
        assert manifest["annotations"]["mask"] is args.provide_mask
        assert saved["task"] == args.task
        assert saved["pre_move_enabled"] is args.pre_move
        assert saved["start_condition"] == ready["start_condition"]
        assert private_audit.stat().st_mode & 0o777 == 0o600
        assert saved["replay_video"].get("codec_name") == "h264", saved["replay_video"]
        assert saved["replay_video"].get("pix_fmt") == "yuv420p"
        print(
            f"REAL_BENCHMARK_SMOKE_OK task={args.task} profile=6 "
            f"start_condition={ready['start_condition']} bbox={args.provide_bbox} "
            f"mask={args.provide_mask} run_dir={run_dir}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        client.reader_thread.join(timeout=2)
        if run_dir.is_dir() and client.output_lines:
            (run_dir / "acceptance_process.log").write_text(
                "".join(client.output_lines), encoding="utf-8"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"REAL_BENCHMARK_SMOKE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
