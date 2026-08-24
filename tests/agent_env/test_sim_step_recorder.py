from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent_env.sim_step_recorder import (
    H264FrameWriter,
    PANEL_ORDER,
    SimStepRecorderConfig,
    SimulationStepRecorder,
    compose_sim_step_frame,
)


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> dict[str, object]:
        return {
            "path": "sim_step_composite_h264.mp4",
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "recorded_frame_count": len(self.frames),
        }


def panels() -> dict[str, np.ndarray]:
    return {
        name: np.full((24, 32, 3), index * 50, dtype=np.uint8)
        for index, name in enumerate(PANEL_ORDER)
    }


def test_recorder_configuration_resolves_sampling_and_record_window() -> None:
    resolved = SimStepRecorderConfig(
        frames_per_second=10.0,
        post_action_record_steps=60,
    ).resolve(1 / 120)
    assert resolved["sample_interval_physics_steps"] == 12
    assert resolved["encoded_frames_per_second"] == pytest.approx(10.0)
    assert resolved["post_action_record_physics_steps"] == 60
    assert resolved["post_action_record_seconds"] == pytest.approx(0.5)
    assert resolved["agent_visible"] is False

    with pytest.raises(ValueError, match="positive"):
        SimStepRecorderConfig(frames_per_second=0)
    with pytest.raises(ValueError, match="non-negative"):
        SimStepRecorderConfig(post_action_record_steps=-1)
    with pytest.raises(ValueError, match="integer"):
        SimStepRecorderConfig(post_action_record_steps=0.5)  # type: ignore[arg-type]


def test_composite_contains_all_four_sensor_panels() -> None:
    frame = compose_sim_step_frame(
        panels(),
        header_lines=("step_eef 01", "phase control"),
    )
    assert frame.shape == (648, 960, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[100, 10] == 0)
    assert np.all(frame[100, 500] == 50)
    assert np.all(frame[400, 10] == 100)
    assert np.all(frame[400, 500] == 150)


def test_streaming_writer_produces_browser_compatible_h264(tmp_path: Path) -> None:
    video_path = tmp_path / "sim_step_composite_h264.mp4"
    writer = H264FrameWriter(video_path, frames_per_second=10.0)
    for index in range(3):
        writer.write(np.full((64, 96, 3), index * 80, dtype=np.uint8))
    receipt = writer.close()

    assert receipt is not None
    assert receipt["path"] == video_path.name
    assert receipt["codec_name"] == "h264"
    assert receipt["pix_fmt"] == "yuv420p"
    assert receipt["recorded_frame_count"] == 3
    assert video_path.is_file()


def test_action_windows_sample_only_advanced_simulation_steps(tmp_path: Path) -> None:
    writer = FakeWriter()
    recorder = SimulationStepRecorder(
        run_dir=tmp_path,
        physics_dt=0.1,
        config=SimStepRecorderConfig(
            frames_per_second=5.0,
            post_action_record_steps=3,
        ),
        frame_supplier=panels,
        writer=writer,
    )
    recorder.begin_step_eef(
        action_index=1,
        prior_observation_id="obs_000",
        action={
            "delta_position_world_m": [0.01, 0.0, 0.0],
            "delta_rpy_world_rad": [0.0, 0.0, 0.1],
            "delta_gripper_m": -0.005,
        },
        sim_step=10,
    )
    recorder.on_sim_step(11)
    recorder.on_sim_step(12)
    recorder.begin_post_action_settle(sim_step=12)
    recorder.on_sim_step(13)
    recorder.on_sim_step(14)
    recorder.end_step_eef(
        sim_step=15,
        execution_succeeded=True,
        control_route="all",
        control_wall_seconds=1.25,
    )
    receipt = recorder.finalize()

    assert len(writer.frames) == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "sim_step_frames.jsonl").read_text().splitlines()
    ]
    assert [row["sim_step"] for row in rows] == [10, 12, 14, 15]
    assert [row["phase"] for row in rows] == [
        "pre_action",
        "control",
        "post_action_settle",
        "post_action_settle",
    ]
    manifest = json.loads(
        (tmp_path / "sim_step_recorder_manifest.json").read_text()
    )
    assert manifest["segments"] == [
        {
            "segment_index": 0,
            "start_sim_step": 10,
            "start_frame_index": 0,
            "kind": "step_eef",
            "action_index": 1,
            "prior_observation_id": "obs_000",
            "action": {
                "delta_position_world_m": [0.01, 0.0, 0.0],
                "delta_rpy_world_rad": [0.0, 0.0, 0.1],
                "delta_gripper_m": -0.005,
            },
            "control_end_sim_step": 12,
            "post_action_record_end_sim_step": 15,
            "end_sim_step": 15,
            "end_frame_index_exclusive": 4,
            "post_action_settle_physics_steps_executed": 3,
            "post_action_record_physics_steps_configured": 3,
            "execution_succeeded": True,
            "control_route": "all",
            "control_wall_seconds": 1.25,
        }
    ]
    assert receipt["frame_count"] == 4
    assert receipt["error"] is None


def test_recorded_tail_can_be_shorter_than_environment_settling(tmp_path: Path) -> None:
    recorder = SimulationStepRecorder(
        run_dir=tmp_path,
        physics_dt=0.1,
        config=SimStepRecorderConfig(
            frames_per_second=5.0,
            post_action_record_steps=1,
        ),
        frame_supplier=panels,
        writer=FakeWriter(),
    )
    recorder.begin_step_eef(
        action_index=1,
        prior_observation_id="obs_000",
        action={
            "delta_position_world_m": [0, 0, 0],
            "delta_rpy_world_rad": [0, 0, 0],
            "delta_gripper_m": 0,
        },
        sim_step=10,
    )
    recorder.on_sim_step(12)
    recorder.begin_post_action_settle(sim_step=12)
    for sim_step in range(13, 16):
        recorder.on_sim_step(sim_step)
    recorder.end_step_eef(
        sim_step=15,
        execution_succeeded=True,
        control_route="no_op",
        control_wall_seconds=0.1,
    )
    recorder.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / "sim_step_frames.jsonl").read_text().splitlines()
    ]
    assert [row["sim_step"] for row in rows] == [10, 12, 13]
    manifest = json.loads(
        (tmp_path / "sim_step_recorder_manifest.json").read_text()
    )
    segment = manifest["segments"][0]
    assert segment["post_action_settle_physics_steps_executed"] == 3
    assert segment["post_action_record_physics_steps_configured"] == 1
    assert segment["post_action_record_end_sim_step"] == 13


def test_terminal_settle_is_a_separate_segment(tmp_path: Path) -> None:
    writer = FakeWriter()
    recorder = SimulationStepRecorder(
        run_dir=tmp_path,
        physics_dt=0.1,
        config=SimStepRecorderConfig(
            frames_per_second=10,
            post_action_record_steps=0,
        ),
        frame_supplier=panels,
        writer=writer,
    )
    recorder.begin_terminal_settle(observation_id="obs_003", sim_step=20)
    recorder.on_sim_step(21)
    recorder.end_terminal_settle(sim_step=22)
    recorder.finalize()

    manifest = json.loads(
        (tmp_path / "sim_step_recorder_manifest.json").read_text()
    )
    assert manifest["segments"][0]["kind"] == "finish_settle"
    assert manifest["segments"][0]["observation_id"] == "obs_003"
    assert manifest["frame_count"] == 3


def test_unexpected_control_error_closes_segment_for_a_retry(tmp_path: Path) -> None:
    recorder = SimulationStepRecorder(
        run_dir=tmp_path,
        physics_dt=0.1,
        config=SimStepRecorderConfig(
            frames_per_second=10,
            post_action_record_steps=0,
        ),
        frame_supplier=panels,
        writer=FakeWriter(),
    )
    action = {
        "delta_position_world_m": [0, 0, 0],
        "delta_rpy_world_rad": [0, 0, 0],
        "delta_gripper_m": 0,
    }
    recorder.begin_step_eef(
        action_index=1,
        prior_observation_id="obs_000",
        action=action,
        sim_step=3,
    )
    recorder.abort_step_eef(
        sim_step=4,
        error=RuntimeError("unexpected simulator error"),
        control_wall_seconds=0.25,
    )
    recorder.begin_step_eef(
        action_index=1,
        prior_observation_id="obs_000",
        action=action,
        sim_step=4,
    )
    recorder.end_step_eef(
        sim_step=4,
        execution_succeeded=False,
        control_route=None,
        control_wall_seconds=0.1,
    )
    recorder.finalize()

    manifest = json.loads(
        (tmp_path / "sim_step_recorder_manifest.json").read_text()
    )
    assert manifest["segments"][0]["aborted"] is True
    assert manifest["segments"][0]["abort_error_type"] == "RuntimeError"
    assert "aborted" not in manifest["segments"][1]


def test_recorder_failure_is_diagnostic_and_does_not_escape(tmp_path: Path) -> None:
    class BrokenWriter(FakeWriter):
        def write(self, frame: np.ndarray) -> None:
            raise RuntimeError("encoder unavailable")

    recorder = SimulationStepRecorder(
        run_dir=tmp_path,
        physics_dt=0.1,
        config=SimStepRecorderConfig(),
        frame_supplier=panels,
        writer=BrokenWriter(),
    )
    recorder.begin_step_eef(
        action_index=1,
        prior_observation_id="obs_000",
        action={
            "delta_position_world_m": [0, 0, 0],
            "delta_rpy_world_rad": [0, 0, 0],
            "delta_gripper_m": 0,
        },
        sim_step=1,
    )
    recorder.on_sim_step(2)
    receipt = recorder.finalize()

    assert receipt["frame_count"] == 0
    assert receipt["error"] == {
        "error_type": "RuntimeError",
        "message": "encoder unavailable",
    }
    assert recorder.enabled is False
