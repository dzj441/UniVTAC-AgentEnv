from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from agent_env import expert_trajectory
from agent_env.expert_trajectory import (
    ExpertTrajectoryError,
    build_fixed_expert_manifest,
    inspect_expert_hdf5,
)


def write_trajectory(path: Path, phases: list[str]) -> None:
    frame_count = len(phases)
    joint = np.zeros((frame_count, 9), dtype=np.float32)
    joint[:, 0] = np.linspace(0.0, 0.2, frame_count)
    ee = np.zeros((frame_count, 7), dtype=np.float32)
    ee[:, 0] = np.linspace(0.4, 0.5, frame_count)
    ee[:, 3] = 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=np.arange(frame_count))
        handle.create_dataset("embodiment/joint", data=joint)
        handle.create_dataset("embodiment/ee", data=ee)
        handle.create_dataset(
            "collection/phase",
            data=np.asarray(phases, dtype="S16"),
        )


def test_full_expert_hdf5_has_one_pre_move_to_task_transition(tmp_path: Path) -> None:
    path = tmp_path / "7.hdf5"
    write_trajectory(path, ["pre_move", "pre_move", "pre_move", "task", "task"])

    summary = inspect_expert_hdf5(path)
    assert summary["frame_count"] == 5
    assert summary["pre_move_frame_count"] == 3
    assert summary["task_frame_count"] == 2
    assert summary["task_phase_start_frame"] == 3
    assert len(summary["sha256"]) == 64


@pytest.mark.parametrize(
    "phases",
    [
        ["task", "task"],
        ["pre_move", "pre_move"],
        ["pre_move", "task", "pre_move"],
        ["pre_move", "other", "task"],
    ],
)
def test_expert_hdf5_rejects_missing_or_ambiguous_phases(
    tmp_path: Path, phases: list[str]
) -> None:
    path = tmp_path / "bad.hdf5"
    write_trajectory(path, phases)
    with pytest.raises(ExpertTrajectoryError):
        inspect_expert_hdf5(path)


def test_expert_hdf5_rejects_non_increasing_sim_steps(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_step.hdf5"
    write_trajectory(path, ["pre_move", "pre_move", "task"])
    with h5py.File(path, "r+") as handle:
        handle["step"][:] = [0, 1, 1]
    with pytest.raises(ExpertTrajectoryError, match="increase strictly"):
        inspect_expert_hdf5(path)


def test_manifest_requires_successful_ungrasped_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "11.hdf5"
    write_trajectory(source, ["pre_move", "pre_move", "task", "task"])
    source_video = tmp_path / "source.mp4"
    replay_video = tmp_path / "replay.mp4"
    source_video.write_bytes(b"source-video")
    replay_video.write_bytes(b"replay-video")
    report_path = tmp_path / "report.json"
    report = {
        "schema_version": "univtac.fixed_expert_replay.v1",
        "task": "pull_out_key",
        "seed": 11,
        "source_hdf5": str(source.resolve()),
        "source_trajectory": inspect_expert_hdf5(source),
        "trajectory_includes_pre_move": True,
        "privileged_pre_move_executed_before_replay": False,
        "source_frame_count": 4,
        "stride": 1,
        "applied_frame_count": 4,
        "physics_action_count": 4,
        "timing_mode": "recorded_step_interpolation",
        "execution_succeeded": True,
        "official_task_success": True,
        "evaluator_checks": {"base_task_success": True, "settle_steps": 60},
        "replay_video": str(replay_video),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        expert_trajectory,
        "_ffprobe_video",
        lambda path: {"path": str(Path(path).resolve()), "codec_name": "h264"},
    )

    manifest = build_fixed_expert_manifest(
        task="pull_out_key",
        seed=11,
        source_hdf5=source,
        source_video=source_video,
        replay_report_path=report_path,
    )
    assert manifest["positive_example"] is True
    assert manifest["agent_visible_checker_details"] is False
    assert manifest["replay_verification"]["official_task_success"] is True

    report["official_task_success"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExpertTrajectoryError, match="did not verify task success"):
        build_fixed_expert_manifest(
            task="pull_out_key",
            seed=11,
            source_hdf5=source,
            source_video=source_video,
            replay_report_path=report_path,
        )


def test_manifest_requires_full_timing_and_bottle_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "3.hdf5"
    write_trajectory(source, ["pre_move", "pre_move", "task", "task"])
    source_video = tmp_path / "source.mp4"
    replay_video = tmp_path / "replay.mp4"
    source_video.write_bytes(b"source-video")
    replay_video.write_bytes(b"replay-video")
    report_path = tmp_path / "report.json"
    report = {
        "schema_version": "univtac.fixed_expert_replay.v1",
        "task": "put_bottle_in_shelf",
        "seed": 3,
        "source_hdf5": str(source.resolve()),
        "source_trajectory": inspect_expert_hdf5(source),
        "trajectory_includes_pre_move": True,
        "privileged_pre_move_executed_before_replay": False,
        "source_frame_count": 4,
        "stride": 1,
        "applied_frame_count": 4,
        "physics_action_count": 4,
        "timing_mode": "recorded_step_interpolation",
        "execution_succeeded": True,
        "official_task_success": True,
        "evaluator_checks": {
            "base_task_success": True,
            "settle_steps": 60,
            "released": True,
            "stable_after_release": True,
        },
        "replay_video": str(replay_video),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        expert_trajectory,
        "_ffprobe_video",
        lambda path: {"path": str(Path(path).resolve()), "codec_name": "h264"},
    )

    manifest = build_fixed_expert_manifest(
        task="put_bottle_in_shelf",
        seed=3,
        source_hdf5=source,
        source_video=source_video,
        replay_report_path=report_path,
    )
    assert manifest["positive_example"] is True

    report["evaluator_checks"]["stable_after_release"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExpertTrajectoryError, match="not stable after release"):
        build_fixed_expert_manifest(
            task="put_bottle_in_shelf",
            seed=3,
            source_hdf5=source,
            source_video=source_video,
            replay_report_path=report_path,
        )
