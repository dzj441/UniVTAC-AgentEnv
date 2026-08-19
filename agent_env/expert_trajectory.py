"""Validation and provenance manifests for fixed expert ICL trajectories."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .artifacts import file_sha256


class ExpertTrajectoryError(ValueError):
    pass


def _decode_phases(value: Any) -> list[str]:
    result: list[str] = []
    for item in np.asarray(value).reshape(-1):
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def inspect_expert_hdf5(path: Path) -> dict[str, Any]:
    """Validate that one HDF5 episode spans ungrasped setup and task motion."""

    path = path.resolve()
    if not path.is_file():
        raise ExpertTrajectoryError(f"Expert HDF5 does not exist: {path}")
    required = (
        "step",
        "embodiment/joint",
        "embodiment/ee",
        "collection/phase",
    )
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ExpertTrajectoryError(
                f"Expert HDF5 is missing required datasets: {missing}"
            )
        steps = np.asarray(handle["step"][()])
        joint = np.asarray(handle["embodiment/joint"][()])
        ee = np.asarray(handle["embodiment/ee"][()])
        phases = _decode_phases(handle["collection/phase"][()])

    frame_count = len(steps)
    if frame_count < 2:
        raise ExpertTrajectoryError("Expert trajectory must contain at least two frames")
    if len(joint) != frame_count or len(ee) != frame_count or len(phases) != frame_count:
        raise ExpertTrajectoryError("Expert trajectory datasets have inconsistent lengths")
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ExpertTrajectoryError(f"Expected joint trajectory Nx9+, got {joint.shape}")
    if ee.ndim != 2 or ee.shape[1] < 7:
        raise ExpertTrajectoryError(f"Expected EEF trajectory Nx7+, got {ee.shape}")
    if not np.isfinite(joint).all() or not np.isfinite(ee).all():
        raise ExpertTrajectoryError("Expert trajectory contains non-finite robot state")
    steps = np.asarray(steps).reshape(-1)
    step_deltas = np.diff(steps)
    if not np.all(step_deltas > 0):
        raise ExpertTrajectoryError("Expert trajectory simulation steps must increase strictly")
    if phases[0] != "pre_move":
        raise ExpertTrajectoryError("The first expert frame must be the ungrasped pre_move phase")
    transitions = [
        index
        for index in range(1, frame_count)
        if phases[index] != phases[index - 1]
    ]
    if len(transitions) != 1 or phases[transitions[0]] != "task":
        raise ExpertTrajectoryError(
            "Expert trajectory must have exactly one pre_move -> task transition"
        )
    transition = transitions[0]
    if set(phases) != {"pre_move", "task"}:
        raise ExpertTrajectoryError(f"Unexpected collection phases: {sorted(set(phases))}")

    qpos_delta = np.diff(joint[:, :8], axis=0)
    ee_translation_delta = np.diff(ee[:, :3], axis=0)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "frame_count": frame_count,
        "first_sim_step": int(steps[0]),
        "last_sim_step": int(steps[-1]),
        "max_recorded_sim_step_gap": int(step_deltas.max()),
        "pre_move_frame_count": transition,
        "task_frame_count": frame_count - transition,
        "task_phase_start_frame": transition,
        "max_recorded_qpos_step": float(np.linalg.norm(qpos_delta, axis=1).max()),
        "max_recorded_ee_translation_step_m": float(
            np.linalg.norm(ee_translation_delta, axis=1).max()
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertTrajectoryError(f"Could not read replay report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpertTrajectoryError(f"Replay report is not an object: {path}")
    return value


def _ffprobe_video(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ExpertTrajectoryError(f"Replay video does not exist: {path}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        beside_python = Path(sys.executable).resolve().parent / "ffprobe"
        ffprobe = str(beside_python) if beside_python.is_file() else None
    if ffprobe is None:
        raise ExpertTrajectoryError("ffprobe is required to validate replay video")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ExpertTrajectoryError(f"Expected one replay video stream: {path}")
    stream = streams[0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise ExpertTrajectoryError(f"Replay video is not H.264/yuv420p: {stream}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        **stream,
    }


def build_fixed_expert_manifest(
    *,
    task: str,
    seed: int,
    source_hdf5: Path,
    source_video: Path,
    replay_report_path: Path,
) -> dict[str, Any]:
    """Freeze one positive expert trajectory only after independent replay success."""

    trajectory = inspect_expert_hdf5(source_hdf5)
    replay_report_path = replay_report_path.resolve()
    replay = _read_json(replay_report_path)
    if replay.get("schema_version") != "univtac.fixed_expert_replay.v1":
        raise ExpertTrajectoryError("Replay report has an unsupported schema")
    if replay.get("task") != task or replay.get("seed") != seed:
        raise ExpertTrajectoryError("Replay report task/seed does not match the source")
    replay_source = replay.get("source_hdf5")
    if not isinstance(replay_source, str) or Path(replay_source).resolve() != source_hdf5.resolve():
        raise ExpertTrajectoryError("Replay report refers to a different source HDF5")
    replay_trajectory = replay.get("source_trajectory")
    if not isinstance(replay_trajectory, dict):
        raise ExpertTrajectoryError("Replay report omitted source trajectory provenance")
    if replay_trajectory.get("sha256") != trajectory["sha256"]:
        raise ExpertTrajectoryError("Replay source HDF5 hash does not match the frozen source")
    if replay.get("trajectory_includes_pre_move") is not True:
        raise ExpertTrajectoryError("Replay did not start from the ungrasped condition")
    if replay.get("privileged_pre_move_executed_before_replay") is not False:
        raise ExpertTrajectoryError("Replay executed privileged pre_move before qpos playback")
    if replay.get("execution_succeeded") is not True:
        raise ExpertTrajectoryError("Replay could not execute the complete recorded trajectory")
    if replay.get("official_task_success") is not True:
        raise ExpertTrajectoryError("Replay checker did not verify task success")
    if replay.get("timing_mode") != "recorded_step_interpolation":
        raise ExpertTrajectoryError("Replay did not preserve recorded simulation timing")
    if replay.get("stride") != 1:
        raise ExpertTrajectoryError("Replay skipped recorded expert frames")
    if replay.get("source_frame_count") != trajectory["frame_count"]:
        raise ExpertTrajectoryError("Replay source frame count does not match the HDF5")
    if replay.get("applied_frame_count") != trajectory["frame_count"]:
        raise ExpertTrajectoryError("Replay did not apply every recorded expert frame")
    expected_physics_actions = (
        trajectory["last_sim_step"] - trajectory["first_sim_step"] + 1
    )
    if replay.get("physics_action_count") != expected_physics_actions:
        raise ExpertTrajectoryError("Replay did not reconstruct every recorded physics step")
    evaluator_checks = replay.get("evaluator_checks")
    if not isinstance(evaluator_checks, dict):
        raise ExpertTrajectoryError("Replay report omitted evaluator checks")
    if evaluator_checks.get("base_task_success") is not True:
        raise ExpertTrajectoryError("Replay did not pass the task's base checker")
    if evaluator_checks.get("settle_steps", 0) < 60:
        raise ExpertTrajectoryError("Replay stability window was shorter than 60 steps")
    if task == "put_bottle_in_shelf":
        if evaluator_checks.get("released") is not True:
            raise ExpertTrajectoryError("Bottle replay did not verify gripper release")
        if evaluator_checks.get("stable_after_release") is not True:
            raise ExpertTrajectoryError("Bottle replay was not stable after release")
    replay_video = replay.get("replay_video")
    if not isinstance(replay_video, str):
        raise ExpertTrajectoryError("Replay report omitted its video path")

    return {
        "schema_version": "univtac.fixed_expert_trajectory.v1",
        "task": task,
        "seed": int(seed),
        "positive_example": True,
        "agent_visible_outcome": "successful expert demonstration",
        "agent_visible_checker_details": False,
        "source": {
            "trajectory": trajectory,
            "collection_video": _ffprobe_video(source_video),
        },
        "replay_verification": {
            "report": {
                "path": str(replay_report_path),
                "sha256": file_sha256(replay_report_path),
            },
            "video": _ffprobe_video(Path(replay_video)),
            "official_task_success": True,
            "evaluator_checks": replay.get("evaluator_checks"),
        },
        "icl_projection_policy": {
            "allowed": ["public observations selected by the active profile", "step_eef-compatible actions"],
            "forbidden": ["actor poses", "planner targets", "contact points", "IK/joint internals", "checker details"],
        },
    }
