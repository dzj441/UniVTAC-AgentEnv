"""Evaluator-private continuous sensor recording around simulator actions."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifacts import (
    _h264_encoder_args,
    _media_executable,
    file_sha256,
)


SIM_STEP_VIDEO_NAME = "sim_step_composite_h264.mp4"
SIM_STEP_FRAME_INDEX_NAME = "sim_step_frames.jsonl"
SIM_STEP_MANIFEST_NAME = "sim_step_recorder_manifest.json"
SIM_STEP_RECORDER_SCHEMA = "univtac.sim_step_recorder.v1"
SIM_STEP_FRAME_SCHEMA = "univtac.sim_step_recorder_frame.v1"
DEFAULT_FRAMES_PER_SECOND = 10.0
DEFAULT_POST_ACTION_SETTLE_STEPS = 60
PANEL_ORDER = (
    "head_rgb",
    "wrist_rgb",
    "left_tactile_rgb",
    "right_tactile_rgb",
)


class FrameWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class SimStepRecorderConfig:
    enabled: bool = True
    frames_per_second: float = DEFAULT_FRAMES_PER_SECOND
    post_action_record_steps: int = DEFAULT_POST_ACTION_SETTLE_STEPS

    def __post_init__(self) -> None:
        if not np.isfinite(self.frames_per_second) or self.frames_per_second <= 0:
            raise ValueError("sim-step recorder FPS must be a positive finite number")
        if (
            isinstance(self.post_action_record_steps, bool)
            or not isinstance(self.post_action_record_steps, (int, np.integer))
            or self.post_action_record_steps < 0
        ):
            raise ValueError("sim-step recorder post-action steps must be a non-negative integer")

    def resolve(self, physics_dt: float) -> dict[str, Any]:
        if not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be a positive finite number")
        interval = max(1, int(round(1.0 / (physics_dt * self.frames_per_second))))
        post_action_steps = int(self.post_action_record_steps)
        return {
            "enabled": bool(self.enabled),
            "agent_visible": False,
            "scope": "step_eef_control_selected_post_action_window_and_terminal_settle",
            "sensor_modalities": list(PANEL_ORDER),
            "physics_dt_seconds": float(physics_dt),
            "requested_frames_per_second": float(self.frames_per_second),
            "sample_interval_physics_steps": interval,
            "encoded_frames_per_second": float(1.0 / (interval * physics_dt)),
            "post_action_record_physics_steps": post_action_steps,
            "post_action_record_seconds": float(post_action_steps * physics_dt),
            "agent_thinking_time_recorded": False,
            "changes_agent_visible_observation_schema": False,
        }


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Recorder panel must be HWC RGB, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(
        np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        .clip(0, 255)
        .astype(np.uint8)
    )


def compose_sim_step_frame(
    panels: dict[str, np.ndarray],
    *,
    header_lines: tuple[str, str],
) -> np.ndarray:
    """Compose head, wrist, and bilateral tactile RGB into one audit frame."""

    missing = [name for name in PANEL_ORDER if name not in panels]
    if missing:
        raise ValueError(f"Recorder panels are missing: {missing}")
    panel_width, panel_height = 480, 270
    header_height, label_height = 60, 24
    canvas = Image.new(
        "RGB",
        (panel_width * 2, header_height + (panel_height + label_height) * 2),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_height), fill=(18, 24, 31))
    draw.text((10, 6), header_lines[0], fill=(235, 241, 246), font=_font(17))
    draw.text((10, 32), header_lines[1], fill=(154, 205, 255), font=_font(14))
    for index, name in enumerate(PANEL_ORDER):
        column, row = index % 2, index // 2
        x = column * panel_width
        y = header_height + row * (panel_height + label_height)
        draw.text((x + 7, y + 5), name, fill="white", font=_font(13))
        image = Image.fromarray(_rgb_array(panels[name]), mode="RGB").resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (x, y + label_height))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


class H264FrameWriter:
    """Stream fixed-size RGB frames to a browser-compatible H.264 MP4."""

    def __init__(self, video_path: Path, *, frames_per_second: float) -> None:
        self.video_path = video_path
        self.frames_per_second = float(frames_per_second)
        self._process: subprocess.Popen[bytes] | None = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._frame_count = 0

    def _start(self, frame: np.ndarray) -> None:
        height, width, channels = frame.shape
        if channels != 3 or width % 2 or height % 2:
            raise ValueError(
                "H.264 recorder frames must be even-sized HWC RGB arrays; "
                f"got {frame.shape}"
            )
        ffmpeg = _media_executable("ffmpeg")
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{self.frames_per_second:.12g}",
                "-i",
                "-",
                *_h264_encoder_args(ffmpeg),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.video_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._frame_shape = frame.shape

    def write(self, frame: np.ndarray) -> None:
        frame = _rgb_array(frame)
        if self._process is None:
            self._start(frame)
        if frame.shape != self._frame_shape:
            raise ValueError(
                f"Recorder frame shape changed from {self._frame_shape} to {frame.shape}"
            )
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("H.264 recorder process is unavailable")
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            message = ""
            if self._process.stderr is not None:
                message = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"H.264 recorder pipe failed: {message}") from exc
        self._frame_count += 1

    def close(self) -> dict[str, Any] | None:
        if self._process is None:
            return None
        process, self._process = self._process, None
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"H.264 recorder exited with status {return_code}: {stderr.strip()}"
            )
        ffprobe = _media_executable("ffprobe")
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,pix_fmt,width,height,nb_frames",
                "-of",
                "json",
                str(self.video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise RuntimeError(f"Unexpected simulator recorder video format: {stream}")
        return {
            "path": self.video_path.name,
            "sha256": file_sha256(self.video_path),
            "frames_per_second": self.frames_per_second,
            "recorded_frame_count": self._frame_count,
            **stream,
        }


class SimulationStepRecorder:
    """Record only simulator-active action windows, excluding Agent think time."""

    def __init__(
        self,
        *,
        run_dir: Path,
        physics_dt: float,
        config: SimStepRecorderConfig,
        frame_supplier: Callable[[], dict[str, np.ndarray]],
        writer: FrameWriter | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config.resolve(physics_dt)
        self.frame_supplier = frame_supplier
        self.writer = writer or H264FrameWriter(
            self.run_dir / SIM_STEP_VIDEO_NAME,
            frames_per_second=self.config["encoded_frames_per_second"],
        )
        self.frame_index_path = self.run_dir / SIM_STEP_FRAME_INDEX_NAME
        self.manifest_path = self.run_dir / SIM_STEP_MANIFEST_NAME
        self._active_segment: dict[str, Any] | None = None
        self._active_phase = "inactive"
        self._next_capture_step: int | None = None
        self._last_capture_step: int | None = None
        self._post_action_record_until_step: int | None = None
        self._segments: list[dict[str, Any]] = []
        self._frame_count = 0
        self._error: dict[str, str] | None = None
        self._final_receipt: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"] and self._error is None)

    @property
    def post_action_record_steps(self) -> int:
        return int(self.config["post_action_record_physics_steps"])

    def contract_manifest(self) -> dict[str, Any]:
        return dict(self.config)

    def _action_header(self, segment: dict[str, Any]) -> str:
        if segment["kind"] == "finish_settle":
            return f"finish_episode | terminal settle | obs {segment['observation_id']}"
        action = segment["action"]
        dp = ",".join(f"{value:+.3f}" for value in action["delta_position_world_m"])
        dr = ",".join(f"{value:+.3f}" for value in action["delta_rpy_world_rad"])
        return (
            f"step_eef {segment['action_index']:02d} | obs {segment['prior_observation_id']} "
            f"| dp[{dp}] dr[{dr}] dg{action['delta_gripper_m']:+.4f}"
        )

    def _capture(self, sim_step: int, *, force: bool = False) -> None:
        if not self.enabled or self._active_segment is None:
            return
        if (
            self._active_phase == "post_action_settle"
            and self._post_action_record_until_step is not None
            and sim_step > self._post_action_record_until_step
        ):
            return
        if not force and self._next_capture_step is not None and sim_step < self._next_capture_step:
            return
        if force and self._last_capture_step == sim_step:
            return
        try:
            panels = self.frame_supplier()
            segment = self._active_segment
            frame = compose_sim_step_frame(
                panels,
                header_lines=(
                    self._action_header(segment),
                    (
                        f"phase {self._active_phase} | sim step {sim_step} | "
                        f"sim time {sim_step * self.config['physics_dt_seconds']:.3f} s"
                    ),
                ),
            )
            self.writer.write(frame)
            record = {
                "schema_version": SIM_STEP_FRAME_SCHEMA,
                "video_frame_index": self._frame_count,
                "video_time_seconds": (
                    self._frame_count / self.config["encoded_frames_per_second"]
                ),
                "segment_index": segment["segment_index"],
                "segment_kind": segment["kind"],
                "action_index": segment.get("action_index"),
                "prior_observation_id": segment.get("prior_observation_id"),
                "phase": self._active_phase,
                "sim_step": int(sim_step),
                "sim_time_seconds": float(sim_step * self.config["physics_dt_seconds"]),
            }
            with self.frame_index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._frame_count += 1
            self._last_capture_step = int(sim_step)
            interval = int(self.config["sample_interval_physics_steps"])
            self._next_capture_step = int(sim_step) + interval
        except Exception as exc:
            self._error = {"error_type": type(exc).__name__, "message": str(exc)}

    def _begin_segment(self, segment: dict[str, Any], *, sim_step: int) -> None:
        if not self.enabled:
            return
        if self._active_segment is not None:
            raise RuntimeError("A simulator recorder segment is already active")
        segment = {
            "segment_index": len(self._segments),
            "start_sim_step": int(sim_step),
            "start_frame_index": self._frame_count,
            **segment,
        }
        self._active_segment = segment
        self._active_phase = "pre_action" if segment["kind"] == "step_eef" else "terminal_settle"
        self._next_capture_step = int(sim_step)
        self._last_capture_step = None
        self._post_action_record_until_step = None
        self._capture(sim_step, force=True)

    def begin_step_eef(
        self,
        *,
        action_index: int,
        prior_observation_id: str,
        action: dict[str, Any],
        sim_step: int,
    ) -> None:
        self._begin_segment(
            {
                "kind": "step_eef",
                "action_index": int(action_index),
                "prior_observation_id": prior_observation_id,
                "action": action,
            },
            sim_step=sim_step,
        )
        if self._active_segment is not None:
            self._active_phase = "control"

    def on_sim_step(self, sim_step: int) -> None:
        sim_step = int(sim_step)
        force = bool(
            self._active_phase == "post_action_settle"
            and self._post_action_record_until_step == sim_step
        )
        self._capture(sim_step, force=force)

    def begin_post_action_settle(self, *, sim_step: int) -> None:
        if self._active_segment is not None:
            # Preserve the exact control/post-action boundary even when the
            # regular sampling cadence did not land on this simulator step.
            self._capture(int(sim_step), force=True)
            self._active_segment["control_end_sim_step"] = int(sim_step)
            self._post_action_record_until_step = (
                int(sim_step) + self.post_action_record_steps
            )
            self._active_segment["post_action_record_end_sim_step"] = (
                self._post_action_record_until_step
            )
            self._active_phase = "post_action_settle"

    def end_step_eef(
        self,
        *,
        sim_step: int,
        execution_succeeded: bool,
        control_route: str | None,
        control_wall_seconds: float,
    ) -> None:
        if self._active_segment is None:
            return
        self._capture(int(sim_step), force=True)
        self._active_segment.update(
            {
                "end_sim_step": int(sim_step),
                "end_frame_index_exclusive": self._frame_count,
                "post_action_settle_physics_steps_executed": max(
                    0,
                    int(sim_step)
                    - int(self._active_segment.get("control_end_sim_step", sim_step)),
                ),
                "post_action_record_physics_steps_configured": (
                    self.post_action_record_steps
                ),
                "execution_succeeded": bool(execution_succeeded),
                "control_route": control_route,
                "control_wall_seconds": float(control_wall_seconds),
            }
        )
        self._segments.append(self._active_segment)
        self._active_segment = None
        self._active_phase = "inactive"
        self._next_capture_step = None
        self._post_action_record_until_step = None

    def abort_step_eef(
        self,
        *,
        sim_step: int,
        error: BaseException,
        control_wall_seconds: float,
    ) -> None:
        """Close an active action segment after an unexpected bridge error."""

        if self._active_segment is None:
            return
        self._capture(int(sim_step), force=True)
        self._active_segment.update(
            {
                "end_sim_step": int(sim_step),
                "end_frame_index_exclusive": self._frame_count,
                "aborted": True,
                "abort_error_type": type(error).__name__,
                "control_wall_seconds": float(control_wall_seconds),
            }
        )
        self._segments.append(self._active_segment)
        self._active_segment = None
        self._active_phase = "inactive"
        self._next_capture_step = None
        self._post_action_record_until_step = None

    def begin_terminal_settle(self, *, observation_id: str, sim_step: int) -> None:
        self._begin_segment(
            {
                "kind": "finish_settle",
                "observation_id": observation_id,
            },
            sim_step=sim_step,
        )

    def end_terminal_settle(self, *, sim_step: int) -> None:
        if self._active_segment is None:
            return
        self._capture(int(sim_step), force=True)
        self._active_segment.update(
            {
                "end_sim_step": int(sim_step),
                "end_frame_index_exclusive": self._frame_count,
            }
        )
        self._segments.append(self._active_segment)
        self._active_segment = None
        self._active_phase = "inactive"
        self._next_capture_step = None
        self._post_action_record_until_step = None

    def finalize(self) -> dict[str, Any]:
        if self._final_receipt is not None:
            return self._final_receipt
        if self._active_segment is not None:
            self._active_segment.update(
                {
                    "end_sim_step": self._last_capture_step,
                    "end_frame_index_exclusive": self._frame_count,
                    "aborted": True,
                }
            )
            self._segments.append(self._active_segment)
            self._active_segment = None
        video: dict[str, Any] | None = None
        try:
            video = self.writer.close()
        except Exception as exc:
            if self._error is None:
                self._error = {"error_type": type(exc).__name__, "message": str(exc)}
        manifest = {
            "schema_version": SIM_STEP_RECORDER_SCHEMA,
            "config": self.contract_manifest(),
            "frame_count": self._frame_count,
            "frame_index": (
                {
                    "path": self.frame_index_path.name,
                    "sha256": file_sha256(self.frame_index_path),
                }
                if self.frame_index_path.is_file()
                else None
            ),
            "segments": self._segments,
            "video": video,
            "error": self._error,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._final_receipt = {
            "manifest": {
                "path": self.manifest_path.name,
                "sha256": file_sha256(self.manifest_path),
            },
            "frame_count": self._frame_count,
            "video": video,
            "error": self._error,
        }
        return self._final_receipt
