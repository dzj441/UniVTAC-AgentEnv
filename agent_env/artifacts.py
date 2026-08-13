"""Media and audit-artifact helpers shared by AgentEnv simulator runners."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def tensor_to_rgb(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB tensor, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgb(path: Path, array: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def marker_grid_health(array: np.ndarray) -> dict[str, Any]:
    """Reject the broken non-grid projection observed in early trials."""

    dark_mask = (array.astype(np.float32).mean(axis=2) < 45.0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    plausible = areas[(areas >= 25) & (areas <= 400)]
    return {
        "expected_markers": 63,
        "plausible_marker_components": int(len(plausible)),
        "all_dark_components": int(component_count - 1),
        "dark_pixel_count": int(dark_mask.sum()),
        "healthy": bool(len(plausible) >= 40),
    }


def save_composite(
    panels: list[tuple[str, np.ndarray]],
    path: Path,
    *,
    columns: int = 2,
) -> dict[str, str]:
    if not panels:
        raise ValueError("At least one panel is required")
    panel_width, panel_height, label_height = 480, 270, 24
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB",
        (panel_width * columns, (panel_height + label_height) * rows),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        column, row = index % columns, index // columns
        x = column * panel_width
        y = row * (panel_height + label_height)
        draw.text((x + 6, y + 5), label, fill="white")
        image = Image.fromarray(array, mode="RGB").resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _media_executable(name: str) -> str:
    on_path = shutil.which(name)
    if on_path is not None:
        return on_path
    beside_python = Path(sys.executable).resolve().parent / name
    if beside_python.is_file():
        return str(beside_python)
    raise FileNotFoundError(f"Could not find {name!r} on PATH or beside {sys.executable}")


def _h264_encoder_args(ffmpeg: str) -> list[str]:
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-profile:v", "main", "-crf", "23"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-profile:v", "main", "-b:v", "2M"]
    raise RuntimeError("No H.264 encoder is available in the active ffmpeg build")


def encode_observation_video(observation_root: Path, video_path: Path) -> dict[str, Any]:
    frame_pattern = str(observation_root / "obs_*" / "composite.png")
    ffmpeg = _media_executable("ffmpeg")
    ffprobe = _media_executable("ffprobe")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-pattern_type",
            "glob",
            "-i",
            frame_pattern,
            *_h264_encoder_args(ffmpeg),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
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
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"Unexpected replay video format: {stream}")
    return {"path": str(video_path.resolve()), "sha256": file_sha256(video_path), **stream}
